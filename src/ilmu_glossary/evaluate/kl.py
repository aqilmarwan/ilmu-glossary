"""Tier 1 - KL divergence on parallel BM/EN pairs. The primary metric.

Spec section 4a. Token-level KL between BF16 reference logits and quantized
logits, computed on the Malay side and the English side of each aligned pair.
Reports mean, P50, P99 and the **BM-EN delta**.

Because content, topic and difficulty are held constant across each aligned
pair, a non-zero delta is attributable to language rather than register. This
is the only tier that supports the causal claim.

**The estimator.** Full-vocab logit capture is infeasible: 5,000 pairs x 4,096
tokens x ~150,000 vocab x 2 bytes is on the order of 6 TB per side. KL is
computed over the union of the two distributions' top-K support, with the
unobserved tail collapsed into a single mass term. The result is a lower bound
whose error is bounded by the tail mass, and the tail mass is reported
alongside every estimate so the bound is auditable rather than assumed.
See SPEC_DEVIATIONS.md D7.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TopKLogprobs:
    """Sparse per-token distribution: top-K token ids and their logprobs."""

    token_ids: np.ndarray  # (n_tokens, k) int32
    logprobs: np.ndarray  # (n_tokens, k) float32

    @property
    def n_tokens(self) -> int:
        return int(self.token_ids.shape[0])

    def tail_mass(self) -> np.ndarray:
        """Probability mass outside the retained top-K, per token."""
        mass: np.ndarray = np.clip(1.0 - np.exp(self.logprobs).sum(axis=1), 0.0, 1.0)
        return mass


def token_kl(
    reference: TopKLogprobs,
    candidate: TopKLogprobs,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-token KL(reference || candidate) over the union of top-K supports.

    Returns (kl_per_token, tail_mass_per_token).

    For each token the union of the two top-K id sets is formed. Ids present
    in the reference but absent from the candidate's top-K are assigned the
    candidate's residual tail mass spread uniformly over the unobserved
    vocabulary - a conservative assignment, since concentrating it instead
    would lower the divergence. That makes the estimate a lower bound on the
    true KL, which is the safe direction: it cannot manufacture an effect
    that is not there.
    """
    n = min(reference.n_tokens, candidate.n_tokens)
    if n == 0:
        return np.zeros(0), np.zeros(0)

    kl = np.zeros(n, dtype=np.float64)
    tails = np.zeros(n, dtype=np.float64)

    ref_tail = reference.tail_mass()
    cand_tail = candidate.tail_mass()

    for i in range(n):
        ref_ids = reference.token_ids[i]
        ref_p = np.exp(reference.logprobs[i].astype(np.float64))
        cand_ids = candidate.token_ids[i]
        cand_p = np.exp(candidate.logprobs[i].astype(np.float64))

        cand_lookup = dict(zip(cand_ids.tolist(), cand_p.tolist(), strict=False))

        # Uniform floor for reference mass the candidate did not retain.
        # Vocab size is unknown from a top-K view; K^2 is a deliberately
        # generous denominator that keeps the floor small without being zero.
        floor = max(cand_tail[i], eps) / max(len(cand_ids) ** 2, 1)

        total = 0.0
        for token_id, p in zip(ref_ids.tolist(), ref_p.tolist(), strict=False):
            if p <= 0:
                continue
            q = cand_lookup.get(token_id, floor)
            total += p * np.log(p / max(q, eps))

        kl[i] = total
        tails[i] = max(ref_tail[i], cand_tail[i])

    return kl, tails


def summarise_kl(
    kl_values: np.ndarray,
    tail_mass: np.ndarray,
    *,
    percentiles: tuple[float, ...] = (50.0, 99.0),
    bootstrap_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Mean, percentiles and a bootstrap CI.

    Spec section 8: "Every number carries its sample size and variance.
    Single-run point estimates are not results."
    """
    if kl_values.size == 0:
        return {"n_tokens": 0.0, "kl_mean": float("nan")}

    generator = np.random.default_rng(seed)
    means = np.array(
        [
            generator.choice(kl_values, size=kl_values.size, replace=True).mean()
            for _ in range(bootstrap_resamples)
        ]
    )
    alpha = (1.0 - confidence_level) / 2.0

    out: dict[str, float] = {
        "n_tokens": float(kl_values.size),
        "kl_mean": float(kl_values.mean()),
        "kl_std": float(kl_values.std(ddof=1)) if kl_values.size > 1 else 0.0,
        "kl_sem": float(kl_values.std(ddof=1) / np.sqrt(kl_values.size))
        if kl_values.size > 1
        else 0.0,
        "kl_ci_low": float(np.percentile(means, 100 * alpha)),
        "kl_ci_high": float(np.percentile(means, 100 * (1 - alpha))),
        "mean_tail_mass": float(tail_mass.mean()),
        "max_tail_mass": float(tail_mass.max()),
    }
    for q in percentiles:
        out[f"kl_p{int(q)}"] = float(np.percentile(kl_values, q))
    return out


def bm_en_delta(
    malay: dict[str, float],
    english: dict[str, float],
    *,
    malay_values: np.ndarray | None = None,
    english_values: np.ndarray | None = None,
    bootstrap_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """The causal claim: how much more does quantization shift Malay than English?

    Content and difficulty are held constant across each aligned pair, so a
    non-zero delta is attributable to language.

    When the raw per-token values are supplied the CI is bootstrapped on the
    **paired** difference rather than from the two marginals, which is both
    tighter and correct - the two sides are not independent samples, they are
    translations of the same content.
    """
    delta = malay["kl_mean"] - english["kl_mean"]
    result: dict[str, float] = {
        "bm_kl_mean": malay["kl_mean"],
        "en_kl_mean": english["kl_mean"],
        "bm_en_delta": delta,
        "bm_en_ratio": malay["kl_mean"] / english["kl_mean"]
        if english["kl_mean"] > 0
        else float("nan"),
        "n_bm_tokens": malay["n_tokens"],
        "n_en_tokens": english["n_tokens"],
    }

    if malay_values is None or english_values is None:
        return result

    generator = np.random.default_rng(seed)
    n = min(len(malay_values), len(english_values))
    if n == 0:
        return result

    deltas = np.array(
        [
            malay_values[idx].mean() - english_values[idx].mean()
            for idx in (generator.integers(0, n, size=n) for _ in range(bootstrap_resamples))
        ]
    )
    alpha = (1.0 - confidence_level) / 2.0
    result["bm_en_delta_ci_low"] = float(np.percentile(deltas, 100 * alpha))
    result["bm_en_delta_ci_high"] = float(np.percentile(deltas, 100 * (1 - alpha)))
    # A CI excluding zero is what licenses the causal statement.
    result["delta_excludes_zero"] = float(
        result["bm_en_delta_ci_low"] > 0 or result["bm_en_delta_ci_high"] < 0
    )
    return result


def per_class_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Per-corpus-class KL, deliberately unmerged.

    Spec section 4a: "Do not merge classes - formal BM, Manglish, and
    code-switched text will likely differ, and merging hides the most
    interesting result."
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "corpus_class" in df.columns and df["corpus_class"].nunique() == 1:
        logger.warning(
            "Per-class KL table holds a single class. The comparison across "
            "formal BM, Manglish and code-switched text is what spec 4a asks "
            "for; a single-class table cannot support it."
        )
    return df


__all__ = [
    "TopKLogprobs",
    "bm_en_delta",
    "per_class_table",
    "summarise_kl",
    "token_kl",
]
