"""Tier 2 - perplexity delta on held-out corpus slices.

Spec section 4b. Held-out 20% slices per corpus class, never used in
calibration. Reports absolute PPL and delta versus BF16. Split disjointness is
asserted before running.

Tier 2 answers "how large is the damage", not "is it caused by language" -
these slices differ from each other in domain as well as language, so a
difference between formal_bm and english_control here is confounded. Tier 1
carries the causal claim; this tier carries magnitude.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ilmu_glossary.config import Config
from ilmu_glossary.io import read_jsonl
from ilmu_glossary.seeds import rng
from ilmu_glossary.splits import load_splits

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerplexityResult:
    corpus_class: str
    perplexity: float
    mean_nll: float
    nll_std: float
    n_documents: int
    n_tokens: int

    def to_row(self) -> dict[str, Any]:
        return {
            "corpus_class": self.corpus_class,
            "perplexity": self.perplexity,
            "mean_nll": self.mean_nll,
            "nll_std": self.nll_std,
            "n_documents": self.n_documents,
            "n_tokens": self.n_tokens,
        }


def load_held_out(cfg: Config, corpus_class: str) -> list[str]:
    """Documents from the evaluation 20% of one class.

    Raises rather than silently returning the whole class if the split is
    missing - a perplexity number computed on calibration data would be
    meaningless and would not look wrong in a table.
    """
    splits = load_splits(cfg.paths.resolve("splits"))
    if corpus_class not in splits:
        raise KeyError(f"No persisted split for {corpus_class}; run phase 0")

    path = cfg.paths.resolve("stratified") / f"{corpus_class}.jsonl"
    eval_indices = set(splits[corpus_class].eval_indices)

    texts = [
        record.get("text") or record.get("malay", "")
        for i, record in enumerate(read_jsonl(path))
        if i in eval_indices
    ]

    limit = cfg.eval.ppl_max_docs_per_class
    if len(texts) > limit:
        generator = rng(cfg.seed, "phase4", "ppl", corpus_class)
        idx = generator.choice(len(texts), size=limit, replace=False)
        texts = [texts[i] for i in sorted(idx.tolist())]

    logger.info("%s: %d held-out documents for perplexity", corpus_class, len(texts))
    return texts


def perplexity_from_nlls(
    nlls: list[float], corpus_class: str, n_documents: int
) -> PerplexityResult:
    """Aggregate per-token negative log-likelihoods into a perplexity.

    Perplexity is exp of the *token-weighted* mean NLL, so long documents
    contribute proportionally. Averaging per-document perplexities instead
    would over-weight short documents, which in the Manglish class are the
    majority.
    """
    if not nlls:
        return PerplexityResult(corpus_class, float("nan"), float("nan"), 0.0, n_documents, 0)

    array = np.asarray(nlls, dtype=np.float64)
    mean_nll = float(array.mean())
    return PerplexityResult(
        corpus_class=corpus_class,
        perplexity=float(math.exp(min(mean_nll, 70.0))),  # guard the exp overflow
        mean_nll=mean_nll,
        nll_std=float(array.std(ddof=1)) if array.size > 1 else 0.0,
        n_documents=n_documents,
        n_tokens=int(array.size),
    )


def score_texts(handle: Any, texts: list[str], *, stride: int) -> list[float]:
    """Per-token NLLs for a set of documents, via the served model."""
    from ilmu_glossary.evaluate.server import completion_logprobs

    nlls: list[float] = []
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        try:
            tokens = completion_logprobs(handle, text[: stride * 8], top_k=1, max_tokens=0)
        except Exception as exc:
            logger.warning("Scoring failed for document %d: %r", i, exc)
            continue
        # The first token has no conditional logprob.
        nlls.extend(-t["logprob"] for t in tokens if t["logprob"] is not None)
        if i % 200 == 0 and i:
            logger.info("  scored %d/%d documents", i, len(texts))
    return nlls


def delta_table(
    quantized: dict[str, PerplexityResult],
    reference: dict[str, PerplexityResult],
) -> pd.DataFrame:
    """PPL and delta versus the BF16 reference, per class."""
    rows: list[dict[str, Any]] = []
    for corpus_class, result in quantized.items():
        base = reference.get(corpus_class)
        row: dict[str, Any] = {**result.to_row()}
        if base is not None:
            row["reference_perplexity"] = base.perplexity
            row["ppl_delta"] = result.perplexity - base.perplexity
            row["ppl_delta_pct"] = (
                100.0 * (result.perplexity - base.perplexity) / base.perplexity
                if base.perplexity
                else float("nan")
            )
            row["nll_delta"] = result.mean_nll - base.mean_nll
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "PerplexityResult",
    "delta_table",
    "load_held_out",
    "perplexity_from_nlls",
    "score_texts",
]
