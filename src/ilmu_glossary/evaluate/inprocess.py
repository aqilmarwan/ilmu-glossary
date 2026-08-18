"""In-process evaluation, without exporting or serving a checkpoint.

ModelOpt can quantize Nemotron's fused MoE experts but **cannot export them**:

    NotImplementedError: MoE model with experts type 'QuantNemotronHExperts'
    is not supported in export.

and `save_pretrained` fails separately under transformers 5.x on ModelOpt's
0-dim amax tensors. So there is no way to persist a quantized Nemotron
checkpoint with this stack, and therefore no way to serve one on vLLM.

The primary metric does not need one. Tier 1 compares BF16 logits against
quantized logits; both are obtainable from the model in memory. This module
scores directly against a HuggingFace model so the study can produce its
headline number while export remains blocked.

Two incidental advantages over the served path:

  * Token ids are real vocabulary indices rather than the hash surrogates the
    completions API forces, so the KL support is exact.
  * Reference and candidate are captured in the same process from the same
    tokenizer, removing every source of drift between them.

Only tier 4e (throughput) genuinely requires a servable checkpoint and stays
blocked.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np

from ilmu_glossary.evaluate import kl as kl_mod

logger = logging.getLogger(__name__)


def capture_topk(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    top_k: int,
    max_tokens: int,
    device: str | None = None,
) -> kl_mod.TopKLogprobs:
    """Top-K next-token logprobs for every position in `text`.

    Position i predicts token i+1, so the final position is dropped - it
    predicts nothing observed. The first position is kept: unlike the
    completions API, a forward pass does give a distribution there.
    """
    import torch

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    input_ids = encoded["input_ids"]
    if input_ids.shape[1] < 2:
        return kl_mod.TopKLogprobs(
            token_ids=np.zeros((0, top_k), dtype=np.int32),
            logprobs=np.zeros((0, top_k), dtype=np.float32),
        )

    target = device or str(getattr(model, "device", "cuda"))
    encoded = {k: v.to(target) for k, v in encoded.items()}

    with torch.inference_mode():
        logits = model(**encoded).logits[0]

    # Drop the last position; it predicts a token beyond the sequence.
    logits = logits[:-1].float()
    k = min(top_k, logits.shape[-1])
    logprobs = torch.log_softmax(logits, dim=-1)
    top = torch.topk(logprobs, k=k, dim=-1)

    return kl_mod.TopKLogprobs(
        token_ids=top.indices.to("cpu").numpy().astype(np.int32),
        logprobs=top.values.to("cpu").numpy().astype(np.float32),
    )


def sequence_nlls(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    max_tokens: int,
    device: str | None = None,
) -> list[float]:
    """Per-token negative log-likelihoods of the observed continuation.

    This is what perplexity aggregates. Computed here rather than derived from
    the top-K capture, because the observed token can fall outside the top-K.
    """
    import torch

    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = encoded["input_ids"]
    if input_ids.shape[1] < 2:
        return []

    target = device or str(getattr(model, "device", "cuda"))
    encoded = {k: v.to(target) for k, v in encoded.items()}

    with torch.inference_mode():
        logits = model(**encoded).logits[0].float()

    logprobs = torch.log_softmax(logits[:-1], dim=-1)
    observed = input_ids[0, 1:].to(logprobs.device)
    picked = logprobs.gather(-1, observed.unsqueeze(-1)).squeeze(-1)
    nlls: list[float] = (-picked).to("cpu").tolist()
    return nlls


def capture_pairs(
    model: Any,
    tokenizer: Any,
    pairs: Iterable[dict[str, Any]],
    *,
    top_k: int,
    max_tokens: int,
    label: str,
) -> dict[str, kl_mod.TopKLogprobs]:
    """Capture both sides of every parallel pair."""
    captured: dict[str, kl_mod.TopKLogprobs] = {}
    for i, pair in enumerate(pairs):
        for side in ("malay", "english"):
            text = pair.get(side, "")
            if not text.strip():
                continue
            captured[f"{pair['id']}::{side}"] = capture_topk(
                model, tokenizer, text, top_k=top_k, max_tokens=max_tokens
            )
        if i % 50 == 0 and i:
            logger.info("  %s: captured %d pairs", label, i)
    logger.info("%s: captured %d pair-sides", label, len(captured))
    return captured


def compare(
    reference: dict[str, kl_mod.TopKLogprobs],
    candidate: dict[str, kl_mod.TopKLogprobs],
    *,
    cfg: Any,
    checkpoint: str,
    variant: str,
    contaminated: bool,
) -> dict[str, Any]:
    """Tier-1 BM-EN KL delta from two in-process captures.

    Content and difficulty are held constant across each aligned pair, so a
    non-zero delta is attributable to language rather than register. This is
    the only tier that supports the causal claim.
    """
    sides: dict[str, list[float]] = {"malay": [], "english": []}
    tails: dict[str, list[float]] = {"malay": [], "english": []}

    for key, ref in reference.items():
        cand = candidate.get(key)
        if cand is None:
            continue
        side = key.rsplit("::", 1)[-1]
        if side not in sides:
            continue
        values, tail = kl_mod.token_kl(ref, cand)
        sides[side].extend(values.tolist())
        tails[side].extend(tail.tolist())

    summaries = {
        side: kl_mod.summarise_kl(
            np.array(values),
            np.array(tails[side]),
            percentiles=cfg.eval.kl_percentiles,
            bootstrap_resamples=cfg.eval.bootstrap_resamples,
            confidence_level=cfg.eval.confidence_level,
            seed=cfg.seed,
        )
        for side, values in sides.items()
    }

    delta = kl_mod.bm_en_delta(
        summaries["malay"],
        summaries["english"],
        malay_values=np.array(sides["malay"]),
        english_values=np.array(sides["english"]),
        bootstrap_resamples=cfg.eval.bootstrap_resamples,
        confidence_level=cfg.eval.confidence_level,
        seed=cfg.seed,
    )

    return {
        "checkpoint": checkpoint,
        "variant": variant,
        "contaminated": contaminated,
        "top_k": cfg.eval.kl_top_k,
        "n_pairs": len(reference) // 2,
        "measured_by": "inprocess",
        **{f"bm_{k}": v for k, v in summaries["malay"].items()},
        **{f"en_{k}": v for k, v in summaries["english"].items()},
        **delta,
    }


__all__ = ["capture_pairs", "capture_topk", "compare", "sequence_nlls"]
