"""Phase 4 - evaluation.

Four tiers, each controlling for something different. **Their roles are not
interchangeable** (spec section 7):

  Tier 1  KL on parallel BM/EN pairs   controls content/domain/difficulty
                                       -> carries the CAUSAL CLAIM
  Tier 2  PPL on held-out slices       controls distribution -> magnitude
  Tier 3  Cross-MMLU (parallel)        controls content/difficulty -> accuracy
  Tier 4  MalayMMLU (native)           controls nothing -> RELEVANCE argument
  Tier 4e Throughput via AIPerf        confirms recalibration is free

Only tier 1 supports the claim that quantization shifts Malay *because it is
Malay*. Tiers 2 and 4 differ in domain as well as language, so a difference
there is confounded; they answer "how much" and "does it show up on the
benchmark people cite", which are different questions.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ilmu_glossary import tracking
from ilmu_glossary.config import CalibVariant, Config, MambaStateDtype
from ilmu_glossary.evaluate import kl as kl_mod
from ilmu_glossary.evaluate import mmlu as mmlu_mod
from ilmu_glossary.evaluate import ppl as ppl_mod
from ilmu_glossary.evaluate import throughput as tp_mod
from ilmu_glossary.evaluate.server import ServerHandle, serve
from ilmu_glossary.io import read_jsonl, write_json, write_parquet
from ilmu_glossary.seeds import seed_everything
from ilmu_glossary.splits import load_splits

logger = logging.getLogger(__name__)

TIERS = ("kl", "ppl", "cross_mmlu", "malay_mmlu", "throughput")
REFERENCE_CHECKPOINT = "bf16_reference"


# --------------------------------------------------------------------------
# checkpoint resolution
# --------------------------------------------------------------------------


def resolve_checkpoint_path(cfg: Config, checkpoint: str) -> str:
    """Map a checkpoint label to a servable path or repo id."""
    if cfg.dry_run:
        # Every dry-run checkpoint resolves to the same small dense model, so
        # tier 1 compares it against itself. KL(model || itself) must come out
        # at ~0 and the BM-EN delta with it - a real correctness check on the
        # estimator, which a genuine quantized comparison could not provide
        # this cheaply.
        return cfg.dry_run_eval_model
    if checkpoint == REFERENCE_CHECKPOINT:
        return cfg.effective_model_repo()
    if checkpoint == "nvfp4_shipped":
        return cfg.model.nvfp4_reference_repo
    path = cfg.paths.resolve("checkpoints") / checkpoint
    if not path.exists():
        raise FileNotFoundError(f"No quantized checkpoint at {path}; run phase 3")
    return str(path)


def _variant_of(checkpoint: str) -> CalibVariant | None:
    """Recover the calibration variant from a checkpoint label.

    Labels are `{family}_{variant}_{n}`. Used to propagate the contamination
    flag into every results row, so `oracle_contaminated` carries its label
    through to the report without anyone having to remember.
    """
    for variant in CalibVariant:
        if f"_{variant.value}_" in f"_{checkpoint}_":
            return variant
    return None


# --------------------------------------------------------------------------
# tier 1
# --------------------------------------------------------------------------


def stable_token_id(piece: str) -> int:
    """Deterministic surrogate id for a token string.

    The completions API keys top-logprobs by token *string*, not vocabulary
    index, so the two sides of a KL comparison need a shared id space. Python's
    builtin `hash()` cannot supply one: string hashing is randomised per
    interpreter, and the BF16 reference cache is written in one Modal container
    and read in another, so surrogate ids would not survive the round trip and
    every reference lookup would miss.

    blake2b over the UTF-8 bytes is stable across processes, machines and
    runs. 31 bits over a ~150k vocabulary gives a collision probability around
    1e-5 per pair, which is immaterial against per-token KL averaged over
    millions of tokens.
    """
    digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def _capture_topk(
    handle: ServerHandle, text: str, top_k: int, max_tokens: int
) -> kl_mod.TopKLogprobs:
    """Score one text and pack its per-token top-K logprobs."""
    from ilmu_glossary.evaluate.server import completion_logprobs

    tokens = completion_logprobs(handle, text, top_k=top_k, max_tokens=0)[:max_tokens]
    if not tokens:
        return kl_mod.TopKLogprobs(
            token_ids=np.zeros((0, top_k), dtype=np.int32),
            logprobs=np.zeros((0, top_k), dtype=np.float32),
        )

    # Tokens with no distribution (the first prompt token under echo) carry no
    # information for a divergence and are dropped rather than contributing a
    # row of zeros, which would dilute every mean.
    scored = [t for t in tokens if isinstance(t.get("top_logprobs"), dict) and t["top_logprobs"]]
    if not scored:
        return kl_mod.TopKLogprobs(
            token_ids=np.zeros((0, top_k), dtype=np.int32),
            logprobs=np.zeros((0, top_k), dtype=np.float32),
        )

    ids = np.zeros((len(scored), top_k), dtype=np.int32)
    lps = np.full((len(scored), top_k), -1e9, dtype=np.float32)

    for i, token in enumerate(scored):
        pairs = sorted(token["top_logprobs"].items(), key=lambda kv: -kv[1])[:top_k]
        for j, (piece, logprob) in enumerate(pairs):
            ids[i, j] = np.int32(stable_token_id(piece))
            lps[i, j] = np.float32(logprob)

    return kl_mod.TopKLogprobs(token_ids=ids, logprobs=lps)


def run_tier1_kl(
    cfg: Config,
    handle: ServerHandle,
    reference_cache: dict[str, kl_mod.TopKLogprobs] | None,
    *,
    checkpoint: str,
) -> tuple[pd.DataFrame, dict[str, kl_mod.TopKLogprobs]]:
    """KL on parallel BM/EN pairs. The primary metric.

    When `reference_cache` is None this run *is* the BF16 reference and its
    logprobs are captured for later comparison. Otherwise KL is computed
    against the cached reference.
    """
    pairs_path = cfg.paths.resolve("stratified") / "parallel_bm_en.jsonl"
    pairs = list(read_jsonl(pairs_path, limit=cfg.eval.kl_max_pairs))
    if not pairs:
        raise RuntimeError(f"No parallel pairs at {pairs_path}; run phase 0")

    is_reference = reference_cache is None
    cache: dict[str, kl_mod.TopKLogprobs] = {}
    bm_values: list[float] = []
    en_values: list[float] = []
    bm_tails: list[float] = []
    en_tails: list[float] = []

    for i, pair in enumerate(pairs):
        for side, text in (("malay", pair["malay"]), ("english", pair["english"])):
            key = f"{pair['id']}::{side}"
            captured = _capture_topk(
                handle, text, cfg.eval.kl_top_k, cfg.eval.kl_max_tokens_per_side
            )
            if is_reference:
                cache[key] = captured
                continue

            reference = reference_cache.get(key) if reference_cache else None
            if reference is None:
                continue
            values, tails = kl_mod.token_kl(reference, captured)
            if side == "malay":
                bm_values.extend(values.tolist())
                bm_tails.extend(tails.tolist())
            else:
                en_values.extend(values.tolist())
                en_tails.extend(tails.tolist())

        if i % 200 == 0 and i:
            logger.info("  tier 1: %d/%d pairs", i, len(pairs))

    if is_reference:
        logger.info("Captured BF16 reference logprobs for %d pair-sides", len(cache))
        return pd.DataFrame(), cache

    bm_summary = kl_mod.summarise_kl(
        np.array(bm_values),
        np.array(bm_tails),
        percentiles=cfg.eval.kl_percentiles,
        bootstrap_resamples=cfg.eval.bootstrap_resamples,
        confidence_level=cfg.eval.confidence_level,
        seed=cfg.seed,
    )
    en_summary = kl_mod.summarise_kl(
        np.array(en_values),
        np.array(en_tails),
        percentiles=cfg.eval.kl_percentiles,
        bootstrap_resamples=cfg.eval.bootstrap_resamples,
        confidence_level=cfg.eval.confidence_level,
        seed=cfg.seed,
    )
    delta = kl_mod.bm_en_delta(
        bm_summary,
        en_summary,
        malay_values=np.array(bm_values),
        english_values=np.array(en_values),
        bootstrap_resamples=cfg.eval.bootstrap_resamples,
        confidence_level=cfg.eval.confidence_level,
        seed=cfg.seed,
    )

    variant = _variant_of(checkpoint)
    row = {
        "checkpoint": checkpoint,
        "variant": variant.value if variant else "",
        "contaminated": bool(variant and variant.is_contaminated),
        "top_k": cfg.eval.kl_top_k,
        "n_pairs": len(pairs),
        **{f"bm_{k}": v for k, v in bm_summary.items()},
        **{f"en_{k}": v for k, v in en_summary.items()},
        **delta,
    }
    return pd.DataFrame([row]), cache


# --------------------------------------------------------------------------
# phase driver
# --------------------------------------------------------------------------


def run_phase4(
    cfg: Config,
    *,
    checkpoint: str = REFERENCE_CHECKPOINT,
    tiers: list[str] | None = None,
    mamba_state: MambaStateDtype = MambaStateDtype.FP16_SR,
) -> dict[str, Any]:
    """Evaluate one checkpoint across the requested tiers."""
    seed_everything(cfg.seed, "phase4", checkpoint)
    fingerprint = cfg.fingerprint()
    eval_dir = cfg.paths.resolve("eval")
    wanted = tiers or list(TIERS)

    # Spec section 3: assert disjointness before each phase. Evaluation must
    # touch only the held-out 20%.
    splits = load_splits(cfg.paths.resolve("splits"))
    logger.info("Split guard: %d classes with persisted indices", len(splits))

    model_path = resolve_checkpoint_path(cfg, checkpoint)
    variant = _variant_of(checkpoint)
    summary: dict[str, Any] = {"checkpoint": checkpoint, "model_path": model_path, "tiers": {}}

    with (
        tracking.run(
            cfg,
            phase="phase4",
            run_name=f"{checkpoint}_{mamba_state.value}",
            tags={
                "checkpoint": checkpoint,
                "variant": variant.value if variant else "",
                "contaminated": bool(variant and variant.is_contaminated),
                "mamba_state": mamba_state.value,
            },
        ),
        serve(cfg, model_path, mamba_state=mamba_state) as handle,
    ):
        # ---------------------------------------------------------- tier 1
        if "kl" in wanted:
            cache_path = eval_dir / "bf16_reference_logprobs.npz"
            reference_cache = None
            if checkpoint != REFERENCE_CHECKPOINT:
                reference_cache = _load_reference_cache(cache_path)
                if reference_cache is None:
                    raise RuntimeError(
                        "Tier 1 needs BF16 reference logprobs. Evaluate "
                        f"{REFERENCE_CHECKPOINT} first."
                    )

            table, cache = run_tier1_kl(cfg, handle, reference_cache, checkpoint=checkpoint)
            if cache:
                _save_reference_cache(cache_path, cache)
            if not table.empty:
                write_parquet(
                    table,
                    eval_dir / f"{checkpoint}_kl.parquet",
                    fingerprint=fingerprint,
                    phase="phase4",
                )
                summary["tiers"]["kl"] = table.to_dict(orient="records")[0]
                tracking.log_metrics(
                    {
                        "bm_kl_mean": table["bm_kl_mean"].iloc[0],
                        "en_kl_mean": table["en_kl_mean"].iloc[0],
                        "bm_en_delta": table["bm_en_delta"].iloc[0],
                    }
                )

        # ---------------------------------------------------------- tier 2
        if "ppl" in wanted:
            results = {}
            for corpus_class in splits:
                if corpus_class == "parallel_bm_en":
                    continue
                texts = ppl_mod.load_held_out(cfg, corpus_class)
                nlls = ppl_mod.score_texts(handle, texts, stride=cfg.eval.ppl_stride)
                results[corpus_class] = ppl_mod.perplexity_from_nlls(nlls, corpus_class, len(texts))
            table = pd.DataFrame([r.to_row() for r in results.values()])
            table["checkpoint"] = checkpoint
            write_parquet(
                table,
                eval_dir / f"{checkpoint}_ppl.parquet",
                fingerprint=fingerprint,
                phase="phase4",
            )
            summary["tiers"]["ppl"] = table.to_dict(orient="records")

        # ---------------------------------------------------------- tier 3
        if "cross_mmlu" in wanted:
            by_language = mmlu_mod.load_cross_mmlu(cfg)
            frames = []
            for language, questions in by_language.items():
                frame = mmlu_mod.evaluate_questions(handle, questions, do_full_answer=False)
                frame["language"] = language
                frames.append(frame)
            if frames:
                combined = pd.concat(frames, ignore_index=True)
                combined["checkpoint"] = checkpoint
                write_parquet(
                    combined,
                    eval_dir / f"{checkpoint}_cross_mmlu.parquet",
                    fingerprint=fingerprint,
                    phase="phase4",
                )
                summary["tiers"]["cross_mmlu"] = mmlu_mod.accuracy_summary(
                    combined, group_by=["language"]
                ).to_dict(orient="records")

        # ---------------------------------------------------------- tier 4
        if "malay_mmlu" in wanted:
            questions = mmlu_mod.load_malay_mmlu(cfg)
            limit = 64 if cfg.dry_run else None
            frame = mmlu_mod.evaluate_questions(handle, questions, limit=limit)
            frame["checkpoint"] = checkpoint
            write_parquet(
                frame,
                eval_dir / f"{checkpoint}_malay_mmlu.parquet",
                fingerprint=fingerprint,
                phase="phase4",
            )
            summary["tiers"]["malay_mmlu"] = mmlu_mod.accuracy_summary(
                frame, group_by=["subject"]
            ).to_dict(orient="records")

        # --------------------------------------------------------- tier 4e
        if "throughput" in wanted:
            table = tp_mod.benchmark_checkpoint(cfg, handle, checkpoint)
            write_parquet(
                table,
                eval_dir / f"{checkpoint}_throughput.parquet",
                fingerprint=fingerprint,
                phase="phase4",
            )
            summary["tiers"]["throughput"] = table.to_dict(orient="records")

    write_json(summary, eval_dir / f"{checkpoint}_summary.json")
    return summary


def _save_reference_cache(path: Path, cache: dict[str, kl_mod.TopKLogprobs]) -> None:
    """Persist BF16 reference logprobs so later checkpoints reuse them.

    Without this every quantized checkpoint would re-serve BF16 to obtain the
    same reference, roughly doubling GPU time across the matrix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for key, value in cache.items():
        payload[f"{key}::ids"] = value.token_ids
        payload[f"{key}::lps"] = value.logprobs
    # savez_compressed's stub types its second positional as `bool`; the
    # **kwargs form is the documented API for named arrays.
    np.savez_compressed(str(path), **payload)  # type: ignore[arg-type]
    logger.info("Cached %d reference distributions to %s", len(cache), path)


def _load_reference_cache(path: Path) -> dict[str, kl_mod.TopKLogprobs] | None:
    if not path.exists():
        return None
    archive = np.load(path)
    keys = {name.rsplit("::", 1)[0] for name in archive.files}
    return {
        key: kl_mod.TopKLogprobs(token_ids=archive[f"{key}::ids"], logprobs=archive[f"{key}::lps"])
        for key in keys
        if f"{key}::ids" in archive.files and f"{key}::lps" in archive.files
    }


__all__ = [
    "REFERENCE_CHECKPOINT",
    "TIERS",
    "resolve_checkpoint_path",
    "run_phase4",
    "run_tier1_kl",
    "stable_token_id",
]
