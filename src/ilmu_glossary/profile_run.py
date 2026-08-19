"""Standalone performance profiling. Runs alone, writes only traces.

This does NOT run alongside a study phase, and no study phase can invoke it.
That separation is the whole point. `torch.profiler` works by installing
callbacks into the dispatcher of the process doing the work, so profiling a
real PTQ run would mean profiling *inside* it - and phase 3 measures its own
`wall_clock_s` and `peak_memory_gb` around exactly that code. A profiled run
would write inflated performance columns into `quantization_runs.parquet`
carrying the same config fingerprint as clean rows, with nothing marking them
non-comparable. Plausible numbers instead of an error.

So this job loads the model itself and samples a bounded slice of each hot
loop. That costs one extra container per profiling session, and buys a
guarantee that no artifact under `results/` was ever produced by an
instrumented process.

Partial sampling loses very little here. Calibration is a uniform loop: every
batch is one forward pass at the same `seq_len` over the same model with the
same quantizers observing, so batch 900 does the work batch 3 does. What
sampling cannot show is the tail - ModelOpt's amax collection after the loop,
and memory growth across a full 1,024-sample run.

Three regions, one trace each:

  bf16_capture       tier-1 logit capture before quantization
  calibration        the PTQ forward loop
  quantized_capture  the same capture with quantized kernels live

**The model this produces is scientifically worthless** - it is calibrated on
a handful of samples purely so the third region traces real quantized
kernels. It is never persisted, never evaluated, and no result may be quoted
from it.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ilmu_glossary import profiling
from ilmu_glossary.config import CalibVariant, Config, RecipeFamily
from ilmu_glossary.evaluate import inprocess
from ilmu_glossary.io import read_jsonl, write_json
from ilmu_glossary.quantize import (
    assert_experts_quantized,
    build_forward_loop,
    build_quant_config,
    load_calibration_texts,
    load_recipe,
)
from ilmu_glossary.seeds import seed_everything

logger = logging.getLogger(__name__)


def run_profile(
    cfg: Config,
    *,
    variant: str = "baseline_en",
    sample_count: int = 1024,
    family: str = "w4a16_shipped",
    calib_batches: int = 8,
    kl_pairs: int = 8,
) -> dict[str, Any]:
    """Trace a bounded sample of the PTQ and capture hot loops.

    `sample_count` selects which calibration set to read - the same one phase
    3 would use - but only the first `calib_batches` batches of it are run.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not cfg.profiling.enabled:
        raise ValueError(
            "run_profile called with profiling disabled; it would load a model "
            "and produce nothing. Enable cfg.profiling.enabled."
        )

    variant_enum = CalibVariant(variant)
    family_enum = RecipeFamily(family)
    seed_everything(cfg.seed, "profile", family, variant, sample_count)

    tag = f"{family}_{variant}_{sample_count}"
    recipe = load_recipe(cfg, family_enum)

    logger.warning(
        "PROFILING RUN - performance only. Calibrating on %d batches instead of "
        "%d samples; the resulting model is not a study artifact and nothing "
        "here is written under results/.",
        calib_batches,
        sample_count,
    )

    pairs = list(
        read_jsonl(cfg.paths.resolve("stratified") / "parallel_bm_en.jsonl", limit=kl_pairs)
    )
    if not pairs:
        raise RuntimeError("No parallel pairs; run phase 0 first")

    repo = cfg.effective_model_repo()
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=cfg.model.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Only the leading slice is ever forwarded. Reading the whole set first
    # keeps the sample identical to the one phase 3 would have started with.
    texts = load_calibration_texts(cfg, variant_enum, sample_count)
    sampled = texts[: max(calib_batches * cfg.quant.calib_batch_size, 1)]

    model = AutoModelForCausalLM.from_pretrained(
        repo,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=cfg.model.trust_remote_code,
    )
    model.eval()  # type: ignore[no-untyped-call]  # transformers is untyped

    timings: dict[str, float] = {}

    # ---- region 1: BF16 capture ------------------------------------------
    started = time.perf_counter()
    with profiling.torch_profile(cfg, f"{tag}_bf16_capture") as region:
        inprocess.capture_pairs(
            model,
            tokenizer,
            pairs,
            top_k=cfg.eval.kl_top_k,
            max_tokens=cfg.eval.kl_max_tokens_per_side,
            label="profile bf16",
        )
    timings["bf16_capture_s"] = time.perf_counter() - started
    traces = {"bf16_capture": _trace_of(region)}

    # ---- region 2: calibration -------------------------------------------
    import modelopt.torch.quantization as mtq

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with profiling.torch_profile(cfg, f"{tag}_calibration") as region:
        model = mtq.quantize(
            model,
            build_quant_config(recipe),
            forward_loop=build_forward_loop(
                sampled,
                tokenizer,
                seq_len=cfg.effective_seq_len(),
                batch_size=cfg.quant.calib_batch_size,
                device=str(model.device),
            ),
        )
    timings["calibration_s"] = time.perf_counter() - started
    traces["calibration"] = _trace_of(region)

    # Fatal, not a warning: a trace of unquantized BF16 kernels looks entirely
    # healthy while answering a different question than the one asked.
    carriers, quantized = assert_experts_quantized(model, family=family)

    # ---- region 3: quantized capture --------------------------------------
    started = time.perf_counter()
    with profiling.torch_profile(cfg, f"{tag}_quantized_capture") as region:
        inprocess.capture_pairs(
            model,
            tokenizer,
            pairs,
            top_k=cfg.eval.kl_top_k,
            max_tokens=cfg.eval.kl_max_tokens_per_side,
            label="profile quantized",
        )
    timings["quantized_capture_s"] = time.perf_counter() - started
    traces["quantized_capture"] = _trace_of(region)

    manifest = {
        "kind": "profile",
        "not_a_study_artifact": True,
        "tag": tag,
        "family": family,
        "variant": variant,
        "calibration_set_sample_count": sample_count,
        "calib_batches_run": len(sampled),
        "kl_pairs_run": len(pairs),
        "seq_len": cfg.effective_seq_len(),
        "expert_carriers": carriers,
        "experts_quantized": quantized,
        "recipe_hash": recipe.identity_hash,
        "model_repo": repo,
        "dry_run": cfg.dry_run,
        "peak_memory_gb": (
            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        ),
        "traces": traces,
        **timings,
    }

    # Beside the traces, never under results/ - these timings are measured
    # with a profiler attached and must not be mistaken for phase 3's.
    write_json(manifest, Path(cfg.profiling.trace_dir) / f"{tag}_profile.json")

    logger.info(
        "profiled %s: bf16 capture %.1fs, calibration %.1fs, quantized capture %.1fs",
        tag,
        timings["bf16_capture_s"],
        timings["calibration_s"],
        timings["quantized_capture_s"],
    )
    return manifest


def _trace_of(region: profiling.ProfiledRegion | None) -> str | None:
    """Trace path as a string, or None when the schedule wrote nothing."""
    if region is None or region.trace_path is None:
        return None
    return str(region.trace_path)


__all__ = ["run_profile"]
