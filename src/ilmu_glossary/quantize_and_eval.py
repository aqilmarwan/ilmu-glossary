"""Quantize and measure tier 1 in a single container, without exporting.

Why this exists: ModelOpt cannot export Nemotron's fused MoE experts
(`NotImplementedError: MoE model with experts type 'QuantNemotronHExperts' is
not supported in export`) and `save_pretrained` fails on its 0-dim amax
tensors under transformers 5.x. A quantized Nemotron checkpoint therefore
cannot be persisted or served with this stack.

The primary metric does not require one. This loads BF16 once, captures the
reference logits, quantizes **in place**, captures the quantized logits, and
computes the BM-EN KL delta - all before the model leaves memory.

That is also a better measurement than the served path:

  * one tokenizer, one process, so nothing drifts between reference and
    candidate;
  * real vocabulary ids rather than the hash surrogates the completions API
    forces, so the KL support is exact;
  * no dependence on a checkpoint that cannot currently be written.

What it cannot do is tier 4e, which genuinely needs a served checkpoint.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from ilmu_glossary import tracking
from ilmu_glossary.config import CalibVariant, Config, RecipeFamily
from ilmu_glossary.evaluate import inprocess
from ilmu_glossary.io import read_jsonl, write_json, write_parquet
from ilmu_glossary.quantize import (
    assert_exclusions_respected,
    assert_experts_quantized,
    build_forward_loop,
    build_quant_config,
    load_calibration_texts,
    load_recipe,
)
from ilmu_glossary.seeds import seed_everything
from ilmu_glossary.splits import load_splits, verify_all

logger = logging.getLogger(__name__)


def run_quantize_and_evaluate(
    cfg: Config,
    *,
    variant: str,
    sample_count: int,
    family: str,
) -> dict[str, Any]:
    """PTQ one cell and measure its tier-1 KL against BF16, in one process."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    variant_enum = CalibVariant(variant)
    family_enum = RecipeFamily(family)
    seed_everything(cfg.seed, "phase3eval", family, variant, sample_count)

    tag = f"{family}_{variant}_{sample_count}"
    results_dir = cfg.paths.resolve("results")
    eval_dir = cfg.paths.resolve("eval")
    recipe = load_recipe(cfg, family_enum)

    # Spec section 3: assert disjointness before evaluating.
    splits = load_splits(cfg.paths.resolve("splits"))
    verify_all(splits, expected_fraction=cfg.data.train_fraction)

    pairs = list(
        read_jsonl(
            cfg.paths.resolve("stratified") / "parallel_bm_en.jsonl",
            limit=cfg.eval.kl_max_pairs,
        )
    )
    if not pairs:
        raise RuntimeError("No parallel pairs; run phase 0 first")

    repo = cfg.effective_model_repo()
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=cfg.model.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = load_calibration_texts(cfg, variant_enum, sample_count)

    with tracking.run(
        cfg,
        phase="phase3eval",
        run_name=tag,
        tags={
            "family": family,
            "variant": variant,
            "contaminated": variant_enum.is_contaminated,
            "measured_by": "inprocess",
        },
    ):
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=cfg.model.trust_remote_code,
        )
        model.eval()  # type: ignore[no-untyped-call]  # transformers is untyped

        # ---- BF16 reference, before anything is touched --------------------
        started = time.perf_counter()
        reference = inprocess.capture_pairs(
            model,
            tokenizer,
            pairs,
            top_k=cfg.eval.kl_top_k,
            max_tokens=cfg.eval.kl_max_tokens_per_side,
            label="bf16 reference",
        )
        reference_s = time.perf_counter() - started

        # ---- quantize in place ---------------------------------------------
        import modelopt.torch.quantization as mtq

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = mtq.quantize(
            model,
            build_quant_config(recipe),
            forward_loop=build_forward_loop(
                texts,
                tokenizer,
                seq_len=cfg.effective_seq_len(),
                batch_size=cfg.quant.calib_batch_size,
                device=str(model.device),
            ),
        )
        quantize_s = time.perf_counter() - started

        carriers, quantized = assert_experts_quantized(model, family=family)
        violations = assert_exclusions_respected(model, recipe, strict=not cfg.dry_run)

        # ---- quantized capture ---------------------------------------------
        started = time.perf_counter()
        candidate = inprocess.capture_pairs(
            model,
            tokenizer,
            pairs,
            top_k=cfg.eval.kl_top_k,
            max_tokens=cfg.eval.kl_max_tokens_per_side,
            label=f"quantized {tag}",
        )
        candidate_s = time.perf_counter() - started

        row = inprocess.compare(
            reference,
            candidate,
            cfg=cfg,
            checkpoint=tag,
            variant=variant,
            contaminated=variant_enum.is_contaminated,
        )
        row.update(
            {
                "family": family,
                "sample_count": sample_count,
                "recipe_hash": recipe.identity_hash,
                "expert_carriers": carriers,
                "experts_quantized": quantized,
                "exclusion_violations": violations,
                "reference_capture_s": reference_s,
                "quantize_s": quantize_s,
                "candidate_capture_s": candidate_s,
                "peak_memory_gb": (
                    torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                ),
            }
        )

        tracking.log_metrics(
            {
                k: float(v)
                for k, v in row.items()
                if isinstance(v, int | float) and not isinstance(v, bool)
            }
        )

    write_parquet(
        pd.DataFrame([row]),
        eval_dir / f"{tag}_kl.parquet",
        fingerprint=cfg.fingerprint(),
        phase="phase3eval",
    )
    write_json(row, results_dir / f"{tag}_inprocess.json")

    logger.info(
        "%s: BM KL %.6f, EN KL %.6f, BM-EN delta %.6f (excludes zero: %s)",
        tag,
        row.get("bm_kl_mean", float("nan")),
        row.get("en_kl_mean", float("nan")),
        row.get("bm_en_delta", float("nan")),
        bool(row.get("delta_excludes_zero", 0.0)),
    )
    return row


__all__ = ["run_quantize_and_evaluate"]
