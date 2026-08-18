"""Phase 3 - quantization.

Runs PTQ for every (family x variant x sample count) combination, changing
**only the calibration data source** within a family. Spec section 6: precision
assignments, block sizes and layer exclusions stay identical across variants,
or the comparison is confounded.

That requirement is enforced rather than trusted. `assert_recipe_identity`
hashes every key of a recipe except `calibration:` and refuses to run if two
variants in the same family disagree. The hash is written into
`quantization_runs.parquet`, so a reader can confirm after the fact that the
comparison was clean.

Two families run (see SPEC_DEVIATIONS.md D2):

  w4a16_shipped    NVIDIA's published recipe. Weight-only experts, so
                   calibration acts only through the static MSE search.
  w4a4_mechanism   Adds per-tensor expert activation scales, so amax
                   estimation genuinely occurs. NOT shipped by NVIDIA.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ilmu_glossary import tracking
from ilmu_glossary.config import CalibVariant, Config, RecipeFamily
from ilmu_glossary.io import read_jsonl, write_json, write_parquet
from ilmu_glossary.seeds import seed_everything

logger = logging.getLogger(__name__)

# Keys excluded from the identity hash - these are what a variant is allowed
# to change. Everything else must match.
MUTABLE_RECIPE_KEYS = frozenset({"calibration", "name", "description"})


class RecipeMismatchError(RuntimeError):
    """Raised when two variants in a family use materially different recipes."""


# --------------------------------------------------------------------------
# recipe handling
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    """A loaded PTQ recipe plus its identity hash."""

    family: RecipeFamily
    path: Path
    payload: dict[str, Any]
    identity_hash: str
    shipped_by_nvidia: bool

    @property
    def algorithm(self) -> str:
        return str(self.payload.get("quantization", {}).get("algorithm", "unknown"))

    @property
    def expert_activation_bits(self) -> int | None:
        """Activation bit width on routed experts.

        16 means weight-only and therefore no amax estimation on experts -
        the distinction that motivates running two families.
        """
        for layer in self.payload.get("quantization", {}).get("layers", []):
            if "experts" in str(layer.get("pattern", "")) and "shared" not in str(
                layer.get("pattern", "")
            ):
                bits = layer.get("activation_bits")
                return int(bits) if bits is not None else None
        return None


def recipe_identity_hash(payload: dict[str, Any]) -> str:
    """Hash everything a variant is NOT allowed to change.

    Canonical JSON with sorted keys, so formatting or key order in the yaml
    cannot change the hash without a real semantic change.
    """
    material = {k: v for k, v in payload.items() if k not in MUTABLE_RECIPE_KEYS}
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def load_recipe(cfg: Config, family: RecipeFamily) -> Recipe:
    path = Path(cfg.quant.recipe_dir) / f"{family.value}.yaml"
    if not path.exists():
        # Recipes live alongside the package when running inside a container.
        alt = Path(__file__).resolve().parents[2] / cfg.quant.recipe_dir / f"{family.value}.yaml"
        if not alt.exists():
            raise FileNotFoundError(f"Recipe not found at {path} or {alt}")
        path = alt

    payload = yaml.safe_load(path.read_text())
    return Recipe(
        family=family,
        path=path,
        payload=payload,
        identity_hash=recipe_identity_hash(payload),
        shipped_by_nvidia=bool(payload.get("shipped_by_nvidia", True)),
    )


def assert_recipe_identity(recipes: list[Recipe]) -> str:
    """Every recipe in a family must hash identically. Spec section 9.

    "Comparing variants with different recipes -> Only the data source
    changes; assert recipe hash identical."
    """
    if not recipes:
        raise ValueError("No recipes to compare")
    hashes = {r.identity_hash for r in recipes}
    if len(hashes) != 1:
        detail = "\n".join(f"  {r.path.name}: {r.identity_hash}" for r in recipes)
        raise RecipeMismatchError(
            "Variants within a family must differ only in their calibration "
            f"data source, but their recipes hash differently:\n{detail}\n"
            "Any accuracy difference measured under these recipes would be "
            "confounded by the precision assignments themselves."
        )
    return hashes.pop()


def materialise_recipe(
    recipe: Recipe,
    *,
    calibration_path: Path,
    num_samples: int,
    seq_len: int,
    batch_size: int,
    out_dir: Path,
) -> Path:
    """Write a run-specific recipe with the calibration source filled in.

    The identity hash is recomputed and asserted unchanged, so this function
    cannot accidentally alter a precision assignment.
    """
    payload = json.loads(json.dumps(recipe.payload, default=str))
    payload["calibration"] = {
        "dataset_path": str(calibration_path),
        "num_samples": num_samples,
        "seq_len": seq_len,
        "batch_size": batch_size,
    }

    rehashed = recipe_identity_hash(payload)
    if rehashed != recipe.identity_hash:
        raise RecipeMismatchError(
            f"Filling in the calibration source changed the recipe identity "
            f"({recipe.identity_hash} -> {rehashed}). This must never happen."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{recipe.family.value}_{calibration_path.stem}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, width=100))
    return path


# --------------------------------------------------------------------------
# calibration data
# --------------------------------------------------------------------------


def load_calibration_texts(cfg: Config, variant: CalibVariant, n: int) -> list[str]:
    """Read one calibration set, asserting it is chat-templated.

    Spec section 9: "Raw text instead of chat-templated calibration -> Pull
    template from tokenizer config; assert formatting before PTQ."
    """
    path = cfg.paths.resolve("calibration_sets") / f"{variant.value}_{n}.jsonl"
    records = list(read_jsonl(path))
    if not records:
        raise RuntimeError(f"Calibration set {path} is empty; run phase 2")

    texts = [r["text"] for r in records]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.effective_model_repo(), trust_remote_code=cfg.model.trust_remote_code
    )
    from ilmu_glossary.data_prep import assert_chat_templated

    assert_chat_templated(texts, tokenizer)
    logger.info("Loaded %d calibration samples from %s", len(texts), path.name)
    return texts


def build_forward_loop(
    texts: list[str],
    tokenizer: Any,
    *,
    seq_len: int,
    batch_size: int,
    device: str,
) -> Any:
    """Return the `forward_loop` ModelOpt calls to drive calibration.

    ModelOpt observes activations during these forward passes and computes
    scales from what it sees. This is the *only* place the calibration data
    enters the pipeline, which is what makes "only the data source changes"
    a meaningful claim.
    """
    import torch

    def forward_loop(model: Any) -> None:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=seq_len,
                    padding=True,
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                model(**encoded)
                if (start // max(batch_size, 1)) % 32 == 0:
                    logger.info("  calibration %d/%d", start, len(texts))

    return forward_loop


# --------------------------------------------------------------------------
# quantization
# --------------------------------------------------------------------------


@dataclass
class QuantizationRun:
    """Everything spec section 6 asks to be logged for one PTQ run."""

    family: str
    variant: str
    sample_count: int
    recipe_hash: str
    algorithm: str
    expert_activation_bits: int | None
    shipped_by_nvidia: bool
    checkpoint_path: str
    wall_clock_s: float = 0.0
    peak_memory_gb: float = 0.0
    n_quantized_modules: int = 0
    fallback_layers: list[str] = field(default_factory=list)
    contaminated: bool = False
    error: str | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "variant": self.variant,
            "sample_count": self.sample_count,
            "recipe_hash": self.recipe_hash,
            "algorithm": self.algorithm,
            "expert_activation_bits": self.expert_activation_bits,
            "shipped_by_nvidia": self.shipped_by_nvidia,
            "checkpoint_path": self.checkpoint_path,
            "wall_clock_s": self.wall_clock_s,
            "peak_memory_gb": self.peak_memory_gb,
            "n_quantized_modules": self.n_quantized_modules,
            "n_fallback_layers": len(self.fallback_layers),
            "fallback_layers": ";".join(self.fallback_layers[:32]),
            "contaminated": self.contaminated,
            "error": self.error,
        }


# ModelOpt config names by expert activation width, most preferred first.
# Verified against nvidia-modelopt 0.40.0 by enumerating `dir(mtq)` inside the
# GPU image - the names in NVIDIA's blog posts (W4A16_NVFP4_CFG) do not exist
# in the installed package.
_QUANT_CONFIG_PREFERENCE: dict[int, tuple[str, ...]] = {
    # Weight-only NVFP4: the shipped Lightning configuration. For an MoE the
    # experts *are* the MLP, so the MLP-scoped config is what reaches them.
    16: ("NVFP4_MLP_WEIGHT_ONLY_CFG", "NVFP4_MLP_ONLY_CFG"),
    # W4A4 NVFP4: weights and activations, so per-tensor amax estimation
    # genuinely occurs on the experts. This is the mechanism family.
    4: ("NVFP4_DEFAULT_CFG", "NVFP4_AFFINE_KV_CFG"),
}


def _resolve_quant_config(recipe: Recipe) -> Any:
    """Map a recipe to a ModelOpt quantization config.

    Falling back to a wrong config silently would collapse the two families
    into one and make the whole mechanism contrast vacuous, so an
    unresolvable width raises and names what the installed package offers.
    """
    import modelopt.torch.quantization as mtq

    bits = recipe.expert_activation_bits
    candidates = _QUANT_CONFIG_PREFERENCE.get(bits or -1)
    if candidates is None:
        raise ValueError(f"Unsupported expert activation width: {bits}")

    for name in candidates:
        config = getattr(mtq, name, None)
        if config is not None:
            logger.info("%s -> mtq.%s", recipe.family.value, name)
            return config

    available = sorted(n for n in dir(mtq) if n.endswith("_CFG"))
    raise RuntimeError(
        f"None of {candidates} exist in nvidia-modelopt {getattr(mtq, '__version__', '?')}. "
        f"The {recipe.family.value} family cannot run.\n"
        f"Available configs: {available}\n"
        "Either repin ModelOpt, update _QUANT_CONFIG_PREFERENCE, or drop the "
        "family and record the omission - do not substitute a config with a "
        "different activation width, which would make the two families identical."
    )


def run_single_quantization(
    cfg: Config,
    *,
    variant: str,
    sample_count: int,
    family: str,
) -> dict[str, Any]:
    """One PTQ run. Idempotent - returns immediately if the checkpoint exists."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    variant_enum = CalibVariant(variant)
    family_enum = RecipeFamily(family)
    seed_everything(cfg.seed, "phase3", family, variant, sample_count)

    tag = f"{family}_{variant}_{sample_count}"
    checkpoint_dir = cfg.paths.resolve("checkpoints") / tag
    results_dir = cfg.paths.resolve("results")

    recipe = load_recipe(cfg, family_enum)
    run = QuantizationRun(
        family=family,
        variant=variant,
        sample_count=sample_count,
        recipe_hash=recipe.identity_hash,
        algorithm=recipe.algorithm,
        expert_activation_bits=recipe.expert_activation_bits,
        shipped_by_nvidia=recipe.shipped_by_nvidia,
        checkpoint_path=str(checkpoint_dir),
        contaminated=variant_enum.is_contaminated,
    )

    if (checkpoint_dir / "config.json").exists():
        logger.info("%s already quantized; skipping", tag)
        return {"skipped": True, **run.to_row()}

    with tracking.run(
        cfg,
        phase="phase3",
        run_name=tag,
        tags={
            "family": family,
            "variant": variant,
            "contaminated": variant_enum.is_contaminated,
            "shipped_recipe": recipe.shipped_by_nvidia,
        },
    ):
        tracking.log_params(
            {
                "recipe_hash": recipe.identity_hash,
                "algorithm": recipe.algorithm,
                "expert_activation_bits": recipe.expert_activation_bits,
                "sample_count": sample_count,
                "seq_len": cfg.effective_seq_len(),
            }
        )

        texts = load_calibration_texts(cfg, variant_enum, sample_count)
        materialise_recipe(
            recipe,
            calibration_path=cfg.paths.resolve("calibration_sets")
            / f"{variant}_{sample_count}.jsonl",
            num_samples=sample_count,
            seq_len=cfg.effective_seq_len(),
            batch_size=cfg.quant.calib_batch_size,
            out_dir=cfg.paths.resolve("results") / "materialised_recipes",
        )

        repo = cfg.effective_model_repo()
        tokenizer = AutoTokenizer.from_pretrained(
            repo, trust_remote_code=cfg.model.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=cfg.model.trust_remote_code,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        started = time.perf_counter()
        try:
            import modelopt.torch.quantization as mtq

            quant_cfg = _resolve_quant_config(recipe)
            forward_loop = build_forward_loop(
                texts,
                tokenizer,
                seq_len=cfg.effective_seq_len(),
                batch_size=cfg.quant.calib_batch_size,
                device=str(model.device),
            )
            model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

            run.n_quantized_modules, run.fallback_layers = _inspect_quantized(model)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

        except Exception as exc:
            run.error = repr(exc)
            logger.exception("PTQ failed for %s", tag)
        finally:
            run.wall_clock_s = time.perf_counter() - started
            if torch.cuda.is_available():
                run.peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9

        tracking.log_metrics(
            {
                "wall_clock_s": run.wall_clock_s,
                "peak_memory_gb": run.peak_memory_gb,
                "n_quantized_modules": float(run.n_quantized_modules),
                "n_fallback_layers": float(len(run.fallback_layers)),
            }
        )

    row = run.to_row()
    _append_run_row(results_dir / "quantization_runs.parquet", row, cfg.fingerprint())
    write_json(row, checkpoint_dir / "quantization_run.json")
    logger.info(
        "%s: %.0fs, %.1f GB peak, %d modules quantized, %d fallbacks",
        tag,
        run.wall_clock_s,
        run.peak_memory_gb,
        run.n_quantized_modules,
        len(run.fallback_layers),
    )
    return row


def _inspect_quantized(model: Any) -> tuple[int, list[str]]:
    """Count quantized modules and identify layers that fell back.

    Spec section 6 requires logging "any layers that fell back to higher
    precision". A module carrying a quantizer whose amax was never populated
    saw no calibration tokens - which for a routed expert is exactly the
    coverage failure this study is about, so it is recorded, not ignored.
    """
    quantized = 0
    fallbacks: list[str] = []

    for name, module in model.named_modules():
        weight_quantizer = getattr(module, "weight_quantizer", None)
        if weight_quantizer is None:
            continue
        if getattr(weight_quantizer, "is_enabled", True):
            quantized += 1
        else:
            fallbacks.append(f"{name}:disabled")
            continue

        input_quantizer = getattr(module, "input_quantizer", None)
        if input_quantizer is not None and getattr(input_quantizer, "is_enabled", False):
            amax = getattr(input_quantizer, "amax", None)
            if amax is None:
                # An activation quantizer with no amax never saw a token.
                fallbacks.append(f"{name}:no_amax")

    return quantized, fallbacks


def _append_run_row(path: Path, row: dict[str, Any], fingerprint: str) -> None:
    """Append one run to the runs table, preserving earlier rows.

    Runs fan out across separate Modal containers, so each writes its own row
    into a shared table rather than the whole table being rewritten.
    """
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    # A rerun of the same cell supersedes the earlier attempt.
    combined = combined.drop_duplicates(subset=["family", "variant", "sample_count"], keep="last")
    write_parquet(combined, path, fingerprint=fingerprint, phase="phase3")


# --------------------------------------------------------------------------
# matrix driver
# --------------------------------------------------------------------------


def run_phase3(
    cfg: Config,
    *,
    families: list[str] | None = None,
    variants: list[str] | None = None,
    sample_counts: list[int] | None = None,
) -> dict[str, Any]:
    """Run the PTQ matrix locally (Modal fans this out instead)."""
    wanted_families = [RecipeFamily(f) for f in families] if families else list(cfg.quant.families)
    wanted_variants = (
        [CalibVariant(v) for v in variants] if variants else list(cfg.calibration.variants)
    )
    wanted_counts = sample_counts or list(cfg.effective_sample_counts())

    # Assert up front that each family's recipe is self-consistent, before any
    # GPU time is spent discovering otherwise.
    for family in wanted_families:
        recipe = load_recipe(cfg, family)
        assert_recipe_identity([recipe])
        logger.info(
            "%s: recipe hash %s, algorithm=%s, expert activations=%s bits, shipped=%s",
            family.value,
            recipe.identity_hash,
            recipe.algorithm,
            recipe.expert_activation_bits,
            recipe.shipped_by_nvidia,
        )

    rows = [
        run_single_quantization(cfg, variant=variant.value, sample_count=n, family=family.value)
        for family in wanted_families
        for variant in wanted_variants
        for n in wanted_counts
    ]
    return {"n_runs": len(rows), "runs": rows}


__all__ = [
    "MUTABLE_RECIPE_KEYS",
    "_QUANT_CONFIG_PREFERENCE",
    "QuantizationRun",
    "Recipe",
    "RecipeMismatchError",
    "assert_recipe_identity",
    "build_forward_loop",
    "load_calibration_texts",
    "load_recipe",
    "materialise_recipe",
    "recipe_identity_hash",
    "run_phase3",
    "run_single_quantization",
]
