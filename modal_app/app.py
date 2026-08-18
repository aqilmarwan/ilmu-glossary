"""Modal app - one remote function per phase.

Every function mounts the same persistent Volume at /vol, so phase N reads
exactly the artifacts phase N-1 wrote and nothing is recomputed. The Volume
also holds the MLflow file store, so `mlflow ui` pointed at a local sync of
/vol/mlruns shows every run.

Run the whole staged pipeline:

    modal run modal_app/app.py::run_all

Run one phase:

    modal run modal_app/app.py::phase0_data_prep
"""

from __future__ import annotations

import logging
from typing import Any

import modal

from modal_app.images import cpu_image, gpu_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# HF streaming issues one ranged GET per chunk and logs each at INFO, which
# buries our own progress lines under tens of thousands of signed URLs.
for _noisy in ("httpx", "httpcore", "urllib3", "filelock", "fsspec"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

APP_NAME = "ilmu-glossary"

app = modal.App(APP_NAME)

# Artifacts: stratified corpora, calibration sets, quantized checkpoints,
# results parquet, MLflow store. Survives across runs - this is what makes
# "no phase re-runs a previous phase" true in practice.
volume = modal.Volume.from_name("ilmu-glossary-vol", create_if_missing=True)

# Separate cache so a Volume reset does not force a re-download of the
# 60 GB BF16 checkpoint.
hf_cache = modal.Volume.from_name("ilmu-glossary-hf-cache", create_if_missing=True)

VOLUMES = {"/vol": volume, "/hf-cache": hf_cache}
SECRETS = [modal.Secret.from_name("huggingface-secret")]

HOURS = 60 * 60


def _csv(value: str | None) -> list[str] | None:
    """Parse a comma-separated CLI argument into a list.

    Modal function signatures double as the `modal run` CLI surface, and its
    parser rejects `list[str] | None` ("unparseable annotation"). Scalars in,
    lists parsed here.
    """
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_int(value: str | None) -> list[int] | None:
    parsed = _csv(value)
    return [int(item) for item in parsed] if parsed else None


def _load_config(config_yaml: str | None, dry_run: bool) -> Any:
    """Rebuild the Config inside the container.

    The config is passed as YAML text rather than a pickled object so that a
    container running an older image cannot silently deserialise into a
    different schema.
    """
    import tempfile
    from pathlib import Path

    from ilmu_glossary.config import Config

    if config_yaml is None:
        return Config(dry_run=dry_run)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(config_yaml)
        tmp = Path(fh.name)
    cfg = Config.load(tmp, dry_run=dry_run)
    tmp.unlink(missing_ok=True)
    return cfg


# --------------------------------------------------------------------------
# Phase 0 - data acquisition and stratification (CPU, network-bound)
# --------------------------------------------------------------------------


@app.function(
    image=cpu_image,
    volumes=VOLUMES,
    secrets=SECRETS,
    cpu=8.0,
    memory=32_768,
    timeout=8 * HOURS,
)
def phase0_data_prep(config_yaml: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    from ilmu_glossary.data_prep import run_phase0

    cfg = _load_config(config_yaml, dry_run)
    result = run_phase0(cfg)
    volume.commit()
    return result


# --------------------------------------------------------------------------
# Phase 1 - routing analysis (GPU, forward passes only, no generation)
# --------------------------------------------------------------------------


@app.function(
    image=gpu_image,
    gpu="B200",
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=6 * HOURS,
)
def phase1_routing(config_yaml: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    from ilmu_glossary.routing_analysis import run_phase1

    cfg = _load_config(config_yaml, dry_run)
    result = run_phase1(cfg)
    volume.commit()
    return result


# --------------------------------------------------------------------------
# Phase 2 - calibration set construction (CPU; greedy selection is the cost)
# --------------------------------------------------------------------------


@app.function(
    image=cpu_image,
    volumes=VOLUMES,
    secrets=SECRETS,
    cpu=16.0,
    memory=65_536,
    timeout=4 * HOURS,
)
def phase2_calib_select(
    config_yaml: str | None = None,
    dry_run: bool = False,
    variants: str | None = None,
    sample_counts: str | None = None,
) -> dict[str, Any]:
    """`variants` and `sample_counts` are comma-separated, e.g. "256,1024"."""
    from ilmu_glossary.calib_select import run_phase2

    cfg = _load_config(config_yaml, dry_run)
    result = run_phase2(cfg, variants=_csv(variants), sample_counts=_csv_int(sample_counts))
    volume.commit()
    return result


# --------------------------------------------------------------------------
# Phase 3 - quantization (GPU, the expensive one)
# --------------------------------------------------------------------------


@app.function(
    image=gpu_image,
    gpu="B200",
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=6 * HOURS,
)
def phase3_quantize(
    config_yaml: str | None = None,
    dry_run: bool = False,
    variant: str = "baseline_en",
    sample_count: int = 1024,
    family: str = "w4a16_shipped",
) -> dict[str, Any]:
    """One PTQ run. Fan out over the matrix with `.starmap`."""
    from ilmu_glossary.quantize import run_single_quantization

    cfg = _load_config(config_yaml, dry_run)
    result = run_single_quantization(cfg, variant=variant, sample_count=sample_count, family=family)
    volume.commit()
    return result


# --------------------------------------------------------------------------
# Phase 4 - evaluation (GPU, serves each checkpoint on vLLM)
# --------------------------------------------------------------------------


@app.function(
    image=gpu_image,
    gpu="B200",
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=6 * HOURS,
)
def phase4_evaluate(
    config_yaml: str | None = None,
    dry_run: bool = False,
    checkpoint: str = "bf16_reference",
    tiers: str | None = None,
) -> dict[str, Any]:
    """`tiers` is comma-separated, e.g. "kl,ppl". Omit to run all of them."""
    from ilmu_glossary.evaluate import run_phase4

    cfg = _load_config(config_yaml, dry_run)
    result = run_phase4(cfg, checkpoint=checkpoint, tiers=_csv(tiers))
    volume.commit()
    return result


# --------------------------------------------------------------------------
# Phase 5 - analysis and report (CPU)
# --------------------------------------------------------------------------


@app.function(
    image=cpu_image,
    volumes=VOLUMES,
    cpu=4.0,
    memory=16_384,
    timeout=1 * HOURS,
)
def phase5_analyze(config_yaml: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    from ilmu_glossary.analyze import run_phase5

    cfg = _load_config(config_yaml, dry_run)
    result = run_phase5(cfg)
    volume.commit()
    return result


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


@app.function(
    image=cpu_image,
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=24 * HOURS,
)
def run_all(config_yaml: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    """The staged pipeline, honouring the spec's decision gates.

    Order matters and is not negotiable:

      1. Phase 0, then phase 1.
      2. `oracle_contaminated` at N=1024 only, quantized and evaluated.
      3. Headroom gate. If oracle does not beat baseline_en by the configured
         margin, calibration is not the lever - the run stops and the null
         result is the finding (spec section 5).
      4. Only if the gate opens: the full variant x sample-count matrix.
      5. Mamba state arm on the winner, then analysis.
    """
    from ilmu_glossary.orchestrate import staged_pipeline

    cfg = _load_config(config_yaml, dry_run)
    return staged_pipeline(
        cfg,
        phase0=phase0_data_prep,
        phase1=phase1_routing,
        phase2=phase2_calib_select,
        phase3=phase3_quantize,
        phase4=phase4_evaluate,
        phase5=phase5_analyze,
    )


@app.local_entrypoint()
def main(config: str = "configs/config.yaml", dry_run: bool = False) -> None:
    """`modal run modal_app/app.py --config configs/config.yaml`."""
    from pathlib import Path

    config_yaml = Path(config).read_text() if Path(config).exists() else None
    result = run_all.remote(config_yaml=config_yaml, dry_run=dry_run)
    print(result)
