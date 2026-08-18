"""MLflow tracking.

Parquet under results/ is authoritative - MLflow exists so that hourly-billed
B200 jobs can be watched while they run rather than only inspected afterwards.
Every function degrades to a no-op when tracking is disabled or MLflow is
absent, so no phase needs a conditional around its logging.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ilmu_glossary.config import Config

logger = logging.getLogger(__name__)


def _mlflow() -> Any | None:
    try:
        import mlflow
    except ImportError:
        logger.debug("mlflow not installed; tracking disabled")
        return None
    return mlflow


@contextmanager
def run(
    cfg: Config,
    *,
    phase: str,
    run_name: str,
    tags: Mapping[str, Any] | None = None,
    nested: bool = False,
) -> Iterator[None]:
    """Scope a phase or a single matrix cell as an MLflow run.

    Usage::

        with tracking.run(cfg, phase="phase3", run_name="bm_only_1024"):
            tracking.log_params({"variant": "bm_only", "n": 1024})
            ...
            tracking.log_metrics({"wall_clock_s": elapsed})
    """
    mlflow = _mlflow() if cfg.tracking.enabled else None
    if mlflow is None:
        yield
        return

    try:
        mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
        mlflow.set_experiment(cfg.tracking.experiment_name)
    except Exception as exc:
        logger.warning("MLflow unavailable (%r); continuing without tracking", exc)
        yield
        return

    base_tags = {
        "phase": phase,
        "config_fingerprint": cfg.fingerprint(),
        "seed": str(cfg.seed),
        "dry_run": str(cfg.dry_run),
        **{k: str(v) for k, v in (tags or {}).items()},
    }
    with mlflow.start_run(run_name=run_name, nested=nested, tags=base_tags):
        try:
            yield
        except Exception as exc:
            log_params({"failed": True, "error": type(exc).__name__})
            raise


def log_params(params: Mapping[str, Any]) -> None:
    mlflow = _mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    try:
        # MLflow truncates long values; keep them readable rather than silently cut.
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
    except Exception as exc:
        logger.debug("log_params failed: %r", exc)


def log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
    mlflow = _mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    numeric = {k: float(v) for k, v in metrics.items() if _is_finite_number(v)}
    if not numeric:
        return
    try:
        mlflow.log_metrics(numeric, step=step)
    except Exception as exc:
        logger.debug("log_metrics failed: %r", exc)


def log_artifact(path: Path, *, cfg: Config | None = None) -> None:
    """Only uploads when explicitly enabled - the volume already holds these."""
    if cfg is not None and not cfg.tracking.log_artifacts:
        return
    mlflow = _mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    try:
        mlflow.log_artifact(str(path))
    except Exception as exc:
        logger.debug("log_artifact failed: %r", exc)


def _is_finite_number(value: Any) -> bool:
    import math

    if isinstance(value, bool):
        return False
    if not isinstance(value, int | float):
        return False
    return math.isfinite(float(value))


__all__ = ["log_artifact", "log_metrics", "log_params", "run"]
