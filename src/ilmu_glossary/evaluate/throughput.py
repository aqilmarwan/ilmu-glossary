"""Tier 4e - throughput regression check, measured with NVIDIA AIPerf.

Spec section 4e: tokens/sec and TTFT at concurrency 1 / 8 / 32 / 64 for the
BF16 baseline, the shipped NVFP4 checkpoint, and the best recalibrated
variant. The purpose is to confirm recalibration costs no throughput - it
should not, since only scale values change and the kernels are identical.

A *negative* result here would be the interesting one: if a recalibrated
checkpoint were slower, something other than scale values changed, and the
accuracy comparison would be suspect.

AIPerf drives vLLM's OpenAI-compatible endpoint. A minimal in-process
fallback exists for when AIPerf is unavailable, clearly marked in the output
so the two are never silently mixed.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ilmu_glossary.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThroughputResult:
    checkpoint: str
    concurrency: int
    output_tokens_per_sec: float
    request_throughput: float
    ttft_mean_ms: float
    ttft_p99_ms: float
    itl_mean_ms: float
    n_requests: int
    measured_by: str
    mamba_state: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "concurrency": self.concurrency,
            "output_tokens_per_sec": self.output_tokens_per_sec,
            "request_throughput": self.request_throughput,
            "ttft_mean_ms": self.ttft_mean_ms,
            "ttft_p99_ms": self.ttft_p99_ms,
            "itl_mean_ms": self.itl_mean_ms,
            "n_requests": self.n_requests,
            "measured_by": self.measured_by,
            "mamba_state": self.mamba_state,
        }


def aiperf_available() -> bool:
    return shutil.which("aiperf") is not None


def run_aiperf(
    cfg: Config,
    handle: Any,
    *,
    concurrency: int,
    checkpoint_label: str,
    n_requests: int | None = None,
) -> ThroughputResult | None:
    """Drive one concurrency level through NVIDIA AIPerf."""
    if not aiperf_available():
        return None

    requests = n_requests or max(concurrency * 8, 32)
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        cmd = [
            "aiperf",
            "profile",
            "--model",
            handle.model_path,
            "--url",
            handle.base_url,
            "--endpoint-type",
            "completions",
            "--concurrency",
            str(concurrency),
            "--request-count",
            str(requests),
            "--warmup-request-count",
            str(cfg.eval.throughput_warmup_requests),
            "--synthetic-input-tokens-mean",
            str(cfg.eval.throughput_input_tokens),
            "--output-tokens-mean",
            str(cfg.eval.throughput_output_tokens),
            "--streaming",
            "--artifact-dir",
            str(out_dir),
        ]
        logger.info("AIPerf c=%d: %s", concurrency, " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
        except subprocess.CalledProcessError as exc:
            logger.error("AIPerf failed (c=%d): %s", concurrency, exc.stderr[-2000:])
            return None
        except subprocess.TimeoutExpired:
            logger.error("AIPerf timed out at concurrency %d", concurrency)
            return None

        parsed = _parse_aiperf_output(out_dir)

    if parsed is None:
        return None

    return ThroughputResult(
        checkpoint=checkpoint_label,
        concurrency=concurrency,
        output_tokens_per_sec=parsed.get("output_token_throughput", float("nan")),
        request_throughput=parsed.get("request_throughput", float("nan")),
        ttft_mean_ms=parsed.get("time_to_first_token_avg", float("nan")),
        ttft_p99_ms=parsed.get("time_to_first_token_p99", float("nan")),
        itl_mean_ms=parsed.get("inter_token_latency_avg", float("nan")),
        n_requests=requests,
        measured_by="aiperf",
        mamba_state=handle.mamba_state.value,
    )


def _parse_aiperf_output(artifact_dir: Path) -> dict[str, float] | None:
    """Pull metrics out of AIPerf's JSON export.

    The exact key layout has shifted between AIPerf releases, so the parser
    walks the structure rather than indexing fixed paths.
    """
    candidates = list(artifact_dir.rglob("*.json"))
    for path in candidates:
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        flat = _flatten_metrics(payload)
        if "output_token_throughput" in flat or "request_throughput" in flat:
            return flat
    logger.warning("Could not locate AIPerf metrics in %s", artifact_dir)
    return None


def _flatten_metrics(payload: Any, prefix: str = "") -> dict[str, float]:
    """Collapse AIPerf's nested metric structure into flat name -> value.

    Handles both `{"metric": {"avg": 1.0, "p99": 2.0}}` and
    `{"metric": 1.0}` shapes.
    """
    out: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            name = f"{prefix}_{key}" if prefix else str(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                out[name] = float(value)
            elif isinstance(value, dict):
                for stat in ("avg", "mean", "p50", "p90", "p99"):
                    if stat in value and isinstance(value[stat], int | float):
                        out[f"{name}_{stat}"] = float(value[stat])
                out.update(_flatten_metrics(value, name))
    return out


def run_fallback_benchmark(
    cfg: Config,
    handle: Any,
    *,
    concurrency: int,
    checkpoint_label: str,
) -> ThroughputResult:
    """Minimal in-process benchmark for when AIPerf is unavailable.

    Marked `measured_by="fallback"` in the output. Fallback and AIPerf numbers
    must not be compared with each other - they measure differently - so the
    column exists to make any such mix visible in the results table.
    """
    import concurrent.futures

    from ilmu_glossary.evaluate.server import post_json

    prompt = "Terangkan sejarah Malaysia secara ringkas. " * 40
    requests = max(concurrency * 4, 16)

    def one_request() -> tuple[float, float, int]:
        started = time.perf_counter()
        response = post_json(
            handle.completions_url,
            {
                "model": handle.model_path,
                "prompt": prompt,
                "max_tokens": cfg.eval.throughput_output_tokens,
                "temperature": 0.0,
            },
            timeout=600,
        )
        elapsed = time.perf_counter() - started
        tokens = int(response.get("usage", {}).get("completion_tokens", 0))
        return elapsed, elapsed, tokens

    # Warm up so kernel autotuning does not land in the measurement.
    for _ in range(min(cfg.eval.throughput_warmup_requests, 4)):
        try:
            one_request()
        except Exception as exc:
            logger.debug("warmup request failed: %r", exc)

    latencies: list[float] = []
    total_tokens = 0
    started = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one_request) for _ in range(requests)]
        for future in concurrent.futures.as_completed(futures):
            try:
                latency, _, tokens = future.result()
            except Exception as exc:
                logger.warning("benchmark request failed: %r", exc)
                continue
            latencies.append(latency)
            total_tokens += tokens

    wall = time.perf_counter() - started
    array = np.array(latencies) if latencies else np.array([float("nan")])

    return ThroughputResult(
        checkpoint=checkpoint_label,
        concurrency=concurrency,
        output_tokens_per_sec=total_tokens / wall if wall else float("nan"),
        request_throughput=len(latencies) / wall if wall else float("nan"),
        ttft_mean_ms=float(array.mean() * 1000),
        ttft_p99_ms=float(np.percentile(array, 99) * 1000),
        itl_mean_ms=float("nan"),
        n_requests=len(latencies),
        measured_by="fallback",
        mamba_state=handle.mamba_state.value,
    )


def benchmark_checkpoint(cfg: Config, handle: Any, checkpoint_label: str) -> pd.DataFrame:
    """Sweep every configured concurrency level for one served checkpoint."""
    rows: list[dict[str, Any]] = []
    for concurrency in cfg.eval.throughput_concurrencies:
        result = run_aiperf(cfg, handle, concurrency=concurrency, checkpoint_label=checkpoint_label)
        if result is None:
            logger.warning(
                "AIPerf unavailable or failed at c=%d; using the in-process "
                "fallback. Do not compare these numbers against AIPerf ones.",
                concurrency,
            )
            result = run_fallback_benchmark(
                cfg, handle, concurrency=concurrency, checkpoint_label=checkpoint_label
            )
        rows.append(result.to_row())
        logger.info(
            "%s c=%d: %.1f tok/s, TTFT %.0f ms (%s)",
            checkpoint_label,
            concurrency,
            result.output_tokens_per_sec,
            result.ttft_mean_ms,
            result.measured_by,
        )
    return pd.DataFrame(rows)


def regression_check(
    df: pd.DataFrame, *, reference_checkpoint: str, tolerance: float = 0.05
) -> pd.DataFrame:
    """Flag any recalibrated checkpoint whose throughput moved materially.

    Recalibration changes scale values, not kernels, so throughput should be
    flat. A flagged row means something else changed and the accuracy
    comparison for that checkpoint needs explaining before it is trusted.
    """
    if df.empty or reference_checkpoint not in set(df["checkpoint"]):
        return df

    reference = df[df["checkpoint"] == reference_checkpoint].set_index("concurrency")
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        base = reference["output_tokens_per_sec"].get(row["concurrency"])
        relative = (
            (row["output_tokens_per_sec"] - base) / base
            if base and np.isfinite(base) and base != 0
            else float("nan")
        )
        rows.append(
            {
                **row.to_dict(),
                "throughput_vs_reference": relative,
                "regression_flagged": bool(np.isfinite(relative) and abs(relative) > tolerance),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "ThroughputResult",
    "aiperf_available",
    "benchmark_checkpoint",
    "regression_check",
    "run_aiperf",
    "run_fallback_benchmark",
]
