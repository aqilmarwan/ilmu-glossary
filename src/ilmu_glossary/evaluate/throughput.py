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
import re
import shutil
import subprocess
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
    artifact_root: Path,
    n_requests: int | None = None,
) -> ThroughputResult | None:
    """Drive one concurrency level through NVIDIA AIPerf."""
    if not aiperf_available():
        return None

    requests = n_requests or max(concurrency * 8, 32)
    # Persist artifacts rather than using a temp dir: when parsing fails the
    # only way to find out what AIPerf actually wrote is to look at it, and a
    # TemporaryDirectory deletes the evidence before anyone can.
    out_dir = artifact_root / f"{checkpoint_label}_c{concurrency}"
    out_dir.mkdir(parents=True, exist_ok=True)
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


# AIPerf exports metrics as .csv / .json / .jsonl into --artifact-dir. Key
# names have shifted between releases, so metrics are matched by substring
# rather than exact path.
_THROUGHPUT_KEYS = ("output_token_throughput", "request_throughput", "token_throughput")


def _parse_aiperf_output(artifact_dir: Path) -> dict[str, float] | None:
    """Pull metrics out of AIPerf's export, whichever format it used."""
    for path in sorted(artifact_dir.rglob("*.json")) + sorted(artifact_dir.rglob("*.jsonl")):
        if path.name.endswith("_raw.jsonl"):
            continue  # per-request records, not the summary
        try:
            text = path.read_text()
            payload = json.loads(text if path.suffix == ".json" else text.splitlines()[0])
        except (json.JSONDecodeError, OSError, IndexError):
            continue
        flat = _flatten_metrics(payload)
        if any(any(k in name for k in _THROUGHPUT_KEYS) for name in flat):
            return _canonicalise(flat)

    for path in sorted(artifact_dir.rglob("*.csv")):
        flat = _parse_csv_metrics(path)
        if flat:
            return _canonicalise(flat)

    found = sorted(p.name for p in artifact_dir.rglob("*") if p.is_file())
    logger.warning(
        "Could not locate AIPerf metrics in %s. Files present: %s. The "
        "artifacts are kept for inspection; extend _THROUGHPUT_KEYS or "
        "_parse_csv_metrics once the layout is known.",
        artifact_dir,
        found or "(none - AIPerf wrote nothing)",
    )
    return None


def _parse_csv_metrics(path: Path) -> dict[str, float]:
    """AIPerf's CSV export: a Metric column plus statistic columns."""
    import csv

    out: dict[str, float] = {}
    try:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                name = (row.get("Metric") or row.get("metric") or "").strip().lower()
                if not name:
                    continue
                slug = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
                for column, value in row.items():
                    if column is None or column.lower() in {"metric", ""}:
                        continue
                    try:
                        out[f"{slug}_{column.strip().lower()}"] = float(str(value).replace(",", ""))
                    except (TypeError, ValueError):
                        continue
    except OSError:
        return {}
    return out


def _canonicalise(flat: dict[str, float]) -> dict[str, float]:
    """Map whatever names AIPerf used onto the ones ThroughputResult reads."""
    aliases = {
        "output_token_throughput": ("output_token_throughput", "output_token_throughput_avg"),
        "request_throughput": ("request_throughput", "request_throughput_avg"),
        "time_to_first_token_avg": ("time_to_first_token_avg", "time_to_first_token_mean"),
        "time_to_first_token_p99": ("time_to_first_token_p99",),
        "inter_token_latency_avg": ("inter_token_latency_avg", "inter_token_latency_mean"),
    }
    out = dict(flat)
    for canonical, candidates in aliases.items():
        if canonical in out:
            continue
        for name, value in flat.items():
            if any(name.endswith(c) or name == c for c in candidates):
                out[canonical] = value
                break
    return out


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


def _warm_up(cfg: Config, handle: Any, artifact_root: Path) -> None:
    """Absorb one-off startup costs before anything is measured.

    A freshly launched container pays FlashInfer cubin downloads and Triton
    JIT on its first requests. Measured on the dry run, the first checkpoint
    benchmarked came out 51-61% slower than an identical model measured later
    with warm caches - enough to swamp any real throughput difference and to
    trip the regression check on nothing.
    """
    result = run_aiperf(
        cfg,
        handle,
        concurrency=max(cfg.eval.throughput_concurrencies),
        checkpoint_label="_warmup",
        artifact_root=artifact_root,
        n_requests=cfg.eval.throughput_warmup_requests * 2,
    )
    logger.info(
        "Throughput warm-up complete (%s); its numbers are discarded",
        "aiperf" if result else "fallback",
    )
    if result is None:
        run_fallback_benchmark(cfg, handle, concurrency=1, checkpoint_label="_warmup")


def benchmark_checkpoint(cfg: Config, handle: Any, checkpoint_label: str) -> pd.DataFrame:
    """Sweep every configured concurrency level for one served checkpoint."""
    artifact_root = cfg.paths.resolve("results") / "aiperf"
    _warm_up(cfg, handle, artifact_root)
    rows: list[dict[str, Any]] = []
    for concurrency in cfg.eval.throughput_concurrencies:
        result = run_aiperf(
            cfg,
            handle,
            concurrency=concurrency,
            checkpoint_label=checkpoint_label,
            artifact_root=artifact_root,
        )
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

    Caveat this tier carries: reference and checkpoint are measured in
    separate containers, and cross-container variance was large on the dry run
    even for an identical model. `_warm_up` removes the dominant cause, but a
    flag near the tolerance should be reproduced before being believed.
    """
    if df.empty or reference_checkpoint not in set(df["checkpoint"]):
        return df

    # Never difference an AIPerf measurement against a fallback one: AIPerf
    # reports streaming TTFT while the fallback reports whole-request latency,
    # so the two differ by orders of magnitude on the same server.
    if "measured_by" in df.columns and df["measured_by"].nunique() > 1:
        logger.warning(
            "Throughput rows were produced by different instruments (%s). They "
            "measure different quantities and are not comparable, so no "
            "regression verdict is issued.",
            sorted(df["measured_by"].dropna().unique()),
        )
        return df.assign(throughput_vs_reference=float("nan"), regression_flagged=False)

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
                **{str(k): v for k, v in row.to_dict().items()},
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
