"""Profiling surface: a separate GPU job, plus the tools to read its traces.

Adapted from https://modal.com/docs/examples/torch_profiling. The example's
`profile()` wrapper calls its target `steps` times inside one profiler, which
works because its target runs in milliseconds. A phase-3 cell here runs for
hours on a 30B model, so two things change:

  * the workload reports its own step boundaries, one per calibration batch
    or KL pair (see `ilmu_glossary.profiling`);
  * profiling runs as its **own** job over a bounded sample, never inside a
    phase that writes artifacts - phase 3 times itself around the code a
    profiler would instrument, so an in-band profiler would silently inflate
    `wall_clock_s` and `peak_memory_gb` in `quantization_runs.parquet`.

Nothing in `modal_app/app.py` references this module. That is deliberate: it
makes "no study artifact was produced by an instrumented process" structural
rather than a convention.

    make modal-profile                      # trace a dry-run cell, download it
    make modal-tensorboard                  # serve TensorBoard over the traces

    uv run modal run modal_app/profiling.py::profile_phase --no-dry-run
    uv run modal volume ls  ilmu-glossary-traces
    uv run modal volume get ilmu-glossary-traces <path> ./traces/

Traces are `*.pt.trace.json`: drag one onto https://ui.perfetto.dev, or use
the TensorBoard app below.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import modal

from modal_app.app import HOURS, SECRETS, VOLUMES, _load_config, app
from modal_app.images import PYTHON_VERSION, cpu_image, ptq_image

logger = logging.getLogger(__name__)

# torch.profiler traces. Deliberately not /vol: a trace is debugging exhaust,
# not a study artifact, and one can run to hundreds of megabytes. Keeping it
# off the results Volume means a profiling session cannot perturb the
# provenance of anything under results/.
traces = modal.Volume.from_name("ilmu-glossary-traces", create_if_missing=True)
TRACE_DIR = "/traces"

# Returning a trace through the function boundary is convenient but not free.
# Past this size, `modal volume get` is the better route and the entrypoint
# says so rather than silently moving hundreds of megabytes.
MAX_INLINE_TRACE_BYTES = 64 * 1024 * 1024


@app.function(
    image=ptq_image,
    gpu="B200",
    volumes={**VOLUMES, TRACE_DIR: traces},
    secrets=SECRETS,
    # Model load plus a handful of sampled batches. Nowhere near phase 3's 12
    # hours, and a profiling session that overruns this is measuring the wrong
    # thing.
    timeout=2 * HOURS,
)
def profile_gpu(
    config_yaml: str | None = None,
    dry_run: bool = False,
    variant: str = "baseline_en",
    sample_count: int = 1024,
    family: str = "w4a16_shipped",
    calib_batches: int = 8,
    kl_pairs: int = 8,
) -> dict[str, Any]:
    """Trace a bounded sample of the PTQ and capture loops. Writes no results.

    This is the only place `profiling.enabled` is ever turned on.
    """
    from ilmu_glossary.profile_run import run_profile

    cfg = _load_config(config_yaml, dry_run)
    cfg = cfg.model_copy(
        update={
            "profiling": cfg.profiling.model_copy(
                update={"enabled": True, "trace_dir": Path(TRACE_DIR)}
            )
        }
    )

    result = run_profile(
        cfg,
        variant=variant,
        sample_count=sample_count,
        family=family,
        calib_batches=calib_batches,
        kl_pairs=kl_pairs,
    )
    # Only the traces Volume is committed. The results Volume is mounted
    # read-through for the corpus and calibration sets, and nothing on this
    # path writes to it.
    traces.commit()
    return result


@app.function(image=cpu_image, volumes={TRACE_DIR: traces}, timeout=15 * 60)
def list_traces() -> list[dict[str, Any]]:
    """Every trace on the Volume, newest first."""
    traces.reload()
    root = Path(TRACE_DIR)
    found = [
        {
            "path": str(path.relative_to(root)),
            "size_mb": round(path.stat().st_size / 1e6, 2),
            "mtime": path.stat().st_mtime,
        }
        for path in root.glob("**/*.pt.trace.json*")
    ]
    return sorted(found, key=lambda row: row["mtime"], reverse=True)


@app.function(image=cpu_image, volumes={TRACE_DIR: traces}, timeout=15 * 60)
def read_trace(relative_path: str | None = None) -> tuple[str, bytes]:
    """Read one trace off the Volume. Defaults to the newest.

    Returns (path, bytes) rather than text: `tensorboard_trace_handler` gzips
    its output when the profiler is built with compression, so the file is
    not always valid UTF-8.
    """
    traces.reload()
    root = Path(TRACE_DIR)

    if relative_path:
        target = root / relative_path
        if not target.is_file():
            raise FileNotFoundError(f"{relative_path} is not on the traces Volume")
    else:
        candidates = sorted(
            root.glob("**/*.pt.trace.json*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                "No traces on the Volume. Run `make modal-profile` first, and check "
                "the sample was long enough to pass profiling.wait + warmup + active steps."
            )
        target = candidates[0]

    size = target.stat().st_size
    if size > MAX_INLINE_TRACE_BYTES:
        raise ValueError(
            f"{target.relative_to(root)} is {size / 1e6:.0f} MB, above the "
            f"{MAX_INLINE_TRACE_BYTES / 1e6:.0f} MB inline limit. Fetch it with:\n"
            f"  uv run modal volume get ilmu-glossary-traces {target.relative_to(root)}"
        )
    return str(target.relative_to(root)), target.read_bytes()


# --------------------------------------------------------------------------
# TensorBoard, served off the traces Volume
# --------------------------------------------------------------------------
#
# Its own image: the PTQ image has no tensorboard, and adding one there would
# rebuild a multi-gigabyte CUDA image for a plotting dependency.

tb_image = modal.Image.debian_slim(python_version=PYTHON_VERSION).pip_install(
    "tensorboard==2.18.0", "torch-tb-profiler==0.4.3"
)


class VolumeMiddleware:
    """Reload the Volume on page load, so a trace written by a running job
    appears without restarting the container."""

    def __init__(self, wsgi_app: Any) -> None:
        self.wsgi_app = wsgi_app

    def __call__(self, environ: Any, start_response: Any) -> Any:
        if (route := environ.get("PATH_INFO")) in ("/", "/modal-volume-reload"):
            try:
                traces.reload()
            except Exception as exc:
                logger.warning("could not reload traces Volume: %s", exc)
            if route == "/modal-volume-reload":
                environ["PATH_INFO"] = "/"
        return self.wsgi_app(environ, start_response)


@app.function(
    image=tb_image,
    volumes={TRACE_DIR: traces},
    max_containers=1,
    scaledown_window=5 * 60,
)
@modal.concurrent(max_inputs=100)
@modal.wsgi_app()
def tensorboard() -> Any:
    import tensorboard

    board = tensorboard.program.TensorBoard()
    board.configure(logdir=TRACE_DIR)
    data_provider, deprecated_multiplexer = board._make_data_provider()
    wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
        board.flags,
        board.plugin_loaders,
        data_provider,
        board.assets_zip_provider,
        deprecated_multiplexer,
        experimental_middlewares=[VolumeMiddleware],
    )
    return wsgi_app._create_wsgi_app()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


# Not `main`: app.py already registers one on this App, and Modal rejects a
# duplicate entrypoint name outright.
@app.local_entrypoint()
def profile_phase(
    config: str = "configs/config.yaml",
    dry_run: bool = True,
    variant: str = "baseline_en",
    sample_count: int = 1024,
    family: str = "w4a16_shipped",
    calib_batches: int = 8,
    kl_pairs: int = 8,
    output_dir: str = "traces",
) -> None:
    """Profile a bounded sample on a B200, then download the newest trace.

    Defaults to `--dry-run`: a B200 hour is a real cost, and the dry run
    exercises the same code path on a small model, which is what a profiling
    change should be checked against before it is trusted on the 30B run.
    """
    config_yaml = Path(config).read_text() if Path(config).exists() else None

    print(f"profiling {calib_batches} calibration batches, {kl_pairs} pairs (dry_run={dry_run})")
    result = profile_gpu.remote(
        config_yaml=config_yaml,
        dry_run=dry_run,
        variant=variant,
        sample_count=sample_count,
        family=family,
        calib_batches=calib_batches,
        kl_pairs=kl_pairs,
    )
    print(result)

    available = list_traces.remote()
    if not available:
        print(
            "No trace was written. Each region needs at least "
            "profiling.wait + warmup + active steps; raise --calib-batches "
            "and --kl-pairs, or lower the schedule in the config."
        )
        return

    print(f"{len(available)} trace(s) on the Volume, newest first:")
    for row in available[:10]:
        print(f"  {row['size_mb']:>8.2f} MB  {row['path']}")

    try:
        relative, payload = read_trace.remote()
    except ValueError as exc:  # too large to inline
        print(exc)
        return

    local = Path(output_dir) / Path(relative).name
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(payload)
    print(f"\ntrace saved to {local}\nopen it at https://ui.perfetto.dev")
