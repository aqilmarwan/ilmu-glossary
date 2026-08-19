"""Bounded `torch.profiler` instrumentation for the GPU phases.

Adapted from Modal's torch profiling example
(https://modal.com/docs/examples/torch_profiling), which profiles a function
by calling it `steps` times inside a `torch.profiler.schedule`:

    with torch.profiler.profile(schedule=..., on_trace_ready=...) as prof:
        for _ in range(steps):
            function.local(**kwargs)
            prof.step()

That shape does not transfer to this pipeline. One phase-3 cell downloads a
30B checkpoint, calibrates on 1,024 sequences of 32,768 tokens and captures
logits twice; re-running it three times to satisfy a schedule is not a
measurement, it is a bill. The example's own target (`underutilize`) runs in
milliseconds, which is what makes repeat-the-whole-function viable there.

So the direction is inverted: the **workload** reports its own step
boundaries. `torch_profile()` opens a profiler around a labelled region and
the hot loops inside it call `step()` once per natural unit of work - one
calibration batch, one KL capture. With `repeat=1` the schedule records a
single wait/warmup/active cycle at the start of the region and nothing
afterwards, so a 12-hour run pays for a handful of profiled steps.

Everything here is inert unless `cfg.profiling.enabled`. `step()` and
`record()` cost one module-global lookup when it is off, which is why they
can sit inside a loop that runs tens of thousands of times.

Traces land on the traces Volume as `*.pt.trace.json`, readable in the
Perfetto UI (ui.perfetto.dev) or TensorBoard with the torch-tb-profiler
plugin - see `modal_app/profiling.py`.
"""

from __future__ import annotations

import contextlib
import logging
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ilmu_glossary.config import Config

logger = logging.getLogger(__name__)


@dataclass
class ProfiledRegion:
    """Handle for the region currently being profiled.

    `steps` counts every `step()` call, including those the schedule spent in
    its wait and warmup phases, so it is the number of units of work the
    region saw rather than the number recorded.
    """

    label: str
    output_dir: Path
    prof: Any
    steps: int = 0
    trace_path: Path | None = field(default=None)

    def step(self) -> None:
        self.steps += 1
        self.prof.step()


# One region at a time. torch.profiler.profile is not reentrant - nesting two
# raises - and the hot loops are called from inside an outer region anyway.
_ACTIVE: ProfiledRegion | None = None


def active() -> ProfiledRegion | None:
    """The region being profiled, or None. Cheap enough to call in a loop."""
    return _ACTIVE


def step() -> None:
    """Mark one unit of work. No-op unless a region is open."""
    if _ACTIVE is not None:
        _ACTIVE.step()


def record(name: str) -> contextlib.AbstractContextManager[Any]:
    """Label a span inside the trace. No-op unless a region is open.

    Without this, a trace of a transformers forward pass is a wall of aten
    ops with no indication of which phase of the run produced them.
    """
    if _ACTIVE is None:
        return contextlib.nullcontext()
    import torch

    # torch is untyped under this config, so name the type here rather than
    # letting Any escape into every caller.
    span: contextlib.AbstractContextManager[Any] = torch.profiler.record_function(name)
    return span


def _latest_trace(output_dir: Path) -> Path | None:
    """The trace `tensorboard_trace_handler` just wrote, if it wrote one.

    The handler names files by hostname and timestamp, so the newest match is
    the one this region produced. Returns None when the schedule closed
    without ever reaching its active phase.
    """
    traces = sorted(
        output_dir.glob("**/*.pt.trace.json*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return traces[0] if traces else None


@contextlib.contextmanager
def torch_profile(
    cfg: Config, label: str, *, run_id: str | None = None
) -> Iterator[ProfiledRegion | None]:
    """Profile the enclosed region, if profiling is enabled.

    Yields the region handle, or None when disabled or already inside another
    region. Callers do not need the handle: the loops they call use the
    module-level `step()` and `record()`.
    """
    global _ACTIVE

    if not cfg.profiling.enabled:
        yield None
        return

    if _ACTIVE is not None:
        # A nested region would raise inside torch; the outer one is already
        # recording this work, so silently ride along.
        logger.debug("profiling: %r nested inside %r, reusing outer region", label, _ACTIVE.label)
        yield None
        return

    import torch

    settings = cfg.profiling
    output_dir = Path(settings.trace_dir) / label / (run_id or uuid.uuid4().hex[:12])
    output_dir.mkdir(parents=True, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    else:
        # Worth saying out loud: a CPU-only trace of a GPU pipeline shows
        # dispatch overhead and nothing about the kernels themselves.
        logger.warning("profiling %r without CUDA; trace will contain no kernel activity", label)

    schedule = torch.profiler.schedule(
        wait=settings.wait,
        warmup=settings.warmup,
        active=settings.active,
        repeat=settings.repeat,
    )

    logger.info(
        "profiling %r: wait=%d warmup=%d active=%d repeat=%d -> %s",
        label,
        settings.wait,
        settings.warmup,
        settings.active,
        settings.repeat,
        output_dir,
    )

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir)),
        record_shapes=settings.record_shapes,
        profile_memory=settings.profile_memory,
        with_stack=settings.with_stack,
    ) as prof:
        region = ProfiledRegion(label=label, output_dir=output_dir, prof=prof)
        _ACTIVE = region
        try:
            yield region
        finally:
            _ACTIVE = None

    region.trace_path = _latest_trace(output_dir)

    if region.trace_path is None:
        # The usual cause is a region shorter than wait+warmup+active: the
        # schedule never reached its active phase, so nothing was written.
        logger.warning(
            "profiling %r: %d steps produced no trace (schedule needs %d); "
            "lower profiling.wait/warmup or profile a longer region",
            label,
            region.steps,
            settings.wait + settings.warmup + settings.active,
        )
    else:
        logger.info(
            "profiling %r: %d steps, trace at %s (%.1f MB)",
            label,
            region.steps,
            region.trace_path,
            region.trace_path.stat().st_size / 1e6,
        )

    if settings.print_rows:
        sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        try:
            logger.info(
                "profiling %r:\n%s",
                label,
                prof.key_averages().table(sort_by=sort_key, row_limit=settings.print_rows),
            )
        except (AssertionError, KeyError, RuntimeError) as exc:
            # key_averages raises when nothing was recorded. A profiler that
            # cannot summarise itself must not take the run down with it.
            logger.warning("profiling %r: no summary available (%s)", label, exc)


__all__ = ["ProfiledRegion", "active", "record", "step", "torch_profile"]
