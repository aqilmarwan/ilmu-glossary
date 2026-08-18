"""Staged pipeline with the spec's decision gates.

The order is not a convenience - two gates in the spec exist specifically to
stop the run before it spends money on a question already answered:

  Spec section 4  If routing across Malay registers and the English control is
                  statistically indistinguishable, the central hypothesis is
                  weakened. **Record and continue** - corpus substitution is
                  still worth testing, but the report reframes around the null.

  Spec section 5  Run `oracle_contaminated` first at N=1024. If it performs
                  similarly to `baseline_en`, calibration is not the lever.
                  **Record it as the finding and stop** rather than running
                  the full matrix.

Every stage is resumable. A killed Modal container loses no completed work,
because each phase checks for its own artifacts before doing anything.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from ilmu_glossary.config import CalibVariant, Config, MambaStateDtype, RecipeFamily
from ilmu_glossary.io import is_complete, write_json

logger = logging.getLogger(__name__)


class RemoteFunction(Protocol):
    """A Modal function, or any callable exposing `.remote`."""

    def remote(self, *args: Any, **kwargs: Any) -> Any: ...


def _call(fn: Any, /, **kwargs: Any) -> Any:
    """Invoke a Modal function remotely, or a plain callable directly.

    Lets `staged_pipeline` be unit-tested with ordinary functions instead of
    requiring a Modal session.
    """
    remote = getattr(fn, "remote", None)
    return remote(**kwargs) if callable(remote) else fn(**kwargs)


def staged_pipeline(
    cfg: Config,
    *,
    phase0: Any,
    phase1: Any,
    phase2: Any,
    phase3: Any,
    phase4: Any,
    phase5: Any,
    config_yaml: str | None = None,
) -> dict[str, Any]:
    """Run the study, honouring both decision gates."""
    fingerprint = cfg.fingerprint()
    results_dir = cfg.paths.resolve("results")
    common = {"config_yaml": config_yaml, "dry_run": cfg.dry_run}
    journal: dict[str, Any] = {"config_fingerprint": fingerprint, "stages": {}}

    def record(stage: str, payload: Any) -> None:
        journal["stages"][stage] = payload
        write_json(journal, results_dir / "pipeline_journal.json")

    # ------------------------------------------------------------- phase 0
    if is_complete(results_dir / "corpus_stats.parquet", fingerprint=fingerprint):
        logger.info("Phase 0 already complete; skipping")
        record("phase0", {"skipped": True})
    else:
        record("phase0", _call(phase0, **common))

    # ------------------------------------------------------------- phase 1
    if is_complete(results_dir / "routing_comparison.parquet", fingerprint=fingerprint):
        logger.info("Phase 1 already complete; skipping")
        record("phase1", {"skipped": True})
    else:
        record("phase1", _call(phase1, **common))

    # Gate 1 - routing. Recorded, never blocking (spec section 4).
    routing_gate = _read_routing_gate(cfg)
    record("routing_gate", routing_gate)
    if routing_gate.get("hypothesis_weakened"):
        logger.warning(
            "ROUTING GATE: %s. The central hypothesis is weakened. Continuing - "
            "corpus-substitution calibration is still worth testing - but the "
            "report will be framed around the null result.",
            routing_gate.get("reason", ""),
        )

    # --------------------------------------- phase 2/3/4 for the gate cell
    gate_variant = cfg.calibration.gate_variant
    gate_n = cfg.calibration.gate_sample_count
    baseline = CalibVariant.BASELINE_EN
    # The headroom comparison needs both arms, so both are built before the gate.
    gate_pair = [gate_variant.value, baseline.value]

    record(
        "phase2_gate",
        _call(phase2, **common, variants=gate_pair, sample_counts=[gate_n]),
    )

    # The gate runs on the shipped family. It is the configuration people
    # actually deploy, so a null there is the operationally meaningful null.
    gate_family = RecipeFamily.W4A16_SHIPPED

    # BF16 reference must be evaluated first - tier 1 has nothing to diff against
    # otherwise, and its logprob cache is reused by every later checkpoint.
    record("phase4_reference", _call(phase4, **common, checkpoint="bf16_reference"))

    gate_checkpoints = []
    for variant in gate_pair:
        tag = f"{gate_family.value}_{variant}_{gate_n}"
        record(
            f"phase3_{tag}",
            _call(phase3, **common, variant=variant, sample_count=gate_n, family=gate_family.value),
        )
        record(f"phase4_{tag}", _call(phase4, **common, checkpoint=tag))
        gate_checkpoints.append(tag)

    # ------------------------------------------------ gate 2 - headroom
    from ilmu_glossary.analyze import headroom_analysis, load_artifacts

    artifacts = load_artifacts(cfg)
    headroom = headroom_analysis(artifacts.kl, cfg)
    record("headroom_gate", headroom)

    if headroom.get("status") == "computed" and not headroom["gate_open"]:
        logger.warning(
            "HEADROOM GATE CLOSED: %s Stopping before the full matrix. "
            "Relative headroom %.4f is below the %.2f threshold.",
            headroom["verdict"],
            headroom["relative_headroom"],
            headroom["threshold"],
        )
        record("phase5", _call(phase5, **common))
        journal["stopped_early"] = True
        journal["stop_reason"] = headroom["verdict"]
        write_json(journal, results_dir / "pipeline_journal.json")
        return journal

    if headroom.get("status") != "computed":
        logger.warning(
            "Headroom gate could not be evaluated (%s). Proceeding with the "
            "full matrix, but the upper bound on recovery is unknown.",
            headroom.get("reason"),
        )

    # ------------------------------------------------------- full matrix
    record("phase2_full", _call(phase2, **common))

    matrix: list[str] = []
    for family in cfg.quant.families:
        for variant in cfg.calibration.variants:
            for n in cfg.effective_sample_counts():
                tag = f"{family.value}_{variant.value}_{n}"
                if tag in gate_checkpoints:
                    continue  # already done for the gate
                record(
                    f"phase3_{tag}",
                    _call(
                        phase3,
                        **common,
                        variant=variant.value,
                        sample_count=n,
                        family=family.value,
                    ),
                )
                matrix.append(tag)

    for tag in matrix:
        record(f"phase4_{tag}", _call(phase4, **common, checkpoint=tag))

    # ------------------------------- Mamba state arm on the winner only
    winner = _select_winner(cfg)
    record("winner", winner)
    if winner:
        for arm in cfg.quant.mamba_state_arms:
            if arm is MambaStateDtype.FP16_SR:
                continue  # already measured; it is the serving default
            record(
                f"phase4_{winner}_{arm.value}",
                _call(phase4, **common, checkpoint=winner, tiers=["throughput", "kl"]),
            )

    # ------------------------------------------------------------ phase 5
    record("phase5", _call(phase5, **common))
    journal["stopped_early"] = False
    write_json(journal, results_dir / "pipeline_journal.json")
    return journal


def _read_routing_gate(cfg: Config) -> dict[str, Any]:
    from ilmu_glossary.io import read_json

    try:
        summary = read_json(cfg.paths.resolve("results") / "phase1_summary.json")
    except FileNotFoundError:
        return {"verdict": "unavailable", "hypothesis_weakened": False}
    gate: dict[str, Any] = summary.get("routing_gate", {})
    return gate


def _select_winner(cfg: Config) -> str | None:
    """Pick the best *legitimate* variant by tier-1 BM-EN delta.

    `oracle_contaminated` is excluded by construction - it is not a legitimate
    result and must never be selected as the winner of anything.
    """
    from ilmu_glossary.analyze import load_artifacts

    kl = load_artifacts(cfg).kl
    if kl is None or kl.empty or "bm_en_delta" not in kl.columns:
        return None

    legitimate = kl[~kl["variant"].isin([CalibVariant.ORACLE_CONTAMINATED.value])]
    if legitimate.empty:
        return None

    # Smallest absolute BM-EN delta is best - least language-specific damage.
    best_position = int(legitimate["bm_en_delta"].abs().to_numpy().argmin())
    best = legitimate.iloc[best_position]
    winner = str(best["checkpoint"])
    logger.info(
        "Winner: %s (BM-EN delta %.6f), excluding contaminated variants",
        winner,
        float(best["bm_en_delta"]),
    )
    return winner


__all__ = ["RemoteFunction", "staged_pipeline"]
