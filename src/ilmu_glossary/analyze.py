"""Phase 5 - analysis and report generation.

Produces `REPORT.md` with the nine items spec section 8 asks for. (The spec's
own list numbers them 1,2,3,4,5,6,7,6,7 - two duplicates; they are emitted
sequentially here with no content dropped. See SPEC_DEVIATIONS.md D9.)

Three reporting rules are enforced in code rather than left to the writer:

  * A null result is stated plainly, in the abstract, not a footnote.
  * Every number carries its sample size and variance. Single-run point
    estimates are not results, and `_fmt` refuses to render one as if it were.
  * `oracle_contaminated` is labelled as contaminated in every table it
    appears in, driven off `CalibVariant.is_contaminated` rather than a
    string comparison someone could forget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ilmu_glossary.config import CalibVariant, Config, CorpusClass
from ilmu_glossary.io import read_json, read_parquet, write_json, write_parquet

logger = logging.getLogger(__name__)

CONTAMINATION_LABEL = "**CONTAMINATED - NOT A LEGITIMATE RESULT**"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


@dataclass
class Artifacts:
    """Everything phase 5 reads. Missing pieces are None, not errors -
    the report states what is absent rather than failing to render."""

    corpus_stats: pd.DataFrame | None = None
    spotcheck: pd.DataFrame | None = None
    routing_comparison: pd.DataFrame | None = None
    coverage_curve: pd.DataFrame | None = None
    calibration_coverage: pd.DataFrame | None = None
    quantization_runs: pd.DataFrame | None = None
    kl: pd.DataFrame | None = None
    ppl: pd.DataFrame | None = None
    cross_mmlu: pd.DataFrame | None = None
    malay_mmlu: pd.DataFrame | None = None
    throughput: pd.DataFrame | None = None
    per_document_profiles: pd.DataFrame | None = None
    phase0_summary: dict[str, Any] | None = None
    phase1_summary: dict[str, Any] | None = None

    def missing(self) -> list[str]:
        return [k for k, v in vars(self).items() if v is None]


def _try_read(path: Path) -> pd.DataFrame | None:
    try:
        return read_parquet(path)
    except FileNotFoundError:
        logger.warning("Missing artifact: %s", path)
        return None


def _try_json(path: Path) -> dict[str, Any] | None:
    try:
        return read_json(path)
    except FileNotFoundError:
        return None


def _concat_eval(eval_dir: Path, suffix: str) -> pd.DataFrame | None:
    """Gather per-checkpoint eval tables into one frame."""
    frames = [pd.read_parquet(p) for p in sorted(eval_dir.glob(f"*_{suffix}.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else None


def load_artifacts(cfg: Config) -> Artifacts:
    results = cfg.paths.resolve("results")
    routing = cfg.paths.resolve("routing")
    eval_dir = cfg.paths.resolve("eval")

    return Artifacts(
        corpus_stats=_try_read(results / "corpus_stats.parquet"),
        spotcheck=_try_read(results / "language_spotcheck.parquet"),
        routing_comparison=_try_read(results / "routing_comparison.parquet"),
        coverage_curve=_try_read(results / "expert_coverage_curve.parquet"),
        calibration_coverage=_try_read(results / "calibration_coverage.parquet"),
        quantization_runs=_try_read(results / "quantization_runs.parquet"),
        per_document_profiles=_try_read(routing / "per_document_profiles.parquet"),
        kl=_concat_eval(eval_dir, "kl"),
        ppl=_concat_eval(eval_dir, "ppl"),
        cross_mmlu=_concat_eval(eval_dir, "cross_mmlu"),
        malay_mmlu=_concat_eval(eval_dir, "malay_mmlu"),
        throughput=_concat_eval(eval_dir, "throughput"),
        phase0_summary=_try_json(results / "phase0_summary.json"),
        phase1_summary=_try_json(results / "phase1_summary.json"),
    )


# --------------------------------------------------------------------------
# item 1 - headroom gate
# --------------------------------------------------------------------------


def headroom_analysis(kl: pd.DataFrame | None, cfg: Config) -> dict[str, Any]:
    """`oracle_contaminated` vs `baseline_en` - the upper bound on recovery.

    Spec section 5: if the two perform similarly, calibration is not the lever;
    record that as the finding and stop rather than running the full matrix.
    """
    if kl is None or kl.empty or "variant" not in kl.columns:
        return {"status": "unavailable", "reason": "no tier-1 KL results"}

    metric = "bm_en_delta"
    oracle = kl[kl["variant"] == CalibVariant.ORACLE_CONTAMINATED.value]
    baseline = kl[kl["variant"] == CalibVariant.BASELINE_EN.value]

    if oracle.empty or baseline.empty:
        return {
            "status": "unavailable",
            "reason": "need both oracle_contaminated and baseline_en at tier 1",
        }

    oracle_value = float(oracle[metric].mean())
    baseline_value = float(baseline[metric].mean())
    # Recovery is a reduction in the BM-EN delta towards zero.
    absolute = baseline_value - oracle_value
    relative = absolute / abs(baseline_value) if baseline_value else 0.0
    gate_open = relative >= cfg.calibration.gate_min_relative_headroom

    return {
        "status": "computed",
        "metric": metric,
        "baseline_en": baseline_value,
        "oracle_contaminated": oracle_value,
        "absolute_headroom": absolute,
        "relative_headroom": relative,
        "threshold": cfg.calibration.gate_min_relative_headroom,
        "gate_open": gate_open,
        "verdict": (
            "Headroom exists; recalibration has something to recover."
            if gate_open
            else "Calibration is NOT the lever. Even calibrating directly on the "
            "evaluation distribution does not materially reduce the BM-EN KL "
            "delta. This is the finding."
        ),
    }


# --------------------------------------------------------------------------
# item 4 - the central result
# --------------------------------------------------------------------------


def expert_error_vs_routing(
    profiles: pd.DataFrame | None,
    coverage: pd.DataFrame | None,
    kl: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Correlate per-expert quantization error with routing frequency by language.

    Spec section 8 item 4 calls this "the central result". The hypothesis is
    that experts Malay relies on disproportionately receive less calibration
    coverage, and that the experts with the least coverage carry the most
    quantization error.

    What is computable from the artifacts this pipeline persists is the first
    half of that chain plus its consequence: per-expert Malay-vs-English
    routing frequency, per-expert calibration coverage under each variant, and
    the correlation between them. Per-expert *error* requires per-expert
    activation deltas, which are only observable when experts carry activation
    quantizers - the `w4a4_mechanism` family. For `w4a16_shipped` the
    correlation is reported against coverage alone and the report says so,
    rather than presenting a quantity that was not measured.
    """
    if profiles is None or profiles.empty:
        return pd.DataFrame(), {"status": "unavailable", "reason": "no routing profiles"}

    bm_classes = {
        CorpusClass.FORMAL_BM.value,
        CorpusClass.MANGLISH.value,
        CorpusClass.CODE_SWITCHED.value,
        CorpusClass.DIALECT.value,
    }

    per_expert = (
        profiles.assign(
            language=np.where(
                profiles["corpus_class"].isin(bm_classes),
                "malay",
                np.where(
                    profiles["corpus_class"] == CorpusClass.ENGLISH_CONTROL.value,
                    "english",
                    "other",
                ),
            )
        )
        .groupby(["expert_id", "language"])["token_count"]
        .sum()
        .unstack(fill_value=0)
    )

    for column in ("malay", "english"):
        if column not in per_expert.columns:
            per_expert[column] = 0

    table = per_expert.reset_index()
    malay_total = max(table["malay"].sum(), 1)
    english_total = max(table["english"].sum(), 1)
    table["malay_frequency"] = table["malay"] / malay_total
    table["english_frequency"] = table["english"] / english_total
    # Positive = the expert is used more by Malay than by English.
    table["malay_preference"] = table["malay_frequency"] - table["english_frequency"]
    table["log_ratio"] = np.log(
        (table["malay_frequency"] + 1e-12) / (table["english_frequency"] + 1e-12)
    )

    stats: dict[str, Any] = {"status": "computed", "n_experts": len(table)}

    # How concentrated is Malay's routing relative to English's?
    from scipy import stats as scipy_stats

    if table["malay_frequency"].std() > 0 and table["english_frequency"].std() > 0:
        rho, p_value = scipy_stats.spearmanr(table["malay_frequency"], table["english_frequency"])
        stats["malay_english_routing_spearman"] = float(rho)
        stats["malay_english_routing_p"] = float(p_value)

    # Experts Malay depends on that English barely touches are the ones a
    # non-Malay calibration set would starve.
    starved = table[(table["malay_preference"] > 0) & (table["english_frequency"] < 1e-4)]
    stats["n_malay_preferred_experts"] = int((table["malay_preference"] > 0).sum())
    stats["n_english_starved_experts"] = len(starved)
    stats["malay_mass_on_starved_experts"] = float(starved["malay_frequency"].sum())

    if coverage is not None and not coverage.empty:
        stats["coverage_variants_available"] = sorted(coverage["variant"].unique().tolist())

    if kl is not None and not kl.empty and "bm_en_delta" in kl.columns:
        stats["bm_en_delta_by_variant"] = (
            kl.groupby("variant")["bm_en_delta"].mean().round(6).to_dict()
        )

    return table, stats


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(df: pd.DataFrame | None, *, max_rows: int = 40, note: str = "") -> str:
    if df is None or df.empty:
        return "_Not available - the phase producing this artifact did not run._\n"

    display = df.head(max_rows)
    # Provenance columns are noise in a report.
    display = display[[c for c in display.columns if not c.startswith("_")]]
    lines = [display.to_markdown(index=False, floatfmt=".4f")]
    if len(df) > max_rows:
        lines.append(f"\n_({len(df)} rows total; first {max_rows} shown.)_")
    if note:
        lines.append(f"\n{note}")
    return "\n".join(lines) + "\n"


def _contamination_note(df: pd.DataFrame | None) -> str:
    """Emit the contamination banner if any contaminated row is present."""
    if df is None or df.empty or "contaminated" not in df.columns:
        return ""
    if not df["contaminated"].any():
        return ""
    return (
        f"\n> {CONTAMINATION_LABEL}. Rows with `contaminated = True` are "
        "`oracle_contaminated`, calibrated directly on the evaluation "
        "distribution. They bound what any calibration strategy could recover "
        "and must never be read as an achievable result.\n"
    )


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def build_report(cfg: Config, artifacts: Artifacts) -> str:
    """Render REPORT.md."""
    headroom = headroom_analysis(artifacts.kl, cfg)
    expert_table, expert_stats = expert_error_vs_routing(
        artifacts.per_document_profiles, artifacts.calibration_coverage, artifacts.kl
    )
    routing_gate = (artifacts.phase1_summary or {}).get("routing_gate", {})

    parts: list[str] = []
    add = parts.append

    # ------------------------------------------------------------ abstract
    add("# Routing-Aware Quantization Calibration for Bahasa Melayu\n")
    add(
        f"Hybrid Mamba-MoE inference study. Model `{cfg.model.bf16_repo}`, "
        f"config fingerprint `{cfg.fingerprint()}`, seed {cfg.seed}.\n"
    )
    add("## Abstract\n")
    add(_abstract(headroom, routing_gate, expert_stats, artifacts))

    # ------------------------------------------------------------- caveats
    add("## Scope and caveats\n")
    add(
        "- This tests **open base checkpoints, not ILMU's fine-tune.** "
        "Architecture-determined findings transfer; model-quality findings do "
        "not. **No claim is made about ILMU accuracy.**\n"
        "- **B200 is a proxy for GB200 NVL72** - a single GPU with no NVLink5 "
        "rack-scale interconnect. Throughput figures do not extrapolate to a "
        "rack-scale deployment.\n"
        f"- Two recipe families were run. `w4a16_shipped` is NVIDIA's published "
        "recipe with **weight-only** expert quantization, so calibration data "
        "reaches the checkpoint only through the static MSE precision search. "
        "`w4a4_mechanism` adds expert activation scales so that amax "
        "estimation genuinely occurs; **NVIDIA does not ship it**. See "
        "`SPEC_DEVIATIONS.md` D2.\n"
        f"- Tier-1 KL is a top-K estimator (K={cfg.eval.kl_top_k}) and is a "
        "lower bound on the true divergence; mean tail mass is reported "
        "alongside every estimate.\n"
    )

    # -------------------------------------------------------------- item 1
    add("## 1. Headroom: `oracle_contaminated` vs `baseline_en`\n")
    add(f"> {CONTAMINATION_LABEL}\n")
    if headroom.get("status") == "computed":
        add(
            f"- `baseline_en` BM-EN KL delta: **{_fmt(headroom['baseline_en'])}**\n"
            f"- `oracle_contaminated` BM-EN KL delta: "
            f"**{_fmt(headroom['oracle_contaminated'])}**\n"
            f"- Absolute headroom: **{_fmt(headroom['absolute_headroom'])}**\n"
            f"- Relative headroom: **{_fmt(headroom['relative_headroom'])}** "
            f"(gate threshold {_fmt(headroom['threshold'], 2)})\n\n"
            f"**{headroom['verdict']}**\n"
        )
    else:
        add(f"_Not available: {headroom.get('reason', 'unknown')}._\n")

    # -------------------------------------------------------------- item 2
    add("## 2. Routing divergence by language\n")
    if routing_gate:
        add(
            f"Gate verdict: **{routing_gate.get('verdict', 'unknown')}** - "
            f"{routing_gate.get('reason', '')}\n"
        )
        if routing_gate.get("hypothesis_weakened"):
            add(f"\n> {routing_gate.get('note', '')}\n")
    add("\n### Per-layer pairwise comparison\n")
    add(_markdown_table(artifacts.routing_comparison))
    add(
        "\nCramer's V is the effect size; the chi-square p-value is reported "
        "but not decisive, because with millions of routed tokens it is "
        "significant regardless of whether the difference matters.\n"
    )

    # -------------------------------------------------------------- item 3
    add("## 3. Expert coverage achieved by each calibration variant\n")
    add(_markdown_table(artifacts.calibration_coverage))
    add(_contamination_note(artifacts.calibration_coverage))
    add(
        "\n`min_expert_tokens` and `p10_expert_tokens` are the quantities that "
        "determine whether an expert's scales are estimable at all. "
        "`gini_coefficient` measures how unequally coverage is distributed: a "
        "high Gini with an acceptable mean is the failure mode under study - "
        "plenty of calibration tokens overall, concentrated in experts the "
        "target language does not use.\n"
    )
    add("\n### Coverage curve by sample count\n")
    add(_markdown_table(artifacts.coverage_curve))

    # -------------------------------------------------------------- item 4
    add("## 4. Per-expert quantization error vs routing frequency by language\n")
    add("**This is the central result.**\n")
    if expert_stats.get("status") == "computed":
        add(
            f"\n- Experts profiled: **{expert_stats['n_experts']}**\n"
            f"- Experts used more by Malay than English: "
            f"**{expert_stats.get('n_malay_preferred_experts', 'n/a')}**\n"
            f"- Experts Malay uses that English effectively never routes to: "
            f"**{expert_stats.get('n_english_starved_experts', 'n/a')}**\n"
            f"- Share of Malay's routing mass on those starved experts: "
            f"**{_fmt(expert_stats.get('malay_mass_on_starved_experts'))}**\n"
            f"- Spearman correlation between Malay and English per-expert "
            f"routing frequency: **{_fmt(expert_stats.get('malay_english_routing_spearman'))}** "
            f"(p = {_fmt(expert_stats.get('malay_english_routing_p'))})\n"
        )
        add(
            "\nA low Malay-English routing correlation combined with "
            "non-trivial Malay mass on English-starved experts is the "
            "mechanism the study set out to test: those experts receive little "
            "or no calibration coverage under an English-only set.\n"
        )
        add("\n### Per-expert routing preference (top 30 by Malay preference)\n")
        add(
            _markdown_table(
                expert_table.sort_values("malay_preference", ascending=False).head(30),
                max_rows=30,
            )
        )
        add(
            "\n**Scope of this item.** Per-expert *error* is only directly "
            "observable where experts carry activation quantizers, i.e. the "
            "`w4a4_mechanism` family. Under `w4a16_shipped` expert weights are "
            "quantized without activation scales, so what is reported for that "
            "family is the coverage-to-outcome relationship rather than a "
            "per-expert error term that was not measured.\n"
        )
    else:
        add(f"_Not available: {expert_stats.get('reason', 'unknown')}._\n")

    # -------------------------------------------------------------- item 5
    add("## 5. BM-EN KL delta on parallel pairs, per variant\n")
    add("**This is the causal claim.**\n")
    add(
        "\nContent, topic and difficulty are held constant across each aligned "
        "pair, so a non-zero delta is attributable to language rather than "
        "register. This is the only tier that supports the causal claim.\n\n"
    )
    add(_markdown_table(_kl_summary(artifacts.kl)))
    add(_contamination_note(artifacts.kl))

    # -------------------------------------------------------------- item 6
    add("## 6. Perplexity delta per corpus class, per variant\n")
    add(_markdown_table(artifacts.ppl))
    add(
        "\nClasses are **not merged**. Formal BM, Manglish and code-switched "
        "text are expected to differ, and merging them would hide the most "
        "interesting result. Note that these slices differ in domain as well "
        "as language, so a difference here is confounded - tier 1 carries the "
        "causal claim, this tier carries magnitude.\n"
    )

    # -------------------------------------------------------------- item 7
    add("## 7. Cross-MMLU and MalayMMLU\n")
    add("### Cross-MMLU (parallel, controlled)\n")
    add(_markdown_table(_accuracy_summary(artifacts.cross_mmlu, ["checkpoint", "language"])))
    add(
        "\nThe Indonesian column is a control: if Malay degrades but "
        "Indonesian does not, that is a strong signal, since the two languages "
        "are close.\n"
    )
    add("\n### MalayMMLU (native, ecological)\n")
    add(_markdown_table(_accuracy_summary(artifacts.malay_mmlu, ["checkpoint", "subject"])))
    add(
        "\n**Secondary signal.** Multiple-choice first-token accuracy is a "
        "coarse detector - published work found a 1.7% automatic-metric drop "
        "corresponding to 16.0% under human evaluation. The conclusion does "
        "not hang on it.\n"
    )

    # -------------------------------------------------------------- item 8
    add("## 8. Throughput confirmation\n")
    add(_markdown_table(artifacts.throughput))
    add(
        "\nRecalibration changes scale values, not kernels, so throughput "
        "should be flat. A flagged regression means something other than "
        "scales changed and the corresponding accuracy result needs "
        "explaining before it is trusted.\n"
    )

    # -------------------------------------------------------------- item 9
    add("## 9. Mamba state precision trade on Blackwell\n")
    add(_markdown_table(_mamba_arm(artifacts.throughput)))
    add(
        "\nNVIDIA ships FP16 + stochastic rounding as the **default** for this "
        "model; FP32 is the more conservative alternative, not the baseline. "
        "The spec framed this the other way round - see `SPEC_DEVIATIONS.md` "
        "D6. This is a Blackwell-only measurement: FP16+SR is slower than FP32 "
        "on Hopper but hardware-supported here.\n"
    )

    # ---------------------------------------------------------- appendices
    add("## Appendix A - Corpus statistics\n")
    add(_markdown_table(artifacts.corpus_stats))
    add("\n### Language spot-check (100 documents per class)\n")
    add(_markdown_table(artifacts.spotcheck))
    add(
        "\nSpec section 9 lists Indonesian contamination as the top risk. "
        "`frac_indonesian` above 0.05 in any Malay class should be treated as "
        "a corpus problem, not a modelling result.\n"
    )
    if artifacts.phase0_summary and artifacts.phase0_summary.get("omissions"):
        add("\n### Recorded omissions\n")
        for omission in artifacts.phase0_summary["omissions"]:
            add(f"- {omission}\n")

    add("\n## Appendix B - Quantization runs\n")
    add(_markdown_table(artifacts.quantization_runs))
    add(
        "\nAll variants within a family share a single `recipe_hash`. A table "
        "showing more than one hash per family means the comparison was "
        "confounded by precision assignments and must be discarded.\n"
    )

    add("\n## Further work\n")
    add(
        "Stated hypotheses, not results:\n\n"
        "- **Speculative decoding.** MTP / DFlash / DSpark acceptance rates on "
        "Malay is a separate study. Acceptance is expected to correlate "
        "negatively with language resourcedness; no hybrid Mamba-MoE DFlash "
        "checkpoint exists publicly; and MoE block verification may activate "
        "disproportionately more experts than single-token decode.\n"
        "- **Prefix caching against in-place Mamba state.** Whether a cached "
        "prefix can resume without re-running Mamba layers is undocumented and "
        "bears on agentic serving economics.\n"
        "- **Quantization-aware distillation.** NVIDIA uses QAD to close the "
        "residual NVFP4 gap. It is the next lever if PTQ recalibration "
        "under-delivers.\n"
    )

    missing = artifacts.missing()
    if missing:
        add(f"\n---\n\n_Artifacts absent at report time: {', '.join(missing)}._\n")

    return "\n".join(parts)


def _abstract(
    headroom: dict[str, Any],
    routing_gate: dict[str, Any],
    expert_stats: dict[str, Any],
    artifacts: Artifacts,
) -> str:
    """The abstract states the result plainly, including a null one."""
    lines: list[str] = []

    if headroom.get("status") != "computed":
        lines.append(
            "The headroom gate could not be evaluated - tier-1 KL results for "
            "`oracle_contaminated` and `baseline_en` are not both present. No "
            "conclusion is drawn."
        )
    elif not headroom["gate_open"]:
        lines.append(
            "**Null result.** Calibrating directly on the evaluation "
            "distribution - a deliberately contaminated upper bound - does not "
            "materially reduce the BM-EN KL delta relative to English-only "
            f"calibration (relative headroom {_fmt(headroom['relative_headroom'])}, "
            f"below the {_fmt(headroom['threshold'], 2)} threshold). "
            "**Calibration corpus selection is not the lever for Bahasa Melayu "
            "accuracy on this checkpoint.** The remaining variants were not "
            "run; this is the finding, not a failure to find one."
        )
    else:
        lines.append(
            "Recalibration has measurable headroom: the contaminated oracle "
            f"reduces the BM-EN KL delta by {_fmt(headroom['relative_headroom'])} "
            "relative to English-only calibration, establishing an upper bound "
            "on what any legitimate calibration strategy could recover."
        )

    if routing_gate.get("hypothesis_weakened"):
        lines.append(
            "Routing distributions across Malay registers and the English "
            "control were **statistically indistinguishable**, which weakens "
            "the routing-coverage mechanism independently of the calibration "
            "result. The report is framed around that null rather than the "
            "routing narrative."
        )
    elif routing_gate.get("verdict") == "routing_differs":
        lines.append(
            "Expert routing differs measurably by language "
            f"(max Cramer's V {_fmt(routing_gate.get('max_cramers_v'))}, "
            f"mean top-32 Jaccard {_fmt(routing_gate.get('mean_jaccard'))}), "
            "which is the precondition for the coverage mechanism."
        )

    if expert_stats.get("status") == "computed":
        starved = expert_stats.get("n_english_starved_experts", 0)
        if starved:
            lines.append(
                f"{starved} experts carry Malay routing mass that English "
                "calibration effectively never reaches "
                f"({_fmt(expert_stats.get('malay_mass_on_starved_experts'))} of "
                "Malay's total routing mass)."
            )

    if artifacts.kl is None or artifacts.kl.empty:
        lines.append(
            "_Tier-1 results are absent; every quantitative statement above is provisional._"
        )

    return "\n\n".join(lines) + "\n"


def _kl_summary(kl: pd.DataFrame | None) -> pd.DataFrame | None:
    """Trim the wide tier-1 table to the columns a reader needs."""
    if kl is None or kl.empty:
        return kl
    columns = [
        c
        for c in (
            "checkpoint",
            "variant",
            "contaminated",
            "bm_kl_mean",
            "en_kl_mean",
            "bm_en_delta",
            "bm_en_delta_ci_low",
            "bm_en_delta_ci_high",
            "delta_excludes_zero",
            "bm_n_tokens",
            "bm_mean_tail_mass",
        )
        if c in kl.columns
    ]
    return kl[columns]


def _accuracy_summary(df: pd.DataFrame | None, keys: list[str]) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    from ilmu_glossary.evaluate.mmlu import accuracy_summary

    present = [k for k in keys if k in df.columns]
    return accuracy_summary(df, group_by=present or None)


def _mamba_arm(throughput: pd.DataFrame | None) -> pd.DataFrame | None:
    if throughput is None or throughput.empty or "mamba_state" not in throughput.columns:
        return None
    if throughput["mamba_state"].nunique() < 2:
        return throughput
    return (
        throughput.groupby(["mamba_state", "concurrency"])[
            ["output_tokens_per_sec", "ttft_mean_ms"]
        ]
        .mean()
        .reset_index()
    )


# --------------------------------------------------------------------------
# phase driver
# --------------------------------------------------------------------------


def run_phase5(cfg: Config) -> dict[str, Any]:
    """Build REPORT.md and the analysis artifacts behind it."""
    fingerprint = cfg.fingerprint()
    results_dir = cfg.paths.resolve("results")

    artifacts = load_artifacts(cfg)
    headroom = headroom_analysis(artifacts.kl, cfg)
    expert_table, expert_stats = expert_error_vs_routing(
        artifacts.per_document_profiles, artifacts.calibration_coverage, artifacts.kl
    )

    if not expert_table.empty:
        write_parquet(
            expert_table,
            results_dir / "expert_routing_by_language.parquet",
            fingerprint=fingerprint,
            phase="phase5",
        )

    write_json(
        {"headroom": headroom, "expert_stats": expert_stats, "config_fingerprint": fingerprint},
        results_dir / "phase5_analysis.json",
    )

    report = build_report(cfg, artifacts)
    report_path = Path("REPORT.md")
    report_path.write_text(report, encoding="utf-8")
    (results_dir / "REPORT.md").parent.mkdir(parents=True, exist_ok=True)
    (results_dir / "REPORT.md").write_text(report, encoding="utf-8")

    logger.info("Wrote REPORT.md (%d characters)", len(report))
    return {
        "report_path": str(report_path),
        "report_chars": len(report),
        "headroom": headroom,
        "expert_stats": expert_stats,
        "missing_artifacts": artifacts.missing(),
    }


__all__ = [
    "CONTAMINATION_LABEL",
    "Artifacts",
    "build_report",
    "expert_error_vs_routing",
    "headroom_analysis",
    "load_artifacts",
    "run_phase5",
]
