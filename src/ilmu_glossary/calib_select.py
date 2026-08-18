"""Phase 2 - calibration set construction.

Six variants at each of 256 / 512 / 1024 / 2048 samples (spec section 5):

  baseline_en           English only - proxy for NVIDIA's calibration
  bm_only               Formal BM only
  mixed_5050            50/50 BM/English, randomly sampled
  coverage_greedy       Routing-coverage-driven (the method under test)
  mixed_weighted        Proportional to a stated assumed traffic mix
  oracle_contaminated   Calibrated on the eval distribution - NOT a result

`oracle_contaminated` is deliberately contaminated. It exists to establish the
upper bound on what any calibration strategy could recover, is run first at
N=1024 to answer whether the project has headroom at all, and is labelled as
contaminated everywhere it appears.

The coverage-greedy variant targets the **minimum** per-expert token count,
because a per-tensor scale estimated from a handful of tokens is unreliable
regardless of how well-covered its neighbours are - the worst-covered expert
bounds the quality of the whole checkpoint.

It does not *optimise* that quantity directly. Spec section 5's scoring rule
(`score(d) = increase in min(C)`) is degenerate: with top-k routing a single
document touches only k of E experts, so `min(C)` is pinned at zero until
every expert is covered and every candidate scores 0.0. Selection collapses to
random - exactly what this variant exists to beat. `coverage_greedy_select`
optimises a saturated-coverage surrogate instead and still reports min/P10.
See D11 in SPEC_DEVIATIONS.md and that function's docstring.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ilmu_glossary import tracking
from ilmu_glossary.config import CalibVariant, Config, CorpusClass
from ilmu_glossary.io import read_jsonl, read_parquet, write_json, write_jsonl, write_parquet
from ilmu_glossary.seeds import rng
from ilmu_glossary.splits import assert_disjoint, load_splits

logger = logging.getLogger(__name__)

BM_CLASSES = (
    CorpusClass.FORMAL_BM.value,
    CorpusClass.MANGLISH.value,
    CorpusClass.CODE_SWITCHED.value,
    CorpusClass.DIALECT.value,
)
EN_CLASSES = (CorpusClass.ENGLISH_CONTROL.value,)

# Below this fraction of selected documents carrying a routing profile, a
# variant's absolute coverage numbers are not comparable with other variants.
PROFILE_COVERAGE_FLOOR = 0.9


# --------------------------------------------------------------------------
# coverage-greedy
# --------------------------------------------------------------------------


@dataclass
class CoverageState:
    """Running per-expert token coverage during greedy selection."""

    coverage: np.ndarray
    selected: list[int]

    @property
    def minimum(self) -> float:
        return float(self.coverage.min())

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.coverage, q))


def saturation_threshold(profiles: np.ndarray, n_select: int) -> float:
    """Per-expert token count at which an expert counts as adequately covered.

    Set to what each expert would receive if `n_select` documents routed
    perfectly uniformly across experts. Above this the marginal value of more
    tokens for scale estimation is negligible; below it an expert's scale is
    estimated from too few observations. It is the knee the objective needs.
    """
    n_experts = int(profiles.shape[1])
    mean_doc_tokens = float(profiles.sum(axis=1).mean()) if profiles.size else 0.0
    return max(1.0, n_select * mean_doc_tokens / max(n_experts, 1))


def _saturated_gain(coverage: np.ndarray, profile: np.ndarray, tau: float) -> float:
    """Marginal gain in saturated coverage from adding one document.

    f(S) = sum_e min(coverage_e(S), tau)

    This is monotone and submodular, which matters twice over. It gives the
    greedy a (1 - 1/e) approximation guarantee, and it makes the lazy
    priority queue below *exact* rather than heuristic, because submodularity
    is precisely the property that marginal gains never increase.
    """
    gained: np.floating[Any] = (
        np.minimum(coverage + profile, tau).sum() - np.minimum(coverage, tau).sum()
    )
    return float(gained)


def coverage_greedy_select(
    profiles: np.ndarray,
    n_select: int,
    *,
    tiebreak_percentile: float = 10.0,
    candidate_cap: int | None = None,
    saturation: float | None = None,
    seed: int = 0,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Select `n_select` documents maximising per-expert calibration coverage.

    `profiles` is (n_docs, n_experts) of per-document token counts from phase 1.

    **Objective.** Spec section 5 states the target as "maximise the MINIMUM
    per-expert token count", scoring each document by `increase in min(C)`.
    Implemented literally that does not work: with top-k routing a single
    document touches only k of E experts, so `min(C)` stays pinned at zero
    until every expert has been covered at least once. Every candidate scores
    exactly 0.0, the argmax is arbitrary, and the selection degenerates to
    random - which is the failure this variant exists to beat. The P10
    tie-break is flat in the same region and does not rescue it.

    What is optimised instead is **saturated coverage**,
    `f(S) = sum_e min(C_e(S), tau)`, which is monotone submodular and has
    gradient from the first document. `tau` is the per-expert count expected
    under uniform routing (see `saturation_threshold`). Maximising it drives
    the same outcome the spec asks for - it rewards tokens for under-covered
    experts and stops rewarding experts already past `tau` - while remaining
    optimisable.

    Min and P10 per-expert token counts are still what gets *reported*, per
    spec section 5. Only the search objective changed.

    **Runtime.** Naive scoring is O(pool x N x experts). This is CELF lazy
    greedy: cached gains are upper bounds that submodularity guarantees can
    only decrease, so the first popped entry whose gain was recomputed this
    step is provably the argmax.

    Returns (selected_indices, per_step_trace).
    """
    n_docs, n_experts = profiles.shape
    if n_select > n_docs:
        logger.warning(
            "Requested %d calibration documents but the pool holds %d; selecting all of them",
            n_select,
            n_docs,
        )
        n_select = n_docs

    tau = saturation if saturation is not None else saturation_threshold(profiles, n_select)
    state = CoverageState(coverage=np.zeros(n_experts, dtype=np.float64), selected=[])
    generator = np.random.default_rng(seed)

    # Initial gains at C=0 reduce to sum(min(profile, tau)) - vectorised.
    initial = np.minimum(profiles, tau).sum(axis=1)
    heap: list[tuple[float, int, int]] = [(-float(initial[doc]), 0, doc) for doc in range(n_docs)]
    heapq.heapify(heap)

    chosen: set[int] = set()
    trace: list[dict[str, Any]] = []

    for step in range(n_select):
        rescored = 0
        picked: int | None = None

        while heap:
            _cached, stamp, doc = heapq.heappop(heap)
            if doc in chosen:
                continue
            if stamp == step:
                picked = doc
                break

            gain = _saturated_gain(state.coverage, profiles[doc], tau)
            heapq.heappush(heap, (-gain, step, doc))
            rescored += 1

            # Spec section 5 permits capping candidates per iteration when
            # runtime becomes prohibitive. The cap bounds rescoring work, not
            # the pool, so quality degrades gracefully instead of the pool
            # being silently truncated.
            if candidate_cap is not None and rescored >= candidate_cap:
                pool = [d for _, _, d in heap if d not in chosen][:candidate_cap]
                if pool:
                    picked = max(
                        pool,
                        key=lambda d: _saturated_gain(state.coverage, profiles[d], tau),
                    )
                break

        if picked is None:
            # Everything remaining is saturated and adds nothing. Fall back to
            # a random unselected document so the requested N is still met,
            # and record that this happened.
            remaining = [d for d in range(n_docs) if d not in chosen]
            if not remaining:
                break
            picked = int(generator.choice(remaining))

        chosen.add(picked)
        state.selected.append(picked)
        state.coverage += profiles[picked]
        trace.append(
            {
                "step": step,
                "doc_index": int(picked),
                "min_coverage": state.minimum,
                "p10_coverage": state.percentile(10.0),
                "mean_coverage": float(state.coverage.mean()),
                "saturated_coverage": float(np.minimum(state.coverage, tau).sum()),
                "experts_unseen": int((state.coverage == 0).sum()),
                "rescored": rescored,
            }
        )

    logger.info(
        "coverage_greedy: %d docs, tau=%.1f, min=%.0f P10=%.0f, %d/%d experts unseen",
        len(state.selected),
        tau,
        state.minimum,
        state.percentile(tiebreak_percentile),
        int((state.coverage == 0).sum()),
        n_experts,
    )
    return state.selected, trace


# --------------------------------------------------------------------------
# variant construction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateDoc:
    """One document eligible for calibration."""

    doc_id: str
    corpus_class: str
    doc_index: int
    text: str


def load_candidates(cfg: Config) -> list[CandidateDoc]:
    """Load the train-split documents phase 1 profiled.

    Calibration must draw only from the 80% train split. Reading the split
    file rather than the whole class file is what enforces that.
    """
    splits = load_splits(cfg.paths.resolve("splits"))
    stratified = cfg.paths.resolve("stratified")
    candidates: list[CandidateDoc] = []

    for class_name, split in splits.items():
        path = stratified / f"{class_name}.jsonl"
        if not path.exists():
            continue
        train_indices = set(split.train_indices)
        for i, record in enumerate(read_jsonl(path)):
            if i not in train_indices:
                continue
            text = record.get("templated") or record.get("text") or record.get("malay", "")
            candidates.append(
                CandidateDoc(
                    doc_id=record.get("id", f"{class_name}:{i}"),
                    corpus_class=class_name,
                    doc_index=i,
                    text=text,
                )
            )
    return candidates


def load_eval_distribution(cfg: Config) -> list[CandidateDoc]:
    """Documents from the **evaluation** split.

    Used exclusively by `oracle_contaminated`. Isolated in its own function so
    that the one place contamination is legitimate is obvious in a diff.
    """
    splits = load_splits(cfg.paths.resolve("splits"))
    stratified = cfg.paths.resolve("stratified")
    docs: list[CandidateDoc] = []

    for class_name in BM_CLASSES:
        path = stratified / f"{class_name}.jsonl"
        if not path.exists() or class_name not in splits:
            continue
        eval_indices = set(splits[class_name].eval_indices)
        for i, record in enumerate(read_jsonl(path)):
            if i not in eval_indices:
                continue
            text = record.get("templated") or record.get("text", "")
            docs.append(
                CandidateDoc(
                    doc_id=record.get("id", f"{class_name}:{i}"),
                    corpus_class=class_name,
                    doc_index=i,
                    text=text,
                )
            )
    return docs


def _sample(pool: list[CandidateDoc], n: int, *, seed: int, tag: str) -> list[CandidateDoc]:
    if not pool:
        logger.error("Empty pool for %s; cannot sample %d documents", tag, n)
        return []
    generator = rng(seed, "phase2", tag)
    if n >= len(pool):
        logger.warning("%s: pool holds %d, requested %d - taking all", tag, len(pool), n)
        return list(pool)
    idx = generator.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in idx]


def build_variant(
    cfg: Config,
    variant: CalibVariant,
    n: int,
    candidates: list[CandidateDoc],
    profiles_by_doc: dict[str, np.ndarray] | None,
) -> tuple[list[CandidateDoc], dict[str, Any]]:
    """Construct one calibration variant at one sample count."""
    by_class: dict[str, list[CandidateDoc]] = {}
    for doc in candidates:
        by_class.setdefault(doc.corpus_class, []).append(doc)

    bm_pool = [d for d in candidates if d.corpus_class in BM_CLASSES]
    en_pool = [d for d in candidates if d.corpus_class in EN_CLASSES]
    meta: dict[str, Any] = {"variant": variant.value, "n_requested": n}

    if variant is CalibVariant.BASELINE_EN:
        selected = _sample(en_pool, n, seed=cfg.seed, tag=f"baseline_en_{n}")

    elif variant is CalibVariant.BM_ONLY:
        formal = by_class.get(CorpusClass.FORMAL_BM.value, [])
        selected = _sample(formal, n, seed=cfg.seed, tag=f"bm_only_{n}")

    elif variant is CalibVariant.MIXED_5050:
        half = n // 2
        selected = _sample(bm_pool, half, seed=cfg.seed, tag=f"mixed_bm_{n}") + _sample(
            en_pool, n - half, seed=cfg.seed, tag=f"mixed_en_{n}"
        )

    elif variant is CalibVariant.MIXED_WEIGHTED:
        selected = []
        composition: dict[str, int] = {}
        for class_name, weight in cfg.calibration.assumed_traffic_mix.items():
            take = round(n * weight)
            pool = by_class.get(class_name, [])
            got = _sample(pool, take, seed=cfg.seed, tag=f"weighted_{class_name}_{n}")
            composition[class_name] = len(got)
            selected.extend(got)
        meta["composition"] = composition
        # Spec section 5: the assumption must be documented, not just applied.
        meta["traffic_mix_assumption"] = dict(cfg.calibration.assumed_traffic_mix)
        meta["traffic_mix_rationale"] = cfg.calibration.traffic_mix_rationale

    elif variant is CalibVariant.COVERAGE_GREEDY:
        if not profiles_by_doc:
            raise RuntimeError(
                "coverage_greedy requires per-document routing profiles from "
                "phase 1. Run phase 1 before phase 2."
            )
        eligible = [d for d in candidates if d.doc_id in profiles_by_doc]
        if not eligible:
            raise RuntimeError("No candidate documents have routing profiles")
        matrix = np.vstack([profiles_by_doc[d.doc_id] for d in eligible])
        indices, trace = coverage_greedy_select(
            matrix,
            n,
            tiebreak_percentile=cfg.calibration.greedy_tiebreak_percentile,
            candidate_cap=cfg.calibration.greedy_candidate_cap,
            seed=cfg.seed,
        )
        selected = [eligible[i] for i in indices]
        meta["greedy_final_min_coverage"] = trace[-1]["min_coverage"] if trace else 0.0
        meta["greedy_final_p10_coverage"] = trace[-1]["p10_coverage"] if trace else 0.0
        meta["greedy_experts_unseen"] = trace[-1]["experts_unseen"] if trace else -1
        meta["greedy_trace"] = trace

    elif variant is CalibVariant.ORACLE_CONTAMINATED:
        # The one legitimate use of the evaluation split.
        eval_docs = load_eval_distribution(cfg)
        selected = _sample(eval_docs, n, seed=cfg.seed, tag=f"oracle_{n}")
        meta["contaminated"] = True
        meta["contamination_note"] = (
            "Calibrated directly on the evaluation distribution. NOT a "
            "legitimate result. Establishes the upper bound on what any "
            "calibration strategy could recover."
        )

    else:
        raise ValueError(f"Unhandled variant {variant}")

    meta["n_selected"] = len(selected)
    meta["class_breakdown"] = (
        pd.Series([d.corpus_class for d in selected]).value_counts().to_dict() if selected else {}
    )
    return selected, meta


# --------------------------------------------------------------------------
# coverage reporting
# --------------------------------------------------------------------------


def coverage_statistics(
    selected: list[CandidateDoc],
    profiles_by_doc: dict[str, np.ndarray] | None,
    n_experts: int,
) -> dict[str, float]:
    """Min and P10 per-expert token count for a selected set.

    Spec section 5: "Report the min and P10 per-expert token count for every
    variant - this table is a result in its own right."
    """
    if not profiles_by_doc:
        return {}
    vectors = [profiles_by_doc[d.doc_id] for d in selected if d.doc_id in profiles_by_doc]
    if not vectors:
        logger.warning(
            "No selected document carries a routing profile. Coverage for this "
            "variant is unmeasured, not zero - it must not be reported as a "
            "coverage failure."
        )
        return {
            "min_expert_tokens": float("nan"),
            "p10_expert_tokens": float("nan"),
            "median_expert_tokens": float("nan"),
            "mean_expert_tokens": float("nan"),
            "frac_experts_unseen": float("nan"),
            "n_docs_with_profile": 0.0,
            "profile_coverage_frac": 0.0,
            "min_expert_tokens_per_profiled_doc": float("nan"),
            "comparable": False,
            "gini_coefficient": float("nan"),
        }

    coverage = np.vstack(vectors).sum(axis=0).astype(np.float64)
    if coverage.shape[0] < n_experts:
        coverage = np.pad(coverage, (0, n_experts - coverage.shape[0]))

    # Coverage is summed over the selected documents that phase 1 actually
    # profiled. `coverage_greedy` selects only from profiled documents by
    # construction, so it is always measured at 1.0 while a randomly sampled
    # variant may be measured over a fraction of its documents. Comparing
    # absolute token counts across rows with different fractions overstates
    # coverage_greedy. Callers must condition on this column.
    profile_frac = len(vectors) / max(len(selected), 1)
    if profile_frac < PROFILE_COVERAGE_FLOOR:
        logger.warning(
            "Only %.0f%% of selected documents (%d/%d) carry routing profiles. "
            "Absolute coverage numbers for this variant are computed over that "
            "subset and are NOT comparable with a variant measured at a higher "
            "fraction. Enlarge data.candidate_pool_size so phase 1 profiles the "
            "whole train split.",
            100 * profile_frac,
            len(vectors),
            len(selected),
        )

    return {
        "min_expert_tokens": float(coverage.min()),
        "p10_expert_tokens": float(np.percentile(coverage, 10)),
        "median_expert_tokens": float(np.median(coverage)),
        "mean_expert_tokens": float(coverage.mean()),
        "max_expert_tokens": float(coverage.max()),
        "frac_experts_unseen": float((coverage == 0).mean()),
        "n_docs_with_profile": float(len(vectors)),
        "profile_coverage_frac": profile_frac,
        # Tokens per profiled document - the like-for-like figure. Absolute
        # min/P10 are what spec section 5 asks for and are kept, but this is
        # what may be compared across variants when the fractions differ.
        "min_expert_tokens_per_profiled_doc": float(coverage.min()) / max(len(vectors), 1),
        "comparable": profile_frac >= PROFILE_COVERAGE_FLOOR,
        "gini_coefficient": _gini(coverage),
    }


def _gini(values: np.ndarray) -> float:
    """Inequality of coverage across experts.

    A high Gini with an acceptable mean is the failure mode the study is
    looking for: plenty of calibration tokens overall, concentrated in experts
    the target language does not use.
    """
    if values.sum() == 0:
        return 0.0
    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n)


# --------------------------------------------------------------------------
# phase driver
# --------------------------------------------------------------------------


def load_profiles(cfg: Config) -> tuple[dict[str, np.ndarray], int]:
    """Rebuild per-document expert profiles from phase 1's sparse table."""
    path = cfg.paths.resolve("routing") / "per_document_profiles.parquet"
    df = read_parquet(path)
    n_experts = int(df["expert_id"].max()) + 1 if not df.empty else 0

    # Sum over layers: an expert's scale estimate draws on tokens it saw
    # anywhere in the network, and selection is per whole document.
    grouped = df.groupby(["doc_id", "expert_id"])["token_count"].sum().reset_index()
    profiles: dict[str, np.ndarray] = {}
    for doc_id, group in grouped.groupby("doc_id"):
        vector = np.zeros(n_experts, dtype=np.float64)
        vector[group["expert_id"].to_numpy()] = group["token_count"].to_numpy()
        profiles[str(doc_id)] = vector

    logger.info("Loaded routing profiles for %d documents (%d experts)", len(profiles), n_experts)
    return profiles, n_experts


def run_phase2(
    cfg: Config,
    *,
    variants: list[str] | None = None,
    sample_counts: list[int] | None = None,
) -> dict[str, Any]:
    """Build calibration variants and their coverage statistics."""
    fingerprint = cfg.fingerprint()
    out_dir = cfg.paths.resolve("calibration_sets")
    results_dir = cfg.paths.resolve("results")

    wanted_variants = (
        [CalibVariant(v) for v in variants] if variants else list(cfg.calibration.variants)
    )
    wanted_counts = sample_counts or list(cfg.effective_sample_counts())

    candidates = load_candidates(cfg)
    if not candidates:
        raise RuntimeError("No calibration candidates; run phase 0 first")

    try:
        profiles_by_doc, n_experts = load_profiles(cfg)
    except FileNotFoundError:
        if CalibVariant.COVERAGE_GREEDY in wanted_variants:
            raise
        logger.warning("No routing profiles; coverage statistics will be empty")
        profiles_by_doc, n_experts = {}, 0

    splits = load_splits(cfg.paths.resolve("splits"))
    eval_ids = {f"{name}:{i}" for name, split in splits.items() for i in split.eval_indices}

    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}

    for variant in wanted_variants:
        for n in wanted_counts:
            tag = f"{variant.value}_{n}"
            with tracking.run(cfg, phase="phase2", run_name=tag, tags={"variant": variant.value}):
                selected, meta = build_variant(cfg, variant, n, candidates, profiles_by_doc)

                # Spec section 3: assert disjointness before each phase.
                # oracle_contaminated must acknowledge its violation explicitly.
                assert_disjoint(
                    {d.doc_id for d in selected},
                    eval_ids,
                    context=f"phase2/{tag}",
                    allow_contamination=variant.is_contaminated,
                )

                path = out_dir / f"{tag}.jsonl"
                write_jsonl(
                    (
                        {
                            "id": d.doc_id,
                            "corpus_class": d.corpus_class,
                            "text": d.text,
                            "variant": variant.value,
                            "contaminated": variant.is_contaminated,
                        }
                        for d in selected
                    ),
                    path,
                )

                stats = coverage_statistics(selected, profiles_by_doc, n_experts)
                rows.append(
                    {
                        "variant": variant.value,
                        "n_samples": n,
                        "n_selected": len(selected),
                        "contaminated": variant.is_contaminated,
                        "path": str(path),
                        **stats,
                    }
                )
                # The greedy trace is large; keep it out of the metadata blob.
                metadata[tag] = {k: v for k, v in meta.items() if k != "greedy_trace"}
                tracking.log_metrics({k: v for k, v in stats.items() if isinstance(v, float)})
                logger.info(
                    "%s: %d docs, min expert tokens %.0f, P10 %.0f, %.1f%% experts unseen",
                    tag,
                    len(selected),
                    stats.get("min_expert_tokens", 0),
                    stats.get("p10_expert_tokens", 0),
                    100 * stats.get("frac_experts_unseen", 0),
                )

    coverage_table = pd.DataFrame(rows)
    write_parquet(
        coverage_table,
        results_dir / "calibration_coverage.parquet",
        fingerprint=fingerprint,
        phase="phase2",
    )
    write_json(
        {"variants": metadata, "n_experts": n_experts, "config_fingerprint": fingerprint},
        results_dir / "phase2_summary.json",
    )

    return {
        "n_variants": len(wanted_variants),
        "n_sample_counts": len(wanted_counts),
        "n_sets": len(rows),
        "n_experts": n_experts,
        "coverage_table": coverage_table.to_dict(orient="records"),
    }


__all__ = [
    "BM_CLASSES",
    "EN_CLASSES",
    "CandidateDoc",
    "CoverageState",
    "build_variant",
    "coverage_greedy_select",
    "coverage_statistics",
    "load_candidates",
    "load_eval_distribution",
    "load_profiles",
    "run_phase2",
]
