"""Tests for the invariants that would silently corrupt results if broken.

These are not coverage tests. Each one guards a failure mode that produces a
plausible-looking number rather than an error - the kind that survives review
and ends up in a report.
"""

from __future__ import annotations

from itertools import pairwise
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from ilmu_glossary.config import CalibVariant, Config, CorpusClass, LidConfig, RecipeFamily
from ilmu_glossary.seeds import derive_seed, rng
from ilmu_glossary.splits import (
    ContaminationError,
    assert_disjoint,
    assert_internally_consistent,
    make_split,
)

# --------------------------------------------------------------------------
# contamination
# --------------------------------------------------------------------------


class TestContaminationGuards:
    def test_split_is_disjoint(self) -> None:
        split = make_split("formal_bm", 10_000, base_seed=42)
        assert_internally_consistent(split)
        assert not set(split.train_indices) & set(split.eval_indices)
        assert len(split.train_indices) + len(split.eval_indices) == 10_000

    def test_split_is_deterministic(self) -> None:
        a = make_split("formal_bm", 5_000, base_seed=42)
        b = make_split("formal_bm", 5_000, base_seed=42)
        assert a.train_indices == b.train_indices

    def test_split_differs_per_class(self) -> None:
        """Adding a class must not reshuffle the splits of existing ones."""
        a = make_split("formal_bm", 5_000, base_seed=42)
        b = make_split("manglish", 5_000, base_seed=42)
        assert a.train_indices != b.train_indices

    def test_overlap_raises(self) -> None:
        with pytest.raises(ContaminationError, match="both the calibration set"):
            assert_disjoint({"a", "b"}, {"b", "c"}, context="test")

    def test_contamination_must_be_explicit(self) -> None:
        """oracle_contaminated may violate disjointness, but only by saying so."""
        overlapping = ({"a", "b"}, {"b", "c"})
        with pytest.raises(ContaminationError):
            assert_disjoint(*overlapping, context="test", allow_contamination=False)
        assert_disjoint(*overlapping, context="test", allow_contamination=True)

    def test_only_oracle_is_contaminated(self) -> None:
        contaminated = [v for v in CalibVariant if v.is_contaminated]
        assert contaminated == [CalibVariant.ORACLE_CONTAMINATED]


# --------------------------------------------------------------------------
# recipe identity
# --------------------------------------------------------------------------


class TestRecipeIdentity:
    @pytest.fixture
    def cfg(self) -> Config:
        return Config()

    def test_families_differ(self, cfg: Config) -> None:
        """A vacuous contrast would make the whole two-family design pointless."""
        from ilmu_glossary.quantize import load_recipe

        shipped = load_recipe(cfg, RecipeFamily.W4A16_SHIPPED)
        mechanism = load_recipe(cfg, RecipeFamily.W4A4_MECHANISM)
        assert shipped.identity_hash != mechanism.identity_hash
        assert shipped.expert_activation_bits == 16
        assert mechanism.expert_activation_bits == 4
        assert shipped.shipped_by_nvidia
        assert not mechanism.shipped_by_nvidia

    def test_calibration_change_preserves_hash(self, cfg: Config) -> None:
        """Changing the data source is the ONE thing a variant may do."""
        import copy

        from ilmu_glossary.quantize import load_recipe, recipe_identity_hash

        recipe = load_recipe(cfg, RecipeFamily.W4A16_SHIPPED)
        a = copy.deepcopy(recipe.payload)
        b = copy.deepcopy(recipe.payload)
        a["calibration"] = {"dataset_path": "/x/bm_only_256.jsonl", "num_samples": 256}
        b["calibration"] = {"dataset_path": "/x/baseline_en_2048.jsonl", "num_samples": 2048}
        assert recipe_identity_hash(a) == recipe_identity_hash(b)

    def test_precision_change_breaks_hash(self, cfg: Config) -> None:
        import copy

        from ilmu_glossary.quantize import load_recipe, recipe_identity_hash

        recipe = load_recipe(cfg, RecipeFamily.W4A16_SHIPPED)
        mutated = copy.deepcopy(recipe.payload)
        mutated["quantization"]["layers"][0]["block_size"] = 32
        assert recipe_identity_hash(mutated) != recipe.identity_hash

    def test_key_order_is_irrelevant(self, cfg: Config) -> None:
        from ilmu_glossary.quantize import load_recipe, recipe_identity_hash

        recipe = load_recipe(cfg, RecipeFamily.W4A16_SHIPPED)
        reordered = {k: recipe.payload[k] for k in reversed(list(recipe.payload))}
        assert recipe_identity_hash(reordered) == recipe.identity_hash

    def test_mismatched_recipes_rejected(self, cfg: Config) -> None:
        from ilmu_glossary.quantize import RecipeMismatchError, assert_recipe_identity, load_recipe

        recipes = [load_recipe(cfg, f) for f in RecipeFamily]
        with pytest.raises(RecipeMismatchError, match="only in their calibration"):
            assert_recipe_identity(recipes)

    def test_routers_excluded_from_quantization(self, cfg: Config) -> None:
        """A perturbed router changes which experts fire, confounding everything.

        Nemotron-H names the router `mixer.gate`; other architectures call it
        `router`. Either spelling satisfies the requirement - what matters is
        that the routing distribution cannot move while expert numerics do.
        """
        from ilmu_glossary.quantize import load_recipe

        for family in RecipeFamily:
            excluded = load_recipe(cfg, family).payload["quantization"]["exclude"]
            assert any("gate" in pattern or "router" in pattern for pattern in excluded), (
                f"{family.value} does not exclude the router: {excluded}"
            )

    def test_routed_experts_are_quantized_by_both_families(self, cfg: Config) -> None:
        """Both families must actually target routed experts, or the contrast
        between them says nothing."""
        from ilmu_glossary.quantize import load_recipe

        for family in RecipeFamily:
            layers = load_recipe(cfg, family).payload["quantization"]["layers"]
            expert_patterns = [
                layer["pattern"]
                for layer in layers
                if "experts" in layer["pattern"] and "shared" not in layer["pattern"]
            ]
            assert expert_patterns, f"{family.value} quantizes no routed experts"

    def test_lm_head_matches_nvidia(self, cfg: Config) -> None:
        """NVIDIA quantizes lm_head - their 'faithful lm_head'. An earlier draft
        excluded it, which would have diverged from the shipped checkpoint."""
        from ilmu_glossary.quantize import load_recipe

        for family in RecipeFamily:
            payload = load_recipe(cfg, family).payload["quantization"]
            patterns = [layer["pattern"] for layer in payload["layers"]]
            assert "lm_head" in patterns
            assert "lm_head" not in payload["exclude"]


# --------------------------------------------------------------------------
# coverage-greedy
# --------------------------------------------------------------------------


class TestCoverageGreedy:
    @staticmethod
    def _zipf_profiles(n_docs: int = 1200, n_experts: int = 64, top_k: int = 8) -> np.ndarray:
        """Realistic MoE routing: skewed popularity, every expert reachable.

        A quarter of documents prefer the unpopular tail, standing in for a
        language whose routing differs from the calibration corpus.
        """
        generator = np.random.default_rng(11)
        head = 1.0 / np.arange(1, n_experts + 1) ** 1.1
        head /= head.sum()
        tail = head[::-1].copy()

        profiles = np.zeros((n_docs, n_experts))
        for i in range(n_docs):
            p = tail if i % 4 == 0 else head
            idx = generator.choice(n_experts, size=top_k, replace=False, p=p)
            profiles[i, idx] = generator.integers(20, 200, size=top_k)
        return profiles

    def test_beats_random_on_min_coverage(self) -> None:
        """The whole point of the variant. Spec 5's literal rule fails this."""
        from ilmu_glossary.calib_select import coverage_greedy_select

        profiles = self._zipf_profiles()
        selected, _ = coverage_greedy_select(profiles, 128, seed=1)
        greedy = profiles[selected].sum(axis=0)

        random_mins = [
            profiles[np.random.default_rng(s).choice(len(profiles), 128, replace=False)]
            .sum(axis=0)
            .min()
            for s in range(10)
        ]
        assert greedy.min() > np.mean(random_mins) * 1.5

    def test_leaves_no_expert_unseen(self) -> None:
        from ilmu_glossary.calib_select import coverage_greedy_select

        profiles = self._zipf_profiles()
        selected, _ = coverage_greedy_select(profiles, 256, seed=1)
        assert (profiles[selected].sum(axis=0) == 0).sum() == 0

    def test_selects_exactly_n_unique(self) -> None:
        from ilmu_glossary.calib_select import coverage_greedy_select

        profiles = self._zipf_profiles()
        for n in (16, 64, 256):
            selected, _ = coverage_greedy_select(profiles, n, seed=1)
            assert len(selected) == n
            assert len(set(selected)) == n

    def test_saturated_coverage_is_monotone(self) -> None:
        """Submodularity requires monotonicity; it is what makes CELF exact."""
        from ilmu_glossary.calib_select import coverage_greedy_select

        _, trace = coverage_greedy_select(self._zipf_profiles(), 64, seed=1)
        values = [step["saturated_coverage"] for step in trace]
        assert all(a <= b + 1e-9 for a, b in pairwise(values))

    def test_deterministic(self) -> None:
        from ilmu_glossary.calib_select import coverage_greedy_select

        profiles = self._zipf_profiles()
        a, _ = coverage_greedy_select(profiles, 64, seed=1)
        b, _ = coverage_greedy_select(profiles, 64, seed=1)
        assert a == b

    def test_handles_oversized_request(self) -> None:
        from ilmu_glossary.calib_select import coverage_greedy_select

        profiles = self._zipf_profiles(n_docs=32)
        selected, _ = coverage_greedy_select(profiles, 100, seed=1)
        assert len(selected) == 32


# --------------------------------------------------------------------------
# KL estimator
# --------------------------------------------------------------------------


class TestKL:
    @staticmethod
    def _dist(probs: list[float]):  # type: ignore[no-untyped-def]
        from ilmu_glossary.evaluate.kl import TopKLogprobs

        p = np.array(probs, dtype=np.float32)
        order = np.argsort(-p)
        return TopKLogprobs(
            token_ids=order[None, :].astype(np.int32),
            logprobs=np.log(p[order])[None, :].astype(np.float32),
        )

    def test_identical_is_zero(self) -> None:
        from ilmu_glossary.evaluate.kl import token_kl

        d = self._dist([0.6, 0.3, 0.1])
        kl, _ = token_kl(d, d)
        assert kl[0] == pytest.approx(0.0, abs=1e-9)

    def test_different_is_positive(self) -> None:
        from ilmu_glossary.evaluate.kl import token_kl

        kl, _ = token_kl(self._dist([0.8, 0.15, 0.05]), self._dist([0.05, 0.15, 0.8]))
        assert kl[0] > 0

    def test_tail_mass_reported(self) -> None:
        from ilmu_glossary.evaluate.kl import TopKLogprobs

        truncated = TopKLogprobs(
            token_ids=np.array([[0, 1]], dtype=np.int32),
            logprobs=np.log(np.array([[0.5, 0.2]], dtype=np.float32)),
        )
        assert truncated.tail_mass()[0] == pytest.approx(0.3, abs=1e-6)

    def test_token_id_stable_across_processes(self) -> None:
        """Builtin hash() is randomised per interpreter; the cache crosses
        containers, so an unstable id would make every lookup miss."""
        import subprocess
        import sys

        code = "from ilmu_glossary.evaluate import stable_token_id; print(stable_token_id('Kuala'))"
        outputs = {
            subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                env={"PYTHONHASHSEED": str(seed), "PATH": "/usr/bin:/bin"},
                check=True,
            ).stdout.strip()
            for seed in (0, 1, 2)
        }
        assert len(outputs) == 1

    def test_null_delta_ci_includes_zero(self) -> None:
        from ilmu_glossary.evaluate.kl import bm_en_delta, summarise_kl

        generator = np.random.default_rng(0)
        a = generator.gamma(2.0, 0.04, 3000)
        b = generator.gamma(2.0, 0.04, 3000)
        tails = np.zeros(3000)
        delta = bm_en_delta(
            summarise_kl(a, tails, seed=1),
            summarise_kl(b, tails, seed=1),
            malay_values=a,
            english_values=b,
            seed=2,
        )
        assert delta["delta_excludes_zero"] == 0.0

    def test_real_delta_ci_excludes_zero(self) -> None:
        from ilmu_glossary.evaluate.kl import bm_en_delta, summarise_kl

        generator = np.random.default_rng(0)
        malay = generator.gamma(2.0, 0.05, 3000)
        english = generator.gamma(2.0, 0.03, 3000)
        tails = np.zeros(3000)
        delta = bm_en_delta(
            summarise_kl(malay, tails, seed=1),
            summarise_kl(english, tails, seed=1),
            malay_values=malay,
            english_values=english,
            seed=2,
        )
        assert delta["delta_excludes_zero"] == 1.0
        assert delta["bm_en_delta"] > 0


# --------------------------------------------------------------------------
# routing statistics
# --------------------------------------------------------------------------


class TestRouting:
    def test_kl_zero_for_identical(self) -> None:
        from ilmu_glossary.routing_analysis import kl_divergence

        p = np.array([0.5, 0.3, 0.2])
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)

    def test_jensen_shannon_symmetric(self) -> None:
        from ilmu_glossary.routing_analysis import jensen_shannon

        p, q = np.array([0.7, 0.2, 0.1]), np.array([0.1, 0.3, 0.6])
        assert jensen_shannon(p, q) == pytest.approx(jensen_shannon(q, p), abs=1e-12)

    def test_jaccard_bounds(self) -> None:
        from ilmu_glossary.routing_analysis import top_k_jaccard

        p = np.array([0.5, 0.3, 0.15, 0.05])
        assert top_k_jaccard(p, p, k=2) == 1.0
        assert top_k_jaccard(p, p[::-1].copy(), k=2) == 0.0

    def test_effect_size_not_p_value_drives_gate(self) -> None:
        """With millions of tokens chi-square is always significant, so a
        p-value-driven gate would open every time."""
        from ilmu_glossary.routing_analysis import evaluate_routing_gate

        indistinguishable = pd.DataFrame(
            [
                {
                    "class_a": CorpusClass.FORMAL_BM.value,
                    "class_b": CorpusClass.ENGLISH_CONTROL.value,
                    "layer_index": 0,
                    "cramers_v": 0.001,
                    "top32_jaccard": 1.0,
                    "p_value": 1e-40,  # significant, yet no real difference
                }
            ]
        )
        gate = evaluate_routing_gate(indistinguishable)
        assert gate["hypothesis_weakened"]
        assert gate["verdict"] == "routing_indistinguishable"

    def test_gate_opens_on_real_divergence(self) -> None:
        from ilmu_glossary.routing_analysis import evaluate_routing_gate

        divergent = pd.DataFrame(
            [
                {
                    "class_a": CorpusClass.FORMAL_BM.value,
                    "class_b": CorpusClass.ENGLISH_CONTROL.value,
                    "layer_index": 0,
                    "cramers_v": 0.31,
                    "top32_jaccard": 0.55,
                    "p_value": 1e-30,
                }
            ]
        )
        assert not evaluate_routing_gate(divergent)["hypothesis_weakened"]

    def test_router_layer_index_parsing(self) -> None:
        from ilmu_glossary.routing_analysis import _layer_index_from_name

        assert _layer_index_from_name("model.layers.17.mlp.gate") == 17
        assert _layer_index_from_name("backbone.blocks.3.moe.router") == 3
        assert _layer_index_from_name("no_index.gate") == -1


# --------------------------------------------------------------------------
# language identification
# --------------------------------------------------------------------------


class TestLanguageID:
    def test_marker_lists_disjoint(self) -> None:
        """A shared form contributes to both ratios and discriminates nothing."""
        cfg = LidConfig()
        assert not set(cfg.indonesian_markers) & set(cfg.malaysian_markers)

    def test_markers_discriminate(self) -> None:
        from ilmu_glossary.lid import marker_ratios

        cfg = LidConfig()
        indonesian, _, _ = marker_ratios("Saya bisa naik mobil kamu, gimana kalau di kantor", cfg)
        _, malaysian, _ = marker_ratios(
            "Saya nak naik kereta awak, macam mana kalau kat pejabat", cfg
        )
        assert indonesian > 0
        assert malaysian > 0
        # And the opposite ratio is clean in each case.
        _, my_in_id, _ = marker_ratios("Saya bisa naik mobil kamu, gimana kalau di kantor", cfg)
        id_in_my, _, _ = marker_ratios(
            "Saya nak naik kereta awak, macam mana kalau kat pejabat", cfg
        )
        assert my_in_id == 0.0
        assert id_in_my == 0.0

    def test_intrasentential_only(self) -> None:
        """Spec 3.3 excludes documents that merely alternate monolingual sentences."""
        from ilmu_glossary.lid import detect_intrasentential_switching

        intra = (
            "Saya nak pergi meeting tapi the traffic is very bad today lah. "
            "Boleh you tolong reschedule the appointment untuk saya?"
        )
        inter = (
            "Saya akan pergi ke pejabat pada waktu pagi esok hari. "
            "The weather today is quite pleasant and rather warm outside."
        )
        assert detect_intrasentential_switching(intra)[0]
        assert not detect_intrasentential_switching(inter)[0]

    def test_dialect_needs_multiple_markers(self) -> None:
        from ilmu_glossary.lid import detect_dialect

        assert detect_dialect("Gapo demo buat kito ambo sokmo")[0] == "kelantan"
        assert detect_dialect("Saya pergi ke pasar pagi tadi")[0] is None


# --------------------------------------------------------------------------
# deduplication and alignment
# --------------------------------------------------------------------------


class TestDataPrep:
    def test_dedup_catches_whitespace_variants(self) -> None:
        from ilmu_glossary.data_prep import Deduplicator

        dedup = Deduplicator(ngram=3, threshold=0.8)
        text = "Kerajaan Malaysia mengumumkan bajet baharu untuk tahun hadapan rakyat"
        assert dedup.add_if_new(text)
        assert not dedup.add_if_new(text)
        assert not dedup.add_if_new(text.replace(" ", "  "))
        assert dedup.add_if_new("Pasukan bola sepak negara menang perlawanan akhir semalam")

    def test_alignment_rejects_truncation(self) -> None:
        from ilmu_glossary.data_prep import check_alignment

        assert not check_alignment(
            "Kerajaan telah mengumumkan bajet 2026 untuk rakyat Malaysia semua hari ini",
            "The government announced",
        ).passed

    def test_alignment_rejects_off_topic(self) -> None:
        from ilmu_glossary.data_prep import check_alignment

        result = check_alignment(
            "Bajet 2026 diumumkan pada 5 Oktober oleh menteri kewangan negara kita",
            "The 1999 report was published on 12 March by a different ministry here",
        )
        assert not result.passed
        assert result.reason == "numeric_content_diverges"

    def test_alignment_accepts_good_pair(self) -> None:
        from ilmu_glossary.data_prep import check_alignment

        assert check_alignment(
            "Kerajaan telah mengumumkan bajet 2026 untuk semua rakyat Malaysia",
            "The government has announced the 2026 budget for all Malaysians",
        ).passed


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


class TestConfig:
    def test_fingerprint_stable(self) -> None:
        assert Config().fingerprint() == Config().fingerprint()

    def test_fingerprint_changes_with_settings(self) -> None:
        assert Config().fingerprint() != Config(seed=43).fingerprint()

    def test_traffic_mix_must_sum_to_one(self) -> None:
        from ilmu_glossary.config import CalibrationConfig

        with pytest.raises(ValueError, match=r"must sum to 1\.0"):
            Config(calibration=CalibrationConfig(assumed_traffic_mix={"formal_bm": 0.5}))

    def test_gate_sample_count_must_be_in_sweep(self) -> None:
        from ilmu_glossary.config import CalibrationConfig

        with pytest.raises(ValueError, match="not in"):
            Config(calibration=CalibrationConfig(sample_counts=(256, 512), gate_sample_count=1024))

    def test_seq_len_is_lightnings_not_the_specs(self) -> None:
        """32,768 is what NVIDIA tuned for Lightning; 65,536 is a Nemotron 3
        figure the spec carried over. See SPEC_DEVIATIONS.md D3."""
        assert Config().model.calib_seq_len == 32_768

    def test_derived_seeds_are_scoped(self) -> None:
        assert derive_seed(42, "phase2", "a") != derive_seed(42, "phase2", "b")
        assert derive_seed(42, "phase2", "a") == derive_seed(42, "phase2", "a")

    def test_rng_reproducible(self) -> None:
        a = rng(42, "x").random(5)
        b = rng(42, "x").random(5)
        assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


class TestReporting:
    @staticmethod
    def _kl_frame(oracle_delta: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "checkpoint": "w4a16_shipped_baseline_en_1024",
                    "variant": "baseline_en",
                    "contaminated": False,
                    "bm_en_delta": 0.040,
                },
                {
                    "checkpoint": "w4a16_shipped_oracle_contaminated_1024",
                    "variant": "oracle_contaminated",
                    "contaminated": True,
                    "bm_en_delta": oracle_delta,
                },
            ]
        )

    def test_gate_closes_when_oracle_matches_baseline(self) -> None:
        from ilmu_glossary.analyze import headroom_analysis

        result = headroom_analysis(self._kl_frame(0.039), Config())
        assert not result["gate_open"]
        assert "NOT the lever" in result["verdict"]

    def test_gate_opens_on_real_headroom(self) -> None:
        from ilmu_glossary.analyze import headroom_analysis

        assert headroom_analysis(self._kl_frame(0.008), Config())["gate_open"]

    def test_null_result_is_in_the_abstract(self) -> None:
        """Spec 8: 'State it in the abstract, not a footnote.'"""
        from ilmu_glossary.analyze import Artifacts, build_report

        report = build_report(Config(), Artifacts(kl=self._kl_frame(0.039)))
        abstract = report.split("## Abstract")[1].split("## Scope")[0]
        assert "Null result" in abstract
        assert "not the lever" in abstract.lower()

    def test_contamination_labelled(self) -> None:
        from ilmu_glossary.analyze import Artifacts, build_report

        report = build_report(Config(), Artifacts(kl=self._kl_frame(0.008)))
        assert "CONTAMINATED" in report

    def test_no_ilmu_accuracy_claim(self) -> None:
        """Spec 8 requires this statement explicitly."""
        from ilmu_glossary.analyze import Artifacts, build_report

        report = build_report(Config(), Artifacts(kl=self._kl_frame(0.008)))
        assert "No claim is made about ILMU accuracy" in report
        assert "not ILMU's fine-tune" in report

    def test_b200_proxy_stated(self) -> None:
        from ilmu_glossary.analyze import Artifacts, build_report

        assert "GB200 NVL72" in build_report(Config(), Artifacts())

    def test_renders_with_no_artifacts(self) -> None:
        from ilmu_glossary.analyze import Artifacts, build_report

        report = build_report(Config(), Artifacts())
        assert "No conclusion is drawn" in report

    def test_winner_excludes_contaminated(self) -> None:
        """oracle_contaminated must never win anything."""
        from ilmu_glossary.orchestrate import _select_winner

        frame = pd.DataFrame(
            [
                {"checkpoint": "a", "variant": "oracle_contaminated", "bm_en_delta": 0.001},
                {"checkpoint": "b", "variant": "coverage_greedy", "bm_en_delta": 0.010},
            ]
        )

        class FakeArtifacts:
            kl = frame

        import ilmu_glossary.analyze as analyze_mod

        original = analyze_mod.load_artifacts
        analyze_mod.load_artifacts = lambda _cfg: FakeArtifacts()  # type: ignore[assignment]
        try:
            assert _select_winner(Config()) == "b"
        finally:
            analyze_mod.load_artifacts = original


# --------------------------------------------------------------------------
# staged pipeline
# --------------------------------------------------------------------------


class TestStagedPipeline:
    """The gates exist to stop the run before it spends money on a question
    already answered. If they stop firing, the study silently costs ~24x more."""

    @staticmethod
    def _run(tmp_path, oracle_delta: float) -> tuple[dict, int]:  # type: ignore[no-untyped-def]
        from ilmu_glossary import orchestrate
        from ilmu_glossary.config import PathsConfig
        from ilmu_glossary.io import write_json, write_parquet

        cfg = Config(paths=PathsConfig(root=tmp_path))
        results = cfg.paths.resolve("results")
        eval_dir = cfg.paths.resolve("eval")
        results.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            {"routing_gate": {"verdict": "routing_differs", "hypothesis_weakened": False}},
            results / "phase1_summary.json",
        )

        calls: list[str] = []
        quantized: list[str] = []

        def stub(name: str):  # type: ignore[no-untyped-def]
            def fn(**kwargs: object) -> dict[str, str]:
                calls.append(name)
                if name == "phase3":
                    quantized.append(
                        f"{kwargs.get('family')}_{kwargs.get('variant')}_{kwargs.get('sample_count')}"
                    )
                checkpoint = str(kwargs.get("checkpoint", ""))
                if name == "phase4" and "oracle" in checkpoint:
                    write_parquet(
                        pd.DataFrame(
                            [
                                {
                                    "checkpoint": "o",
                                    "variant": "oracle_contaminated",
                                    "contaminated": True,
                                    "bm_en_delta": oracle_delta,
                                }
                            ]
                        ),
                        eval_dir / "o_kl.parquet",
                        fingerprint=cfg.fingerprint(),
                        phase="phase4",
                    )
                if name == "phase4" and "baseline_en" in checkpoint:
                    write_parquet(
                        pd.DataFrame(
                            [
                                {
                                    "checkpoint": "b",
                                    "variant": "baseline_en",
                                    "contaminated": False,
                                    "bm_en_delta": 0.040,
                                }
                            ]
                        ),
                        eval_dir / "b_kl.parquet",
                        fingerprint=cfg.fingerprint(),
                        phase="phase4",
                    )
                return {"ok": name}

            return fn

        journal = orchestrate.staged_pipeline(
            cfg,
            phase0=stub("phase0"),
            phase1=stub("phase1"),
            phase2=stub("phase2"),
            phase3=stub("phase3"),
            phase4=stub("phase4"),
            phase5=stub("phase5"),
        )
        return journal, quantized

    def test_null_headroom_stops_before_the_matrix(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        journal, quantized = self._run(tmp_path, oracle_delta=0.039)
        assert journal["stopped_early"] is True
        assert "not the lever" in journal["stop_reason"].lower()
        # Only the two gate cells. The full matrix is 48 PTQ runs on an
        # hourly-billed B200; the gate is what makes a null result cheap.
        assert len(quantized) == 2
        assert set(quantized) == {
            "w4a16_shipped_oracle_contaminated_1024",
            "w4a16_shipped_baseline_en_1024",
        }

    def test_real_headroom_runs_the_matrix(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        journal, quantized = self._run(tmp_path, oracle_delta=0.008)
        assert journal["stopped_early"] is False
        # 2 families x 6 variants x 4 sample counts, gate cells included once.
        assert len(quantized) == 2 * 6 * 4

    def test_gate_cells_are_not_requantized(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The matrix must skip what the gate already built, or two of the
        48 runs are paid for twice."""
        _, quantized = self._run(tmp_path, oracle_delta=0.008)
        assert len(quantized) == len(set(quantized))

    def test_phase5_runs_even_when_gate_closes(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A null result still needs its report written."""
        journal, _ = self._run(tmp_path, oracle_delta=0.039)
        assert "phase5" in journal["stages"]


@pytest.mark.network
class TestLidLabelsMatchModels:
    """The label constants must cover what the models actually emit.

    An unrecognised label does not raise - it falls through to "other" and the
    document is dropped. That is how `english_control` came out with 0 of
    20,000 records on the first dry run: the code looked for `eng` while v2
    emits `standard-english`. The model cards are unreliable here (v2's omits
    `manglish`, `standard-indonesian` and both mandarin labels), so this reads
    the labels from the model files themselves.
    """

    REPOS = (
        "mesolitica/fasttext-language-detection-v1",
        "mesolitica/fasttext-language-detection-v2",
        "mesolitica/fasttext-language-detection-ms-id",
    )

    def test_every_emitted_label_is_known(self) -> None:
        fasttext = pytest.importorskip("fasttext")
        from huggingface_hub import hf_hub_download

        from ilmu_glossary.lid import KNOWN_LABELS

        unknown: dict[str, set[str]] = {}
        for repo in self.REPOS:
            path = hf_hub_download(repo, "fasttext.ftz")
            labels = {label.removeprefix("__label__") for label in fasttext.load_model(path).labels}
            missing = labels - KNOWN_LABELS
            if missing:
                unknown[repo] = missing

        assert not unknown, (
            f"LID models emit labels the pipeline does not recognise: {unknown}. "
            "Documents carrying them are silently dropped."
        )


class TestExpertQuantizationGuard:
    """A checkpoint whose routed experts were never quantized still loads,
    still serves, and still yields a full set of evaluation numbers - numbers
    that say nothing about the hypothesis. This is the most dangerous failure
    mode in the pipeline, so it must be fatal rather than a warning."""

    class _Quantizer:
        def __init__(self, enabled: bool) -> None:
            self.is_enabled = enabled

    class _FakeModule:
        def __init__(self, params: dict[str, object], quantizer: object | None = None) -> None:
            self._params = params
            if quantizer is not None:
                self.weight_quantizer = quantizer

        def named_parameters(self, recurse: bool = True) -> list[tuple[str, object]]:
            del recurse
            return list(self._params.items())

    class _FakeModel:
        def __init__(self, modules: dict[str, object]) -> None:
            self._modules_by_name = modules

        def named_modules(self) -> list[tuple[str, object]]:
            return list(self._modules_by_name.items())

    def _model(self, *, experts_quantized: bool):  # type: ignore[no-untyped-def]
        q = self._Quantizer(True) if experts_quantized else None
        return self._FakeModel(
            {
                # transformers 5.x fused layout: one carrier, 3D batched params.
                "model.layers.0.mlp.experts": self._FakeModule(
                    {"gate_up_proj": object(), "down_proj": object()}, q
                ),
                "model.layers.0.mlp.shared_expert.gate_proj": self._FakeModule(
                    {"weight": object()}, self._Quantizer(True)
                ),
                "model.layers.0.self_attn.q_proj": self._FakeModule(
                    {"weight": object()}, self._Quantizer(True)
                ),
            }
        )

    def test_shared_experts_do_not_satisfy_the_guard(self) -> None:
        """Shared experts see every token regardless of routing, so quantizing
        them proves nothing about routing-driven coverage."""
        from ilmu_glossary.quantize import UnquantizedExpertsError, assert_experts_quantized

        with pytest.raises(UnquantizedExpertsError, match="NONE quantized"):
            assert_experts_quantized(self._model(experts_quantized=False), family="w4a16_shipped")

    def test_passes_when_routed_experts_are_quantized(self) -> None:
        from ilmu_glossary.quantize import assert_experts_quantized

        carriers, quantized = assert_experts_quantized(
            self._model(experts_quantized=True), family="w4a16_shipped"
        )
        assert carriers == 1
        assert quantized == 1

    def test_no_carriers_at_all_is_fatal(self) -> None:
        from ilmu_glossary.quantize import UnquantizedExpertsError, assert_experts_quantized

        empty = self._FakeModel(
            {"model.layers.0.self_attn.q_proj": self._FakeModule({"weight": object()})}
        )
        with pytest.raises(UnquantizedExpertsError, match="No modules carrying"):
            assert_experts_quantized(empty, family="w4a16_shipped")

    def test_fused_3d_carrier_is_detected(self) -> None:
        """The fused layout must be counted; a Linear-only detector reports
        zero carriers and the guard would misdiagnose the failure."""
        from ilmu_glossary.quantize import count_routed_expert_carriers

        carriers, _ = count_routed_expert_carriers(self._model(experts_quantized=True))
        assert carriers == 1


class TestMalayMmluPrompt:
    """MalayMMLU ships options already lettered AND already embedded in the
    `prompt` field. Rendering them again yields "A. A. kunci (keys)" under a
    duplicated option block - a malformed prompt that still produces
    accuracies, so it would never announce itself."""

    RECORD: ClassVar[dict[str, object]] = {
        "id": 1,
        "prompt": (
            "Pasangan algoritma yang digunakan untuk melakukan penyulitan "
            "dan nyahsulit dikenali sebagai\nA. kunci (keys)\nB. Sifer (cipher)\n"
            "C. Teks sifer (ciphertext)"
        ),
        "answer": "B. Sifer (cipher)",
        "options": ["A. kunci (keys)", "B. Sifer (cipher)", "C. Teks sifer (ciphertext)"],
        "key": "B",
        "subject": "Sains Komputer",
        "category": "STEM",
        "level": "Secondary",
    }

    def _parsed(self):  # type: ignore[no-untyped-def]
        from ilmu_glossary.evaluate.mmlu import _parse_malay_mmlu_record

        question = _parse_malay_mmlu_record(self.RECORD)
        assert question is not None
        return question

    def test_options_are_stored_without_their_letter(self) -> None:
        assert self._parsed().options == (
            "kunci (keys)",
            "Sifer (cipher)",
            "Teks sifer (ciphertext)",
        )

    def test_prompt_renders_each_option_exactly_once(self) -> None:
        prompt = self._parsed().prompt()
        for text in ("kunci (keys)", "Sifer (cipher)", "Teks sifer (ciphertext)"):
            assert prompt.count(text) == 1, f"{text!r} appears {prompt.count(text)} times"

    def test_no_doubled_letter_prefix(self) -> None:
        import re

        assert re.search(r"\b([A-E])\.\s+\1\.\s", self._parsed().prompt()) is None

    def test_answer_letter_comes_from_key(self) -> None:
        question = self._parsed()
        assert question.answer_letter == "B"
        assert question.options[question.answer_index] == "Sifer (cipher)"

    def test_embedding_is_detected_not_assumed(self) -> None:
        """A revision that stops embedding options must still get them."""
        from ilmu_glossary.evaluate.mmlu import _parse_malay_mmlu_record

        bare = {**self.RECORD, "prompt": "Pasangan algoritma ... dikenali sebagai"}
        question = _parse_malay_mmlu_record(bare)
        assert question is not None
        assert not question.options_in_question
        prompt = question.prompt()
        for text in ("kunci (keys)", "Sifer (cipher)"):
            assert prompt.count(text) == 1

    def test_strip_option_letter_variants(self) -> None:
        from ilmu_glossary.evaluate.mmlu import strip_option_letter

        assert strip_option_letter("A. kunci") == ("A", "kunci")
        assert strip_option_letter("B) Sifer") == ("B", "Sifer")
        assert strip_option_letter("(C) Teks") == ("C", "Teks")
        assert strip_option_letter("no letter here") == (None, "no letter here")


class TestCrossMmluParsing:
    """SeaEval keeps each language in a COLUMN of one `test` split, with
    choices lettered "(A) ...". Treating language as a split raises
    Unknown split "malay" and tier 3 silently loads nothing."""

    CELL: ClassVar[dict[str, object]] = {
        "question": "According to Adler, firstborn children are more likely to be",
        "choices": ["(A) responsible", "(B) funny", "(C) sociable", "(D) followers"],
        "answer": "(A) responsible",
    }

    def test_letters_stripped_and_answer_resolved(self) -> None:
        from ilmu_glossary.evaluate.mmlu import _parse_cross_mmlu_cell

        q = _parse_cross_mmlu_cell(self.CELL, "test_1", "english")
        assert q is not None
        assert q.options == ("responsible", "funny", "sociable", "followers")
        assert q.answer_letter == "A"
        assert not q.options_in_question

    def test_options_rendered_once(self) -> None:
        from ilmu_glossary.evaluate.mmlu import _parse_cross_mmlu_cell

        q = _parse_cross_mmlu_cell(self.CELL, "test_1", "english")
        assert q is not None
        prompt = q.prompt()
        for text in ("responsible", "funny", "sociable", "followers"):
            assert prompt.count(text) == 1

    def test_non_parallel_languages_are_flagged(self, caplog) -> None:  # type: ignore[no-untyped-def]
        """Tier 3 is the controlled comparison; misalignment must not pass silently."""
        import logging

        from ilmu_glossary.evaluate.mmlu import MCQuestion, _assert_parallel

        def q(qid: str, lang: str) -> MCQuestion:
            return MCQuestion(qid, "x", ("a", "b"), 0, language=lang)

        with caplog.at_level(logging.WARNING):
            _assert_parallel(
                {"english": [q("1", "english"), q("2", "english")], "malay": [q("1", "malay")]}
            )
        assert "absent from at least one" in caplog.text
