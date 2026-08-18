"""Typed configuration for the routing-aware calibration study.

Every phase reads its settings from here. Nothing else in the codebase is
allowed to hardcode a model id, a corpus name, a sample count, or a path -
if a phase needs a constant, it belongs in this module or in configs/*.yaml.

The config is loaded once, hashed, and the hash is written into every results
artifact so a parquet file can always be traced back to the settings that
produced it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------
# enums
# --------------------------------------------------------------------------


class CorpusClass(StrEnum):
    """The six stratification classes from spec section 3, plus the parallel set.

    `dialect` is optional: spec section 3.4 permits dropping it if fewer than
    2,000 documents can be sourced, and requires the omission be recorded
    rather than padded with formal BM.
    """

    FORMAL_BM = "formal_bm"
    MANGLISH = "manglish"
    CODE_SWITCHED = "code_switched"
    DIALECT = "dialect"
    ENGLISH_CONTROL = "english_control"
    CODE_CONTROL = "code_control"

    @classmethod
    def required(cls) -> tuple[CorpusClass, ...]:
        return tuple(c for c in cls if c is not cls.DIALECT)


class CalibVariant(StrEnum):
    """Calibration set compositions from spec section 5."""

    BASELINE_EN = "baseline_en"
    BM_ONLY = "bm_only"
    MIXED_5050 = "mixed_5050"
    COVERAGE_GREEDY = "coverage_greedy"
    MIXED_WEIGHTED = "mixed_weighted"
    ORACLE_CONTAMINATED = "oracle_contaminated"

    @property
    def is_contaminated(self) -> bool:
        """True for variants that must be labelled as not-a-legitimate-result.

        Spec section 5 requires this label appear in every table, plot and
        caption. Reporting code keys off this property rather than string
        comparison so the label can never be forgotten.
        """
        return self is CalibVariant.ORACLE_CONTAMINATED


class RecipeFamily(StrEnum):
    """Two precision families, run in parallel (see SPEC_DEVIATIONS.md).

    W4A16_SHIPPED reproduces NVIDIA's published Lightning recipe. Expert
    weights are quantized without activation scales, so calibration data
    influences the checkpoint only through the static MSE precision search.

    W4A4_MECHANISM adds per-tensor activation quantization on the routed
    experts. This is the configuration in which spec section 0's stated
    mechanism - sparse routing starving per-tensor amax estimates - can
    actually operate. It is not a configuration NVIDIA ships.
    """

    W4A16_SHIPPED = "w4a16_shipped"
    W4A4_MECHANISM = "w4a4_mechanism"


class MambaStateDtype(StrEnum):
    """Spec section 6 additional arm - Blackwell-only measurement."""

    FP32 = "float32"
    FP16_SR = "float16_stochastic_rounding"


# --------------------------------------------------------------------------
# sub-configs
# --------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Model identity.

    Spec section 1 warns against hardcoding HuggingFace ids. The ids below were
    resolved by web search on `resolved_on` and are additionally re-validated
    at run time by `ilmu_glossary.resolve.assert_model_resolves`, which fails
    loudly if a repo stops resolving or a newer revision appears.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bf16_repo: str = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    nvfp4_reference_repo: str = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
    revision: str = "main"
    resolved_on: date = date(2026, 8, 18)
    trust_remote_code: bool = True

    # Fallback if Lightning's ModelOpt recipes become unavailable (spec s1).
    fallback_bf16_repo: str = "nvidia/Nemotron-3-Super-120B-A12B-BF16"

    # NVIDIA calibrated Lightning at 32,768 tokens after sweeping 8k-128k.
    # The spec's 65,536 figure belongs to Nemotron 3, not this model.
    calib_seq_len: int = 32_768
    # Serving context for evaluation. Full 1M is not needed and costs cache.
    eval_max_model_len: int = 65_536

    @property
    def supported_languages(self) -> tuple[str, ...]:
        """Languages NVIDIA post-trained for. Malay is deliberately absent -
        that absence is the premise of this study."""
        return ("en", "es", "fr", "de", "it", "ja")


class VllmConfig(BaseModel):
    """Serving flags, mirrored from vLLM's day-0 Lightning post (2026-08-10).

    These are held constant across every checkpoint. Only the model path
    changes, or the throughput comparison in tier 4e is meaningless.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_version: str = "0.27.1"
    image_tag: str = "vllm/vllm-openai:v0.27.1"
    mamba_backend: str = "flashinfer"
    moe_backend: str = "humming"
    linear_backend: str = "humming"
    mamba_ssu_algorithm: str = "horizontal"
    mamba_cache_mode: str = "align"
    mamba_cache_philox_rounds: int = 5
    reasoning_parser: str = "nemotron_v3"
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 32_768
    gpu_memory_utilization: float = 0.85
    enable_prefix_caching: bool = True
    async_scheduling: bool = True
    port: int = 8000
    # Skip torch.compile and CUDA graph capture. Startup on the dry-run proxy
    # hung for 16 minutes in graph capture after compilation completed, which
    # is pure cost for a run whose purpose is plumbing rather than
    # performance. Throughput measured under eager is NOT representative, so
    # tier 4e records the mode alongside its numbers.
    enforce_eager: bool = False

    def serve_args(
        self,
        model_path: str,
        *,
        mamba_state: MambaStateDtype,
        max_model_len: int,
        hybrid_mamba: bool = True,
        max_logprobs: int | None = None,
    ) -> list[str]:
        """Build the `vllm serve` argv. Single source of truth for serving.

        `hybrid_mamba=False` drops every Nemotron- and Mamba-specific flag.
        The dry-run proxy is an ordinary transformer MoE with no SSM layers,
        and vLLM refuses to start when handed `--mamba-backend` or
        `--reasoning-parser nemotron_v3` for a model that has neither.
        """
        args = [
            "vllm",
            "serve",
            model_path,
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--max-num-batched-tokens",
            str(self.max_num_batched_tokens),
            "--max-model-len",
            str(max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            # Tier 1 asks for the top-K logprobs per token. vLLM rejects any
            # request whose `logprobs` exceeds this with a bare HTTP 400, so it
            # must be raised to at least eval.kl_top_k or the primary metric
            # cannot be collected at all.
            "--max-logprobs",
            str(max(max_logprobs or 20, 20)),
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
        ]
        if self.enable_prefix_caching:
            args.append("--enable-prefix-caching")
        if self.enforce_eager:
            args.append("--enforce-eager")

        if not hybrid_mamba:
            return args

        args += [
            "--mamba-backend",
            self.mamba_backend,
            "--moe-backend",
            self.moe_backend,
            "--linear-backend",
            self.linear_backend,
            "--mamba-ssu-algorithm",
            self.mamba_ssu_algorithm,
            "--mamba-cache-mode",
            self.mamba_cache_mode,
            "--reasoning-parser",
            self.reasoning_parser,
        ]
        if self.async_scheduling:
            args.append("--async-scheduling")

        # Spec section 6 arm: FP32 vs FP16+stochastic-rounding SSM state.
        # NVIDIA ships FP16+SR by default; FP32 is the spec's stated baseline.
        if mamba_state is MambaStateDtype.FP16_SR:
            args += [
                "--mamba-ssm-cache-dtype",
                "float16",
                "--enable-mamba-cache-stochastic-rounding",
                "--mamba-cache-philox-rounds",
                str(self.mamba_cache_philox_rounds),
            ]
        else:
            args += ["--mamba-ssm-cache-dtype", "float32"]
        return args


class RecordFormat(StrEnum):
    """Physical shape of one line in an upstream JSONL file.

    These repos are bare JSONL dumps rather than curated datasets, and the
    shapes genuinely differ - verified 2026-08-18 by reading the first line
    of each. Assuming `{"text": ...}` everywhere is what made phase 0 fail
    with "'str' object is not a mapping".
    """

    OBJECT = "object"  # {"text": "..."}          fineweb-edu, codeparrot
    BARE_STRING = "bare_string"  # "..."          malaysia-ai/dedup-text-dataset
    JSON_ARRAY = "json_array"  # [tag, text, {...}]  chatgpt-noisy-translation
    INSTRUCTION_PAIR = "instruction_pair"  # {prompt_input, input, output}


class SourceSpec(BaseModel):
    """One upstream dataset feeding one or more corpus classes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo_id: str
    kind: Literal["hf_dataset", "hf_collection", "github"]
    config_name: str | None = None
    split: str = "train"
    text_field: str = "text"

    # `datasets` schema inference fails on bare-JSONL repos with hundreds of
    # heterogeneous files, so those are read line by line instead.
    loader: Literal["datasets", "raw_jsonl"] = "datasets"
    data_files: tuple[str, ...] = ()
    record_format: RecordFormat = RecordFormat.OBJECT
    array_text_index: int = 1
    # Streaming avoids materialising the 90B-token pretrain corpus. Spec
    # reproducibility is preserved by pinning `revision` and recording the
    # number of records consumed per shard in the manifest.
    streaming: bool = True
    revision: str | None = None
    licence: str = "unverified"
    provenance: Literal["malaysian", "english", "code", "mixed"] = "malaysian"
    notes: str = ""


class DataConfig(BaseModel):
    """Phase 0 settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_docs_per_class: int = 10_000
    min_dialect_docs: int = 2_000
    min_parallel_pairs: int = 5_000
    alignment_check_sample: int = 200
    language_spotcheck_sample: int = 100
    candidate_pool_size: int = 50_000  # spec section 4
    train_fraction: float = 0.8
    dedup_ngram: int = 13
    dedup_threshold: float = 0.8
    # Documents shorter than this contribute little to a 32K calibration
    # sample and distort the fertility statistic.
    min_doc_chars: int = 200
    # Hard bound on records consumed from any single upstream source. The bulk
    # Malaysian corpus is effectively unbounded when streamed, so without this
    # phase 0 never reaches its later sources.
    max_records_per_source: int = 400_000

    sources: dict[str, list[SourceSpec]] = Field(default_factory=dict)


class LidConfig(BaseModel):
    """Bahasa Melayu vs Bahasa Indonesia discrimination (spec section 9, top risk).

    mesolitica's fastText models are used in preference to GlotLID/OpenLID:
    they are trained specifically on the ms/id boundary (96.58% on that pair)
    and v2 emits `manglish` and `rojak` labels that the `manglish` and
    `code_switched` classes depend on. General-purpose LID collapses those
    into a single `msa` bucket.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    primary_model: str = "mesolitica/fasttext-language-detection-v2"
    ms_id_model: str = "mesolitica/fasttext-language-detection-ms-id"
    quantized: bool = True
    # Both repos publish `fasttext.ftz` / `fasttext.bin`, not the `model.*`
    # convention. Verified against the Hub 2026-08-18.
    weights_filename: str = "fasttext.ftz"
    weights_filename_full: str = "fasttext.bin"
    # When the fastText models cannot be loaded the identifier degrades to
    # lexicon-only. In that mode a source whose provenance is declared
    # English or code is trusted rather than rejected - otherwise the control
    # classes come out empty, which is what happened on the first dry run.
    trust_provenance_when_degraded: bool = True
    # A document must clear this to be admitted to a Malay class.
    min_malay_prob: float = 0.70
    # And must not look more Indonesian than Malay by this margin.
    max_indonesian_prob: float = 0.20
    min_english_prob: float = 0.80  # for english_control

    # Layer 3: discriminative lexicon. Presence of Indonesian-only forms is
    # evidence against Malaysian provenance even when fastText is confident.
    #
    # The two lists MUST stay disjoint. A form common to both languages
    # ("saya", "tidak", "macam", "kenapa", "sangat", "cuma", "buat", "anda")
    # contributes to both ratios and cancels out, adding noise while
    # discriminating nothing. `_check_markers_disjoint` enforces this.
    indonesian_markers: tuple[str, ...] = (
        # lexical: standard Indonesian where Malay uses a different word
        "bisa",
        "mobil",
        "kamu",
        "mau",
        "kenapa sih",
        "sepeda",
        "ban",
        "cewek",
        "cowok",
        "ngomong",
        "bicara",
        "kantor",
        "bilang",
        # colloquial Jakarta forms with no Malaysian counterpart
        "gimana",
        "banget",
        "nggak",
        "enggak",
        "udah",
        "aja",
        "doang",
        "kayak",
        "bikin",
        "ngapain",
        "lu",
        "gue",
        "sih",
        "dong",
        "kok",
        "deh",
        "nih",
        "tuh",
        "banget banget",
    )
    malaysian_markers: tuple[str, ...] = (
        # lexical: Malaysian standard where Indonesian differs
        "kereta",
        "basikal",
        "tayar",
        "pejabat",
        "cakap",
        "jom",
        "kedai",
        "wang",
        "cikgu",
        "rasuah",
        "gaduh",
        # colloquial Malaysian particles and contractions
        "nak",
        "tak",
        "dah",
        "je",
        "korang",
        "awak",
        "macam mana",
        "apa khabar",
        "sahaja",
        "boleh",
        "tengok",
        "kat",
    )
    max_indonesian_marker_ratio: float = 0.02

    @model_validator(mode="after")
    def _check_markers_disjoint(self) -> Self:
        overlap = set(self.indonesian_markers) & set(self.malaysian_markers)
        if overlap:
            raise ValueError(
                "indonesian_markers and malaysian_markers overlap on "
                f"{sorted(overlap)}. A shared form contributes to both ratios "
                "and discriminates nothing."
            )
        return self


class CalibrationConfig(BaseModel):
    """Phase 2 settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_counts: tuple[int, ...] = (256, 512, 1024, 2048)
    variants: tuple[CalibVariant, ...] = tuple(CalibVariant)

    # Spec section 5: run this first at this N, then gate on the headroom.
    gate_variant: CalibVariant = CalibVariant.ORACLE_CONTAMINATED
    gate_sample_count: int = 1024
    # If oracle beats baseline_en by less than this on the primary metric
    # (BM-EN KL delta reduction), calibration is not the lever - record and stop.
    gate_min_relative_headroom: float = 0.10

    # `mixed_weighted` composition. This is an assumption, not a measurement,
    # and spec section 5 requires it be documented as such. Rationale: a
    # Malaysian assistant deployment is assumed to see mostly colloquial and
    # code-switched Malay, with a substantial English tail from technical and
    # transactional queries.
    assumed_traffic_mix: dict[str, float] = Field(
        default_factory=lambda: {
            CorpusClass.FORMAL_BM.value: 0.25,
            CorpusClass.MANGLISH.value: 0.30,
            CorpusClass.CODE_SWITCHED.value: 0.20,
            CorpusClass.ENGLISH_CONTROL.value: 0.20,
            CorpusClass.CODE_CONTROL.value: 0.05,
        }
    )
    traffic_mix_rationale: str = (
        "Assumed, not measured. Derived from the register distribution of "
        "Malaysian consumer-assistant traffic described in mesolitica's social "
        "media snapshot collection. No production traffic was observed. "
        "Treat as a stated prior; sensitivity is not tested."
    )

    # Coverage-greedy runtime guards (spec section 5).
    greedy_candidate_cap: int = 4_000
    greedy_lazy_queue: bool = True
    greedy_tiebreak_percentile: float = 10.0


class QuantConfig(BaseModel):
    """Phase 3 settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    families: tuple[RecipeFamily, ...] = tuple(RecipeFamily)
    recipe_dir: Path = Path("recipes")
    # Upstream recipe this study forks. Only the calibration data source may
    # differ between variants within a family; `assert_recipe_identity` enforces it.
    upstream_recipe_path: str = (
        "modelopt_recipes/huggingface/models/nvidia/"
        "Nemotron-3.5-Lightning-30B-A3B-BF16/ptq/w4a16_nvfp4_4o6.yaml"
    )
    upstream_repo: str = "https://github.com/NVIDIA/Model-Optimizer"
    calib_batch_size: int = 1
    # Mamba state arm runs on the winning variant only (spec section 6).
    mamba_state_arms: tuple[MambaStateDtype, ...] = tuple(MambaStateDtype)


class EvalConfig(BaseModel):
    """Phase 4 settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Tier 1 - primary metric. Full-vocab logit capture is infeasible
    # (5k pairs x 32k tokens x ~150k vocab). Top-K logprobs plus tail mass
    # bound the KL; see evaluate/kl.py for the estimator and its error bound.
    kl_top_k: int = 512
    kl_max_pairs: int = 5_000
    kl_max_tokens_per_side: int = 4_096
    kl_percentiles: tuple[float, ...] = (50.0, 99.0)

    # Tier 2
    ppl_max_docs_per_class: int = 2_000
    ppl_stride: int = 2_048

    # Tier 3
    cross_mmlu_repo: str = "SeaEval/SeaEval_datasets"
    cross_mmlu_config: str = "cross_mmlu"
    cross_mmlu_languages: tuple[str, ...] = ("english", "malay", "indonesian", "chinese")

    # Tier 4 - MalayMMLU. Loaded file-by-file: calling load_dataset on the
    # repo root raises DatasetGenerationCastError because MalayMMLU_1shot.json
    # carries two columns the other files lack (spec section 4d).
    malay_mmlu_repo: str = "UMxYTLAILabs/MalayMMLU"
    malay_mmlu_files: tuple[str, ...] = ("MalayMMLU_0shot.json",)
    malay_mmlu_shots: int = 0
    malay_mmlu_reference_impl: str = (
        "https://github.com/mesolitica/malaya/tree/master/session/qwen2.5/evaluate-malaymmlu"
    )

    # Tier 4e - throughput, measured with NVIDIA AIPerf.
    throughput_concurrencies: tuple[int, ...] = (1, 8, 32, 64)
    throughput_input_tokens: int = 2_048
    throughput_output_tokens: int = 256
    throughput_warmup_requests: int = 16

    # Every reported number carries a sample size and a variance (spec section 8).
    bootstrap_resamples: int = 1_000
    confidence_level: float = 0.95


class PathsConfig(BaseModel):
    """Filesystem layout. Root differs between laptop and Modal Volume."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root: Path = Path("/vol")
    raw: Path = Path("data/raw")
    stratified: Path = Path("data/stratified")
    calibration_sets: Path = Path("data/calibration_sets")
    splits: Path = Path("data/splits")
    checkpoints: Path = Path("checkpoints")
    results: Path = Path("results")
    routing: Path = Path("results/routing")
    eval: Path = Path("results/eval")
    mlruns: Path = Path("mlruns")

    def resolve(self, field: str) -> Path:
        """Absolute path for a named directory, rooted at `root`."""
        value = getattr(self, field)
        if not isinstance(value, Path):
            raise TypeError(f"{field} is not a path field")
        return self.root / value


class ModalConfig(BaseModel):
    """Modal deployment settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    app_name: str = "ilmu-glossary"
    volume_name: str = "ilmu-glossary-vol"
    hf_cache_volume: str = "ilmu-glossary-hf-cache"
    gpu: str = "B200"
    gpu_count: int = 1
    cpu_only_cpus: float = 8.0
    cpu_only_memory_mb: int = 32_768
    # Long PTQ and eval jobs; Modal's default 5 min is far too short.
    gpu_timeout_s: int = 6 * 60 * 60
    cpu_timeout_s: int = 8 * 60 * 60
    secrets: tuple[str, ...] = ("huggingface-secret",)


class TrackingConfig(BaseModel):
    """MLflow. Parquet under results/ stays authoritative; MLflow is for
    live monitoring of hourly-billed GPU jobs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    experiment_name: str = "routing-aware-calibration"
    tracking_uri: str = "file:///vol/mlruns"
    log_artifacts: bool = False  # parquet already persists to the volume


# --------------------------------------------------------------------------
# root config
# --------------------------------------------------------------------------


class Config(BaseModel):
    """Root config. Load with `Config.load()`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = 42
    dry_run: bool = False
    # Tiny-model, tiny-N mode that exercises every phase for a few dollars.
    dry_run_model: str = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
    # Phase 4 is model-agnostic: it exercises server lifecycle, top-K logprob
    # capture, the KL estimator, perplexity and the MMLU scorers, none of which
    # care whether the model is an MoE. Serving the MoE proxy hung in vLLM's
    # FlashInfer MoE path, which is incidental to this study since Nemotron
    # serves through a different backend, so the dry run validates that logic
    # against a small dense model instead.
    dry_run_eval_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dry_run_sample_counts: tuple[int, ...] = (8, 16)
    dry_run_seq_len: int = 1_024

    model: ModelConfig = Field(default_factory=ModelConfig)
    vllm: VllmConfig = Field(default_factory=VllmConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    lid: LidConfig = Field(default_factory=LidConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    quant: QuantConfig = Field(default_factory=QuantConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    modal: ModalConfig = Field(default_factory=ModalConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @model_validator(mode="after")
    def _check_gate_is_reachable(self) -> Self:
        if self.calibration.gate_sample_count not in self.calibration.sample_counts:
            raise ValueError(
                f"gate_sample_count={self.calibration.gate_sample_count} is not in "
                f"sample_counts={self.calibration.sample_counts}; the headroom gate "
                "would run a configuration the matrix never revisits"
            )
        if self.calibration.gate_variant not in self.calibration.variants:
            raise ValueError("gate_variant must be present in variants")
        return self

    @model_validator(mode="after")
    def _check_traffic_mix_sums(self) -> Self:
        total = sum(self.calibration.assumed_traffic_mix.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"assumed_traffic_mix must sum to 1.0, got {total}")
        unknown = set(self.calibration.assumed_traffic_mix) - {c.value for c in CorpusClass}
        if unknown:
            raise ValueError(f"assumed_traffic_mix references unknown classes: {sorted(unknown)}")
        return self

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, path: Path | str | None = None, **overrides: Any) -> Config:
        """Load from YAML with optional keyword overrides.

        Absent `path`, the defaults encoded above are used verbatim - the
        YAML file exists to make the settings visible and diffable, not to
        hold values the code cannot run without.
        """
        data: dict[str, Any] = {}
        if path is not None:
            data = yaml.safe_load(Path(path).read_text()) or {}
        data.update(overrides)
        return cls.model_validate(data)

    # ------------------------------------------------------------- provenance

    def fingerprint(self) -> str:
        """Stable hash of the full config.

        Written into every results artifact. Two parquet files with different
        fingerprints were not produced by the same settings and must not be
        compared without saying so.
        """
        payload = self.model_dump(mode="json", exclude={"dry_run"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def effective_sample_counts(self) -> tuple[int, ...]:
        return self.dry_run_sample_counts if self.dry_run else self.calibration.sample_counts

    def effective_seq_len(self) -> int:
        return self.dry_run_seq_len if self.dry_run else self.model.calib_seq_len

    def effective_model_repo(self) -> str:
        return self.dry_run_model if self.dry_run else self.model.bf16_repo

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, width=100)


__all__ = [
    "CalibVariant",
    "CalibrationConfig",
    "Config",
    "CorpusClass",
    "DataConfig",
    "EvalConfig",
    "LidConfig",
    "MambaStateDtype",
    "ModalConfig",
    "ModelConfig",
    "PathsConfig",
    "QuantConfig",
    "RecipeFamily",
    "RecordFormat",
    "SourceSpec",
    "TrackingConfig",
    "VllmConfig",
]
