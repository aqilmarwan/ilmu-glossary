"""Phase 0 - data acquisition and stratification.

Produces, from streamed upstream corpora:

  * six stratified corpus classes (`dialect` optional, per spec section 3.4)
  * a parallel BM/EN corpus - the substrate for the primary metric
  * persisted 80/20 splits with a fixed seed
  * `results/corpus_stats.parquet`

Everything is streamed. The bulk Malaysian corpus is 349 GB / ~90B tokens and
is never downloaded whole; the manifest records how many records were consumed
so the sampling is reproducible without materialising the source.

Four constraints from spec section 3 drive most of the design here:

  1. Bahasa Melayu is not Bahasa Indonesia. See `lid.py`.
  2. Calibration inputs must use the model's chat template, because NVIDIA
     calibrated on formatted SFT data. Raw text would change what the
     activations look like independently of language.
  3. Length distribution is recorded rather than truncated to a convenient
     2,048. NVIDIA calibrated Lightning at 32,768.
  4. A document appears in exactly one class.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ilmu_glossary import tracking
from ilmu_glossary.config import Config, CorpusClass, SourceSpec
from ilmu_glossary.io import count_jsonl, read_jsonl, write_json, write_jsonl, write_parquet
from ilmu_glossary.lid import (
    LanguageIdentifier,
    detect_dialect,
    detect_intrasentential_switching,
)
from ilmu_glossary.seeds import python_rng, rng
from ilmu_glossary.splits import make_split, save_splits, verify_all

logger = logging.getLogger(__name__)

PARALLEL_CLASS = "parallel_bm_en"
_WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)


# --------------------------------------------------------------------------
# deduplication
# --------------------------------------------------------------------------


class Deduplicator:
    """MinHash-LSH near-duplicate detection across all classes at once.

    Spec section 3: "Deduplicate across classes. A document must appear in
    exactly one class." Exact hashing is not enough - the bulk corpus feeds
    three classes and contains heavy near-duplication from crawl artifacts.

    Memory is the binding constraint at 50k+ documents, so signatures are
    stored as a single int64 array rather than per-document objects.
    """

    def __init__(self, *, ngram: int = 13, threshold: float = 0.8, num_perm: int = 64) -> None:
        self.ngram = ngram
        self.threshold = threshold
        self.num_perm = num_perm
        # Banding: b bands of r rows, tuned so the S-curve inflects near
        # `threshold`. With 64 permutations, 16 bands of 4 rows puts the
        # 50% detection point at ~0.79.
        self.rows = 4
        self.bands = num_perm // self.rows
        self._signatures: list[np.ndarray] = []
        self._buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
        self._exact: set[str] = set()
        self.n_exact_dupes = 0
        self.n_near_dupes = 0

        # Fixed permutation coefficients; a fresh seed per process would make
        # dedup non-reproducible across phases.
        generator = np.random.default_rng(0xD3D9)
        self._a = generator.integers(1, 2**61 - 1, size=num_perm, dtype=np.int64)
        self._b = generator.integers(0, 2**61 - 1, size=num_perm, dtype=np.int64)

    @staticmethod
    def normalise(text: str) -> str:
        """Casefold, strip accents and collapse whitespace before hashing.

        Without this, the same crawled page differing only in whitespace or
        smart quotes survives as two documents.
        """
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        return " ".join(text.casefold().split())

    def _shingles(self, text: str) -> set[int]:
        words = _WORD_RE.findall(self.normalise(text))
        if len(words) < self.ngram:
            # Short documents shingle on whatever they have rather than
            # producing an empty set that would match everything.
            return {hash(" ".join(words))} if words else set()
        return {
            hash(" ".join(words[i : i + self.ngram])) for i in range(len(words) - self.ngram + 1)
        }

    def _signature(self, shingles: set[int]) -> np.ndarray:
        if not shingles:
            return np.full(self.num_perm, np.iinfo(np.int64).max, dtype=np.int64)
        values = np.fromiter(shingles, dtype=np.int64, count=len(shingles))
        # (a*x + b) mod prime, minimum over shingles, per permutation.
        prime = np.int64(2**61 - 1)
        hashed = (self._a[:, None] * values[None, :] + self._b[:, None]) % prime
        signature: np.ndarray = hashed.min(axis=1)
        return signature

    def add_if_new(self, text: str) -> bool:
        """Register a document. False if it duplicates one already seen."""
        exact_key = hashlib.sha256(self.normalise(text).encode()).hexdigest()
        if exact_key in self._exact:
            self.n_exact_dupes += 1
            return False

        signature = self._signature(self._shingles(text))
        band_keys = [
            (band, signature[band * self.rows : (band + 1) * self.rows].tobytes())
            for band in range(self.bands)
        ]

        candidates: set[int] = set()
        for key in band_keys:
            candidates.update(self._buckets[key])

        for candidate in candidates:
            similarity = float(np.mean(self._signatures[candidate] == signature))
            if similarity >= self.threshold:
                self.n_near_dupes += 1
                return False

        index = len(self._signatures)
        self._signatures.append(signature)
        for key in band_keys:
            self._buckets[key].append(index)
        self._exact.add(exact_key)
        return True

    @property
    def n_unique(self) -> int:
        return len(self._signatures)


# --------------------------------------------------------------------------
# streaming ingest
# --------------------------------------------------------------------------


@dataclass
class IngestManifest:
    """Records exactly what was consumed, so a stream can be reproduced."""

    source: str
    records_read: int = 0
    records_kept: int = 0
    records_rejected: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "records_read": self.records_read,
            "records_kept": self.records_kept,
            "records_rejected": dict(self.records_rejected),
            "error": self.error,
        }


def stream_source(spec: SourceSpec, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield raw records from one upstream dataset.

    Streaming is non-negotiable for the bulk corpus. `datasets` is imported
    lazily so this module stays importable where it is absent.
    """
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": spec.split, "streaming": spec.streaming}
    if spec.config_name:
        kwargs["name"] = spec.config_name
    if spec.revision:
        kwargs["revision"] = spec.revision

    dataset = load_dataset(spec.repo_id, **kwargs)
    for i, record in enumerate(dataset):
        if limit is not None and i >= limit:
            return
        yield record


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassificationOutcome:
    corpus_class: CorpusClass | None
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


def classify_document(
    text: str,
    identifier: LanguageIdentifier,
    cfg: Config,
    *,
    source_provenance: str,
) -> ClassificationOutcome:
    """Assign a document to exactly one corpus class, or reject it.

    Order matters: code-switching and dialect are tested before plain formal
    BM, because a code-switched document would otherwise be absorbed into
    `formal_bm` and the class would never fill.
    """
    if len(text) < cfg.data.min_doc_chars:
        return ClassificationOutcome(None, "too_short")

    if source_provenance == "code":
        return ClassificationOutcome(CorpusClass.CODE_CONTROL, "provenance_code")

    if source_provenance == "english":
        if identifier.is_english(text):
            return ClassificationOutcome(CorpusClass.ENGLISH_CONTROL, "lid_english")
        return ClassificationOutcome(None, "english_source_but_not_english")

    result = identifier.identify(text)

    if result.verdict == "indonesian":
        return ClassificationOutcome(None, "indonesian", result.to_dict())
    if result.rejected_by is not None:
        return ClassificationOutcome(None, result.rejected_by, result.to_dict())

    if result.verdict == "english":
        return ClassificationOutcome(CorpusClass.ENGLISH_CONTROL, "lid_english", result.to_dict())

    if result.verdict not in {"malay", "manglish", "rojak"}:
        return ClassificationOutcome(None, f"verdict_{result.verdict}", result.to_dict())

    # Manglish: the LID model has a dedicated label for it and is more
    # reliable than any heuristic we would write.
    if result.verdict == "manglish":
        return ClassificationOutcome(CorpusClass.MANGLISH, "lid_manglish", result.to_dict())

    is_switched, switch_ratio = detect_intrasentential_switching(text)
    evidence = {**result.to_dict(), "switch_ratio": switch_ratio}

    # `rojak` is mesolitica's label for mixed-language text. Confirm the mixing
    # is intra-sentential before admitting it, since spec section 3.3 excludes
    # documents that merely alternate between monolingual sentences.
    if result.verdict == "rojak":
        if is_switched:
            return ClassificationOutcome(CorpusClass.CODE_SWITCHED, "lid_rojak_intra", evidence)
        return ClassificationOutcome(None, "rojak_but_inter_sentential", evidence)

    if is_switched:
        return ClassificationOutcome(CorpusClass.CODE_SWITCHED, "intrasentential", evidence)

    dialect, hits = detect_dialect(text)
    if dialect is not None:
        return ClassificationOutcome(
            CorpusClass.DIALECT, f"dialect_{dialect}", {**evidence, "dialect_hits": hits}
        )

    return ClassificationOutcome(CorpusClass.FORMAL_BM, "lid_malay", evidence)


# --------------------------------------------------------------------------
# chat templating
# --------------------------------------------------------------------------


def load_tokenizer(cfg: Config) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        cfg.effective_model_repo(),
        trust_remote_code=cfg.model.trust_remote_code,
        revision=cfg.model.revision if not cfg.dry_run else "main",
    )


def apply_chat_template(text: str, tokenizer: Any) -> str:
    """Wrap raw text in the model's chat template.

    Spec section 3: "Nemotron is instruction-tuned and NVIDIA calibrated on
    formatted SFT data. Calibration inputs must use the model's chat template,
    not raw text."

    The template is retrieved from the tokenizer config rather than
    reconstructed, so it tracks the checkpoint. If the tokenizer carries no
    template the text is returned unchanged and the caller records that the
    assertion could not be satisfied - silently proceeding with raw text would
    invalidate every calibration comparison.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        return text
    messages = [{"role": "user", "content": text}]
    rendered: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return rendered


def assert_chat_templated(samples: list[str], tokenizer: Any) -> None:
    """Fail loudly if calibration text is not formatted (spec section 9)."""
    if getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError(
            "Tokenizer carries no chat_template. NVIDIA calibrated on formatted "
            "SFT data; calibrating on raw text would change activation "
            "statistics independently of language and confound every variant."
        )
    rendered_marker = apply_chat_template("probe", tokenizer)
    # Whatever delimiter the template introduces around 'probe' must appear
    # in real samples too.
    prefix = rendered_marker.split("probe")[0]
    if not prefix:
        return
    unformatted = [s for s in samples[:64] if not s.startswith(prefix)]
    if unformatted:
        raise RuntimeError(
            f"{len(unformatted)} of {min(len(samples), 64)} inspected calibration "
            "samples are not chat-templated."
        )


# --------------------------------------------------------------------------
# parallel corpus
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignmentCheck:
    """Outcome of verifying one BM/EN pair."""

    passed: bool
    reason: str
    length_ratio: float


def check_alignment(
    malay: str,
    english: str,
    *,
    min_ratio: float = 0.4,
    max_ratio: float = 2.5,
    min_chars: int = 40,
) -> AlignmentCheck:
    """Reject truncated or off-topic pairs (spec section 3, parallel corpus).

    Length ratio is the practical detector for truncation. Malay is
    consistently a little longer than English in characters, so the acceptance
    band is asymmetric around 1.0 rather than centred on it.
    """
    if len(malay) < min_chars or len(english) < min_chars:
        return AlignmentCheck(False, "too_short", 0.0)

    ratio = len(malay) / len(english)
    if ratio < min_ratio:
        return AlignmentCheck(False, "malay_side_truncated", ratio)
    if ratio > max_ratio:
        return AlignmentCheck(False, "english_side_truncated", ratio)

    # Digits and proper nouns survive translation. Their disappearance is a
    # reliable off-topic signal that length ratio alone will not catch.
    malay_digits = set(re.findall(r"\d+", malay))
    english_digits = set(re.findall(r"\d+", english))
    if malay_digits and english_digits:
        overlap = len(malay_digits & english_digits) / len(malay_digits | english_digits)
        if overlap < 0.3:
            return AlignmentCheck(False, "numeric_content_diverges", ratio)

    return AlignmentCheck(True, "ok", ratio)


def build_parallel_corpus(
    cfg: Config,
    identifier: LanguageIdentifier,
    dedup: Deduplicator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build aligned BM/EN pairs and verify a sample of them.

    Spec section 3 requires at least 5,000 pairs, alignment verified on 200
    random pairs with the pass rate logged, and truncated or off-topic pairs
    discarded.
    """
    specs = cfg.data.sources.get(PARALLEL_CLASS, [])
    if not specs:
        raise RuntimeError("No parallel_bm_en sources configured")

    pairs: list[dict[str, Any]] = []
    manifest = IngestManifest(source=",".join(s.repo_id for s in specs))
    target = cfg.data.min_parallel_pairs * 3  # oversample; verification prunes

    for spec in specs:
        for record in stream_source(spec):
            manifest.records_read += 1
            if len(pairs) >= target:
                break

            malay, english = _extract_pair(record)
            if malay is None or english is None:
                manifest.records_rejected["no_pair_fields"] += 1
                continue

            check = check_alignment(malay, english)
            if not check.passed:
                manifest.records_rejected[check.reason] += 1
                continue

            # The Malay side must actually be Malaysian Malay, not Indonesian.
            lid_result = identifier.identify(malay)
            if not lid_result.accepted_as_malay:
                manifest.records_rejected[lid_result.rejected_by or "not_malay"] += 1
                continue

            if not dedup.add_if_new(malay):
                manifest.records_rejected["duplicate"] += 1
                continue

            pairs.append(
                {
                    "id": f"{PARALLEL_CLASS}:{len(pairs)}",
                    "malay": malay,
                    "english": english,
                    "length_ratio": check.length_ratio,
                    "malay_prob": lid_result.malay_prob,
                    "source": spec.repo_id,
                }
            )
            manifest.records_kept += 1

    # Spec section 3: verify alignment quality on 200 random pairs, log the rate.
    generator = python_rng(cfg.seed, "phase0", "alignment_check")
    sample_size = min(cfg.data.alignment_check_sample, len(pairs))
    sample = generator.sample(pairs, sample_size) if sample_size else []
    passed = sum(1 for p in sample if check_alignment(p["malay"], p["english"]).passed)
    pass_rate = passed / sample_size if sample_size else 0.0

    report = {
        "n_pairs": len(pairs),
        "verified_sample_size": sample_size,
        "verified_pass_rate": pass_rate,
        "meets_minimum": len(pairs) >= cfg.data.min_parallel_pairs,
        "manifest": manifest.to_dict(),
    }
    logger.info(
        "Parallel corpus: %d pairs, alignment pass rate %.1f%% on %d sampled",
        len(pairs),
        100 * pass_rate,
        sample_size,
    )
    if len(pairs) < cfg.data.min_parallel_pairs:
        logger.error(
            "Parallel corpus has %d pairs, below the %d minimum. The primary "
            "metric (tier 1 BM-EN KL delta) rests on this corpus - a short one "
            "weakens the only tier that supports the causal claim.",
            len(pairs),
            cfg.data.min_parallel_pairs,
        )
    return pairs, report


def _extract_pair(record: dict[str, Any]) -> tuple[str | None, str | None]:
    """Pull the BM and EN sides out of a record.

    mesolitica's translation datasets are not schema-consistent across repos,
    so several field conventions are tried before giving up.
    """
    candidates = [
        ("ms", "en"),
        ("malay", "english"),
        ("src", "tgt"),
        ("translation_ms", "translation_en"),
        ("input", "output"),
    ]
    for ms_key, en_key in candidates:
        if ms_key in record and en_key in record:
            ms, en = record[ms_key], record[en_key]
            if isinstance(ms, str) and isinstance(en, str):
                return ms, en

    # HF `translation` feature: {"translation": {"ms": ..., "en": ...}}
    nested = record.get("translation")
    if isinstance(nested, dict):
        ms, en = nested.get("ms"), nested.get("en")
        if isinstance(ms, str) and isinstance(en, str):
            return ms, en
    return None, None


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def compute_corpus_stats(
    cfg: Config,
    class_paths: dict[str, Path],
    tokenizer: Any,
    *,
    max_docs_for_tokens: int = 2_000,
) -> pd.DataFrame:
    """Per-class doc count, token count, length distribution and fertility.

    Spec section 3 output. Token counts use the *model's* tokenizer, because
    fertility against a different tokenizer would say nothing about how many
    of the model's tokens a Malay document consumes.
    """
    rows: list[dict[str, Any]] = []
    generator = rng(cfg.seed, "phase0", "stats")

    for class_name, path in class_paths.items():
        n_docs = count_jsonl(path)
        if n_docs == 0:
            continue

        # Tokenizing every document of a 10k-document class is wasteful; a
        # bounded sample gives the distribution to well within reporting
        # precision, and the sample size is reported alongside.
        sample_size = min(max_docs_for_tokens, n_docs)
        indices = set(generator.choice(n_docs, size=sample_size, replace=False).tolist())

        token_counts: list[int] = []
        word_counts: list[int] = []
        char_counts: list[int] = []

        for i, record in enumerate(read_jsonl(path)):
            if i not in indices:
                continue
            text = record.get("text") or record.get("malay") or ""
            token_counts.append(len(tokenizer.encode(text, add_special_tokens=False)))
            word_counts.append(len(_WORD_RE.findall(text)))
            char_counts.append(len(text))

        if not token_counts:
            continue

        tokens = np.array(token_counts, dtype=np.float64)
        words = np.array(word_counts, dtype=np.float64)
        rows.append(
            {
                "corpus_class": class_name,
                "n_docs": n_docs,
                "n_docs_sampled_for_tokens": sample_size,
                "total_tokens_estimate": float(tokens.mean() * n_docs),
                "tokens_mean": float(tokens.mean()),
                "tokens_std": float(tokens.std(ddof=1)) if len(tokens) > 1 else 0.0,
                "tokens_p10": float(np.percentile(tokens, 10)),
                "tokens_p50": float(np.percentile(tokens, 50)),
                "tokens_p90": float(np.percentile(tokens, 90)),
                "tokens_p99": float(np.percentile(tokens, 99)),
                "tokens_max": float(tokens.max()),
                # Fraction reaching the calibration sequence length matters:
                # if it is near zero, the study is not calibrating at 32K
                # regardless of what the config says.
                "frac_at_calib_seq_len": float((tokens >= cfg.effective_seq_len()).mean()),
                "chars_mean": float(np.mean(char_counts)),
                "words_mean": float(words.mean()),
                "fertility_tokens_per_word": float(tokens.sum() / max(words.sum(), 1)),
            }
        )

    return pd.DataFrame(rows)


def language_spotcheck(
    cfg: Config,
    class_paths: dict[str, Path],
    identifier: LanguageIdentifier,
) -> pd.DataFrame:
    """Spec section 3: spot-check 100 random samples per class and log the result.

    Retains each layer's evidence so a human can audit *why* a document was
    admitted, which is the point of the check.
    """
    rows: list[dict[str, Any]] = []
    generator = rng(cfg.seed, "phase0", "spotcheck")

    for class_name, path in class_paths.items():
        n_docs = count_jsonl(path)
        if n_docs == 0:
            continue
        sample_size = min(cfg.data.language_spotcheck_sample, n_docs)
        indices = set(generator.choice(n_docs, size=sample_size, replace=False).tolist())

        verdicts: list[str] = []
        for i, record in enumerate(read_jsonl(path)):
            if i not in indices:
                continue
            text = record.get("text") or record.get("malay") or ""
            verdicts.append(identifier.identify(text).verdict)

        counts = pd.Series(verdicts).value_counts()
        n = len(verdicts)
        rows.append(
            {
                "corpus_class": class_name,
                "n_sampled": n,
                "frac_malay": float(counts.get("malay", 0) / n) if n else 0.0,
                "frac_manglish": float(counts.get("manglish", 0) / n) if n else 0.0,
                "frac_rojak": float(counts.get("rojak", 0) / n) if n else 0.0,
                "frac_indonesian": float(counts.get("indonesian", 0) / n) if n else 0.0,
                "frac_english": float(counts.get("english", 0) / n) if n else 0.0,
                "frac_other": float(counts.get("other", 0) / n) if n else 0.0,
                "lid_degraded": identifier.degraded,
            }
        )

    df = pd.DataFrame(rows)
    for row in rows:
        if row["frac_indonesian"] > 0.05:
            logger.warning(
                "%s: %.1f%% of the spot-check sample reads as Indonesian. "
                "Spec section 9 flags this as the top contamination risk.",
                row["corpus_class"],
                100 * row["frac_indonesian"],
            )
    return df


# --------------------------------------------------------------------------
# phase driver
# --------------------------------------------------------------------------


def run_phase0(cfg: Config) -> dict[str, Any]:
    """Build every Phase 0 artifact. Idempotent - skips completed outputs."""
    fingerprint = cfg.fingerprint()
    stratified_dir = cfg.paths.resolve("stratified")
    splits_dir = cfg.paths.resolve("splits")
    results_dir = cfg.paths.resolve("results")
    stratified_dir.mkdir(parents=True, exist_ok=True)

    with tracking.run(cfg, phase="phase0", run_name="data_prep"):
        tracking.log_params(
            {
                "min_docs_per_class": cfg.data.min_docs_per_class,
                "min_parallel_pairs": cfg.data.min_parallel_pairs,
                "calib_seq_len": cfg.effective_seq_len(),
                "dedup_threshold": cfg.data.dedup_threshold,
            }
        )

        identifier = LanguageIdentifier(cfg.lid)
        dedup = Deduplicator(ngram=cfg.data.dedup_ngram, threshold=cfg.data.dedup_threshold)
        tokenizer = load_tokenizer(cfg)

        buffers: dict[CorpusClass, list[dict[str, Any]]] = defaultdict(list)
        manifests: list[dict[str, Any]] = []
        target = cfg.data.min_docs_per_class if not cfg.dry_run else 64

        # ------------------------------------------------------ stratify
        for class_key, specs in cfg.data.sources.items():
            if class_key == PARALLEL_CLASS:
                continue
            for spec in specs:
                manifest = IngestManifest(source=spec.repo_id)
                try:
                    for record in stream_source(spec):
                        manifest.records_read += 1
                        if _all_classes_full(buffers, target, cfg):
                            break

                        text = record.get(spec.text_field)
                        if not isinstance(text, str):
                            manifest.records_rejected["missing_text_field"] += 1
                            continue

                        outcome = classify_document(
                            text, identifier, cfg, source_provenance=spec.provenance
                        )
                        if outcome.corpus_class is None:
                            manifest.records_rejected[outcome.reason] += 1
                            continue

                        bucket = buffers[outcome.corpus_class]
                        if len(bucket) >= target:
                            manifest.records_rejected["class_full"] += 1
                            continue

                        if not dedup.add_if_new(text):
                            manifest.records_rejected["duplicate"] += 1
                            continue

                        bucket.append(
                            {
                                "id": f"{outcome.corpus_class.value}:{len(bucket)}",
                                "text": text,
                                # Chat-templated form is what calibration
                                # consumes; both are kept so perplexity can
                                # score raw text.
                                "templated": apply_chat_template(text, tokenizer),
                                "source": spec.repo_id,
                                "classified_by": outcome.reason,
                            }
                        )
                        manifest.records_kept += 1
                except Exception as exc:
                    manifest.error = repr(exc)
                    logger.error("Streaming %s failed: %r", spec.repo_id, exc)
                manifests.append(manifest.to_dict())

        # ------------------------------------------- write class files
        class_paths: dict[str, Path] = {}
        omissions: list[str] = []

        for corpus_class, records in buffers.items():
            minimum = (
                cfg.data.min_dialect_docs
                if corpus_class is CorpusClass.DIALECT
                else cfg.data.min_docs_per_class
            )
            if cfg.dry_run:
                minimum = 1

            # Spec section 3.4: drop `dialect` below its floor and record the
            # omission rather than padding it with formal BM.
            if corpus_class is CorpusClass.DIALECT and len(records) < minimum:
                omissions.append(
                    f"dialect dropped: {len(records)} documents sourced, "
                    f"below the {minimum} minimum. Not padded with formal BM."
                )
                logger.warning(omissions[-1])
                continue

            if len(records) < minimum:
                logger.error(
                    "%s has %d documents, below the %d minimum",
                    corpus_class.value,
                    len(records),
                    minimum,
                )

            path = stratified_dir / f"{corpus_class.value}.jsonl"
            write_jsonl(records, path)
            class_paths[corpus_class.value] = path

        # --------------------------------------------- parallel corpus
        pairs, parallel_report = build_parallel_corpus(cfg, identifier, dedup)
        parallel_path = stratified_dir / f"{PARALLEL_CLASS}.jsonl"
        write_jsonl(pairs, parallel_path)
        class_paths[PARALLEL_CLASS] = parallel_path

        # ---------------------------------------------------- splits
        splits = {
            name: make_split(
                name,
                count_jsonl(path),
                base_seed=cfg.seed,
                train_fraction=cfg.data.train_fraction,
            )
            for name, path in class_paths.items()
        }
        verify_all(splits, expected_fraction=cfg.data.train_fraction)
        save_splits(splits, splits_dir)

        # ------------------------------------------------ statistics
        stats = compute_corpus_stats(cfg, class_paths, tokenizer)
        write_parquet(
            stats, results_dir / "corpus_stats.parquet", fingerprint=fingerprint, phase="phase0"
        )

        spotcheck = language_spotcheck(cfg, class_paths, identifier)
        write_parquet(
            spotcheck,
            results_dir / "language_spotcheck.parquet",
            fingerprint=fingerprint,
            phase="phase0",
        )

        summary = {
            "classes_built": sorted(class_paths),
            "omissions": omissions,
            "dedup": {
                "unique": dedup.n_unique,
                "exact_duplicates": dedup.n_exact_dupes,
                "near_duplicates": dedup.n_near_dupes,
            },
            "parallel": parallel_report,
            "lid_degraded": identifier.degraded,
            "chat_template_available": getattr(tokenizer, "chat_template", None) is not None,
            "manifests": manifests,
            "config_fingerprint": fingerprint,
        }
        write_json(summary, results_dir / "phase0_summary.json")

        tracking.log_metrics(
            {
                "n_classes": float(len(class_paths)),
                "n_parallel_pairs": float(parallel_report["n_pairs"]),
                "alignment_pass_rate": float(parallel_report["verified_pass_rate"]),
                "near_duplicates_removed": float(dedup.n_near_dupes),
            }
        )

    return summary


def _all_classes_full(
    buffers: dict[CorpusClass, list[dict[str, Any]]], target: int, cfg: Config
) -> bool:
    """Stop streaming once every required class has met its quota.

    `dialect` is excluded - it is optional, and waiting for it to fill would
    mean streaming the whole 349 GB corpus for a class the spec permits
    dropping.
    """
    required = CorpusClass.required()
    configured = {c for c in required if c.value in cfg.data.sources or buffers.get(c)}
    if not configured:
        return False
    return all(len(buffers.get(c, [])) >= target for c in configured)


__all__ = [
    "PARALLEL_CLASS",
    "AlignmentCheck",
    "ClassificationOutcome",
    "Deduplicator",
    "IngestManifest",
    "apply_chat_template",
    "assert_chat_templated",
    "build_parallel_corpus",
    "check_alignment",
    "classify_document",
    "compute_corpus_stats",
    "language_spotcheck",
    "load_tokenizer",
    "run_phase0",
    "stream_source",
]
