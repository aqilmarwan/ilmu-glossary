"""Language identification, specialised for the Bahasa Melayu / Bahasa
Indonesia boundary.

Spec section 9 lists Indonesian contamination as the top risk: "Many HF
datasets tagged 'Malay' are Indonesian." Generic language ID does not help
here - lid.176 and most GlotLID label sets collapse both into a single `msa`
or `zsm`/`ind` pair trained on formal text, and Malaysian social media is
neither formal nor cleanly separable by a general model.

Three layers, applied in order, each able to reject:

  1. Provenance   - the source dataset must be Malaysian-operated.
  2. fastText     - mesolitica's models, trained on the ms/id boundary
                    specifically (96.58% on that pair) and emitting the
                    `manglish` / `rojak` labels the colloquial and
                    code-switched classes depend on.
  3. Lexicon      - discriminative markers. Indonesian-only forms are
                    evidence against Malaysian provenance even when
                    fastText is confident, because the fastText models were
                    themselves trained on crawled data with some leakage.

Layer 3 is deliberately conservative: it rejects on Indonesian markers rather
than requiring Malaysian ones, because short Manglish documents legitimately
contain neither.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Protocol

from ilmu_glossary.config import LidConfig

logger = logging.getLogger(__name__)

# Labels mesolitica's v2 model emits. Mapped to the study's decision space.
MALAY_LABELS = frozenset({"standard-malay", "local-malay", "malay", "manglish", "rojak"})
INDONESIAN_LABELS = frozenset({"standard-indonesian", "socialmedia-indonesian", "ind"})
ENGLISH_LABELS = frozenset({"eng", "english"})

Verdict = Literal["malay", "indonesian", "english", "manglish", "rojak", "other", "unknown"]

_WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)


@dataclass(frozen=True)
class LidResult:
    """One document's language decision, with every layer's evidence retained.

    The per-layer fields exist so the 100-document-per-class spot-check
    required by spec section 9 can be audited: a reviewer can see *why* a
    document was admitted, not just that it was.
    """

    verdict: Verdict
    malay_prob: float
    indonesian_prob: float
    english_prob: float
    top_label: str
    indonesian_marker_ratio: float
    malaysian_marker_ratio: float
    n_words: int
    rejected_by: str | None = None

    @property
    def accepted_as_malay(self) -> bool:
        return self.verdict in {"malay", "manglish", "rojak"} and self.rejected_by is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "malay_prob": self.malay_prob,
            "indonesian_prob": self.indonesian_prob,
            "english_prob": self.english_prob,
            "top_label": self.top_label,
            "indonesian_marker_ratio": self.indonesian_marker_ratio,
            "malaysian_marker_ratio": self.malaysian_marker_ratio,
            "n_words": self.n_words,
            "rejected_by": self.rejected_by,
        }


class _FastTextModel(Protocol):
    def predict(self, text: str, k: int) -> tuple[list[str], Any]: ...


# --------------------------------------------------------------------------
# lexicon layer
# --------------------------------------------------------------------------


def marker_ratios(text: str, cfg: LidConfig) -> tuple[float, float, int]:
    """Fraction of tokens matching Indonesian-only and Malaysian-only markers.

    Ratios rather than counts so that a long document is not penalised for
    containing one Indonesian loan word.
    """
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return 0.0, 0.0, 0

    # Multi-word markers ("macam mana", "apa khabar") are checked against the
    # raw lowercased text; single tokens against the word list.
    lowered = text.lower()
    id_single = {m for m in cfg.indonesian_markers if " " not in m}
    my_single = {m for m in cfg.malaysian_markers if " " not in m}

    id_hits = sum(1 for w in words if w in id_single)
    my_hits = sum(1 for w in words if w in my_single)
    id_hits += sum(lowered.count(m) for m in cfg.indonesian_markers if " " in m)
    my_hits += sum(lowered.count(m) for m in cfg.malaysian_markers if " " in m)

    n = len(words)
    return id_hits / n, my_hits / n, n


# --------------------------------------------------------------------------
# fastText layer
# --------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _load_fasttext(repo_id: str, quantized: bool) -> _FastTextModel | None:
    """Download and load one mesolitica fastText model.

    Returns None rather than raising so the pipeline can fall back to the
    lexicon-only path with a recorded warning, instead of failing phase 0
    outright because a model repo moved.
    """
    try:
        import fasttext
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.warning("fasttext not installed; LID falls back to lexicon only")
        return None

    filename = "model.ftz" if quantized else "model.bin"
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception:
        # Repos differ in filename convention; try the other one before giving up.
        alt = "model.bin" if quantized else "model.ftz"
        try:
            path = hf_hub_download(repo_id=repo_id, filename=alt)
        except Exception as exc:
            logger.warning("Could not fetch %s from %s: %r", filename, repo_id, exc)
            return None

    try:
        model: _FastTextModel = fasttext.load_model(path)
    except Exception as exc:
        logger.warning("Could not load fastText model %s: %r", repo_id, exc)
        return None
    return model


def _normalise(text: str) -> str:
    """fastText's predict rejects newlines and is sensitive to whitespace runs."""
    return " ".join(text.split())[:2000]


def _predict_probs(model: _FastTextModel | None, text: str) -> dict[str, float]:
    if model is None:
        return {}
    try:
        labels, probs = model.predict(_normalise(text), k=10)
    except Exception as exc:
        logger.debug("fastText predict failed: %r", exc)
        return {}
    return {
        label.removeprefix("__label__"): float(prob)
        for label, prob in zip(labels, probs, strict=False)
    }


def _aggregate(probs: dict[str, float], labels: frozenset[str]) -> float:
    return sum(p for label, p in probs.items() if label in labels)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


class LanguageIdentifier:
    """Stateful identifier holding the loaded fastText models.

    Construct once per worker. Model loading is memoised, so constructing
    several is cheap but pointless.
    """

    def __init__(self, cfg: LidConfig) -> None:
        self.cfg = cfg
        self._primary = _load_fasttext(cfg.primary_model, cfg.quantized)
        self._ms_id = _load_fasttext(cfg.ms_id_model, cfg.quantized)
        self.degraded = self._primary is None
        if self.degraded:
            logger.warning(
                "LID running in degraded lexicon-only mode. Malay/Indonesian "
                "separation will be weaker; record this in corpus_stats."
            )

    def identify(self, text: str) -> LidResult:
        """Classify one document. Never raises."""
        cfg = self.cfg
        id_ratio, my_ratio, n_words = marker_ratios(text, cfg)

        primary = _predict_probs(self._primary, text)
        # The ms/id model is a specialist: consult it when the primary model
        # thinks the text is Malay-or-Indonesian at all, and let it arbitrate.
        ms_id = _predict_probs(self._ms_id, text)
        merged = {**primary}
        for label, prob in ms_id.items():
            # Specialist wins on the labels it was trained to separate.
            if label in MALAY_LABELS or label in INDONESIAN_LABELS:
                merged[label] = max(merged.get(label, 0.0), prob)

        malay_p = _aggregate(merged, MALAY_LABELS)
        indo_p = _aggregate(merged, INDONESIAN_LABELS)
        eng_p = _aggregate(merged, ENGLISH_LABELS)
        top_label = max(merged, key=lambda k: merged[k]) if merged else "unknown"

        verdict: Verdict = "unknown"
        rejected_by: str | None = None

        if merged:
            if top_label in ENGLISH_LABELS and eng_p >= cfg.min_english_prob:
                verdict = "english"
            elif top_label == "manglish":
                verdict = "manglish"
            elif top_label == "rojak":
                verdict = "rojak"
            elif top_label in INDONESIAN_LABELS:
                verdict = "indonesian"
            elif top_label in MALAY_LABELS:
                verdict = "malay"
            else:
                verdict = "other"

        # Layer 2 thresholds.
        if verdict in {"malay", "manglish", "rojak"}:
            if malay_p < cfg.min_malay_prob:
                rejected_by = f"fasttext_malay_prob<{cfg.min_malay_prob}"
            elif indo_p > cfg.max_indonesian_prob:
                rejected_by = f"fasttext_indonesian_prob>{cfg.max_indonesian_prob}"
            # Layer 3: lexicon veto. Applies even when fastText is confident.
            elif id_ratio > cfg.max_indonesian_marker_ratio and id_ratio > my_ratio:
                rejected_by = (
                    f"lexicon_indonesian_markers={id_ratio:.3f}>{cfg.max_indonesian_marker_ratio}"
                )

        return LidResult(
            verdict=verdict,
            malay_prob=malay_p,
            indonesian_prob=indo_p,
            english_prob=eng_p,
            top_label=top_label,
            indonesian_marker_ratio=id_ratio,
            malaysian_marker_ratio=my_ratio,
            n_words=n_words,
            rejected_by=rejected_by,
        )

    def is_malaysian_malay(self, text: str) -> bool:
        return self.identify(text).accepted_as_malay

    def is_english(self, text: str) -> bool:
        result = self.identify(text)
        return result.verdict == "english" and result.english_prob >= self.cfg.min_english_prob


# --------------------------------------------------------------------------
# code-switching detection (spec section 3.3)
# --------------------------------------------------------------------------

# Function words are the reliable signal. Content words borrow freely in both
# directions ("meeting", "handphone"), so keying on them would classify
# ordinary Malay as code-switched.
_EN_FUNCTION_WORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "should",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "from",
        "by",
        "about",
        "into",
        "over",
        "after",
        "before",
        "you",
        "your",
        "they",
        "them",
        "we",
        "our",
        "it",
        "its",
        "not",
        "no",
        "yes",
    ]
)
_MS_FUNCTION_WORDS = frozenset(
    [
        "yang",
        "dan",
        "atau",
        "tetapi",
        "jika",
        "maka",
        "daripada",
        "itu",
        "ini",
        "adalah",
        "ialah",
        "ada",
        "sudah",
        "telah",
        "akan",
        "boleh",
        "dapat",
        "harus",
        "mesti",
        "untuk",
        "dengan",
        "dari",
        "oleh",
        "pada",
        "di",
        "ke",
        "kepada",
        "saya",
        "awak",
        "anda",
        "mereka",
        "kami",
        "kita",
        "dia",
        "tak",
        "tidak",
        "ya",
        "bukan",
        "sangat",
        "juga",
        "lagi",
        "sahaja",
        "saja",
    ]
)

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")


def detect_intrasentential_switching(text: str, *, min_ratio: float = 0.15) -> tuple[bool, float]:
    """True when BM and EN alternate *within* sentences.

    Spec section 3.3 requires code-switching within sentences, not alternating
    documents. Measuring per sentence and requiring a minimum fraction of
    sentences to be mixed is what enforces that distinction - a document of
    alternating monolingual sentences scores zero here.

    Returns (is_code_switched, fraction_of_mixed_sentences).
    """
    sentences = [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]
    if not sentences:
        return False, 0.0

    mixed = 0
    considered = 0
    for sentence in sentences:
        words = {w.lower() for w in _WORD_RE.findall(sentence)}
        # Very short fragments cannot evidence a switch.
        if len(words) < 5:
            continue
        considered += 1
        has_en = bool(words & _EN_FUNCTION_WORDS)
        has_ms = bool(words & _MS_FUNCTION_WORDS)
        if has_en and has_ms:
            mixed += 1

    if considered == 0:
        return False, 0.0
    ratio = mixed / considered
    return ratio >= min_ratio, ratio


# --------------------------------------------------------------------------
# dialect detection (spec section 3.4, optional class)
# --------------------------------------------------------------------------

DIALECT_MARKERS: dict[str, frozenset[str]] = {
    "kelantan": frozenset(
        [
            "gapo",
            "demo",
            "kito",
            "ambo",
            "tubik",
            "gok",
            "kkecek",
            "ttubik",
            "sokmo",
            "doh",
            "nnate",
            "koho",
        ]
    ),
    "kedah": frozenset(
        ["hang", "depa", "hampa", "pi", "mai", "tak", "dan", "cek", "awat", "pasaipa"]
    ),
    "terengganu": frozenset(["dokrok", "mung", "starang", "lok", "bakpe", "ceroh", "htok"]),
    "sarawak": frozenset(["kamek", "kitak", "sik", "nemu", "bok", "aok", "gerek"]),
    "sabah": frozenset(["bah", "kitak", "palui", "gia", "banar", "sia"]),
    "negeri_sembilan": frozenset(["den", "ekau", "opa", "poi", "yo"]),
}


def detect_dialect(text: str, *, min_hits: int = 2) -> tuple[str | None, int]:
    """Identify a regional variant by marker vocabulary.

    Returns (dialect_name, hit_count). Requires `min_hits` distinct markers so
    that a single ambiguous token ("bah", "yo") does not label a document.
    """
    words = {w.lower() for w in _WORD_RE.findall(text)}
    best: tuple[str | None, int] = (None, 0)
    for dialect, markers in DIALECT_MARKERS.items():
        hits = len(words & markers)
        if hits > best[1]:
            best = (dialect, hits)
    return best if best[1] >= min_hits else (None, best[1])


__all__ = [
    "DIALECT_MARKERS",
    "ENGLISH_LABELS",
    "INDONESIAN_LABELS",
    "MALAY_LABELS",
    "LanguageIdentifier",
    "LidResult",
    "Verdict",
    "detect_dialect",
    "detect_intrasentential_switching",
    "marker_ratios",
]
