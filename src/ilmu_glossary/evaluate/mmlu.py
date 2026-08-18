"""Tiers 3 and 4 - Cross-MMLU and MalayMMLU.

Two instruments with different jobs:

  Tier 3, **Cross-MMLU** (SeaEval): parallel English / Chinese / Indonesian /
  Malay versions of the same 900 questions. This is the correct instrument for
  a controlled cross-language accuracy comparison, because MalayMMLU is
  natively Malay with no English counterpart and cannot support one. The
  Indonesian column is a useful bonus control: if Malay degrades but
  Indonesian does not, that is a strong signal, since the two are close.

  Tier 4, **MalayMMLU**: 24,213 natively Malay questions across Malaysian
  primary (Year 1-6) and secondary (Form 1-5) levels, 5 topics, 22 subjects.
  This is the ecological check - the benchmark YTL cites.

MalayMMLU is a **secondary** signal. Multiple-choice first-token accuracy is a
coarse detector; published work found a 1.7% automatic-metric drop
corresponding to 16.0% under human evaluation. The conclusion does not hang
on it.

Scoring follows the official MalayMMLU protocol (first-token and full-answer,
zero-shot, by-letter) so numbers stay comparable with published results, and
the same scorer serves Cross-MMLU.
"""

from __future__ import annotations

import logging
import string
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ilmu_glossary.config import Config

logger = logging.getLogger(__name__)

LETTERS = string.ascii_uppercase

# The Malay instruction is what the official evaluation uses; keeping it in
# Malay matters because an English instruction wrapped around a Malay question
# changes the language mix of the prompt and therefore the routing.
MALAY_PROMPT_TEMPLATE = (
    "Berikut adalah soalan aneka pilihan tentang {subject}. "
    "Jawab dengan memberikan huruf pilihan yang betul sahaja.\n\n"
    "{question}\n{options}\nJawapan:"
)

ENGLISH_PROMPT_TEMPLATE = (
    "The following is a multiple choice question. "
    "Answer with the letter of the correct option only.\n\n"
    "{question}\n{options}\nAnswer:"
)


@dataclass(frozen=True)
class MCQuestion:
    """One multiple-choice item, normalised across both benchmarks."""

    question_id: str
    question: str
    options: tuple[str, ...]
    answer_index: int
    subject: str = ""
    category: str = ""
    level: str = ""
    language: str = "malay"

    @property
    def answer_letter(self) -> str:
        return LETTERS[self.answer_index]

    def format_options(self) -> str:
        return "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(self.options))

    def prompt(self) -> str:
        template = ENGLISH_PROMPT_TEMPLATE if self.language == "english" else MALAY_PROMPT_TEMPLATE
        if self.language == "english":
            return template.format(question=self.question, options=self.format_options())
        return template.format(
            subject=self.subject or "pengetahuan am",
            question=self.question,
            options=self.format_options(),
        )


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_malay_mmlu(cfg: Config) -> list[MCQuestion]:
    """Load MalayMMLU, working around the documented load failure.

    Spec section 4d: "the HF dataset viewer fails with
    DatasetGenerationCastError because MalayMMLU_1shot.json carries two
    columns the other files lack. Load JSON files individually; do not call
    load_dataset on the repo root."

    Confirmed: `MalayMMLU_1shot.json` adds `full_question_1shot` and
    `full_question_1shot_llama`. Each configured file is therefore downloaded
    and parsed on its own.
    """
    import json

    from huggingface_hub import hf_hub_download

    questions: list[MCQuestion] = []

    for filename in cfg.eval.malay_mmlu_files:
        path = hf_hub_download(
            repo_id=cfg.eval.malay_mmlu_repo,
            filename=filename,
            repo_type="dataset",
        )
        with open(path, encoding="utf-8") as fh:  # noqa: PTH123 - path from hub
            records = json.load(fh)

        for record in records:
            question = _parse_malay_mmlu_record(record)
            if question is not None:
                questions.append(question)

    logger.info(
        "MalayMMLU: %d questions from %d files", len(questions), len(cfg.eval.malay_mmlu_files)
    )
    return questions


def _parse_malay_mmlu_record(record: dict[str, Any]) -> MCQuestion | None:
    """Normalise one MalayMMLU record.

    Schema: id, prompt, answer, year, subject, subject_eng, category, level,
    options, num_options, key. `key` is the answer letter; `answer` is the
    answer text. Either can identify the correct option, and both are tried
    because a record missing one is still usable.
    """
    options = record.get("options")
    if not isinstance(options, list) or len(options) < 2:
        return None

    options = [str(o) for o in options]
    answer_index: int | None = None

    key = record.get("key")
    if isinstance(key, str) and key.strip().upper() in LETTERS[: len(options)]:
        answer_index = LETTERS.index(key.strip().upper())

    if answer_index is None:
        answer = record.get("answer")
        if isinstance(answer, str):
            normalised = answer.strip()
            for i, option in enumerate(options):
                if option.strip() == normalised:
                    answer_index = i
                    break

    if answer_index is None:
        return None

    return MCQuestion(
        question_id=str(record.get("id", "")),
        question=str(record.get("prompt", "")),
        options=tuple(options),
        answer_index=answer_index,
        subject=str(record.get("subject", "")),
        category=str(record.get("category", "")),
        level=str(record.get("level", "")),
        language="malay",
    )


def load_cross_mmlu(cfg: Config) -> dict[str, list[MCQuestion]]:
    """Load SeaEval Cross-MMLU, one parallel question list per language.

    Parallelism is the point: the same 900 questions in each language. The
    loader asserts the per-language counts match, because a silently
    misaligned language column would turn a controlled comparison into an
    uncontrolled one.
    """
    from datasets import load_dataset

    by_language: dict[str, list[MCQuestion]] = {}

    for language in cfg.eval.cross_mmlu_languages:
        try:
            dataset = load_dataset(
                cfg.eval.cross_mmlu_repo,
                cfg.eval.cross_mmlu_config,
                split=language,
            )
        except Exception as exc:
            logger.warning("Cross-MMLU split %s unavailable: %r", language, exc)
            continue

        items: list[MCQuestion] = []
        for record in dataset:
            options = record.get("choices") or record.get("options")
            if not isinstance(options, list) or len(options) < 2:
                continue
            answer = record.get("answer")
            index = _resolve_answer_index(answer, [str(o) for o in options])
            if index is None:
                continue
            items.append(
                MCQuestion(
                    question_id=str(record.get("id", len(items))),
                    question=str(record.get("question", "")),
                    options=tuple(str(o) for o in options),
                    answer_index=index,
                    subject=str(record.get("category", "")),
                    language=language,
                )
            )
        by_language[language] = items

    counts = {lang: len(items) for lang, items in by_language.items()}
    if len(set(counts.values())) > 1:
        logger.warning(
            "Cross-MMLU language splits are not parallel: %s. The controlled "
            "cross-language comparison assumes the same questions in each "
            "language; report this alongside tier 3 results.",
            counts,
        )
    logger.info("Cross-MMLU: %s", counts)
    return by_language


def _resolve_answer_index(answer: Any, options: list[str]) -> int | None:
    if isinstance(answer, int) and 0 <= answer < len(options):
        return answer
    if isinstance(answer, str):
        stripped = answer.strip()
        if stripped.upper() in LETTERS[: len(options)]:
            return LETTERS.index(stripped.upper())
        for i, option in enumerate(options):
            if option.strip() == stripped:
                return i
    return None


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def score_first_token(
    handle: Any,
    question: MCQuestion,
    *,
    top_k: int = 20,
) -> tuple[bool, str, dict[str, float]]:
    """First-token accuracy: is the correct letter the most likely next token?

    Read from the logprob distribution over the *option letters only* rather
    than from free generation. Restricting to the valid letters is what the
    official protocol does and it removes a failure mode that has nothing to
    do with quantization - a model that answers "Jawapannya ialah B" would
    otherwise score zero for a correct answer.
    """
    from ilmu_glossary.evaluate.server import post_json

    payload = {
        "model": handle.model_path,
        "prompt": question.prompt(),
        "max_tokens": 1,
        "logprobs": top_k,
        "temperature": 0.0,
    }
    response = post_json(handle.completions_url, payload)
    choice = response["choices"][0]
    top_logprobs = (choice.get("logprobs") or {}).get("top_logprobs") or [{}]
    distribution = top_logprobs[0] if top_logprobs else {}

    letter_scores: dict[str, float] = {}
    for i in range(len(question.options)):
        letter = LETTERS[i]
        # Models emit the letter with or without a leading space depending on
        # the tokenizer; take whichever variant the model actually produced.
        candidates = [distribution.get(letter), distribution.get(f" {letter}")]
        values = [v for v in candidates if v is not None]
        letter_scores[letter] = max(values) if values else float("-inf")

    predicted = max(letter_scores, key=lambda k: letter_scores[k])
    return predicted == question.answer_letter, predicted, letter_scores


def score_full_answer(handle: Any, question: MCQuestion) -> tuple[bool, int, list[float]]:
    """Full-answer accuracy: which option text does the model score highest?

    Each option is appended to the prompt and length-normalised by token
    count. Without normalisation the shortest option wins regardless of
    content, which would make this metric a length detector.
    """
    from ilmu_glossary.evaluate.server import completion_logprobs

    scores: list[float] = []
    for option in question.options:
        text = f"{question.prompt()} {option}"
        try:
            tokens = completion_logprobs(handle, text, top_k=1, max_tokens=0)
        except Exception as exc:
            logger.debug("full-answer scoring failed: %r", exc)
            scores.append(float("-inf"))
            continue
        logprobs = [t["logprob"] for t in tokens if t["logprob"] is not None]
        # Score only the option's own tokens, not the shared prompt.
        option_logprobs = logprobs[-max(len(option.split()), 1) :] if logprobs else []
        scores.append(float(np.mean(option_logprobs)) if option_logprobs else float("-inf"))

    predicted = int(np.argmax(scores))
    return predicted == question.answer_index, predicted, scores


def evaluate_questions(
    handle: Any,
    questions: list[MCQuestion],
    *,
    do_full_answer: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    """Score a question list, returning one row per item."""
    subset = questions[:limit] if limit else questions
    rows: list[dict[str, Any]] = []

    for i, question in enumerate(subset):
        try:
            first_correct, predicted_letter, _ = score_first_token(handle, question)
        except Exception as exc:
            logger.warning("first-token scoring failed on %s: %r", question.question_id, exc)
            continue

        row: dict[str, Any] = {
            "question_id": question.question_id,
            "language": question.language,
            "subject": question.subject,
            "category": question.category,
            "level": question.level,
            "first_token_correct": first_correct,
            "predicted_letter": predicted_letter,
            "answer_letter": question.answer_letter,
            "n_options": len(question.options),
        }
        if do_full_answer:
            full_correct, predicted_index, _ = score_full_answer(handle, question)
            row["full_answer_correct"] = full_correct
            row["full_answer_predicted"] = predicted_index

        rows.append(row)
        if i % 500 == 0 and i:
            logger.info("  scored %d/%d questions", i, len(subset))

    return pd.DataFrame(rows)


def accuracy_summary(df: pd.DataFrame, *, group_by: list[str] | None = None) -> pd.DataFrame:
    """Accuracy with sample size and a binomial standard error.

    Spec section 8: every number carries its sample size and variance. A
    per-subject accuracy over 40 questions and one over 4,000 are not
    comparable without it.
    """
    if df.empty:
        return df

    keys = group_by or []
    metrics = [c for c in ("first_token_correct", "full_answer_correct") if c in df.columns]

    def _summarise(group: pd.DataFrame) -> pd.Series:
        out: dict[str, float] = {"n": float(len(group))}
        for metric in metrics:
            values = group[metric].astype(float)
            accuracy = float(values.mean())
            out[metric.replace("_correct", "_accuracy")] = accuracy
            out[metric.replace("_correct", "_stderr")] = (
                float(np.sqrt(accuracy * (1 - accuracy) / len(group))) if len(group) else 0.0
            )
        return pd.Series(out)

    if not keys:
        return _summarise(df).to_frame().T

    # An explicit loop rather than groupby.apply: apply's return shape depends
    # on what the callable returns, so the column layout of a results table
    # would become an implementation detail of pandas.
    rows: list[dict[str, Any]] = []
    for group_key, group in df.groupby(keys):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        summary = {str(k): v for k, v in _summarise(group).to_dict().items()}
        rows.append({**dict(zip(keys, values, strict=True)), **summary})
    return pd.DataFrame(rows)


__all__ = [
    "ENGLISH_PROMPT_TEMPLATE",
    "LETTERS",
    "MALAY_PROMPT_TEMPLATE",
    "MCQuestion",
    "accuracy_summary",
    "evaluate_questions",
    "load_cross_mmlu",
    "load_malay_mmlu",
    "score_first_token",
    "score_full_answer",
]
