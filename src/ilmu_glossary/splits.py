"""Held-out splits and contamination guards.

Spec section 3: "Split every class 80/20 before anything else runs. Calibration
draws only from the 80%; evaluation touches only the 20%. Persist the split
indices with a fixed seed and assert disjointness before each phase."

The assertion is not decorative. `oracle_contaminated` deliberately calibrates
on the evaluation distribution, and it would be easy for that contamination to
leak into a variant that is supposed to be clean. `assert_disjoint` is called
at the top of phases 2, 3 and 4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ilmu_glossary.io import read_json, write_json
from ilmu_glossary.seeds import rng

logger = logging.getLogger(__name__)

SPLIT_FILENAME = "splits.json"


class ContaminationError(AssertionError):
    """Raised when calibration and evaluation index sets intersect."""


@dataclass(frozen=True)
class Split:
    """Persisted 80/20 index split for one corpus class."""

    corpus_class: str
    train_indices: tuple[int, ...]
    eval_indices: tuple[int, ...]
    total: int
    seed: int

    @property
    def train_fraction(self) -> float:
        return len(self.train_indices) / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_class": self.corpus_class,
            "train_indices": list(self.train_indices),
            "eval_indices": list(self.eval_indices),
            "total": self.total,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Split:
        """Rebuild from the persisted JSON.

        Indices are re-read as ints rather than trusted, so a hand-edited
        splits.json produces a clear failure instead of a silently wrong split.
        """
        return cls(
            corpus_class=str(payload["corpus_class"]),
            train_indices=tuple(int(i) for i in payload["train_indices"]),
            eval_indices=tuple(int(i) for i in payload["eval_indices"]),
            total=int(payload["total"]),
            seed=int(payload["seed"]),
        )


def make_split(
    corpus_class: str,
    total: int,
    *,
    base_seed: int,
    train_fraction: float = 0.8,
) -> Split:
    """Deterministic 80/20 split of `total` document indices.

    Seeded per class so that adding a class later does not reshuffle the
    splits of classes already built.
    """
    if total <= 0:
        raise ValueError(f"{corpus_class}: cannot split {total} documents")

    generator = rng(base_seed, "split", corpus_class)
    permuted = generator.permutation(total)
    n_train = round(total * train_fraction)
    # Guarantee both sides are non-empty; a class with one held-out document
    # cannot support a perplexity estimate but should fail loudly later, not here.
    n_train = min(max(n_train, 1), total - 1)

    return Split(
        corpus_class=corpus_class,
        train_indices=tuple(sorted(int(i) for i in permuted[:n_train])),
        eval_indices=tuple(sorted(int(i) for i in permuted[n_train:])),
        total=total,
        seed=base_seed,
    )


def save_splits(splits: dict[str, Split], directory: Path) -> Path:
    path = directory / SPLIT_FILENAME
    write_json({name: split.to_dict() for name, split in splits.items()}, path)
    logger.info("Persisted %d splits to %s", len(splits), path)
    return path


def load_splits(directory: Path) -> dict[str, Split]:
    path = directory / SPLIT_FILENAME
    payload = read_json(path)
    return {name: Split.from_dict(value) for name, value in payload.items()}


def assert_internally_consistent(split: Split) -> None:
    """Every index appears exactly once, on exactly one side."""
    train, evaluation = set(split.train_indices), set(split.eval_indices)
    overlap = train & evaluation

    if overlap:
        raise ContaminationError(
            f"{split.corpus_class}: {len(overlap)} indices appear in both sides "
            f"of its own split, e.g. {sorted(overlap)[:5]}"
        )
    if len(train) + len(evaluation) != split.total:
        raise ContaminationError(
            f"{split.corpus_class}: split covers {len(train) + len(evaluation)} "
            f"indices but the class holds {split.total}"
        )
    if len(split.train_indices) != len(train) or len(split.eval_indices) != len(evaluation):
        raise ContaminationError(f"{split.corpus_class}: split contains duplicate indices")


def assert_disjoint(
    calibration_ids: set[str],
    evaluation_ids: set[str],
    *,
    context: str,
    allow_contamination: bool = False,
) -> None:
    """Assert no document is used for both calibration and evaluation.

    `allow_contamination` is set only by `oracle_contaminated`, whose whole
    purpose is to violate this. The flag exists so the violation must be
    stated explicitly at the call site rather than achieved by skipping
    the check.
    """
    overlap = calibration_ids & evaluation_ids
    if not overlap:
        logger.info(
            "%s: disjointness verified (%d calibration / %d evaluation docs)",
            context,
            len(calibration_ids),
            len(evaluation_ids),
        )
        return

    if allow_contamination:
        logger.warning(
            "%s: CONTAMINATED BY DESIGN - %d documents (%.1f%% of calibration) "
            "are drawn from the evaluation distribution. This variant is not a "
            "legitimate result and must be labelled as such wherever it appears.",
            context,
            len(overlap),
            100.0 * len(overlap) / max(len(calibration_ids), 1),
        )
        return

    raise ContaminationError(
        f"{context}: {len(overlap)} documents appear in both the calibration set "
        f"and the evaluation set, e.g. {sorted(overlap)[:5]}. "
        "Calibration must draw only from the 80% train split."
    )


def verify_all(splits: dict[str, Split], *, expected_fraction: float, tol: float = 0.02) -> None:
    """Full pre-phase check: consistency plus the expected train fraction."""
    for split in splits.values():
        assert_internally_consistent(split)
        drift = abs(split.train_fraction - expected_fraction)
        if drift > tol:
            raise ContaminationError(
                f"{split.corpus_class}: train fraction {split.train_fraction:.3f} "
                f"deviates from the configured {expected_fraction:.3f} by {drift:.3f}"
            )
    logger.info("All %d splits verified", len(splits))


def eval_index_set(splits: dict[str, Split]) -> dict[str, set[int]]:
    """Evaluation indices per class, for building document-id guards."""
    return {name: set(split.eval_indices) for name, split in splits.items()}


def train_index_set(splits: dict[str, Split]) -> dict[str, set[int]]:
    return {name: set(split.train_indices) for name, split in splits.items()}


def summarise(splits: dict[str, Split]) -> np.ndarray:
    """Compact array view for logging: rows of (total, n_train, n_eval)."""
    return np.array(
        [[s.total, len(s.train_indices), len(s.eval_indices)] for s in splits.values()],
        dtype=np.int64,
    )


__all__ = [
    "SPLIT_FILENAME",
    "ContaminationError",
    "Split",
    "assert_disjoint",
    "assert_internally_consistent",
    "eval_index_set",
    "load_splits",
    "make_split",
    "save_splits",
    "summarise",
    "train_index_set",
    "verify_all",
]
