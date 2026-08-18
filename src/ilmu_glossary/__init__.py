"""Routing-aware quantization calibration for Bahasa Melayu.

Tests whether post-training quantization calibrated on non-Malay text imposes
a measurable, recoverable accuracy penalty on Bahasa Melayu workloads, and
whether routing-coverage-driven calibration sample selection recovers it
better than corpus substitution alone.

This studies open base checkpoints. It makes no claim about ILMU's fine-tune.
"""

from __future__ import annotations

from ilmu_glossary.config import (
    CalibVariant,
    Config,
    CorpusClass,
    MambaStateDtype,
    RecipeFamily,
)

__version__ = "0.1.0"

__all__ = [
    "CalibVariant",
    "Config",
    "CorpusClass",
    "MambaStateDtype",
    "RecipeFamily",
    "__version__",
]
