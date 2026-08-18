"""Deterministic seeding.

Acceptance criterion: "Entire pipeline reproducible from one entry script with
a fixed seed." Every source of randomness in the pipeline draws from a
seed derived here, never from the global RNG directly, so that running phase 3
does not perturb the stream phase 4 will consume.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator


def derive_seed(base_seed: int, *tags: str | int) -> int:
    """Derive a stable child seed from a base seed and a set of tags.

    Deriving rather than sharing means the sample drawn for
    ``("phase2", "coverage_greedy", 1024)`` is identical whether or not
    phase 1 ran first, and stays identical if an unrelated phase is added.

    >>> derive_seed(42, "phase2", "bm_only") == derive_seed(42, "phase2", "bm_only")
    True
    >>> derive_seed(42, "phase2", "bm_only") == derive_seed(42, "phase2", "baseline_en")
    False
    """
    payload = "|".join([str(base_seed), *(str(t) for t in tags)])
    digest = hashlib.sha256(payload.encode()).digest()
    # numpy seeds must fit in uint32.
    return int.from_bytes(digest[:4], "big")


def rng(base_seed: int, *tags: str | int) -> Generator:
    """A numpy Generator scoped to `tags`."""
    return np.random.default_rng(derive_seed(base_seed, *tags))


def python_rng(base_seed: int, *tags: str | int) -> random.Random:
    """A stdlib Random scoped to `tags`, for shuffling python lists."""
    return random.Random(derive_seed(base_seed, *tags))


def seed_everything(base_seed: int, *tags: str | int) -> int:
    """Seed every global RNG in the process. Returns the derived seed.

    Call once at the top of a phase entrypoint. Torch is seeded only if it is
    importable, so this works unchanged on the host where torch is absent.
    """
    seed = derive_seed(base_seed, *tags)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - intentional global seeding
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism costs throughput on the routing forward passes but those are
    # the runs whose numbers must be reproducible.
    torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


__all__ = ["derive_seed", "python_rng", "rng", "seed_everything"]
