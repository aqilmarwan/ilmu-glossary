"""Artifact persistence.

Spec section 2: "All intermediate artifacts persist to disk. No phase re-runs a
previous phase." Every write goes through this module so that provenance
columns (config fingerprint, phase, git sha) are attached uniformly and so
that resumability has one definition rather than six.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROVENANCE_COLUMNS = ("_config_fingerprint", "_phase", "_git_sha", "_written_at")


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def git_sha(default: str = "unknown") -> str:
    """Short git sha of the working tree, or `default` outside a repo.

    A dirty tree is reported with a `-dirty` suffix so a results file can never
    silently claim to come from a clean commit.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return default
    return f"{sha}-dirty" if dirty else sha


def stamp(df: pd.DataFrame, *, fingerprint: str, phase: str) -> pd.DataFrame:
    """Attach provenance columns. Returns a copy."""
    out = df.copy()
    out["_config_fingerprint"] = fingerprint
    out["_phase"] = phase
    out["_git_sha"] = git_sha()
    out["_written_at"] = datetime.now(UTC).isoformat()
    return out


# --------------------------------------------------------------------------
# parquet
# --------------------------------------------------------------------------


def write_parquet(
    df: pd.DataFrame,
    path: Path,
    *,
    fingerprint: str,
    phase: str,
    overwrite: bool = True,
) -> Path:
    """Write a results table with provenance attached.

    Writes to a temporary sibling then renames, so a killed job never leaves a
    half-written parquet that a resumed run would mistake for complete.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists and overwrite=False")

    stamped = stamp(df, fingerprint=fingerprint, phase=phase)
    tmp = path.with_suffix(path.suffix + ".tmp")
    stamped.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The phase that produces it has not run - "
            "phases never re-run their predecessors, so run it explicitly."
        )
    return pd.read_parquet(path)


def parquet_fingerprint(path: Path) -> str | None:
    """Read back the config fingerprint an artifact was written under."""
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["_config_fingerprint"])
    if df.empty:
        return None
    value = df["_config_fingerprint"].iloc[0]
    return str(value)


# --------------------------------------------------------------------------
# jsonl
# --------------------------------------------------------------------------


def write_jsonl(records: Iterable[dict[str, Any]], path: Path, *, overwrite: bool = True) -> int:
    """Stream records to jsonl. Returns the count written.

    Atomic via temp-then-rename for the same reason as `write_parquet`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists and overwrite=False")

    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


def read_jsonl(path: Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Stream records back. Lazily evaluated - these files can be large."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def count_jsonl(path: Path) -> int:
    """Line count without materialising the file."""
    if not path.exists():
        return 0
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


# --------------------------------------------------------------------------
# json sidecars
# --------------------------------------------------------------------------


def write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    tmp.replace(path)
    return path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    result: dict[str, Any] = json.loads(path.read_text())
    return result


# --------------------------------------------------------------------------
# resumability
# --------------------------------------------------------------------------


def is_complete(path: Path, *, fingerprint: str | None = None) -> bool:
    """True if an artifact exists and matches the current config.

    A fingerprint mismatch counts as incomplete: the artifact was produced by
    different settings, so reusing it would silently mix configurations.
    """
    if not path.exists():
        return False
    if fingerprint is None:
        return True
    if path.suffix != ".parquet":
        return True
    return parquet_fingerprint(path) == fingerprint


__all__ = [
    "PROVENANCE_COLUMNS",
    "count_jsonl",
    "git_sha",
    "is_complete",
    "parquet_fingerprint",
    "read_json",
    "read_jsonl",
    "read_parquet",
    "stamp",
    "write_json",
    "write_jsonl",
    "write_parquet",
]
