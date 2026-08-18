"""Runtime model and dataset id resolution.

Spec section 1: "exact HuggingFace model IDs change... Do not hardcode an ID
from this spec without confirming it resolves."

The ids in config.py were resolved by search on 2026-08-18. This module
re-checks them against the Hub before any expensive phase starts, and records
the resolved commit sha so a results artifact can name the exact revision it
was produced against.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.utils import (
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Resolution:
    """The outcome of checking one repo id."""

    repo_id: str
    repo_type: str
    exists: bool
    gated: bool
    sha: str | None = None
    last_modified: str | None = None
    error: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def usable(self) -> bool:
        return self.exists and not self.gated

    def describe(self) -> str:
        if self.usable:
            return f"{self.repo_id} -> {self.sha} (modified {self.last_modified})"
        if self.gated:
            return f"{self.repo_id} is GATED - accept the licence on the Hub first"
        return f"{self.repo_id} did not resolve: {self.error}"


def resolve_repo(repo_id: str, *, repo_type: str = "model", revision: str = "main") -> Resolution:
    """Check one repo id against the Hub. Never raises."""
    api = HfApi()
    try:
        if repo_type == "model":
            info: Any = api.model_info(repo_id, revision=revision)
        else:
            info = api.dataset_info(repo_id, revision=revision)
    except GatedRepoError as exc:
        return Resolution(repo_id, repo_type, exists=True, gated=True, error=str(exc))
    except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
        return Resolution(repo_id, repo_type, exists=False, gated=False, error=str(exc))
    except Exception as exc:  # network, auth, rate limit
        return Resolution(repo_id, repo_type, exists=False, gated=False, error=repr(exc))

    return Resolution(
        repo_id=repo_id,
        repo_type=repo_type,
        exists=True,
        gated=bool(getattr(info, "gated", False)),
        sha=getattr(info, "sha", None),
        last_modified=str(getattr(info, "last_modified", "")) or None,
        tags=tuple(getattr(info, "tags", ()) or ()),
    )


def search_nvidia_nemotron(pattern: str = "Nemotron-3.5-Lightning") -> list[str]:
    """List nvidia-org models matching `pattern`, newest first.

    Used when the pinned id fails to resolve: rather than guessing, report
    what the org actually publishes so the operator can repin.
    """
    api = HfApi()
    try:
        models = api.list_models(author="nvidia", search=pattern, sort="lastModified", direction=-1)
        return [m.id for m in models]
    except Exception as exc:
        logger.warning("Could not search the nvidia org: %r", exc)
        return []


def assert_model_resolves(
    bf16_repo: str,
    nvfp4_repo: str,
    *,
    revision: str = "main",
    fallback_repo: str | None = None,
) -> dict[str, Resolution]:
    """Validate the study's model ids before any GPU time is spent.

    Raises with an actionable message rather than letting a phase fail hours
    later on a 404 inside a download.
    """
    results = {
        "bf16": resolve_repo(bf16_repo, revision=revision),
        "nvfp4_reference": resolve_repo(nvfp4_repo, revision=revision),
    }
    if fallback_repo:
        results["fallback"] = resolve_repo(fallback_repo, revision=revision)

    for name, res in results.items():
        logger.info("%s: %s", name, res.describe())

    broken = {n: r for n, r in results.items() if n != "fallback" and not r.usable}
    if broken:
        candidates = search_nvidia_nemotron()
        hint = "\n".join(f"    {c}" for c in candidates[:20]) or "    (search returned nothing)"
        detail = "\n".join(f"  {n}: {r.describe()}" for n, r in broken.items())
        raise RuntimeError(
            "Pinned model ids no longer resolve.\n"
            f"{detail}\n\n"
            "Currently published under the nvidia org:\n"
            f"{hint}\n\n"
            "Repin model.bf16_repo / model.nvfp4_reference_repo in configs/config.yaml "
            "and update model.resolved_on."
        )
    return results


def resolve_datasets(repo_ids: list[str]) -> dict[str, Resolution]:
    """Check every dataset the pipeline will stream from.

    Does not raise: a missing optional source (dialect, for instance) is a
    recorded omission per spec section 3.4, not a failure. Callers decide.
    """
    return {rid: resolve_repo(rid, repo_type="dataset") for rid in repo_ids}


__all__ = [
    "Resolution",
    "assert_model_resolves",
    "resolve_datasets",
    "resolve_repo",
    "search_nvidia_nemotron",
]
