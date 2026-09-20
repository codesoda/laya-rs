"""Repository and cache path resolution for baseline tools."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CACHE_ROOT = REPO_ROOT / ".cache" / "laya"
HUB_REPO_ID = "convaiinnovations/laya"
HUB_REVISION = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"
PROFILES = ("english", "multilingual", "typed-decisions")


def cache_root() -> Path:
    """Return $LAYA_HOME when set, otherwise the repository-local cache."""
    value = os.environ.get("LAYA_HOME")
    return Path(value).expanduser().resolve() if value else DEFAULT_CACHE_ROOT


def revision_root() -> Path:
    return cache_root() / "hub" / "convaiinnovations--laya" / HUB_REVISION


def profile_dir(profile: str) -> Path:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    return revision_root() / profile


def manifest_path() -> Path:
    return REPO_ROOT / "manifests" / "sources.json"
