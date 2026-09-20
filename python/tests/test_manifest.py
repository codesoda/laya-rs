from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "manifests" / "sources.json"
EXPECTED_MODEL_BYTES = {
    "english": 842_609_210,
    "multilingual": 643_835_514,
    "typed-decisions": 842_609_220,
}


def _manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_cases() -> list[pytest.param]:
    if not MANIFEST_PATH.exists():
        return [pytest.param("", "", {}, id="manifest-not-created")]
    manifest = _manifest()
    cases = []
    for profile, profile_record in manifest.get("profiles", {}).items():
        for relative, expected in profile_record.get("files", {}).items():
            cases.append(pytest.param(profile, relative, expected, id=f"{profile}-{relative.replace('/', '-') }"))
    return cases or [pytest.param("", "", {}, id="no-profile-files")]


def test_manifest_parses_and_has_expected_profiles() -> None:
    manifest = _manifest()
    assert manifest["schema_version"] == 1
    assert set(manifest["profiles"]) == set(EXPECTED_MODEL_BYTES)
    for profile, expected_bytes in EXPECTED_MODEL_BYTES.items():
        model = manifest["profiles"][profile]["files"]["model.safetensors"]
        assert model["bytes"] == expected_bytes
        assert len(model["sha256"]) == 64


@pytest.mark.parametrize("profile,relative,expected", _local_cases())
def test_locally_present_file_matches_manifest(profile: str, relative: str, expected: dict) -> None:
    if not profile:
        pytest.skip("manifest has no local profile file records")
    profile_record = _manifest()["profiles"][profile]
    path = REPO_ROOT / profile_record["directory"] / relative
    if not path.exists():
        pytest.skip(f"model cache file is not present: {path}")
    assert path.stat().st_size == expected["bytes"]
    assert _sha256(path) == expected["sha256"]
