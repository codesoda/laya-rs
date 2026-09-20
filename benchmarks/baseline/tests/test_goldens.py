"""Model-free integrity checks for committed Python oracle records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from laya_baseline.paths import PROFILES, REPO_ROOT

FIXTURE_DIR = REPO_ROOT / "benchmarks" / "fixtures" / "requests"
GOLDEN_DIR = REPO_ROOT / "benchmarks" / "goldens"


def _fixture_ids() -> list[str]:
    ids = []
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            ids.append(json.load(handle)["id"])
    return ids


def _load(profile: str, device: str, fixture_id: str) -> dict[str, Any]:
    path = GOLDEN_DIR / profile / device / f"{fixture_id}.json"
    assert path.is_file(), f"missing golden: {path}"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("fixture_id", _fixture_ids())
def test_every_fixture_has_cpu_golden(profile: str, fixture_id: str) -> None:
    record = _load(profile, "cpu", fixture_id)
    assert record["meta"]["fixture_id"] == fixture_id
    assert record["meta"]["profile"] == profile
    assert record["native_result"] is not None or record["error"] is not None


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("device", ("cpu", "mps"))
def test_instrumented_matches_native(profile: str, device: str) -> None:
    for fixture_id in _fixture_ids():
        record = _load(profile, device, fixture_id)
        assert record["instrumented_matches_native"] is True, f"{profile}/{device}/{fixture_id}"


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("device", ("cpu", "mps"))
def test_non_error_golden_hidden_state_sidecars(profile: str, device: str) -> None:
    for fixture_id in _fixture_ids():
        record = _load(profile, device, fixture_id)
        if record["error"] is not None:
            continue
        sidecar = GOLDEN_DIR / profile / device / record["hidden_states_npz"]
        assert sidecar.is_file(), f"missing hidden-state sidecar: {sidecar}"
        assert all("hidden_cls" not in row and "hidden_markers" not in row for row in record["model"].values())
        with np.load(sidecar, allow_pickle=False) as arrays:
            assert set(arrays.files) == {"hidden_cls", "hidden_markers", "hidden_markers_mask"}
            assert list(arrays["hidden_cls"].shape) == record["hidden_cls_shape"]
            assert list(arrays["hidden_markers"].shape) == record["hidden_markers_shape"]
            assert arrays["hidden_cls"].dtype == np.float32
            assert arrays["hidden_markers"].dtype == np.float32
            assert arrays["hidden_markers_mask"].dtype == np.bool_
            assert arrays["hidden_markers_mask"].shape == arrays["hidden_markers"].shape[:2]
            assert np.array_equal(arrays["hidden_markers_mask"], np.asarray(record["batch"]["marker_mask"], dtype=bool))
            assert np.all(arrays["hidden_markers"][~arrays["hidden_markers_mask"]] == 0)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("device", ("cpu", "mps"))
def test_non_error_golden_invariants(profile: str, device: str) -> None:
    for fixture_id in _fixture_ids():
        record = _load(profile, device, fixture_id)
        if record["error"] is not None:
            continue
        for qid, serialized in record["serialized"].items():
            assert len(record["tokens"][qid]["markers"]) == len(serialized["rendered_options"]), (
                profile,
                device,
                fixture_id,
                qid,
            )
        attention_sum = sum(sum(row) for row in record["batch"]["attention_mask"])
        assert attention_sum == record["native_result"]["usage"]["input_tokens"]

        for qid, answer in record["native_result"]["answers"].items():
            if answer["type"] == "noul":
                assert answer["noul"] == round(record["model"][qid]["probs_unrounded"][1], 4)
