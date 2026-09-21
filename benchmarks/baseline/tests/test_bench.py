from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from laya_baseline import bench, bench_report


def test_stats_match_numpy() -> None:
    values = [1.0, 2.0, 7.0, 11.0]
    stats = bench.summarize_samples(values, 2)
    assert stats["mean_ms"] == float(np.mean(values))
    assert stats["std_ms"] == float(np.std(values))
    assert stats["p50_ms"] == float(np.percentile(values, 50))
    assert stats["p95_ms"] == float(np.percentile(values, 95))
    assert stats["requests_per_sec"] == 1000.0 / np.percentile(values, 50)
    assert stats["questions_per_sec"] == 2000.0 / np.percentile(values, 50)


def test_workload_resolution() -> None:
    assert bench.resolve_workload("published") == [
        "published-latency-q1", "published-latency-q5", "published-latency-q10", "published-latency-q50"
    ]
    assert bench.resolve_workload("distinct-ml-q1, distinct-ml-q10") == ["distinct-ml-q1", "distinct-ml-q10"]
    assert bench.resolve_workload("primary") == list(bench.WORKLOADS["primary"])


def test_immutability_refusal_precedes_model_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    (fixture_dir / "tiny.json").write_text(json.dumps({"id": "tiny", "state": "x", "questions": {"q": {
        "type": "noul", "instructions": "x"
    }}}))
    output = tmp_path / "benchmarks" / "results" / "run"
    output.mkdir(parents=True)
    (output / "english-cpu-tiny.json").write_text("{}")
    monkeypatch.setattr(bench, "FIXTURE_DIR", fixture_dir)
    monkeypatch.setattr(bench, "REPO_ROOT", tmp_path)
    with pytest.raises(FileExistsError):
        bench.run("english", "cpu", "tiny", reps=1, run_id="run")


def test_report_reevaluates_old_python_record_as_uncontended() -> None:
    record = {
        "contended": True,
        "host_start": {
            "loadavg": {"1": 2.3, "5": 2.0, "15": 2.0},
            "processes": [{
                "pid": 99, "name": "Python",
                "comm": "/opt/homebrew/Cellar/python@3.12/3.12.7/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python",
                "pcpu": 112.8,
            }],
        },
    }
    policy = {
        "load1_max": 3.0, "process_cpu_percent_max": 15.0,
        "allow_comm_patterns": [r".*/Python\.framework/.*Python$"],
    }
    assert bench_report.reevaluate_contention(record, policy) is False


def test_report_renderer_on_synthetic_result() -> None:
    record = {
        "profile": "multilingual", "device": "cpu", "fixture": "published-latency-q1",
        "n_questions": 1, "diagnostics": {"n_tokens": 42}, "contended": True,
        "e2e_ms": [10.0, 20.0, 30.0], "model_ms": [5.0, 6.0, 7.0], "preprocess_ms": [1.0, 2.0, 3.0],
    }
    markdown = bench_report.render_markdown([record], compare_published=True)
    assert "| multilingual | cpu | published-latency-q1 | 1 | 42 | 20.0 |" in markdown
    assert "32.8" in markdown and "38.8" in markdown
    assert "different hardware" in markdown
