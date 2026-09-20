from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

import pytest

from laya_baseline import bench, hostload


POLICY = {
    "schema_version": 1,
    "load1_max": 3.0,
    "process_cpu_percent_max": 15.0,
    "process_rss_mb_report_min": 200,
    "allow": ["pi"],
    "system_prefixes": ["/System/", "/usr/libexec/", "/usr/sbin/", "/sbin/", "com.apple."],
}


def snap(load1: float = 1.0, processes: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {"loadavg": {"1": load1, "5": 1.0, "15": 1.0}, "processes": processes or []}


def process(name: str, pcpu: float, comm: str | None = None) -> dict[str, object]:
    return {"pid": 1, "name": name, "comm": comm or name, "pcpu": pcpu, "rss_mb": 10.0, "user": "test"}


def test_evaluate_idle() -> None:
    assert hostload.evaluate(snap(), POLICY) == {"idle": True, "violations": []}


def test_evaluate_load_violation() -> None:
    result = hostload.evaluate(snap(3.1), POLICY)
    assert result["idle"] is False
    assert result["violations"] == ["load1 3.1 > 3.0"]


def test_evaluate_disallowed_process() -> None:
    result = hostload.evaluate(snap(processes=[process("ctx", 80.0)]), POLICY)
    assert result["violations"] == ["ctx pcpu 80.0 > 15.0 (not in allow list)"]


def test_allowed_process_above_threshold_passes() -> None:
    assert hostload.evaluate(snap(processes=[process("pi", 80.0)]), POLICY)["idle"] is True


def test_system_prefix_is_excluded() -> None:
    result = hostload.evaluate(snap(processes=[process("daemon", 99.0, "/System/Library/daemon")]), POLICY)
    assert result["idle"] is True


def test_fingerprint_is_stable_and_sorted() -> None:
    first = snap(processes=[process("zeta", 50.0), process("alpha", 60.0)])
    second = snap(processes=[process("alpha", 60.0), process("zeta", 50.0)])
    assert hostload.fingerprint(first, POLICY) == hostload.fingerprint(second, POLICY)
    assert hostload.fingerprint(first, POLICY) == "load1-idle|alpha,zeta"


def test_require_idle_exits_before_torch_or_laya_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    policy_path = tmp_path / "benchmarks" / "idle-policy.json"
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text(json.dumps(POLICY), encoding="utf-8")
    violating = snap(processes=[process("ctx", 80.0)])
    monkeypatch.setattr(bench, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bench, "resolve_workload", lambda _: ["tiny"])
    monkeypatch.setattr(bench.hostload, "snapshot", lambda policy: violating)
    monkeypatch.setattr(sys, "argv", ["laya-bench", "--profile", "english", "--device", "cpu",
                                        "--workload", "tiny", "--require-idle"])
    imported: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object):
        if name == "torch" or name == "laya":
            imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 3
    assert imported == []
