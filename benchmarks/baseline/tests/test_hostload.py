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
    "allow_comm_patterns": [r".*/Python\.framework/.*Python$", r".*/\.venv/bin/[^/]+$"],
    "system_prefixes": ["/System/", "/usr/libexec/", "/usr/sbin/", "/sbin/", "com.apple."],
}


def snap(load1: float = 1.0, processes: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {"loadavg": {"1": load1, "5": 1.0, "15": 1.0}, "processes": processes or []}


def process(name: str, pcpu: float, comm: str | None = None) -> dict[str, object]:
    return {"pid": 1, "name": name, "comm": comm or name, "pcpu": pcpu, "rss_mb": 10.0, "user": "test"}


def test_evaluate_idle() -> None:
    assert hostload.evaluate(snap(), POLICY) == {"idle": True, "violations": [], "self_processes": []}


def test_evaluate_load_violation() -> None:
    result = hostload.evaluate(snap(3.1), POLICY)
    assert result["idle"] is False
    assert result["violations"] == ["load1 3.1 > 3.0"]


def test_end_load_is_informational_only() -> None:
    result = hostload.evaluate(snap(3.1), POLICY, phase="end")
    assert result["idle"] is True
    assert result["violations"] == []
    assert result["load1_end_informational"] == "load1 3.1 > 3.0"


def test_self_tree_processes_are_excluded_and_recorded() -> None:
    current = process("ctx", 80.0)
    current["pid"] = 42
    result = hostload.evaluate({**snap(processes=[current]), "self_tree": [42]}, POLICY)
    assert result["idle"] is True
    assert result["self_processes"] == [current]


def test_allow_comm_pattern_matches_full_comm() -> None:
    comm = "/opt/homebrew/Cellar/python@3.12/3.12.7/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python"
    result = hostload.evaluate(snap(processes=[process("Python", 80.0, comm)]), POLICY)
    assert result["idle"] is True


def test_old_snapshot_pattern_is_treated_as_self() -> None:
    comm = "/opt/homebrew/Cellar/python@3.12/3.12.7/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python"
    item = process("Python", 80.0, comm)
    result = hostload.evaluate(snap(processes=[item]), POLICY)
    assert result["idle"] is True
    assert result["self_processes"] == [item]


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


def test_snapshot_records_self_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hostload.os, "getpid", lambda: 42)

    def fake_run(command: tuple[str, ...]) -> tuple[str | None, str | None]:
        if command == ("ps", "-Ao", "pid,ppid"):
            return " PID PPID\n42 1\n43 42\n44 43\n99 1", None
        if command == ("ps", "-Ao", "pid,pcpu,rss,user,comm"):
            return " PID %CPU RSS USER COMM\n42 80 100 user /tmp/python", None
        return None, "unavailable"

    monkeypatch.setattr(hostload, "_run", fake_run)
    result = hostload.snapshot(POLICY)
    assert result["self_pid"] == 42
    assert result["self_tree"] == [42, 43, 44]


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
