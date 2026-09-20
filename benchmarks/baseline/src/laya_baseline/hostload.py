"""Small, dependency-free host load snapshots for comparable benchmarks."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT

DEFAULT_POLICY_PATH = REPO_ROOT / "benchmarks" / "idle-policy.json"


def load_policy(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the committed host policy, or a caller-supplied policy for tests."""
    policy_path = Path(path) if path is not None else DEFAULT_POLICY_PATH
    with policy_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _run(command: tuple[str, ...]) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        return None, str(error)
    text = (result.stdout or result.stderr).strip()
    if result.returncode != 0:
        return text or None, f"exit status {result.returncode}: {text or 'no output'}"
    return text or None, None


def _is_system(comm: str, prefixes: list[str]) -> bool:
    base = Path(comm).name
    return any(comm.startswith(prefix) or base.startswith(prefix) for prefix in prefixes)


def _processes(policy: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    raw, error = _run(("ps", "-Ao", "pid,pcpu,rss,user,comm"))
    if raw is None:
        return [], error
    minimum_rss = float(policy.get("process_rss_mb_report_min", 200))
    prefixes = [str(prefix) for prefix in policy.get("system_prefixes", [])]
    result: list[dict[str, Any]] = []
    for line in raw.splitlines()[1:]:
        fields = line.strip().split(None, 4)
        if len(fields) != 5:
            continue
        try:
            pid = int(fields[0])
            pcpu = float(fields[1])
            rss_mb = float(fields[2]) / 1024.0
        except ValueError:
            continue
        comm = fields[4].strip()
        if not comm or _is_system(comm, prefixes):
            continue
        if pcpu < 1.0 and rss_mb < minimum_rss:
            continue
        result.append({
            "pid": pid,
            "name": Path(comm).name,
            "comm": comm,
            "pcpu": pcpu,
            "rss_mb": rss_mb,
            "user": fields[3],
        })
    result.sort(key=lambda process: (-float(process["pcpu"]), -float(process["rss_mb"]), str(process["name"]), int(process["pid"])))
    return result, error


def _load_values(snapshot_value: dict[str, Any]) -> tuple[float, float, float]:
    values = snapshot_value.get("loadavg")
    if isinstance(values, dict):
        return tuple(float(values.get(key, 0.0)) for key in ("1", "5", "15"))  # type: ignore[return-value]
    if isinstance(values, (list, tuple)):
        padded = list(values) + [0.0, 0.0, 0.0]
        return float(padded[0]), float(padded[1]), float(padded[2])
    return (
        float(snapshot_value.get("load1", snapshot_value.get("load_1", 0.0))),
        float(snapshot_value.get("load5", snapshot_value.get("load_5", 0.0))),
        float(snapshot_value.get("load15", snapshot_value.get("load_15", 0.0))),
    )


def _power(raw: str | None) -> dict[str, Any]:
    value: dict[str, Any] = {"raw": raw}
    if raw:
        first = raw.splitlines()[0].strip() if raw.splitlines() else ""
        if "AC Power" in raw:
            value["source"] = "AC Power"
        elif "Battery Power" in raw:
            value["source"] = "Battery Power"
        elif first:
            value["source"] = first
        for line in raw.splitlines():
            match = re.search(r"(\d+)%", line)
            if match:
                value["battery_percent"] = int(match.group(1))
                break
    return value


def _low_power_mode(raw: str | None) -> int | None:
    if not raw:
        return None
    for line in raw.splitlines():
        if "lowpowermode" in line.lower():
            try:
                return int(line.split()[-1])
            except (ValueError, IndexError):
                return None
    return None


def snapshot(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Capture load, notable processes, GUI applications, thermal and power state."""
    policy = policy or load_policy()
    try:
        loads = os.getloadavg()
    except OSError:
        loads = (0.0, 0.0, 0.0)
    processes, process_error = _processes(policy)
    gui_raw, gui_error = _run((
        "osascript", "-e",
        "tell application \"System Events\" to get name of every process whose background only is false",
    ))
    gui_apps: list[str] | None = None
    if gui_raw is not None and gui_error is None:
        gui_apps = [item.strip() for item in gui_raw.split(",") if item.strip()]
    thermal, thermal_error = _run(("pmset", "-g", "therm"))
    power_raw, power_error = _run(("pmset", "-g", "batt"))
    pmset_raw, pmset_error = _run(("pmset", "-g"))
    timestamp = time.time()
    result: dict[str, Any] = {
        "timestamp": timestamp,
        "loadavg": {"1": float(loads[0]), "5": float(loads[1]), "15": float(loads[2])},
        "load1": float(loads[0]),
        "load5": float(loads[1]),
        "load15": float(loads[2]),
        "processes": processes,
        "gui_apps": gui_apps,
        "thermal": thermal,
        "power": _power(power_raw),
        "lowpowermode": _low_power_mode(pmset_raw),
    }
    errors = {
        "processes": process_error,
        "gui_apps": gui_error,
        "thermal": thermal_error,
        "power": power_error,
        "lowpowermode": pmset_error,
    }
    errors = {key: value for key, value in errors.items() if value}
    if errors:
        result["errors"] = errors
    if gui_error:
        result["gui_apps_error"] = gui_error
    if thermal_error:
        result["thermal_error"] = thermal_error
    if power_error:
        result["power_error"] = power_error
    return result


def evaluate(snapshot_value: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    """Evaluate only declared idle-host gates; RSS-only reports are informational."""
    load1, _, _ = _load_values(snapshot_value)
    load_limit = float(policy.get("load1_max", 3.0))
    cpu_limit = float(policy.get("process_cpu_percent_max", 15.0))
    allow = {str(name) for name in policy.get("allow", [])}
    prefixes = [str(prefix) for prefix in policy.get("system_prefixes", [])]
    violations: list[str] = []
    if load1 > load_limit:
        violations.append(f"load1 {load1:.1f} > {load_limit:.1f}")
    for process in snapshot_value.get("processes", []):
        if not isinstance(process, dict):
            continue
        comm = str(process.get("comm", process.get("name", "")))
        if _is_system(comm, prefixes):
            continue
        name = str(process.get("name") or Path(comm).name)
        try:
            pcpu = float(process.get("pcpu", 0.0))
        except (TypeError, ValueError):
            continue
        if pcpu > cpu_limit and name not in allow:
            violations.append(f"{name} pcpu {pcpu:.1f} > {cpu_limit:.1f} (not in allow list)")
    return {"idle": not violations, "violations": violations}


def fingerprint(snapshot_value: dict[str, Any], policy: dict[str, Any]) -> str:
    """Return a compact stable comparison key for a host snapshot."""
    load1, _, _ = _load_values(snapshot_value)
    load_limit = float(policy.get("load1_max", 3.0))
    cpu_limit = float(policy.get("process_cpu_percent_max", 15.0))
    allow = {str(name) for name in policy.get("allow", [])}
    prefixes = [str(prefix) for prefix in policy.get("system_prefixes", [])]
    names: set[str] = set()
    for process in snapshot_value.get("processes", []):
        if not isinstance(process, dict):
            continue
        comm = str(process.get("comm", process.get("name", "")))
        if _is_system(comm, prefixes):
            continue
        name = str(process.get("name") or Path(comm).name)
        try:
            pcpu = float(process.get("pcpu", 0.0))
        except (TypeError, ValueError):
            continue
        if pcpu > cpu_limit and name not in allow:
            names.add(name)
    load_bucket = "high" if load1 > load_limit else "idle"
    offenders = ",".join(sorted(names)) or "none"
    return f"load1-{load_bucket}|{offenders}"
