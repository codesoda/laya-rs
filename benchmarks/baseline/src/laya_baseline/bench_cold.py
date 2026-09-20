"""Cold process/startup benchmark for the pinned Laya Agent."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import hostload
from .paths import PROFILES, REPO_ROOT, profile_dir

_CHILD = r'''
import contextlib, io, json, resource, time
started = time.perf_counter()
import torch
from laya import Agent
imports_done = time.perf_counter()
output = io.StringIO()
with contextlib.redirect_stdout(output):
    agent = Agent({profile!r}, device={device!r})
constructed = time.perf_counter()
if agent.device.type != {device!r}:
    raise RuntimeError(f"device fallback: requested {device!r}, got {{agent.device.type}}")
with contextlib.redirect_stdout(output):
    first = agent.system_one({state}, {questions})
first_complete = time.perf_counter()
with contextlib.redirect_stdout(output):
    second = agent.system_one({state}, {questions})
second_complete = time.perf_counter()
if first != second:
    raise RuntimeError("cold benchmark result was not deterministic")
print(json.dumps({{
  "imports_done_ms": (imports_done-started)*1000,
  "agent_constructed_ms": (constructed-started)*1000,
  "first_complete_ms": (first_complete-started)*1000,
  "second_complete_ms": (second_complete-started)*1000,
  "import_stage_ms": (imports_done-started)*1000,
  "load_stage_ms": (constructed-imports_done)*1000,
  "first_call_stage_ms": (first_complete-constructed)*1000,
  "second_call_stage_ms": (second_complete-first_complete)*1000,
  "ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  "actual_device": agent.device.type,
  "runtime_stdout": output.getvalue()
}}))
'''


def _child_command(profile: str, device: str, request: dict[str, Any]) -> list[str]:
    code = _CHILD.format(profile=str(profile_dir(profile)), device=device,
                         state=repr(request["state"]), questions=repr(request["questions"]))
    return [sys.executable, "-c", code]


class IdleHostError(RuntimeError):
    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("idle-host policy violation: " + "; ".join(violations))


def run(profile: str, device: str, n: int, run_id: str, contended: str | None = None,
        require_idle: bool = False) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    if device not in ("cpu", "mps"):
        raise ValueError(f"unknown device: {device}")
    if n <= 0:
        raise ValueError("n must be positive")
    output_path = REPO_ROOT / "benchmarks" / "results" / run_id / f"cold-{profile}-{device}.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable result: {output_path}")
    fixture_path = REPO_ROOT / "benchmarks" / "fixtures" / "requests" / "plan-example.json"
    with fixture_path.open(encoding="utf-8") as handle:
        request = json.load(handle)
    policy_path = REPO_ROOT / "benchmarks" / "idle-policy.json"
    policy = hostload.load_policy(policy_path)
    host_start = hostload.snapshot(policy)
    idle_start = hostload.evaluate(host_start, policy)
    if require_idle and not idle_start["idle"]:
        raise IdleHostError(idle_start["violations"])
    effective_contended = contended
    if not idle_start["idle"] and effective_contended is None:
        effective_contended = "auto: " + "; ".join(idle_start["violations"])
        print("warning: host is contended; recording this cold run as contended: " + effective_contended, file=sys.stderr)
    iterations: list[dict[str, Any]] = []
    for index in range(n):
        started = time.perf_counter()
        process = subprocess.run(_child_command(profile, device, request), capture_output=True, text=True, check=False)
        wall_ms = (time.perf_counter() - started) * 1000.0
        if process.returncode != 0:
            raise RuntimeError(
                f"cold iteration {index + 1} failed ({process.returncode}):\n{process.stderr}\n{process.stdout}"
            )
        try:
            child = json.loads(process.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"cold child emitted invalid JSON: {process.stdout!r}") from error
        child["iteration"] = index + 1
        child["wall_process_ms"] = wall_ms
        child["page_cache"] = "cold/unknown on first iteration" if index == 0 else "warm (after first iteration)"
        iterations.append(child)
    host_end = hostload.snapshot(policy)
    idle_end = hostload.evaluate(host_end, policy)
    record = {
        "schema_version": 1,
        "profile": profile,
        "device": device,
        "run_id": run_id,
        "n": n,
        "fixture": "plan-example",
        "contended": bool(effective_contended or not idle_end["idle"]),
        "contended_reason": effective_contended or (("auto: " + "; ".join(idle_end["violations"])) if not idle_end["idle"] else None),
        "host_policy": {"path": str(policy_path.relative_to(REPO_ROOT)), "policy": policy},
        "host_start": host_start,
        "host_end": host_end,
        "idle_evaluation_start": idle_start,
        "idle_evaluation_end": idle_end,
        "env_fingerprint": hostload.fingerprint(host_start, policy),
        "page_cache_note": "The first iteration is not classified; subsequent iterations are noted as warm. The OS page cache was not purged.",
        "iterations": iterations,
        "created_at": time.time(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    partial.write_text(json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(output_path)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--contended", metavar="REASON")
    parser.add_argument("--require-idle", action="store_true")
    args = parser.parse_args()
    try:
        record = run(**vars(args))
    except IdleHostError as error:
        for violation in error.violations:
            print(f"idle-host violation: {violation}", file=sys.stderr)
        raise SystemExit(3) from error
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
