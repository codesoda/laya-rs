"""Run a two-call upstream Laya smoke test on one explicit device."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch

from .env import environment_summary
from .fetch import record_tokenizer_patch
from .paths import PROFILES, REPO_ROOT, profile_dir

REQUEST = {
    "state": {"body": "I was charged twice. Please refund."},
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Payments and refunds", "support": "Other support"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["low", "medium", "high"],
        },
        "refund": {"type": "noul", "instructions": "Is a refund requested?"},
    },
}
FALLBACK_MARKERS = ("falling back", "warning: could not place")


def _synchronize(device: str) -> None:
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _timed(device: str, function: Callable[[], Any]) -> tuple[Any, float]:
    _synchronize(device)
    start = time.perf_counter()
    value = function()
    _synchronize(device)
    return value, time.perf_counter() - start


def _result_path(profile: str, device: str) -> Path:
    return REPO_ROOT / "benchmarks" / "results" / "l0-smoke" / f"{profile}-{device}.json"


def _write(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(profile: str, requested_device: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    from laya import Agent

    output = io.StringIO()
    record: dict[str, Any] = {
        "profile": profile,
        "requested_device": requested_device,
        "env": environment_summary(),
        "load_seconds": None,
        "first_call_seconds": None,
        "second_call_seconds": None,
        "result": None,
        "fallback_detected": False,
        "error": None,
        "traceback": None,
        "runtime_stdout": "",
        "reference_compile": None,
    }
    response: dict[str, Any] | None = None
    try:
        with contextlib.redirect_stdout(output):
            start = time.perf_counter()
            agent = Agent(str(profile_dir(profile)), device=requested_device)
            _synchronize(requested_device)
            record["load_seconds"] = time.perf_counter() - start
            record_tokenizer_patch(profile)

            actual_device = agent.device.type
            record["actual_device"] = actual_device
            compile_setting = getattr(getattr(agent.model, "encoder", None), "config", None)
            record["reference_compile"] = getattr(compile_setting, "reference_compile", None)
            if actual_device != requested_device:
                raise RuntimeError(f"device fallback: requested {requested_device}, agent.device is {actual_device}")
            if record["reference_compile"] is not False:
                raise RuntimeError(
                    f"reference_compile was not disabled (value={record['reference_compile']!r}); refusing smoke run"
                )
            response, first = _timed(
                requested_device,
                lambda: agent.system_one(REQUEST["state"], REQUEST["questions"]),
            )
            record["first_call_seconds"] = first
            second_response, second = _timed(
                requested_device,
                lambda: agent.system_one(REQUEST["state"], REQUEST["questions"]),
            )
            record["second_call_seconds"] = second
            if second_response != response:
                raise RuntimeError("first and second smoke results differ")
            record["result"] = response
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()
    finally:
        captured = output.getvalue()
        record["runtime_stdout"] = captured
        record["fallback_detected"] = any(marker in captured.lower() for marker in FALLBACK_MARKERS)
        if captured:
            print(captured, end="" if captured.endswith("\n") else "\n", file=sys.stderr)
        if record["fallback_detected"] and record["error"] is None:
            record["error"] = "RuntimeError: fallback output detected"
            record["traceback"] = "Fallback marker was emitted by upstream laya."
    return record, response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    args = parser.parse_args()
    path = _result_path(args.profile, args.device)
    record, response = run(args.profile, args.device)
    _write(path, record)
    print(f"wrote {path}", file=sys.stderr)
    if record["error"] is not None:
        print(record["error"], file=sys.stderr)
        raise SystemExit(1)
    json.dump(response, sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
