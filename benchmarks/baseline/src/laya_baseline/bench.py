"""Pinned, synchronized warm benchmark harness for the Laya Python baseline."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from . import hostload
from .paths import PROFILES, REPO_ROOT, profile_dir

FIXTURE_DIR = REPO_ROOT / "benchmarks" / "fixtures" / "requests"
WORKLOADS: dict[str, tuple[str, ...]] = {
    "published": tuple(f"published-latency-q{n}" for n in (1, 5, 10, 50)),
    "primary": ("distinct-ml-q1", "distinct-ml-q10", "distinct-en-q1", "distinct-en-q10"),
    "distinct": (
        "distinct-ml-q1", "distinct-ml-q10", "distinct-en-q1", "distinct-en-q10",
        "distinct-en-q21", "distinct-en-q50",
    ),
    "extended": (
        "nonlatin-mixed-q6", "ragged-q12", "bucket-choice-20", "bucket-choice-2",
        "edge-long-state", "edge-head-overflow",
    ),
}
WORKLOADS["all"] = tuple(dict.fromkeys(WORKLOADS["published"] + WORKLOADS["distinct"] + WORKLOADS["extended"]))
FALLBACK_MARKERS = ("falling back", "warning: could not place")


def percentile(values: Iterable[float], fraction: float) -> float:
    """Return a NumPy-linear percentile (fraction is in the inclusive 0..1 range)."""
    values = list(values)
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), fraction * 100.0))


def mean(samples: Iterable[float]) -> float:
    return float(np.mean(np.asarray(list(samples), dtype=float)))


def std(samples: Iterable[float]) -> float:
    return float(np.std(np.asarray(list(samples), dtype=float)))


def qps(samples: Iterable[float], n_questions: int = 1) -> float:
    values = np.asarray(list(samples), dtype=float)
    return float(1000.0 * n_questions / np.percentile(values, 50))


def summarize_samples(samples: Iterable[float], n_questions: int) -> dict[str, float | int]:
    values = np.asarray(list(samples), dtype=float)
    if values.size == 0:
        raise ValueError("cannot summarize an empty sample list")
    p50 = float(np.percentile(values, 50))
    p95 = float(np.percentile(values, 95))
    p99 = float(np.percentile(values, 99))
    mean = float(np.mean(values))
    std = float(np.std(values))
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    rps = 1000.0 / p50
    qps = 1000.0 * n_questions / p50
    return {
        "n": int(values.size), "mean_ms": mean, "std_ms": std, "min_ms": minimum, "max_ms": maximum,
        "p50_ms": p50, "p95_ms": p95, "p99_ms": p99,
        "mean": mean, "std": std, "min": minimum, "max": maximum,
        "p50": p50, "p95": p95, "p99": p99, "requests_per_sec": rps, "qps": rps,
        "questions_per_sec": qps, "ms_per_question": p50 / n_questions,
    }


def resolve_workload(spec: str) -> list[str]:
    """Resolve named workload sets and/or comma-separated fixture ids, preserving order."""
    result: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        values = WORKLOADS.get(part, (part,))
        for fixture_id in values:
            if fixture_id not in result:
                result.append(fixture_id)
    if not result:
        raise ValueError("workload must name a set or at least one fixture id")
    available = {p.stem for p in FIXTURE_DIR.glob("*.json")}
    unknown = [item for item in result if item not in available]
    if unknown:
        raise ValueError(f"unknown fixture id(s): {', '.join(unknown)}")
    return result


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    # CPU intentionally has a no-op synchronization path for timing symmetry.


def _thermals() -> str | None:
    try:
        import subprocess
        result = subprocess.run(("pmset", "-g", "therm"), capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout or result.stderr).strip()
    return text or None


def _host_snapshot(policy: dict[str, Any] | None = None, *, phase: str = "start") -> dict[str, Any]:
    """Keep fixture snapshots compact while retaining the idle-host evidence."""
    policy = policy or hostload.load_policy()
    full = hostload.snapshot(policy)
    loadavg = full.get("loadavg", {})
    compact = {
        "time": full.get("timestamp", time.time()),
        "loadavg": loadavg,
        "self_pid": full.get("self_pid"),
        "self_tree": full.get("self_tree", []),
        "pmset_therm": full.get("thermal"),
        "top_processes": full.get("processes", [])[:10],
        "idle_evaluation": hostload.evaluate(full, policy, phase=phase),
        "fingerprint": hostload.fingerprint(full, policy),
    }
    return compact


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fixture(fixture_id: str) -> dict[str, Any]:
    path = FIXTURE_DIR / f"{fixture_id}.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_plan(agent: Any, request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    """Build the exact upstream batch and useful token diagnostics."""
    from laya.common import QTYPES, build_sequence, collate_items, render_options, serialize_state

    max_len = int(agent.cfg.get("max_len", 512))
    head_max_len = int(agent.cfg.get("head_max_len", 192))
    items: list[dict[str, Any]] = []
    lengths: list[int] = []
    option_counts: list[int] = []
    state_rows: list[dict[str, int | bool]] = []
    state_text = serialize_state(request["state"]).replace(agent.tok.mask_token, " ")
    state_full = agent.tok(state_text, add_special_tokens=False)["input_ids"]
    for qid, qdef in request["questions"].items():
        internal = agent._to_internal(qdef)
        sequence, markers = build_sequence(agent.tok, request["state"], internal, max_len, head_max_len)
        expected = len(render_options(internal))
        if len(markers) != expected:
            raise ValueError(f"question {qid!r} options exceed head_max_len={head_max_len}")
        # Recompute only the lengths used by upstream build_sequence, so this diagnostic
        # reports the exact state room without changing the measured preprocessing path.
        head_full = agent.tok(
            "%s question: %s" % (internal["t"], str(internal["ins"]).replace(agent.tok.mask_token, " ")),
            add_special_tokens=False,
        )["input_ids"]
        option_lengths = [1 + len(agent.tok(" " + text.replace(agent.tok.mask_token, " "), add_special_tokens=False)["input_ids"][:48])
                          for text in render_options(internal)]
        option_budget = head_max_len - sum(option_lengths)
        if option_budget < 16:
            per = max(4, (head_max_len - 16) // max(1, len(option_lengths)))
            option_lengths = [min(length, per) for length in option_lengths]
            option_budget = head_max_len - sum(option_lengths)
        head_kept = min(len(head_full), max(8, option_budget))
        prefix_len = 1 + head_kept + 1 + sum(option_lengths) + 1
        room = max(0, max_len - prefix_len - 1)
        kept_state = min(len(state_full), room)
        lengths.append(len(sequence))
        option_counts.append(expected)
        state_rows.append({
            "full_tokens": len(state_full),
            "kept_tokens": kept_state,
            "truncated": kept_state < len(state_full),
        })
        items.append({"ids": sequence, "markers": markers, "qtype": QTYPES[internal["t"]]})
    batch = collate_items([items], agent.tok.pad_token_id)
    if batch is None:
        raise ValueError("fixture contains no questions")
    n_tokens = int(batch["attention_mask"].sum().item())
    diagnostics = {
        "sequence_lengths": lengths,
        "padded_batch_shape": list(batch["input_ids"].shape),
        "n_tokens": n_tokens,
        "option_counts": option_counts,
        "state_tokens": state_rows,
        "truncated": any(row["truncated"] for row in state_rows),
    }
    return batch, {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
                   "marker_pos": batch["marker_pos"], "marker_mask": batch["marker_mask"],
                   "qtype": batch["qtype"]}, diagnostics


def _timed(function: Callable[[], Any], device: str) -> tuple[Any, float, float]:
    timestamp = time.time()
    _sync(device)
    start = time.perf_counter()
    value = function()
    _sync(device)
    return value, (time.perf_counter() - start) * 1000.0, timestamp


def _sample_block(function: Callable[[], Any], warmup: int, reps: int, device: str,
                 after_warmup: Callable[[], None] | None = None) -> tuple[list[float], list[float], list[float]]:
    warm: list[float] = []
    for _ in range(warmup):
        _, elapsed, _ = _timed(function, device)
        warm.append(elapsed)
    if after_warmup is not None:
        after_warmup()
    samples: list[float] = []
    timestamps: list[float] = []
    for _ in range(reps):
        _, elapsed, timestamp = _timed(function, device)
        samples.append(elapsed)
        timestamps.append(timestamp)
    return warm, samples, timestamps


def _timed_model(agent: Any, batch: dict[str, torch.Tensor], device: str) -> tuple[Any, float, float]:
    def invoke() -> Any:
        with torch.no_grad():
            return agent.model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                               batch["marker_mask"], batch["qtype"])
    return _timed(invoke, device)


def _run_fixture(agent: Any, profile: str, device: str, request: dict[str, Any], warmup: int, reps: int,
                 contended: str | None, policy: dict[str, Any]) -> dict[str, Any]:
    fixture_id = request["id"]
    before = _host_snapshot(policy, phase="start")
    wall_start = time.perf_counter()
    batch, cpu_batch, diagnostics = _load_plan(agent, request)
    device_batch = {key: value.to(agent.device) for key, value in cpu_batch.items()}

    def preprocess() -> Any:
        return _load_plan(agent, request)[0]

    preprocess_warm, preprocess_samples, preprocess_timestamps = _sample_block(preprocess, warmup, reps, device)
    def model_call() -> Any:
        with torch.no_grad():
            return agent.model(device_batch["input_ids"], device_batch["attention_mask"], device_batch["marker_pos"],
                               device_batch["marker_mask"], device_batch["qtype"])
    model_warm, model_samples, model_timestamps = _sample_block(model_call, warmup, reps, device)

    # Transfer is deliberately separated from the synchronized model invocation. The output
    # tensors are immediately converted exactly as Agent.system_one does.
    def transfer_call() -> Any:
        logits, act = model_call()
        _sync(device)
        start = time.perf_counter()
        value = (logits.float().cpu(), act.float().cpu())
        _sync(device)
        return value, (time.perf_counter() - start) * 1000.0
    transfer_warm: list[float] = []
    for _ in range(warmup):
        _, elapsed = transfer_call()
        transfer_warm.append(elapsed)
    transfer_samples: list[float] = []
    transfer_timestamps: list[float] = []
    for _ in range(reps):
        timestamp = time.time()
        _, elapsed = transfer_call()
        transfer_samples.append(elapsed)
        transfer_timestamps.append(timestamp)

    def e2e() -> Any:
        return agent.system_one(request["state"], request["questions"])
    warmup_memory: dict[str, int] | None = None
    def remember_warmup_memory() -> None:
        nonlocal warmup_memory
        if device == "mps":
            warmup_memory = {
                "current_allocated_memory": int(torch.mps.current_allocated_memory()),
                "driver_allocated_memory": int(torch.mps.driver_allocated_memory()),
            }
    e2e_warm, e2e_samples, e2e_timestamps = _sample_block(e2e, warmup, reps, device, remember_warmup_memory)
    first = e2e()
    second = e2e()
    identical = first == second
    if not identical:
        raise AssertionError(f"non-deterministic second system_one result for {fixture_id}")

    after = _host_snapshot(policy, phase="end")
    nq = len(request["questions"])
    start_evaluation = before["idle_evaluation"]
    end_evaluation = after["idle_evaluation"]
    midrun_violations = list(start_evaluation["violations"]) + [
        violation for violation in end_evaluation["violations"]
        if violation not in start_evaluation["violations"]
    ]
    fixture_contended = bool(contended or midrun_violations)
    fixture_reason = contended
    if midrun_violations and not fixture_reason:
        fixture_reason = "auto: " + "; ".join(midrun_violations)
    result: dict[str, Any] = {
        "schema_version": 1,
        "profile": profile,
        "device": device,
        "fixture": fixture_id,
        "n_questions": nq,
        "warmup": warmup,
        "reps": reps,
        "contended": fixture_contended,
        "contended_reason": fixture_reason,
        "host_policy": {"policy": policy},
        "idle_evaluation_start": start_evaluation,
        "idle_evaluation_end": end_evaluation,
        "env_fingerprint": before["fingerprint"],
        "diagnostics": diagnostics,
        "n_tokens": diagnostics["n_tokens"],
        "sequence_lengths": diagnostics["sequence_lengths"],
        "padded_batch_shape": diagnostics["padded_batch_shape"],
        "option_counts": diagnostics["option_counts"],
        "truncated": diagnostics["truncated"],
        "results_identical": identical,
        "result": first,
        "timing": {
            "preprocess": {"samples_ms": preprocess_samples, "sample_timestamps": preprocess_timestamps,
                            "sample_contended": [fixture_contended] * reps, "contended_reason": fixture_reason,
                            "warmup_ms": preprocess_warm, "stats": summarize_samples(preprocess_samples, nq)},
            "model": {"samples_ms": model_samples, "sample_timestamps": model_timestamps,
                      "sample_contended": [fixture_contended] * reps, "contended_reason": fixture_reason,
                      "warmup_ms": model_warm, "stats": summarize_samples(model_samples, nq)},
            "transfer": {"samples_ms": transfer_samples, "sample_timestamps": transfer_timestamps,
                         "sample_contended": [fixture_contended] * reps, "contended_reason": fixture_reason,
                         "warmup_ms": transfer_warm, "stats": summarize_samples(transfer_samples, nq),
                         "included_in_model_ms": False},
            "e2e": {"samples_ms": e2e_samples, "sample_timestamps": e2e_timestamps,
                    "sample_contended": [fixture_contended] * reps, "contended_reason": fixture_reason,
                    "warmup_ms": e2e_warm, "stats": summarize_samples(e2e_samples, nq)},
        },
        # Stable aliases make raw sample lists obvious to non-Python consumers.
        "preprocess_ms": preprocess_samples,
        "model_ms": model_samples,
        "transfer_ms": transfer_samples,
        "e2e_ms": e2e_samples,
        "stats": {"preprocess": summarize_samples(preprocess_samples, nq),
                  "model": summarize_samples(model_samples, nq),
                  "transfer": summarize_samples(transfer_samples, nq),
                  "e2e": summarize_samples(e2e_samples, nq)},
        "host_before": before,
        "host_after": after,
        "wall_clock_seconds": time.perf_counter() - wall_start,
        "ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if device == "mps":
        result["mps_memory_after_reps"] = {
            "current_allocated_memory": torch.mps.current_allocated_memory(),
            "driver_allocated_memory": torch.mps.driver_allocated_memory(),
        }
    # Approximate the warmup checkpoint after all warmups have completed. Keep this separate
    # from the after-reps value; MPS APIs report unified-memory bytes, not discrete VRAM.
    if device == "mps":
        result["mps_memory_after_warmup"] = warmup_memory
    return result


class IdleHostError(RuntimeError):
    """Raised before model loading when --require-idle cannot be satisfied."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("idle-host policy violation: " + "; ".join(violations))


def run(profile: str, device: str, workload: str, warmup: int = 3, reps: int = 20,
        run_id: str = "l0", label: str | None = None, contended: str | None = None,
        threads: int | None = None, require_idle: bool = False) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    if device not in ("cpu", "mps"):
        raise ValueError(f"unsupported device: {device}")
    if warmup < 0 or reps <= 0:
        raise ValueError("warmup must be non-negative and reps must be positive")
    fixture_ids = resolve_workload(workload)
    output_dir = REPO_ROOT / "benchmarks" / "results" / run_id
    paths = [output_dir / f"{profile}-{device}-{fixture_id}.json" for fixture_id in fixture_ids]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite immutable benchmark result(s): " + ", ".join(existing))
    policy_path = REPO_ROOT / "benchmarks" / "idle-policy.json"
    policy = hostload.load_policy(policy_path)
    host_start = hostload.snapshot(policy)
    idle_start = hostload.evaluate(host_start, policy, phase="start")
    if require_idle and not idle_start["idle"]:
        raise IdleHostError(idle_start["violations"])
    effective_contended = contended
    if not idle_start["idle"] and effective_contended is None:
        effective_contended = "auto: " + "; ".join(idle_start["violations"])
        print("warning: host is contended; recording this run as contended: " + effective_contended, file=sys.stderr)

    # Keep torch/laya imports after the require-idle gate so a rejected run does no model work.
    global torch
    import torch
    from .env import environment_summary
    if threads is not None:
        if threads <= 0:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
    requested_threads = threads
    run_timer_start = time.perf_counter()
    host_policy = {"path": str(policy_path.relative_to(REPO_ROOT)), "policy": policy}
    run_record: dict[str, Any] = {
        "schema_version": 1, "run_id": run_id, "profile": profile, "device": device,
        "workload": fixture_ids, "warmup": warmup, "reps": reps, "label": label,
        "contended": bool(effective_contended), "contended_reason": effective_contended,
        "requested_threads": requested_threads, "torch_threads": torch.get_num_threads(),
        "dtype": "float32", "autocast": False, "interrupted": False,
        "cli_args": list(sys.argv), "started_at": time.time(), "env": environment_summary(),
        "host_policy": host_policy, "host_start": host_start, "host_end": None,
        "idle_evaluation_start": idle_start, "idle_evaluation_end": None,
        "env_fingerprint": hostload.fingerprint(host_start, policy),
    }
    run_json = output_dir / "run.json"
    previous_run: dict[str, Any] | None = None
    if run_json.exists():
        with run_json.open(encoding="utf-8") as handle:
            previous_run = json.load(handle)
    if previous_run is not None:
        prior_invocations = previous_run.get("invocations", [previous_run])
        run_record["invocations"] = prior_invocations + [dict(run_record)]
    _write_json(run_json, run_record)
    current: dict[str, Any] | None = None
    elapsed = 0.0
    stdout = io.StringIO()
    try:
        from laya import Agent
        with contextlib.redirect_stdout(stdout):
            agent = Agent(str(profile_dir(profile)), device=device)
        captured = stdout.getvalue()
        if any(marker in captured.lower() for marker in FALLBACK_MARKERS):
            raise RuntimeError("Laya emitted a fallback warning before measurement: " + captured.strip())
        if agent.device.type != device:
            raise RuntimeError(f"requested {device}, agent.device is {agent.device.type}")
        if str(agent.dtype) != "torch.float32":
            raise RuntimeError(f"expected float32 on {device}, got {agent.dtype}")
        reference_compile = getattr(getattr(agent.model, "encoder", None), "config", None)
        if getattr(reference_compile, "reference_compile", None) is not False:
            raise RuntimeError("agent.model.encoder.config.reference_compile is not False")
        for index, fixture_id in enumerate(fixture_ids):
            request = _fixture(fixture_id)
            fixture_stdout = io.StringIO()
            with contextlib.redirect_stdout(fixture_stdout):
                current = _run_fixture(agent, profile, device, request, warmup, reps, effective_contended, policy)
            fixture_output = fixture_stdout.getvalue()
            if any(marker in fixture_output.lower() for marker in FALLBACK_MARKERS):
                raise RuntimeError("Laya emitted a fallback warning during measurement: " + fixture_output.strip())
            _write_json(output_dir / f"{profile}-{device}-{fixture_id}.json", current)
            elapsed = time.perf_counter() - run_timer_start
            remaining = len(fixture_ids) - index - 1
            warm_e2e = current["timing"]["e2e"]["warmup_ms"]
            estimate = (sum(warm_e2e) / len(warm_e2e) * reps / 1000.0 * remaining) if warm_e2e else 0.0
            print(f"{fixture_id}: elapsed={elapsed:.1f}s estimated_remaining={estimate:.1f}s", file=sys.stderr)
        run_record["finished_at"] = time.time()
    except KeyboardInterrupt:
        run_record["interrupted"] = True
        if current is not None:
            current["interrupted"] = True
            _write_json(output_dir / f"{profile}-{device}-{current['fixture']}.json", current)
        raise
    finally:
        host_end = hostload.snapshot(policy)
        idle_end = hostload.evaluate(host_end, policy, phase="end")
        run_record["host_end"] = host_end
        run_record["idle_evaluation_end"] = idle_end
        if not idle_end["idle"] and contended is None:
            end_reason = "auto: " + "; ".join(idle_end["violations"])
            run_record["contended"] = True
            run_record["contended_reason"] = (effective_contended + "; " + end_reason.removeprefix("auto: ")) if effective_contended else end_reason
        _write_json(output_dir / "run.json", run_record)
        if stdout.getvalue() if 'stdout' in locals() else False:
            print(stdout.getvalue(), file=sys.stderr, end="")
    return run_record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--run-id", default="l0")
    parser.add_argument("--label")
    parser.add_argument("--contended", metavar="REASON")
    parser.add_argument("--require-idle", action="store_true", help="reject the run unless the start host passes idle-policy.json")
    parser.add_argument("--threads", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        run(**vars(args))
    except IdleHostError as error:
        for violation in error.violations:
            print(f"idle-host violation: {violation}", file=sys.stderr)
        raise SystemExit(3) from error


if __name__ == "__main__":
    main()
