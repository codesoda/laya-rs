"""Render immutable Laya benchmark result JSON as a concise Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import hostload
from .paths import REPO_ROOT

PUBLISHED_T4: dict[tuple[str, str], tuple[float, float]] = {
    ("english", "published-latency-q1"): (39.5, 44.8),
    ("english", "published-latency-q5"): (84.5, 86.0),
    ("english", "published-latency-q10"): (158.6, 160.1),
    ("english", "published-latency-q50"): (771.3, 806.7),
    ("multilingual", "published-latency-q1"): (32.8, 38.8),
    ("multilingual", "published-latency-q5"): (40.1, 43.5),
    ("multilingual", "published-latency-q10"): (72.3, 74.4),
    ("multilingual", "published-latency-q50"): (337.4, 693.7),
}


def _samples(record: dict[str, Any], kind: str) -> list[float]:
    direct = record.get(f"{kind}_ms")
    if isinstance(direct, list):
        return [float(value) for value in direct]
    return [float(value) for value in record.get("timing", {}).get(kind, {}).get("samples_ms", [])]


def _p(record: dict[str, Any], kind: str, quantile: float) -> float:
    values = _samples(record, kind)
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), quantile))


def _fmt(value: float | None) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:.1f}"


def _start_snapshot(record: dict[str, Any]) -> dict[str, Any] | None:
    fixture_snapshot = record.get("host_before")
    if isinstance(fixture_snapshot, dict):
        return fixture_snapshot
    run_snapshot = record.get("host_start")
    return run_snapshot if isinstance(run_snapshot, dict) else None


def _recorded_contended(record: dict[str, Any]) -> bool:
    value = record.get("contended", False)
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "none")
    return bool(value)


def reevaluate_contention(record: dict[str, Any], policy: dict[str, Any] | None = None) -> bool:
    """Re-evaluate a stored start snapshot with today's policy without editing it."""
    snapshot = _start_snapshot(record)
    if snapshot is None:
        return _recorded_contended(record)
    current_policy = policy or hostload.load_policy()
    evaluation_snapshot = snapshot
    if "processes" not in snapshot and "top_processes" in snapshot:
        # Fixture snapshots use a compact top_processes key; evaluate the same data
        # without changing the immutable record.
        evaluation_snapshot = dict(snapshot)
        evaluation_snapshot["processes"] = snapshot["top_processes"]
    return not hostload.evaluate(evaluation_snapshot, current_policy, phase="start")["idle"]


def _load1_start(record: dict[str, Any]) -> float | None:
    snapshot = _start_snapshot(record)
    if snapshot is None:
        return None
    load1, _, _ = hostload._load_values(snapshot)
    return load1


def _env_fingerprint(record: dict[str, Any]) -> str:
    value = record.get("env_fingerprint")
    if isinstance(value, str):
        return value
    policy_record = record.get("host_policy")
    start = record.get("host_start")
    if isinstance(policy_record, dict) and isinstance(start, dict):
        policy = policy_record.get("policy", policy_record)
        if isinstance(policy, dict):
            return hostload.fingerprint(start, policy)
    return "—"


def table_rows(records: Iterable[dict[str, Any]], policy: dict[str, Any] | None = None,
               compare_published: bool = False) -> list[dict[str, Any]]:
    """Build the derived report rows; source result JSON is never modified."""
    records = list(records)
    current_policy = policy or hostload.load_policy()
    rows: list[dict[str, Any]] = []
    for record in records:
        e2e_p50 = _p(record, "e2e", 50)
        n_questions = int(record.get("n_questions", 1))
        row: dict[str, Any] = {
            "profile": str(record.get("profile", "")),
            "device": str(record.get("device", "")),
            "fixture": str(record.get("fixture", "")),
            "nq": record.get("n_questions", record.get("nq", "")),
            "n_tokens": record.get("diagnostics", {}).get("n_tokens", record.get("n_tokens", "")),
            "e2e_p50_ms": e2e_p50,
            "e2e_p95_ms": _p(record, "e2e", 95),
            "model_p50_ms": _p(record, "model", 50),
            "preprocess_p50_ms": _p(record, "preprocess", 50),
            "ms_per_question": e2e_p50 / n_questions,
            "contended_recorded": _recorded_contended(record),
            "contended_reevaluated": reevaluate_contention(record, current_policy),
            "load1_start": _load1_start(record),
            "env": _env_fingerprint(record),
        }
        if compare_published:
            published = PUBLISHED_T4.get((str(record.get("profile")), str(record.get("fixture"))))
            row["published_t4_p50_ms"] = published[0] if published else None
            row["published_t4_p95_ms"] = published[1] if published else None
        rows.append(row)
    return rows


def render_markdown(records: Iterable[dict[str, Any]], compare_published: bool = False) -> str:
    rows = table_rows(records, compare_published=compare_published)
    headers = ["profile", "device", "fixture", "nq", "n_tokens", "e2e p50 ms", "e2e p95 ms",
               "model p50 ms", "preprocess p50 ms", "ms/question", "contended (recorded)",
               "contended (re-evaluated)", "load1 start", "env"]
    if compare_published:
        headers.extend(["published T4 p50 ms", "published T4 p95 ms"])
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        values = [
            row["profile"], row["device"], row["fixture"], str(row["nq"]), str(row["n_tokens"]),
            _fmt(row["e2e_p50_ms"]), _fmt(row["e2e_p95_ms"]), _fmt(row["model_p50_ms"]),
            _fmt(row["preprocess_p50_ms"]), _fmt(row["ms_per_question"]),
            "yes" if row["contended_recorded"] else "no",
            "yes" if row["contended_reevaluated"] else "no", _fmt(row["load1_start"]), row["env"],
        ]
        if compare_published:
            values.extend([_fmt(row["published_t4_p50_ms"]), _fmt(row["published_t4_p95_ms"])])
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    if compare_published:
        lines.extend(["", "*Published T4 values are different hardware — not comparable as a speedup.*"])
    return "\n".join(lines) + "\n"


def load_records(run_ids: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_id in run_ids:
        directory = REPO_ROOT / "benchmarks" / "results" / run_id
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.glob("*.json")):
            if path.name == "run.json" or path.name.startswith("cold-"):
                continue
            with path.open(encoding="utf-8") as handle:
                record = json.load(handle)
            if "fixture" in record:
                records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--compare-published", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit derived table rows as JSON")
    args = parser.parse_args()
    records = load_records(args.run_id)
    if args.json:
        print(json.dumps(table_rows(records, compare_published=args.compare_published), ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(render_markdown(records, args.compare_published), end="")


if __name__ == "__main__":
    main()
