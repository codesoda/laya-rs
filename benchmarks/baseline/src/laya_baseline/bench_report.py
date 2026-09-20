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


def _fmt(value: float) -> str:
    return "—" if not np.isfinite(value) else f"{value:.1f}"


def _midrun_contention(record: dict[str, Any]) -> bool:
    start = record.get("idle_evaluation_start")
    end = record.get("idle_evaluation_end")
    return bool(isinstance(start, dict) and start.get("idle") is True
                and isinstance(end, dict) and end.get("idle") is False)


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


def render_markdown(records: Iterable[dict[str, Any]], compare_published: bool = False) -> str:
    records = list(records)
    headers = ["profile", "device", "fixture", "nq", "n_tokens", "e2e p50 ms", "e2e p95 ms",
               "model p50 ms", "preprocess p50 ms", "ms/question", "contended", "env"]
    if compare_published:
        headers.extend(["published T4 p50 ms", "published T4 p95 ms"])
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for record in records:
        e2e_p50 = _p(record, "e2e", 50)
        row = [
            str(record.get("profile", "")), str(record.get("device", "")), str(record.get("fixture", "")),
            str(record.get("n_questions", record.get("nq", ""))),
            str(record.get("diagnostics", {}).get("n_tokens", record.get("n_tokens", ""))),
            _fmt(e2e_p50), _fmt(_p(record, "e2e", 95)), _fmt(_p(record, "model", 50)),
            _fmt(_p(record, "preprocess", 50)), _fmt(e2e_p50 / int(record.get("n_questions", 1))),
            "contended(mid-run)" if _midrun_contention(record) else ("yes" if record.get("contended") else "no"),
            _env_fingerprint(record),
        ]
        if compare_published:
            published = PUBLISHED_T4.get((str(record.get("profile")), str(record.get("fixture"))))
            row.extend([_fmt(published[0]) if published else "—", _fmt(published[1]) if published else "—"])
        lines.append("| " + " | ".join(row) + " |")
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
    args = parser.parse_args()
    print(render_markdown(load_records(args.run_id), args.compare_published), end="")


if __name__ == "__main__":
    main()
