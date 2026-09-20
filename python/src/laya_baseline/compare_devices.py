"""Compare two directories of generated Laya golden records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _resolve(path: str) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    from .paths import REPO_ROOT

    return REPO_ROOT / value


def _load_directory(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for candidate in sorted(path.glob("*.json")):
        with candidate.open(encoding="utf-8") as handle:
            record = json.load(handle)
        fixture_id = record.get("meta", {}).get("fixture_id")
        if fixture_id:
            records[fixture_id] = record
    return records


def _metric(differences: list[float]) -> dict[str, float | int | None]:
    if not differences:
        return {"count": 0, "max_abs_diff": None, "mean_abs_diff": None, "rmse": None}
    values = np.asarray(differences, dtype=np.float64)
    return {
        "count": int(values.size),
        "max_abs_diff": float(values.max()),
        "mean_abs_diff": float(values.mean()),
        "rmse": float(np.sqrt(np.mean(values * values))),
    }


def _flatten_differences(a: list[float], b: list[float], destination: list[float]) -> None:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"numeric shape mismatch: {left.shape} != {right.shape}")
    destination.extend(np.abs(left - right).ravel().tolist())


def _leaf_differences(a: Any, b: Any, prefix: str = "") -> list[tuple[str, Any, Any]]:
    if isinstance(a, dict) and isinstance(b, dict):
        differences: list[tuple[str, Any, Any]] = []
        for key in dict.fromkeys((*a.keys(), *b.keys())):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in a or key not in b:
                differences.append((path, a.get(key), b.get(key)))
            else:
                differences.extend(_leaf_differences(a[key], b[key], path))
        return differences
    if isinstance(a, list) and isinstance(b, list):
        differences = []
        for index in range(max(len(a), len(b))):
            path = f"{prefix}[{index}]"
            if index >= len(a) or index >= len(b):
                differences.append((path, a[index] if index < len(a) else None, b[index] if index < len(b) else None))
            else:
                differences.extend(_leaf_differences(a[index], b[index], path))
        return differences
    return [] if a == b else [(prefix, a, b)]


def compare(a_dir: Path, b_dir: Path) -> dict[str, Any]:
    a_records = _load_directory(a_dir)
    b_records = _load_directory(b_dir)
    a_ids, b_ids = set(a_records), set(b_records)
    if a_ids != b_ids:
        raise ValueError(
            f"fixture sets differ: only in {a_dir}={sorted(a_ids - b_ids)}, only in {b_dir}={sorted(b_ids - a_ids)}"
        )

    exact_names = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    exact: dict[str, dict[str, Any]] = {
        name: {"exact": True, "mismatched_fixtures": []} for name in exact_names
    }
    exact["question_ids"] = {"exact": True, "mismatched_fixtures": []}
    exact["question_markers"] = {"exact": True, "mismatched_fixtures": []}
    option_differences: list[float] = []
    action_differences: list[float] = []
    probability_differences: list[float] = []
    expected_score_differences: list[float] = []
    action_probability_differences: list[float] = []
    argmax_matches = 0
    argmax_total = 0
    rounded_differences: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []
    compared_successes = 0

    for fixture_id in sorted(a_ids):
        a_record, b_record = a_records[fixture_id], b_records[fixture_id]
        if a_record.get("error") or b_record.get("error"):
            error_records.append(
                {
                    "fixture": fixture_id,
                    "a": a_record.get("error"),
                    "b": b_record.get("error"),
                    "equal": a_record.get("error") == b_record.get("error"),
                }
            )
            continue
        compared_successes += 1
        a_batch, b_batch = a_record["batch"], b_record["batch"]
        for name in exact_names:
            if a_batch[name] != b_batch[name]:
                exact[name]["exact"] = False
                exact[name]["mismatched_fixtures"].append(fixture_id)
        for qid in a_record["tokens"]:
            if a_record["tokens"][qid]["ids"] != b_record["tokens"][qid]["ids"]:
                exact["question_ids"]["exact"] = False
                exact["question_ids"]["mismatched_fixtures"].append(f"{fixture_id}/{qid}")
            if a_record["tokens"][qid]["markers"] != b_record["tokens"][qid]["markers"]:
                exact["question_markers"]["exact"] = False
                exact["question_markers"]["mismatched_fixtures"].append(f"{fixture_id}/{qid}")

        for qid in a_record["model"]:
            a_model, b_model = a_record["model"][qid], b_record["model"][qid]
            _flatten_differences(a_model["option_logits_raw"], b_model["option_logits_raw"], option_differences)
            _flatten_differences(a_model["act_logits"], b_model["act_logits"], action_differences)
            _flatten_differences(a_model["probs_unrounded"], b_model["probs_unrounded"], probability_differences)
            _flatten_differences(a_model["act_probs"], b_model["act_probs"], action_probability_differences)
            argmax_total += 1
            argmax_matches += int(a_model["argmax_index"] == b_model["argmax_index"])
            a_score = a_model["expected_score_unrounded"]
            b_score = b_model["expected_score_unrounded"]
            if a_score is not None and b_score is not None:
                expected_score_differences.append(abs(float(a_score) - float(b_score)))

        a_answers = a_record["native_result"]["answers"]
        b_answers = b_record["native_result"]["answers"]
        for qid in dict.fromkeys((*a_answers.keys(), *b_answers.keys())):
            for field, a_value, b_value in _leaf_differences(a_answers.get(qid), b_answers.get(qid)):
                rounded_differences.append(
                    {"fixture": fixture_id, "qid": qid, "field": field, "a": a_value, "b": b_value}
                )

    profile_values = {record["meta"]["profile"] for record in (*a_records.values(), *b_records.values())}
    result = {
        "profile": next(iter(profile_values)) if len(profile_values) == 1 else sorted(profile_values),
        "a": str(a_dir),
        "b": str(b_dir),
        "fixture_count": len(a_ids),
        "compared_success_count": compared_successes,
        "error_records": error_records,
        "exact": exact,
        "metrics": {
            "option_logits_raw": _metric(option_differences),
            "act_logits": _metric(action_differences),
            "probs_unrounded": _metric(probability_differences),
            "expected_score": _metric(expected_score_differences),
            "act_probs": _metric(action_probability_differences),
            "argmax": {
                "matches": argmax_matches,
                "count": argmax_total,
                "agreement_rate": (argmax_matches / argmax_total) if argmax_total else None,
            },
        },
        "native_result_rounded_differences": {
            "count": len(rounded_differences),
            "items": rounded_differences,
        },
    }
    return result


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def _print_summary(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    print("metric                 max_abs       mean_abs      RMSE", file=sys.stderr)
    for name in ("option_logits_raw", "act_logits"):
        row = metrics[name]
        print(
            f"{name:<22} {_fmt(row['max_abs_diff']):>12} {_fmt(row['mean_abs_diff']):>13} {_fmt(row['rmse']):>12}",
            file=sys.stderr,
        )
    print(f"probs max abs         {_fmt(metrics['probs_unrounded']['max_abs_diff'])}", file=sys.stderr)
    print(f"expected score max    {_fmt(metrics['expected_score']['max_abs_diff'])}", file=sys.stderr)
    print(f"act_probs max abs     {_fmt(metrics['act_probs']['max_abs_diff'])}", file=sys.stderr)
    print(
        f"argmax agreement      {metrics['argmax']['matches']}/{metrics['argmax']['count']} "
        f"({_fmt(metrics['argmax']['agreement_rate'])})",
        file=sys.stderr,
    )
    print(
        f"rounded field diffs   {result['native_result_rounded_differences']['count']}",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    args = parser.parse_args()
    a_dir, b_dir = _resolve(args.a), _resolve(args.b)
    result = compare(a_dir, b_dir)
    destination = b_dir.parent / f"{b_dir.name}-vs-{a_dir.name}.json"
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    _print_summary(result)
    print(f"wrote {destination}", file=sys.stderr)


if __name__ == "__main__":
    main()
