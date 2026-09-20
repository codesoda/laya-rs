"""Generate exact preprocessing, model, and native-output Laya oracle records."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .env import LAYA_GIT_SHA
from .paths import PROFILES, REPO_ROOT, manifest_path, profile_dir

FIXTURE_DIR = REPO_ROOT / "benchmarks" / "fixtures" / "requests"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _version(distribution: str) -> str:
    return metadata.version(distribution)


def _json_equal(a: Any, b: Any) -> bool:
    return json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)


def _tensor_list(tensor: torch.Tensor) -> list[Any]:
    return tensor.detach().cpu().tolist()


def _exception_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _load_fixture_paths(fixture_id: str | None) -> list[Path]:
    paths = sorted(FIXTURE_DIR.glob("*.json"))
    if fixture_id is None:
        return paths
    matches = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            if json.load(handle).get("id") == fixture_id:
                matches.append(path)
    if not matches:
        raise ValueError(f"unknown fixture id: {fixture_id}")
    return matches


def _meta(agent: Any, profile: str, device: str, fixture_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    tok = agent.tok
    profile_manifest = manifest["profiles"][profile]
    return {
        "fixture_id": fixture_id,
        "profile": profile,
        "device": device,
        "dtype": str(agent.dtype).removeprefix("torch."),
        "versions": {
            "torch": torch.__version__,
            "transformers": _version("transformers"),
            "tokenizers": _version("tokenizers"),
        },
        "laya_sha": LAYA_GIT_SHA,
        "weights_sha256": profile_manifest["files"]["model.safetensors"]["sha256"],
        "max_len": int(agent.cfg.get("max_len", 512)),
        "head_max_len": int(agent.cfg.get("head_max_len", 192)),
        "tokenizer_special": {
            "cls": {"id": tok.cls_token_id, "string": tok.cls_token},
            "sep": {"id": tok.sep_token_id, "string": tok.sep_token},
            "mask": {"id": tok.mask_token_id, "string": tok.mask_token},
            "pad": {"id": tok.pad_token_id, "string": tok.pad_token},
        },
        "timestamp": _utc_now(),
    }


def _trace_build_sequence(
    tok: Any,
    state: Any,
    internal: dict[str, Any],
    max_len: int,
    head_max_len: int,
) -> tuple[list[int], list[int], dict[str, Any], dict[str, Any]]:
    """Call upstream build_sequence and capture its return-frame locals.

    Full tokenizer outputs are obtained by calling the same tokenizer on the exact strings
    passed by upstream. Kept values come from upstream's own frame rather than a local
    reimplementation of its truncation algorithm.
    """
    from laya.common import build_sequence, render_options, serialize_state

    captured: dict[str, Any] = {}
    target_code = build_sequence.__code__
    previous_trace = sys.gettrace()

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        if frame.f_code is target_code and event == "return":
            captured.update(frame.f_locals)
        return tracer

    sys.settrace(tracer)
    try:
        ids, markers = build_sequence(tok, state, internal, max_len, head_max_len)
    finally:
        sys.settrace(previous_trace)
    if not captured:
        raise RuntimeError("failed to capture upstream build_sequence locals")

    mask_tok = tok.mask_token
    rendered = render_options(internal)
    ins_with_mask_replaced = str(internal["ins"]).replace(mask_tok, " ")
    head_text = "%s question: %s" % (internal["t"], ins_with_mask_replaced)
    option_texts = [" " + option.replace(mask_tok, " ") for option in rendered]
    head_ids_full = tok(head_text, add_special_tokens=False)["input_ids"]
    option_ids_full = [tok(text, add_special_tokens=False)["input_ids"] for text in option_texts]
    state_text = serialize_state(state).replace(mask_tok, " ")
    state_ids_full = tok(state_text, add_special_tokens=False)["input_ids"]

    initial_option_lengths = [1 + min(48, len(option_ids)) for option_ids in option_ids_full]
    opt_budget_initial = head_max_len - sum(initial_option_lengths)
    retruncated = "per" in captured
    sequence_before_slice = list(captured["ids"])
    token_record: dict[str, Any] = {
        "head_ids_full": list(head_ids_full),
        "head_ids_kept": list(captured["head_ids"]),
        "option_ids_full": [list(value) for value in option_ids_full],
        "option_ids_kept": [list(value) for value in captured["opt_ids"]],
        "opt_budget_initial": opt_budget_initial,
        "retruncated": retruncated,
        "per": int(captured["per"]) if retruncated else None,
        "state_ids_full_len": len(state_ids_full),
        "state_ids_kept_len": len(captured["st"]),
        "room": int(captured["room"]),
        "ids": list(ids),
        "markers": list(markers),
        "truncated_state_tokens": len(state_ids_full) - len(captured["st"]),
        "sequence_len": len(ids),
        "hit_max_len": len(sequence_before_slice) == max_len,
    }
    serialized = {
        "internal": internal,
        "rendered_options": rendered,
        "head_text": head_text,
        "option_texts": option_texts,
        "state_text_ref": "$.state_text",
    }
    return list(ids), list(markers), serialized, token_record


def _prepare(agent: Any, fixture: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, list[dict[str, Any]], bool]:
    from laya.common import QTYPES, collate_items, render_options, serialize_state

    max_len = int(agent.cfg.get("max_len", 512))
    head_max_len = int(agent.cfg.get("head_max_len", 192))
    state_text = serialize_state(fixture["state"]).replace(agent.tok.mask_token, " ")
    serialized: dict[str, Any] = {}
    tokens: dict[str, Any] = {}
    items: list[dict[str, Any]] = []
    options_exceeded = False

    for qid, qdef in fixture["questions"].items():
        internal = agent._to_internal(qdef)
        ids, markers, serialized_row, token_row = _trace_build_sequence(
            agent.tok, fixture["state"], internal, max_len, head_max_len
        )
        serialized[qid] = serialized_row
        tokens[qid] = token_row
        if len(markers) != len(render_options(internal)):
            options_exceeded = True
        items.append({"ids": ids, "markers": markers, "qtype": QTYPES[internal["t"]]})

    batch = None if options_exceeded else collate_items([items], agent.tok.pad_token_id)
    batch_record = None
    if batch is not None:
        batch_record = {
            "input_ids": _tensor_list(batch["input_ids"]),
            "attention_mask": _tensor_list(batch["attention_mask"]),
            "marker_pos": _tensor_list(batch["marker_pos"]),
            "marker_mask": _tensor_list(batch["marker_mask"]),
            "qtype": _tensor_list(batch["qtype"]),
            "pad_id": agent.tok.pad_token_id,
            "n_tokens": int(batch["attention_mask"].sum()),
        }
    return {"state_text": state_text, "serialized": serialized, "tokens": tokens}, batch_record, batch, items, options_exceeded


def _forward(
    agent: Any, batch: dict[str, torch.Tensor]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Call the real model while hooks capture exact hidden vectors consumed by heads."""
    captured: dict[str, torch.Tensor] = {}

    def scorer_pre_hook(_module: Any, args: tuple[torch.Tensor, ...]) -> None:
        captured["markers"] = args[0].detach().float().cpu().clone()

    def act_pre_hook(_module: Any, args: tuple[torch.Tensor, ...]) -> None:
        d = int(agent.model.encoder.config.hidden_size)
        captured["cls"] = args[0][:, :d].detach().float().cpu().clone()

    scorer_hook = agent.model.scorer.register_forward_pre_hook(scorer_pre_hook)
    act_hook = agent.model.act_head.register_forward_pre_hook(act_pre_hook)
    try:
        with torch.no_grad():
            logits, act_logits = agent.model(
                batch["input_ids"].to(agent.device),
                batch["attention_mask"].to(agent.device),
                batch["marker_pos"].to(agent.device),
                batch["marker_mask"].to(agent.device),
                batch["qtype"].to(agent.device),
            )
            # Match Agent.system_one: softmax action logits while still on the active device.
            act_probs = torch.softmax(act_logits.float(), -1)
            logits_np = logits.float().cpu().numpy().copy()
            act_logits_np = act_logits.float().cpu().numpy().copy()
            act_probs_np = act_probs.cpu().numpy().copy()
    finally:
        scorer_hook.remove()
        act_hook.remove()
    return logits_np, act_logits_np, act_probs_np, np.array(captured["cls"]), np.array(captured["markers"])


def _postprocess(
    agent: Any,
    fixture: dict[str, Any],
    items: list[dict[str, Any]],
    logits: np.ndarray,
    act_logits: np.ndarray,
    act_probs: np.ndarray,
    n_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Repeat only upstream's output assembly, invoking its calibration helpers directly."""
    from laya.common import QTYPES, confidence_from_probs, temp_bucket

    answers: dict[str, Any] = {}
    model_rows: dict[str, Any] = {}
    for row, (qid, qdef) in enumerate(fixture["questions"].items()):
        internal = agent._to_internal(qdef)
        k = len(items[row]["markers"])
        qtype = QTYPES[internal["t"]]
        bucket = temp_bucket(qtype, k)
        temperature = agent.temperature_by_options.get(bucket, agent.temperature[qtype])
        z = logits[row, :k] / max(1e-3, float(temperature))
        probabilities = np.exp(z - z.max())
        probabilities = probabilities / probabilities.sum()
        confidence = confidence_from_probs(probabilities, k)
        expected_score = float((np.arange(k) * probabilities).sum()) if internal["t"] == "score" else None
        argmax = int(probabilities.argmax())
        extension = {"act_probability": round(float(act_probs[row, 0]), 4)}

        if internal["t"] == "choice":
            keys = list(internal["crit"].keys())
            answer = {
                "type": "choice",
                "choice": keys[argmax],
                "probabilities": {key: round(float(value), 4) for key, value in zip(keys, probabilities)},
                "confidence": round(confidence, 4),
                "action": extension,
            }
        elif internal["t"] == "score":
            answer = {
                "type": "score",
                "score": round(float(expected_score), 4),
                "legend": {str(index): criterion for index, criterion in enumerate(internal["crit"])},
                "probabilities": {str(index): round(float(value), 4) for index, value in enumerate(probabilities)},
                "confidence": round(confidence, 4),
                "action": extension,
            }
        else:
            answer = {
                "type": "noul",
                "noul": round(float(probabilities[1]), 4),
                "confidence": round(max(float(probabilities[1]), 1.0 - float(probabilities[1])), 4),
                "action": extension,
            }
        answers[qid] = answer
        model_rows[qid] = {
            "option_logits_raw": [float(value) for value in logits[row, :k]],
            "option_logits_masked_full": [float(value) for value in logits[row]],
            "act_logits": [float(value) for value in act_logits[row]],
            "act_probs": [float(value) for value in act_probs[row]],
            "temperature": float(temperature),
            "temp_bucket": bucket,
            "probs_unrounded": [float(value) for value in probabilities],
            "confidence_unrounded": float(confidence),
            "expected_score_unrounded": expected_score,
            "argmax_index": argmax,
        }

    result = {
        "model": "laya-rl-agent",
        "answers": answers,
        "usage": {"input_tokens": n_tokens, "output_tokens": 0},
    }
    return result, model_rows


def _call_native(agent: Any, fixture: dict[str, Any]) -> tuple[dict[str, Any] | None, BaseException | None]:
    try:
        return agent.system_one(fixture["state"], fixture["questions"]), None
    except Exception as error:  # oracle exceptions are part of the golden contract
        return None, error


def generate_one(
    agent: Any, profile: str, device: str, fixture: dict[str, Any], manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    fixture_id = fixture["id"]
    record: dict[str, Any] = {
        "meta": _meta(agent, profile, device, fixture_id, manifest),
        "source": fixture.get("source"),
        "expect_error": fixture.get("expect_error"),
    }
    prepared, batch_record, batch, items, options_exceeded = _prepare(agent, fixture)
    record.update(prepared)
    record["batch"] = batch_record

    native_result, native_error = _call_native(agent, fixture)
    if native_error is not None:
        # A second real upstream call establishes that the captured error is reproducible.
        _, instrumented_error = _call_native(agent, fixture)
        record.update(
            {
                "model": None,
                "native_result": None,
                "instrumented_matches_native": (
                    instrumented_error is not None
                    and _exception_record(instrumented_error) == _exception_record(native_error)
                ),
                "error": _exception_record(native_error),
                "determinism": None,
            }
        )
        return record, None

    if options_exceeded or batch is None:
        raise RuntimeError("upstream accepted a fixture whose prepared option markers were incomplete")

    first = _forward(agent, batch)
    second = _forward(agent, batch)
    logits, act_logits, act_probs, hidden_cls, hidden_markers = first
    second_logits, second_act_logits = second[0], second[1]
    option_equal = np.array_equal(logits, second_logits)
    action_equal = np.array_equal(act_logits, second_act_logits)
    differences = np.concatenate(
        [np.abs(logits.astype(np.float64) - second_logits.astype(np.float64)).ravel(),
         np.abs(act_logits.astype(np.float64) - second_act_logits.astype(np.float64)).ravel()]
    )
    instrumented_result, model_rows = _postprocess(
        agent,
        fixture,
        items,
        logits,
        act_logits,
        act_probs,
        int(batch["attention_mask"].sum()),
    )
    matches = _json_equal(instrumented_result, native_result)
    record.update(
        {
            "model": model_rows,
            "native_result": native_result,
            "instrumented_matches_native": matches,
            "error": None,
            "determinism": {
                "device": device,
                "bitwise_identical": bool(option_equal and action_equal),
                "max_abs_diff": float(differences.max(initial=0.0)),
            },
        }
    )
    if not matches:
        raise RuntimeError(f"instrumented result differs from native result for {fixture_id}")

    marker_mask = batch["marker_mask"].detach().cpu().numpy().astype(np.bool_, copy=False)
    hidden_states = {
        "hidden_cls": np.asarray(hidden_cls, dtype=np.float32),
        "hidden_markers": np.where(
            marker_mask[..., np.newaxis], np.asarray(hidden_markers, dtype=np.float32), 0.0
        ).astype(np.float32, copy=False),
        "hidden_markers_mask": marker_mask,
    }
    record.update(
        {
            "hidden_states_npz": f"{fixture_id}.npz",
            "hidden_cls_shape": list(hidden_states["hidden_cls"].shape),
            "hidden_markers_shape": list(hidden_states["hidden_markers"].shape),
        }
    )
    return record, hidden_states


def _write_json(path: Path, value: dict[str, Any], hidden_states: dict[str, np.ndarray] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hidden_states is not None:
        sidecar = path.with_name(value["hidden_states_npz"])
        sidecar_partial = sidecar.with_suffix(sidecar.suffix + ".partial")
        with sidecar_partial.open("wb") as handle:
            np.savez_compressed(handle, **hidden_states)
        sidecar_partial.replace(sidecar)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(encoded)
    partial.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=(*PROFILES, "all"), required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--fixture", help="generate only this fixture id")
    parser.add_argument("--out", default="benchmarks/goldens", help="output root, relative to repository root")
    args = parser.parse_args()

    output_root = Path(args.out)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    fixture_paths = _load_fixture_paths(args.fixture)
    profiles = PROFILES if args.profile == "all" else (args.profile,)
    with manifest_path().open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    from laya import Agent

    failures: list[str] = []
    for profile in profiles:
        print(f"[{profile}/{args.device}] loading {profile_dir(profile)}", file=sys.stderr, flush=True)
        load_started = time.perf_counter()
        agent = Agent(str(profile_dir(profile)), device=args.device)
        if agent.device.type != args.device:
            raise RuntimeError(f"device fallback: requested {args.device}, agent.device is {agent.device.type}")
        if agent.dtype != torch.float32:
            raise RuntimeError(f"expected float32 on {args.device}, got {agent.dtype}")
        print(
            f"[{profile}/{args.device}] loaded in {time.perf_counter() - load_started:.2f}s; "
            f"{len(fixture_paths)} fixtures",
            file=sys.stderr,
            flush=True,
        )
        try:
            for index, fixture_path in enumerate(fixture_paths, 1):
                with fixture_path.open(encoding="utf-8") as handle:
                    fixture = json.load(handle)
                started = time.perf_counter()
                try:
                    record, hidden_states = generate_one(agent, profile, args.device, fixture, manifest)
                    destination = output_root / profile / args.device / f"{fixture['id']}.json"
                    _write_json(destination, record, hidden_states)
                except Exception as error:
                    failures.append(f"{profile}/{args.device}/{fixture['id']}: {type(error).__name__}: {error}")
                    print(f"ERROR {failures[-1]}", file=sys.stderr, flush=True)
                    continue
                elapsed = time.perf_counter() - started
                status = "error-record" if record["error"] else "ok"
                print(
                    f"[{profile}/{args.device}] {index}/{len(fixture_paths)} {fixture['id']}: "
                    f"{status} ({elapsed:.2f}s)",
                    file=sys.stderr,
                    flush=True,
                )
                if elapsed > 60:
                    print(
                        f"WARNING fixture exceeded 60s: {profile}/{args.device}/{fixture['id']} "
                        f"took {elapsed:.2f}s",
                        file=sys.stderr,
                        flush=True,
                    )
        finally:
            del agent
            gc.collect()
            if args.device == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()

    if failures:
        print("golden generation failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
