# Python Laya goldens

These files are generated from the pinned upstream Laya runtime and are the preprocessing and inference oracle for the Rust port. Do not hand-edit them.

## Regenerate

From `baseline/`:

```sh
~/.local/bin/uv sync
~/.local/bin/uv run laya-goldens --profile all --device cpu
~/.local/bin/uv run laya-goldens --profile all --device mps
~/.local/bin/uv run laya-compare-goldens --a benchmarks/goldens/english/cpu --b benchmarks/goldens/english/mps
~/.local/bin/uv run laya-compare-goldens --a benchmarks/goldens/multilingual/cpu --b benchmarks/goldens/multilingual/mps
~/.local/bin/uv run laya-compare-goldens --a benchmarks/goldens/typed-decisions/cpu --b benchmarks/goldens/typed-decisions/mps
~/.local/bin/uv run pytest tests/test_goldens.py
```

Relative `--out`, `--a`, and `--b` paths are resolved from the repository root, not the shell working directory. `--fixture ID` limits generation to one fixture. Each invocation loads a profile once and evaluates every fixture discovered at run time under `benchmarks/fixtures/requests/`.

## Layout and format

`<profile>/<device>/<fixture-id>.json` contains:

- `meta`: fixture/profile/device identity; actual FP32 dtype; package and pinned Laya/weight revisions; model budgets; tokenizer special token IDs and strings; generation timestamp.
- `state_text`: the once-serialized state with literal mask tokens replaced.
- `serialized`: one insertion-ordered row per question with the exact upstream `_to_internal` value, rendered options, head text, option texts, and a `state_text_ref`.
- `tokens`: full and retained head/option token IDs, initial option budget, re-truncation metadata, state lengths and room, final IDs/markers, truncation count, sequence length, and max-length status. `option_ids_full` excludes the marker and precedes the 48-token cut; `option_ids_kept` includes the leading mask marker.
- `batch`: the five tensors passed to the model (`input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, `qtype`) as nested JSON lists, plus padding ID and summed attention tokens.
- `model`: raw/full option logits, action logits/probabilities, selected temperature/bucket, unrounded probabilities/confidence/expected Score, and argmax.
- `hidden_states_npz`, `hidden_cls_shape`, and `hidden_markers_shape`: the sibling sidecar name and declared hidden-state array shapes.
- `native_result`: the independent result of the real `Agent.system_one` call. `instrumented_matches_native` is true only when the instrumented reconstruction has the same JSON serialization.
- `determinism`: bitwise equality and maximum absolute raw-logit difference between two model forwards on the selected device. CPU records are the required CPU determinism evidence.
- `error`: upstream exception type and verbatim message for failing edge fixtures; `native_result` is null. Error reproducibility is checked with a second real upstream call.

JSON uses Python's default full float representation; only `native_result` contains upstream's intentional four-decimal rounding. Every successful record has a compressed, non-pickled `<fixture-id>.npz` sidecar. It contains float32 `hidden_cls` (`[n,d]`), float32 `hidden_markers` (`[n,kmax,d]`, zero-padded), and boolean `hidden_markers_mask` (`[n,kmax]`). Batch tensors and logits remain in JSON.

`<profile>/mps-vs-cpu.json` records exact token/tensor equality, aggregate max/mean/RMSE deltas for option and action logits, probability/action/Score deltas, argmax agreement, and every rounded native-answer field difference.
