# L1 tokenizer/preprocessing parity spike

The standalone project in `spikes/tokenizer-parity/` reproduces the preprocessing boundary in
`laya/common.py` and `Agent._to_internal` without touching the root workspace.

## Crate decision

The Python golden metadata reports `tokenizers==0.23.2`. The matching stable Rust release is
also the `tokenizers` crate `0.23.2` (the upstream repository publishes the Rust crate and Python
wrapper on the same `v0.23.x` release line). The spike pins `=0.23.2`, rather than the newer
`1.0.0-rc.2`, and uses `Tokenizer::encode(..., false)` for the upstream
`add_special_tokens=False` path. `serde_json` enables `preserve_order` and `float_roundtrip`.
The latter is important: it preserves correctly rounded parsed f64 values before the custom
Python spelling is applied.

## Result

`cargo run --bin parity` compares all 29 CPU goldens for each profile, including both error-edge
fixtures (which still contain preprocessing tensors):

```
english: 29/29 fixtures exact
multilingual: 29/29 fixtures exact
typed-decisions: 29/29 fixtures exact
total: 87/87 fixtures exact
```

The comparison covers serialized state, every serialized question field, all token metadata and
IDs/markers, and every batch tensor (`input_ids`, `attention_mask`, `marker_pos`, `marker_mask`,
`qtype`, padding ID, and token count). Mismatch output reports the first differing path/index and
neighboring token ID/piece.

The Rust unit tests also compare 2,000 deterministic Python-generated float cases, including
signed zero, integer-valued floats, subnormals, boundary exponents, and wide magnitudes. The
fixture is `tests/fixtures/python-floats.json`; regenerate it with the script under
`spikes/tokenizer-parity/scripts/` from the locked baseline venv.

## Quirks that mattered

- State JSON uses Python's default `", "`/`": "` spacing, insertion order, UTF-8 text, and
  Python float exponent thresholds/spelling. Non-string instructions intentionally use ASCII
  escaping, including surrogate pairs; criterion rendering uses UTF-8 JSON and the same spacing.
- Literal replacement is profile-token-specific (`[MASK]` or `<mask>`) before encoding. It also
  applies to state, head text, and option text. Tokenizer normalizers/byte-BPE behavior is left
  entirely to the serialized `tokenizer.json`; no Unicode NFC or cleanup postprocessing is added.
- Per-option text is capped at 48 tokens, the shared budget may re-truncate to `per`, the head
  retains `max(8, opt_budget)`, state is right-truncated, and final IDs/markers are filtered after
  `[:max_len]`. Collation pads IDs on the right and leaves padded marker positions at zero.
- `[CLS]`/`[SEP]`/`[MASK]`/`[PAD]` and `<bos>`/`<eos>`/`<mask>`/`<pad>` are obtained from each
  tokenizer vocabulary; no special tokens are added by encoding.

## L2 handoff

`serialize_state`, `render_criterion`, `to_internal`, `render_options`, `build_sequence`, and
`collate` in `src/lib.rs` are small, dependency-light candidates for lifting into `laya-core`
(with the `Tokenizer` ownership moved to the runtime crate). Keep `serde_json` insertion-order
handling, `float_roundtrip`, the ASCII/non-ASCII serializer distinction, and the golden parity
harness as permanent regression tests. Model inference, error-envelope policy, and compatibility
adaptation are intentionally outside this spike.

Verification from the spike directory:

```
cargo fmt --check
cargo clippy -- -D warnings
cargo test
cargo run --bin parity
```
