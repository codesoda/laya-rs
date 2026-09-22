# L1 Candle CPU/Metal feasibility spike

Status: **complete — correctness PASS on CPU and native Metal (fp32), performance NOT viable on Metal as-is.**
Implementation by Sol; reviewed, re-run and written up by Fable. Raw evidence: `benchmarks/results/l1-spike/` (immutable; timestamped files are full 28-fixture runs, unsuffixed files are early 1-fixture smoke runs).

## What was built

`spikes/candle-parity` — a Python-free binary that loads the upstream `model.safetensors` directly (strict name/shape inventory, `inspect`), runs the complete Laya network (ModernBERT/mmBERT encoder, type embedding, two pre-norm ReLU head layers, marker scorer, `-1e4` masking, entropy/top-2/`k/255` features, action head) on Candle CPU or `Device::new_metal(0)`, and compares to the frozen goldens (`parity`), plus a synchronized `timing` sanity subcommand.

Pinned: `candle-core = 0.11.0` (`metal` feature), `candle-nn = 0.11.0`. The ModernBERT module is adapted from candle-transformers at commit `1bbda281093b29a62ab2331b239894778b3aefb2` (Apache-2.0/MIT) rather than depended on, because its `Config` expects flat `global_rope_theta`/`local_rope_theta` while the checkpoints use `rope_parameters.{full,sliding}_attention.rope_theta` and `layer_types`; the adapter (`src/config.rs`) asserts `layer_types` agrees with the every-3-layers rule. Verified against transformers 5.17 `modeling_modernbert.py`: local window is *inclusive* at `local_attention/2` = 64 (`|i-j| > 64` masked; unit-tested), padded keys masked with `f32::MIN`, `norm_bias: false` (bias-free LayerNorm), erf-GELU (`gelu_erf`) in the gated MLP, `sans_pos` is a no-op. Metal residency is checked on every output tensor; `inference_ops_run_on_cpu` is empty in every report.

## Parity (28 fixtures, 203 questions per profile; `edge-single-option` skipped as upstream error)

| profile | device | dtype | result | option_logits_raw max / mean | act_logits max / mean | probs max | hidden_cls max (diagnostic) | argmax | load ms | report |
|---|---|---|---|---|---|---|---|---|---|---|
| english | cpu | f32 | FAIL | 1.65e-04 / 4.11e-06 | 3.32e-02 / 1.62e-03 | 1.36e-05 | 1.12e-02 | 203/203 | 735 | `english-cpu-f32-1790026986.json` |
| english | cpu | f32 | PASS | 1.65e-04 / 4.11e-06 | 3.32e-02 / 1.62e-03 | 1.36e-05 | 1.12e-02 | 203/203 | 298 | `english-cpu-f32-1790029243.json` |
| english | cpu | f32 | PASS | 1.65e-04 / 4.11e-06 | 3.32e-02 / 1.62e-03 | 1.36e-05 | 1.12e-02 | 203/203 | 270 | `english-cpu-f32-1790030438.json` |
| english | metal | f16 | FAIL | 1.26e-01 / 4.09e-03 | 2.67e+01 / 1.76e+00 | 1.99e-02 | 9.17e+00 | 203/203 | 116 | `english-metal-f16-1790030129.json` |
| english | metal | f32 | PASS | 1.03e-04 / 3.66e-06 | 4.52e-02 / 1.65e-03 | 8.52e-06 | 1.40e-02 | 203/203 | 283 | `english-metal-f32-1790027101.json` |
| english | metal | f32 | PASS | 1.03e-04 / 3.66e-06 | 4.52e-02 / 1.65e-03 | 8.52e-06 | 1.40e-02 | 203/203 | 225 | `english-metal-f32-1790029333.json` |
| english | metal | f32 | PASS | 1.03e-04 / 3.66e-06 | 4.52e-02 / 1.65e-03 | 8.52e-06 | 1.40e-02 | 203/203 | 162 | `english-metal-f32-1790030505.json` |
| english | metal | f32 | PASS | 1.03e-04 / 3.66e-06 | 4.52e-02 / 1.65e-03 | 8.52e-06 | 1.40e-02 | 203/203 | 449 | `english-metal-f32-1790032433.json` |
| multilingual | cpu | f32 | PASS | 5.25e-05 / 3.06e-06 | 3.08e-02 / 8.04e-04 | 5.78e-06 | 7.87e-03 | 203/203 | 199 | `multilingual-cpu-f32-1790029465.json` |
| multilingual | cpu | f32 | PASS | 5.25e-05 / 3.06e-06 | 3.08e-02 / 8.04e-04 | 5.78e-06 | 7.87e-03 | 203/203 | 201 | `multilingual-cpu-f32-1790030581.json` |
| multilingual | metal | f32 | PASS | 6.72e-05 / 3.43e-06 | 1.06e-02 / 4.95e-04 | 9.18e-06 | 7.44e-03 | 203/203 | 188 | `multilingual-metal-f32-1790029515.json` |
| multilingual | metal | f32 | PASS | 6.72e-05 / 3.43e-06 | 1.06e-02 / 4.95e-04 | 9.18e-06 | 7.44e-03 | 203/203 | 118 | `multilingual-metal-f32-1790030621.json` |
| typed-decisions | cpu | f32 | PASS | 1.10e-05 / 1.73e-06 | 4.39e-03 / 9.45e-04 | 1.85e-06 | 2.32e-03 | 203/203 | 302 | `typed-decisions-cpu-f32-1790029768.json` |
| typed-decisions | cpu | f32 | PASS | 1.10e-05 / 1.73e-06 | 4.39e-03 / 9.45e-04 | 1.85e-06 | 2.32e-03 | 203/203 | 251 | `typed-decisions-cpu-f32-1790030784.json` |
| typed-decisions | metal | f32 | PASS | 1.32e-05 / 1.90e-06 | 5.37e-03 / 1.05e-03 | 2.24e-06 | 1.95e-03 | 203/203 | 158 | `typed-decisions-metal-f32-1790029844.json` |
| typed-decisions | metal | f32 | PASS | 1.32e-05 / 1.90e-06 | 5.37e-03 / 1.05e-03 | 2.24e-06 | 1.95e-03 | 203/203 | 160 | `typed-decisions-metal-f32-1790030859.json` |

Gates: `rust_cpu_fp32` option_logits_raw max 2e-4 / mean 2e-5, act_logits max 5e-2 / mean 5e-3, probs 2e-5; `rust_metal_fp32` 4e-4 / 4e-5, 5e-2 / 5e-3, 4e-5. All fp32 cells pass with 100% argmax agreement and no near-tie disagreements. The first `english-cpu-f32-1790026986` row is marked FAIL only because that build applied the `act_logits.mean_abs` limit per fixture (2 values); see the 2026-09-22 clarification in `benchmarks/goldens/tolerances.json` — numbers are identical to the later PASS rows. Fable re-ran `english metal f32` independently (`…-1790032433.json`): identical aggregate numbers.

Notes: english option logits on CPU (1.65e-4) sit close to the 2e-4 gate; english `hidden_cls` (1.1–1.4e-2) exceeds the 5e-3 diagnostic target on both devices while all gated outputs pass — same pattern as the mlx-rs spike, so it is a property of accumulating 28 layers in a different op order, not a bug. `f16` (english, full run) FAILS the `rust_metal_fp16_or_bf16` profile (option_logits_raw 1.26e-1 > 5e-2; probs 2e-2 > 5e-3); `bf16` fails on the single smoke fixture. Candle fp16 accumulates in fp16 in these ops; mlx-rs fp16 passed the same profile. Not pursued further.

## Timing sanity — spike, not acceptance (Paseo running; uncontrolled host)

Model-only, warm mean, multilingual:

| fixture | Python MPS fp32 (L0 baseline, model p50) | Candle Metal fp32 | Candle CPU fp32 | Python CPU fp32 (L0) |
|---|---:|---:|---:|---:|
| distinct-ml-q1 | 32.5 ms | **123 ms** | 371 ms | 91.3 ms |
| distinct-ml-q10 | 292.4 ms | **1616 ms** | — | 764 ms |

Candle 0.11 Metal is ~4–5.5× *slower* than PyTorch MPS on this network out of the box (unfused attention/softmax/matmul kernels, per-op dispatch, no flash-attention on Metal), and Candle CPU is ~4× slower than PyTorch CPU. Load from safetensors to model-ready is 120–300 ms (vs 22–25 s for Python `Agent()`), peak RSS 1.6–3.5 GB.

## Recommendation (plan §11 items 1, 3)

- **Correctness/operator coverage: GO.** The complete network is reproducible in Rust within the frozen fp32 tolerances on CPU and native Metal for all three profiles, and Metal fails loudly rather than falling back.
- **Candle as the macOS acceleration backend: NO-GO for the 1.10× target** without substantial custom Metal kernel work; the gap is 4–5× against the primary baseline. mlx-rs (see `docs/L1-MLX-SPIKE.md`) is at rough parity with MPS in the same uncontrolled sanity runs and is the credible Mac path.
- **Candle remains the portable CPU (and later CUDA) backend candidate**, with the caveat that its CPU speed is ~4× behind PyTorch CPU here; CPU performance was never a gated target but should be tracked.

## Open issues

1. Candle fp16/bf16 fail the fp16 tolerance profile; if a Candle half-precision configuration is ever wanted, accumulation dtype needs investigation.
2. `candle-core` 0.11.0 exposes no Metal allocation counter; peak memory on Metal is reported via RSS only.
3. `english-cpu-f32` option_logits_raw max (1.65e-4) has little headroom under the 2e-4 gate; any future op change on CPU must be re-verified against this row.
