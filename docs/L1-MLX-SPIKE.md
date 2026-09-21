# L1 MLX feasibility spike

## Decision summary

**Full-network Metal feasibility: PASS.** On the M3 Pro, upstream `mlx-rs` 0.32.0 executes the complete Laya network (ModernBERT/mmBERT encoder, question-type embedding, both pre-norm ReLU transformer head layers, scorer, masking, and action head) on the MLX Metal GPU. All 28 successful frozen fixtures, 203 questions per checkpoint, pass the frozen `rust_metal_fp32` and `rust_metal_fp16_or_bf16` profiles for English, multilingual, and typed-decisions. Every run has 203/203 argmax agreement and finite outputs. The one upstream-error fixture, `edge-single-option`, is intentionally excluded by the spike contract.

**Distribution: platform archive PASS; single executable FAIL.** The Rust/C++ runtime is statically linked and `otool -L` shows only macOS system libraries, but the 174 MiB `mlx.metallib` is external, not embedded. A copied binary fails a real Metal operation when the build-cache metallib is hidden. The binary plus `mlx.metallib` beside it passes `device` and full `plan-example` parity from `/tmp`, with no Python, Cargo, Xcode, or Homebrew libraries at runtime. Model/tokenizer files remain separate product assets.

**Recommendation:** MLX is a defensible native Metal backend candidate and clears the operator/parity gate. Compare its performance and package size with the parallel Candle result before selecting the product backend. If MLX is selected, ship a two-file signed platform archive (or add a reviewed embed-and-extract mechanism), set a reproducible macOS 14 deployment target, and validate on clean macOS 14 and other Apple Silicon generations before claiming that release family.

## Scope, host, and references

- Implementation: `spikes/mlx-parity/`, a standalone Cargo workspace with its own lockfile.
- Raw evidence: `benchmarks/results/l1-spike-mlx/`.
- Host: Apple M3 Pro, 14 GPU cores, 18 GiB unified memory, macOS 26.2 (25C56), Metal 4; only this machine is validated.
- Rust: 1.95.0 aarch64-apple-darwin. Xcode 26.2 (17C52). CMake 3.28.2.
- Oracle: pinned upstream `common.py::DecisionModel.forward` and frozen Python CPU goldens.
- Independent architecture reference: `laya-mlx` commit `fc1df62828a3fedf4d8229fdac1cbd85f1cdf337` (Apache-2.0), especially `laya_mlx/model.py`. It was used as a readable architecture map, not as a weight source or runtime dependency.

## Implementation and architecture

`mlx-parity` loads each original upstream `model.safetensors` directly. An inspection through the locked baseline environment found 206 tensors for English and typed-decisions and 170 for multilingual. Representative shapes were:

| Tensor | English / typed | Multilingual |
| --- | ---: | ---: |
| `encoder.embeddings.tok_embeddings.weight` | 50368 × 1024 | 256000 × 768 |
| `encoder.layers.0.attn.Wqkv.weight` | 3072 × 1024 | 2304 × 768 |
| `type_emb.weight` | 3 × 1024 | 3 × 768 |
| `head.layers.0.self_attn.in_proj_weight` | 3072 × 1024 | 2304 × 768 |
| `head.layers.0.linear1.weight` | 4096 × 1024 | 3072 × 768 |
| `act_head.0.weight` | 256 × 1028 | 256 × 772 |
| `act_head.2.weight` | 2 × 256 | 2 × 256 |
| `temperature` | 3 | 3 |

The loader checks every expected name and shape and rejects leftovers. The implementation reproduces:

- token embedding and bias-free LayerNorm;
- 28-layer English/typed or 22-layer multilingual ModernBERT, fused QKV, non-traditional RoPE, per-layer local/global theta, inclusive local radius 64, padded-key masks, gated GELU MLP, and final norm;
- three-way type embedding added to every final encoder token;
- two full-sequence pre-norm transformer layers with `d/64` heads, biased projections/norms, default PyTorch **ReLU**, and padding masks;
- marker gather, scorer, `-1e4` padded-option masking, and FP32 logits;
- uncalibrated option softmax, `k=max(marker_count,2)`, top-two margin, normalized entropy, `k/255`, first-token pooling, and the full action head.

All output arrays are explicitly evaluated before comparison or timing. There is no dropout path in this inference-only implementation, which is equivalent to forcing the upstream model to `eval()`. `device` refuses a non-GPU default and evaluates a real Metal addition so device discovery cannot hide a missing metallib.

### Deliberate deviations from `laya-mlx`

- Rust structs and `mlx-rs` operations replace Python MLX modules; no Python code is invoked.
- Original upstream safetensors are loaded directly; `laya-mlx` converted checkpoints are not used.
- This bounded spike consumes the already-frozen `batch.*` tensors rather than implementing tokenization or public request preprocessing.
- Calibration/probability/confidence calculations used for the report are host-side comparison logic; the complete learned network, including action features and action head, runs through MLX.
- No compilation, quantization, pruning, deduplication, cache, or custom kernel is enabled.

## Parity results

Each row aggregates 28 successful fixtures and 203 questions against `benchmarks/goldens/<profile>/cpu`. Reports include per-fixture and aggregate count/max/mean/RMSE for masked logits, raw logits, action logits, action probabilities, calibrated probabilities, confidence, expected score, and `hidden_cls`.

### FP32 — `rust_metal_fp32`

| Profile | Result | Raw logits max / mean | Action logits max / mean | Prob max | Action prob max | `hidden_cls` max | Argmax |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| English | PASS | 1.1635e-4 / 6.3878e-6 | 2.3193e-2 / 2.3858e-3 | 2.4229e-5 | 0 | 7.8125e-3 | 203/203 |
| Multilingual | PASS | 1.0395e-4 / 5.4208e-6 | 1.1475e-2 / 8.0849e-4 | 7.1526e-6 | 0 | 5.9586e-3 | 203/203 |
| Typed-decisions | PASS | 1.5914e-5 / 2.2631e-6 | 1.0254e-2 / 1.9700e-3 | 6.9141e-6 | 0 | 3.7842e-3 | 203/203 |

The English acceptance sequence (`plan-example`, `distinct-en-q10`, `ragged-q12`, `nonlatin-mixed-q6`, `edge-long-state`, `edge-head-overflow`, and `bucket-choice-20`) passed its per-fixture max-error and argmax gates before the complete English sweep. Mean-error limits are applied to the frozen complete-profile aggregate, not to an arbitrarily selected single fixture.

`hidden_cls` is explicitly diagnostic-only in the frozen tolerances. English and multilingual exceed its 5e-3 diagnostic target while all gated final outputs pass; this localizes the residual numerical difference to accumulated encoder/head Metal arithmetic rather than omitted learned layers.

### FP16 — `rust_metal_fp16_or_bf16`

| Profile | Result | Raw logits max | Probability max | Expected-score max | Action prob max | Argmax |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| English | PASS | 2.0501e-2 | 2.6966e-3 | 1.3634e-3 | 0 | 203/203 |
| Multilingual | PASS | 2.7618e-2 | 1.6124e-3 | 6.6002e-4 | 0 | 203/203 |
| Typed-decisions | PASS | 2.4529e-3 | 1.0059e-3 | 5.5698e-4 | 0 | 203/203 |

The FP16 profile does not gate raw action logits. They differ by as much as 5.3674 (English), 3.4867 (multilingual), and 1.0537 (typed), while action probabilities are exactly equal on these saturated outputs and all declared FP16 gates pass. This should remain visible if future checkpoints produce nonsaturated action probabilities.

Raw reports:

- `benchmarks/results/l1-spike-mlx/english-f32.json`
- `benchmarks/results/l1-spike-mlx/english-f16.json`
- `benchmarks/results/l1-spike-mlx/multilingual-f32.json`
- `benchmarks/results/l1-spike-mlx/multilingual-f16.json`
- `benchmarks/results/l1-spike-mlx/typed-decisions-f32.json`
- `benchmarks/results/l1-spike-mlx/typed-decisions-f16.json`

## Timing sanity (not acceptance)

`timing` retains one loaded model, creates a fresh lazy graph per call, evaluates all returned fields before stopping the clock, separates the first call, and reports MLX active/peak memory. A ten-repetition `plan-example` sanity pass is in `timing-sanity.txt`. It ran while other development work was active and showed substantial variability, so it is explicitly **not acceptance evidence**.

| Profile | Dtype | First call ms | Warm p50 ms | Warm mean ms | Peak MLX bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| English | f32 | 429.141 | 171.378 | 162.116 | 2,764,817,335 |
| English | f16 | 715.969 | 317.361 | 270.147 | 2,530,011,111 |
| Multilingual | f32 | 183.819 | 117.758 | 117.803 | 2,111,635,727 |
| Multilingual | f16 | 208.398 | 133.135 | 131.244 | 1,306,487,075 |
| Typed-decisions | f32 | 290.663 | 126.635 | 126.849 | 2,430,374,321 |
| Typed-decisions | f16 | 127.500 | 177.149 | 145.748 | 2,246,820,369 |

Earlier invocations in the same session were materially faster, confirming that this short spike is sensitive to system contention/thermal state. Do not compare this table with L0 Python timings or infer that FP16 is intrinsically slower. A later acceptance benchmark must use the frozen manifest, idle host, alternating order, adequate samples, and complete request timing.

## Build and runtime facts

- crates.io `mlx-rs = 0.32.0` resolves to `mlx-sys = 0.6.0` in the standalone lockfile. The upstream `oxideai`/`oxiglade` crate built successfully; the `pmetal-mlx-rs` fork was not needed.
- `mlx-sys` says its native tuple is mlx-c `c74db5307cc8...` and MLX 0.32.2 (`mlx-sys-0.6.0/README.md:4-6`). Its vendored mlx-c CMake uses `FetchContent` to clone/build MLX tag `v0.32.2`, not download a prebuilt runtime (`src/mlx-c/CMakeLists.txt:25-39`).
- Build prerequisites observed: Rust/Cargo, Git/network for the first MLX source fetch, CMake (MLX itself requires 3.25; 3.28.2 was used), bindgen/libclang, `/usr/bin/cc` and `/usr/bin/c++`, Xcode/CLT, `xcrun`, and a macOS SDK. `mlx-sys/build.rs:20-45,115-123` explicitly invokes Xcode discovery and selects those compilers.
- `mlx`, `mlxc`, and `gguflib` are statically linked (`mlx-sys/build.rs:146-158`); Metal, Accelerate, Foundation, Objective-C, libc++, iconv, and libSystem are system dependencies. No MLX dylib is shipped.
- MLX 0.32.2 source rejects deployment targets below macOS 14 (`target/.../_deps/mlx-src/CMakeLists.txt:206-216`). This host build did not set `CMAKE_OSX_DEPLOYMENT_TARGET`; MLX therefore selected the host 26.2 target. The metallib contains `air64_v28-apple-macosx26.2.0`. Although the final Mach-O load command says 11.0, that does **not** make this host-built metallib a macOS 11 artifact. The supported source minimum is macOS 14; this exact artifact is validated only on 26.2.

## Packaging findings

`mlx-sys/build.rs:97-106` chooses `MLX_RS_METAL_PATH` or `~/.mlx/lib/<source-key>`. Lines 133-138 pass that directory as CMake's `MLX_METAL_PATH`; lines 164-172 merely check that `mlx.metallib` exists. It is not linked as bytes. The compiled executable contains the absolute build-machine cache path, and `mlx-rs/src/metal.rs:45-65` exposes the process-global runtime override used here to prefer a sibling `mlx.metallib`.

Measured release artifacts:

- `mlx-parity`: 16,015,472 bytes (about 15.3 MiB), SHA-256 `69a9f7b5...ad84b5e` for the packaging-tested build.
- `mlx.metallib`: 182,351,120 bytes (about 173.9 MiB), SHA-256 `fec9fad1...193a`.
- `otool -L`: only `/usr/lib` libraries and Apple Foundation, Metal, and Accelerate frameworks; no Homebrew path, Python, or MLX dylib.

Test sequence:

1. Copy only the release binary to `/tmp/laya-mlx-packaging`, temporarily hide the build-cache metallib, and run `device`. It exits 1 on the first real Metal operation: expected proof that kernels are not embedded.
2. Copy `mlx.metallib` beside the binary. Startup code calls `mlx_rs::metal::set_metallib_path` before initializing Metal.
3. Run `device` and English `plan-example` FP32 parity with `DYLD_PRINT_LIBRARIES=1`. Both exit 0; parity passes. No non-system package-manager dylib appears.

Evidence: `packaging.json`, `packaging-binary-only.txt`, `packaging-device-dyld.txt`, `packaging-parity-dyld.txt`, `packaging-otool.txt`, and `packaging-sizes.txt` in the raw-results directory.

**End-user requirement beyond macOS:** Apple Silicon, a compatible Metal GPU/OS, the platform-matched `mlx-parity` executable, the exact colocated `mlx.metallib`, and separately downloaded model/tokenizer assets. Python, Cargo, CMake, Xcode/CLT, and Homebrew are build-time only. A single-file executable is not currently achieved. A product release must also set `MACOSX_DEPLOYMENT_TARGET=14.0`, put each deployment-target-specific metallib in an immutable archive, and test that archive on the oldest claimed OS. The current mlx-sys cache key does not include deployment target, so release builds should use an isolated explicit `MLX_RS_METAL_PATH` to avoid cross-target metallib reuse.

## What `laya-mlx` already tried and measured

Do not repeat these as unmeasured L4 ideas:

- On an M3 Max, its committed eager MLX FP16 end-to-end medians were 13.421/71.068/336.030 ms for English short Q=1/10/50 and 7.390/27.386/127.565 ms for multilingual; FP32 was sometimes faster than Torch MPS but not uniformly (see `PERFORMANCE_RESEARCH.md`).
- Whole-model compilation plus exact final-head output pruning produced mostly modest paired gains, roughly 3–8%; English Q=50 and multilingual long-batch intervals included no improvement. First English compiled shape cost about 2.17 s and new shapes retraced.
- An exact custom Metal GELU×gate kernel matched 27,958,016 FP16 elements but gave no consistent whole-model gain over MLX compilation.
- Naive encoder-only 8-bit/4-bit quantization reduced storage but did not accelerate larger pilots and caused prediction/probability drift (English 8-bit already changed one fixture decision; multilingual 4-bit changed 7/26 distinct decisions).
- Exact local-window attention plus final-head pruning is mathematically bounded to about 2.8–16.5% modeled-FLOP removal on the studied shapes. Sampled low-rank decompositions at a nominal 10× matrix-work budget had about 82–87% best relative Frobenius error. A universal same-checkpoint 10× from a few kernels is not credible.
- The shipped Snake opt-in combination of compilation, 16-token buckets, and bounded tokenized-prefix caching measured 1.065× over 2,400 moves; one of four seeds was slower. It does not cache predictions or contextual hidden states.
- Exact whole-input deduplication can help repeated workloads but must be labeled; a 50-distinct-question benchmark is required. A much smaller distilled/shared-state model is a separate model and quality program.

Sources reviewed: `/tmp/laya-mlx/docs/{PERFORMANCE_RESEARCH,MATH_10X_RESEARCH,ENGINEERING_10X_RESEARCH,SNAKE_OPTIMIZATION}.md` at commit `fc1df628...`.

## Verification and open issues

Passing commands inside `spikes/mlx-parity/`:

```text
cargo fmt --check
cargo clippy -- -D warnings
cargo test
cargo build --release
```

Open issues before a product backend decision:

1. Compare uncontended end-to-end MLX results against the frozen Python MPS manifest and the parallel Candle spike; this task establishes correctness and runtime feasibility, not speed acceptance.
2. Decide whether the ~174 MiB external metallib/two-file archive is acceptable. A single executable requires an upstream embedding facility or a reviewed embedded-bytes extraction design.
3. Produce and test a clean macOS 14-targeted archive. The current host-target metallib cannot establish older-OS compatibility despite the executable's misleading 11.0 Mach-O minimum.
4. Validate M1/M2/M4 and memory pressure; only this M3 Pro is proven.
5. Keep FP16 action-logit drift visible, even though the declared FP16 profile gates action probabilities rather than action logits.
6. CPU feasibility and portable release behavior are outside this MLX-only worker scope and must be combined with the parallel backend result.
