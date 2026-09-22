# L1 MLX Metal JIT follow-up

## Decision

**Yes, with a small embedded residual library.** `MLX_METAL_JIT=ON` shrinks
`mlx.metallib` from 182,351,120 to 1,455,256 bytes (99.20%). MLX still must
load that residual metallib: a copied JIT executable fails a real Metal
operation when both its sibling and compiled absolute-path metallib are absent.
Because it is only 1.39 MiB, `spikes/mlx-jit` embeds it, validates/extracts it
to a content-addressed user cache, calls `set_metallib_path`, and runs as one
shipped file. It passed `device` and English `plan-example` FP32 from `/tmp`
with no sibling metallib.

The final executable is 18,017,552 bytes, SHA-256
`ab187ad2...5b3837`; the residual metallib is 1,455,256 bytes, SHA-256
`44eb25db...e618b`. The prior AOT executable plus metallib was 198,366,592
bytes, so the single shipped file is 90.92% smaller. Model/tokenizer assets
remain separate.

## What JIT changes

MLX 0.32.2 still AOT-compiles nine source groups into the residual library:
`arg_reduce`, `conv`, `dot`, `layer_norm`, `random`, `rms_norm`, `rope`,
`scaled_dot_product_attention`, and `fence`. `xcrun metal-nm` reports 164
kernel symbols. `metal-objdump --syms` did not expose symbols for this archive;
`metal-nm` did.

The other kernels do not require source files at runtime. CMake's
`make_jit_source` preprocesses Metal headers into generated C++ functions that
return source strings; those objects and `jit_kernels.cpp` are linked into
static `libmlx`. `Device::build_library_` passes the strings to
`MTLDevice::newLibrary(source, options, error)` and caches libraries in-process;
Apple persists compiled shader cache entries across processes/reboots.

## mlx-sys patch

`spikes/mlx-jit/Cargo.toml` uses
`[patch.crates-io] mlx-sys = { path = "../vendor/mlx-sys" }`. The exact
`build.rs` delta (also `jit/mlx-sys-build.patch`) is:

```diff
@@ after CMAKE_CXX_COMPILER
+config.define(
+    "CMAKE_OSX_DEPLOYMENT_TARGET",
+    env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "14.0".to_owned()),
+);
+if env::var_os("MLX_RS_METAL_JIT").is_some_and(|value| value == "1") {
+    config.define("MLX_METAL_JIT", "ON");
+}
@@ fn main
 println!("cargo:rerun-if-env-changed=MLX_RS_METAL_PATH");
+println!("cargo:rerun-if-env-changed=MLX_RS_METAL_JIT");
+println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
```

Build command used `MLX_RS_METAL_JIT=1`, an isolated `MLX_RS_METAL_PATH`, and
`MACOSX_DEPLOYMENT_TARGET=14.0`. The observed cached-source release build took
2m56s; this is not a clean-network build-time benchmark.

## Single-file behavior and end-user requirements

At startup, a sibling `mlx.metallib` wins for diagnostics. Otherwise the
embedded bytes are compared exactly with
`$XDG_CACHE_HOME/laya-rs/<sha>/mlx.metallib`, or
`~/Library/Caches/laya-rs/<sha>/mlx.metallib`; missing/corrupt content is
atomically rewritten before `set_metallib_path`. The machine therefore needs
Apple Silicon, macOS 14+, compatible Metal, and a writable cache. It needs no
Python, Cargo, CMake, Xcode, Homebrew, or loose kernel source/metallib. The
first launch also needs enough temporary runtime/compiler resources to build
JIT kernels.

## Cold start (plan-example, ms)

Caches were actually removed before each JIT-cold run with:

```sh
CACHE="$(getconf DARWIN_USER_CACHE_DIR)"
rm -rf "${CACHE}com.apple.metal" "${CACHE}com.apple.metalfe"
killall MTLCompilerService 2>/dev/null || true
```

Both paths were absent immediately afterward; JIT-cold recreated 11,340 KiB
(FP32) or 12,076 KiB (FP16). The large cold/warm change confirms the clear took
effect. Times are first model-forward calls, not process/model-load time.

| Profile | Type | JIT cold | JIT warm-cache | AOT |
|---|---:|---:|---:|---:|
| English | f32 | 2184.730 | 92.675 | 406.638 |
| English | f16 | 2897.598 | 83.820 | 592.621 |
| Multilingual | f32 | 2095.800 | 89.814 | 387.644 |
| Multilingual | f16 | 3149.585 | 65.123 | 608.628 |
| Typed | f32 | 2388.293 | 97.943 | 465.467 |
| Typed | f16 | 3144.414 | 89.281 | 669.090 |

A user's first used shapes/dtypes pay roughly 2.1-3.1 seconds here. System cache
removal is destructive and was used only to measure first-machine behavior.

## Steady state: spike, not acceptance

Each cell is `p50 run1/run2; pooled mean`, with 20 warm repetitions per run in
AOT, JIT, AOT, JIT order (40/backend). Variability is large; do not infer a
stable speedup. The AOT artifact was the prior host-target build, while JIT was
macOS-14-targeted, so NAX availability is also confounded.

| Profile/fixture | Type | AOT ms | JIT ms |
|---|---:|---:|---:|
| multilingual/distinct-ml-q1 | f32 | 35.483/31.170; 33.645 | 31.013/31.083; 31.477 |
| multilingual/distinct-ml-q1 | f16 | 35.127/37.242; 36.199 | 36.098/47.547; 40.542 |
| multilingual/distinct-ml-q10 | f32 | 341.202/331.952; 345.507 | 320.378/281.422; 321.519 |
| multilingual/distinct-ml-q10 | f16 | 337.569/369.418; 360.845 | 323.752/467.693; 393.639 |
| english/plan-example | f32 | 75.076/81.361; 77.032 | 76.665/79.417; 77.309 |
| english/plan-example | f16 | 100.867/74.291; 87.708 | 101.515/74.092; 86.388 |
| english/distinct-en-q10 | f32 | 1039.586/860.315; 939.185 | 1063.995/1132.259; 1078.152 |
| english/distinct-en-q10 | f16 | 1231.389/1272.066; 1359.842 | 1240.017/1218.473; 1215.271 |

## Parity, deployment, and recommendation

Full FP32 `parity --fixture all` passes for English, multilingual, and
typed-decisions. `pass`, fixture/question counts, argmax counts, and every
aggregate metric are exactly equal to the three committed AOT reports.

The JIT Mach-O minimum is 14.0 and its AIR says macOS 14.0. The prior AOT
Mach-O says 11.0 but AIR says 26.2. Target 14 disables MLX's Metal-4 NAX path;
that is expected and makes this artifact honest for macOS 14, but it requires
oldest-OS and M1/M2/M4 validation before release.

**Recommendation:** prefer JIT plus embedded extraction when a ~2-3 second
first-machine/shape cost is acceptable; it removes the 174 MiB distribution
problem with no observed parity change and no clear steady-state regression.
Keep AOT only for products where first-use latency dominates package size.
Treat these timings as a spike, not acceptance.

Raw evidence: `benchmarks/results/l1-spike-mlx/jit/summary.json`,
`cold-start/summary.json`, `steady-state/summary.json`, `parity/`,
`single-file-*.txt`, `metallib-nm.txt`, and `deployment-target.txt`.
