# Progress

## Planning bootstrap

- Created `docs/plans/initial-build.md` before implementation.
- Grounded architecture and benchmarks in pinned Laya main/research sources and HF manifests; no Laya weights downloaded or benchmarks run in this repository.
- Verified native Jev `/v1/systemone` and `/v1/models` schemas against pinned official JS SDK source, plus the AI SDK adapter and its different base-URL convention.
- Added the user's explicit requirement for native Apple M-series Metal libraries; CPU and CoreML are not undisclosed substitutes for the Metal gate.
- Recorded Python-first baselines, exact native parity, inline CLI, resident HTTP, standalone release packaging, local performance criteria and honest blocked outcomes.
- A read-only architecture review identified four issues: inconsistent illustrative confidence values, underspecified speed acceptance, ambiguous truncation language, and the unsupported-Python-MPS comparison case. All were addressed in the plan and implementation prompt.
- Created `docs/plans/implementation-prompt.md` for a Fable-led implementation session with Sol/Luna workers.

Validation at this stage is documentation/source-contract validation only. No Cargo workspace exists; Rust fmt/clippy/test and runtime/benchmark claims are not applicable yet.

Next: a new implementation session starts L0 (pinned Python baselines and compatibility fixtures), then follows reviewed gates L1–L5. Do not start from an assumed ONNX/Metal export or claim existing release binaries.
