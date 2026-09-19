# laya-rs: initial build and verification plan

Status: planning only. No Rust port, exports, local benchmarks, or release binaries exist yet.

## 1. Outcome and hard boundaries

Build a standalone Rust library and `laya` executable for the open Laya decision-model family. **Apple M-series native Metal acceleration is a first-class required target**, alongside a portable CPU path. Support one-shot JSON requests and `laya --serve`: load the selected checkpoint once, retain it in memory, accept HTTP POSTs, and return typed JSON. Existing Jev callers should be able to switch a configurable base URL to localhost with minimal application changes.

Success has four separate proofs:

1. **Python baseline first:** pinned upstream execution, exact input/output goldens, and measured local timings before port implementation.
2. **Rust parity:** identical preprocessing/tensors and equivalent model outputs within predeclared numerical tolerances, with explicit differences between native Laya and Jev wire adapters.
3. **Usable distribution:** download a precompiled executable, run `laya --serve`, download/verify missing assets on first launch, and serve without Python, pip, Cargo, or a compiler installed.
4. **Performance:** repeat the Python workloads on the same machine and report honest model-only, end-to-end, CLI, and warm HTTP latency/throughput. Optimize against measurements rather than assume Rust is faster.

A speedup on Apple M3 Pro is not proof of beating a Tesla T4 result. Beating the published T4 figures requires a matched T4 experiment. Shipping a correct port without a demonstrated speedup is an intermediate milestone, not achievement of the performance objective.

No training, architecture replacement, hidden precision reduction, context shortening, confidence recalibration, remote inference fallback, or changed option ordering may be called a transparent port optimization. Keep Laya separate from `openjev-rs` and `gliner2-rs`; a future backend-neutral `System1` interface may join them, but do not modify those repositories during this build.

## 2. Verified upstream evidence and pins

Freeze these references into a source manifest before development. Recheck availability without silently updating revisions. Record file SHA-256, licenses, Python/package versions, and acquisition commands. Download each weight artifact once into the project cache; do not put weights in Git.

| Source | Reviewed revision / fact |
| --- | --- |
| Laya code, `NandhaKishorM/laya`, main | `6a5819129eb220570792e417e49723d697efd76f` |
| Laya research branch | `28d43add7e47ce502489c9433310d55276c64e0f` |
| Runtime equality | `laya/` runtime was identical between those reviewed revisions |
| Bundled HF hub, `convaiinnovations/laya` | `c5d78730f3493e4fe16d61507ef4b78eef7318cf` |
| Separate multilingual HF repo | `052592a15d198d9ad47da779604259b10b47b7aa` |
| Separate typed-decisions HF repo | `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2` |
| TypeSafe JS SDK | `typesafe-ai/typesafe-sdk-js` at `66880ccded6cb642dc1809620c2b108c33730214`, package `@typesafe-ai/sdk` version `0.6.0` |
| Vercel AI SDK provider source | `vercel/ai` at `20dd00abba618d5a516e0fee40ccd3e18a2bd1fb`; pin published package versions separately after checking registry/source correspondence |

Use the bundled HF hub as the canonical download source where its selected subfolder has verified parity with the corresponding separate checkpoint; do not assume artifact identity from matching names. Root is English; `multilingual/` and `typed-decisions/` contain the other variants. Assets include `rl_agent_config.json`, `encoder/config.json`, tokenizer files, and `model.safetensors`. There is no ordinary root `config.json` classifier to load blindly.

Code and model cards declare Apache-2.0. Audit tokenizer/backbone/export/runtime licenses too. Use Apache-2.0 for this project unless the owner changes that decision; include LICENSE, THIRD_PARTY.md, required notices, source links and modifications. State non-affiliation with Convai Innovations and TypeSafe. Commercial use of openly licensed weights is not a promise of zero hardware cost.

### Model family

| Profile | Encoder | Total parameters, recorded | Context / head budget | Use |
| --- | --- | ---: | --- | --- |
| `english` | ModernBERT-large, 28 layers, width 1024 | 421.29M | 512 / 192 tokens | English |
| `multilingual` | mmBERT-base, 22 layers, width 768 | 321.91M | 1024 / 256 tokens | Multilingual, faster published T4 result |
| `typed-decisions` | ModernBERT-large | about 421M | 1024; verify exact checkpoint head budget | Specialized typed-workflow fine-tune |

Do not raise context to a backbone's theoretical 8k limit without treating it as a new configuration and validating accuracy and performance.

## 3. Architecture: what must actually be ported

Laya is a bidirectional encoder plus trained decision heads, not a GGUF decoder or next-token readout. The public API creates **one full sequence per question**, repeating state in each sequence, then issues one batched model call:

`[CLS] <type> question: <instructions> [SEP] [MASK] <option 0> ... [SEP] <state> [SEP]`

“One forward pass” means one batched invocation. Ten questions entail ten state-containing sequences through the encoder. The state representation depends on the question/options via bidirectional attention. Caching shared **hidden states** would change the architecture; caching state **token IDs** with question-specific truncation is legitimate.

The complete network is:

1. Transformers `AutoModel` encoder using SDPA, with the exact checkpoint's positional embeddings, attention masks/local-global pattern, norms and activations.
2. Add learned three-way question-type embeddings to every token's final hidden state.
3. Usually two full-sequence pre-norm `TransformerEncoderLayer`s: `d/64` heads, FFN width `4d`, padding mask. Preserve actual PyTorch constructor defaults, including its activation; do not assume the head FFN is GELU because the scorer is GELU.
4. Gather option-marker positions; score with LayerNorm → Linear(d,d) → GELU → Linear(d,1). Mask padded options with `-10000`.
5. Action head: first-token hidden state plus maximum probability, top-two margin, normalized entropy and option count/255; Linear(d+4,256) → GELU → Linear(256,n_act).
6. The action head uses **uncalibrated** option probabilities. Its logits/output belong in native-parity testing even when the compact Jev response omits them.

Strictly load every expected trained tensor and report missing/unexpected/mismatched keys. No randomly initialized head or automatic pretrained-backbone download may masquerade as a loaded checkpoint.

### Exact preprocessing and output semantics

Freeze Python-generated fixtures for:

- State strings unchanged; structured state uses Python `json.dumps(..., ensure_ascii=False)` with original object order and default spacing.
- Non-string instructions use the actual `_to_internal` serializer, which currently calls `json.dumps(ins)` with default ASCII escaping. Do not mistakenly reuse state serialization here.
- Criterion JSON rendering, descriptions absent/null/empty, label order, leading spaces before option text, and literal mask-token replacement.
- The initial 48-token per-option truncation, shared head budget allocation, minimum head/option rules, final state budget, special tokens, marker positions, padding, masks and qtype tensor.
- Noul internal option order is **false, true**; returned true probability is index 1. This differs from openjev-rs's yes/no adapter.
- Default right-side state truncation is upstream behavior; preserve that documented token-budget policy and expose truncation counts in native diagnostics. For HTTP compatibility responses, make truncation discoverable through `X-Laya-Truncated` and budget/count headers without changing the default answer schema; CLI emits a stderr warning when truncation occurs unless an explicit documented policy acknowledges it. Reducing budgets beyond the pinned policy is prohibited as an undisclosed optimization. Compatibility mode must not advertise Jev's 32k context.
- Floating-point JSON state, Unicode, nested objects/arrays, escapes, signed zero, numeric boundaries and insertion order. Do not copy openjev-rs's integer-only-state restriction into this port. Either prove accepted float formatting matches Python or fail explicitly on unsupported representations; never silently reformat state.
- Guard pathological nesting/body sizes with controlled errors; reject duplicate raw JSON keys rather than silently collapse options/questions. Document this intentional validation difference.
- Option-count/type temperature bucket, fallback per-type temperature, minimum denominator `1e-3`, stable softmax and first-argmax ordering.
- Choice/Score confidence is Laya's normalized-entropy statistic, not openjev's normalized margin. Noul native confidence is maximum binary probability. Capture precise upstream code as the oracle.
- Native outputs round to four decimal places; Score is expected ordinal level, not winning index. Preserve `legend`, native `action.act_probability`, token usage and rounding behavior. Rounded probabilities need not sum exactly to one; never renormalize silently.

Single-option Choice is a compatibility edge: native action-head `topk(2)` needs two entries, while Jev advertises one-option support. Decide explicitly in the compatibility spike: return a transparent deterministic trivial result without model invocation, or reject as unsupported. Do not pad with a fabricated semantic option and claim upstream parity.

## 4. Benchmarks: Python first, then identical Rust runs

### What the published numbers mean

The reviewed research result and harness report Tesla T4, PyTorch 2.11.0+cu128 and Transformers 5.17.0:

| Questions/request | English p50 | Multilingual p50 |
| ---: | ---: | ---: |
| 1 | 39.5 ms | 32.8 ms |
| 10 | 158.6 ms | 72.3 ms |
| 50 | 771 ms | 337 ms |

72.3 ms is request latency; 7.23 ms/question is amortized work per question, not the latency seen by each independent caller.

The published latency harness uses three warmups and twenty measured repetitions, synchronizes CUDA immediately before/after each timed call, and measures complete `system_one` preprocessing, transfers, inference, CPU postprocessing and result construction. It excludes downloading, loading, router selection and HTTP. It uses one English ticket sentence repeated six times and alternates only two schemas: a 3-option Choice and binary urgency question. It is not a ten-distinct-question workload.

Limitations: no actual input-token lengths, individual samples, GPU clocks, exact weight revisions or installed Laya version are retained. The notebook installs unpinned `laya>=0.1.6`. Current code on a T4 selects FP16 autocast (not BF16); CPU/MPS use FP32. Parameters are not necessarily all FP16, so “FP16 model” is an imprecise baseline description. Reproduce this as a **reconstructed pinned baseline**, not an exact recreation of an incompletely recorded historical environment.

A separate research throughput harness tokenizes before its timer and length-sorts token-budget batches. Its questions/sec is not comparable to end-to-end request throughput.

### Mandatory local baseline procedure

Before writing the Rust inference implementation:

1. Capture host OS, CPU/GPU/RAM, power mode, Python/PyTorch/Transformers/tokenizers/NumPy versions, backend/runtime versions, thread counts, precision/autocast, environment and all artifact hashes. Use an isolated locked Python environment.
2. Run pinned upstream Python on the development Mac (Apple M3 Pro, previously observed 18 GB RAM; remeasure) with CPU and MPS if genuinely supported. Record explicit failure instead of silently switching devices. The upstream exception path can print a warning and move CUDA work to CPU; instrument/reject this in benchmark mode.
3. Reconstruct published workloads at Q=1/10/50; add genuinely distinct questions at Q=1/10/21/50, all three primitives, 2/3/10/20 choices, varied actual token lengths and ragged batches. Test high cardinality as a quality/limit stress case, not a promised performant workload.
4. Record preprocessing-only, synchronized model-only, full warm API, cold load, one-shot process startup, router overhead, and later loopback HTTP timing separately. Model construction/downloads belong outside warm intervals.
5. Three warmups and twenty measured repetitions are the minimum comparable steady-state run; keep raw samples. Include first-run effects separately. Report p50/p95, mean/stddev, requests/sec, questions/sec and peak RAM/VRAM with actual token/option counts.
6. Run workloads serially on an otherwise idle host; no simultaneous builds, tests, other agent inference or model benchmarks. Mark affected samples contended rather than silently treating them as clean. Alternate Python/Rust order once the port exists; report thread/thermal/power differences.
7. Use bounded named experiment batches and save after each case/model/device. Give progress checkpoints during long runs. Start with a pilot to estimate full-suite duration. Do not launch an unbounded multi-hour sweep without communicating its projected cost and success criteria. On interruption retain partial evidence and list remaining cells.
8. T4 is optional external hardware, not assumed available. If unavailable, mark published-T4 speedup **unverified**; never substitute an M3 Pro number into that claim. Do not rent hardware or create paid services without approval.

### Predeclared performance acceptance

Separate a correct release from the speed objective. The initial primary speed target is **multilingual Q=1 and Q=10 genuinely distinct questions on this same M3 Pro**, Rust native Metal versus pinned Python MPS at the same supported precision, tokens and complete network. Require at least **1.10× speedup in median latency** for both workloads, no more than **5% p95 regression**, and all parity/quality gates passing. CPU and all three checkpoints are reported separately; a win on one profile is not a blanket family speedup. Flag >5% regressions on other matched representative workloads and adjudicate them before broad performance claims. The 10% speed-ratio target is an engineering acceptance target, not an expected or already observed gain.

At L0, before collecting Rust performance results, freeze exact primary request IDs, lengths, dtype, host/backend versions, metric calculation and tolerances in a benchmark manifest. If Python baseline evidence requires a different primary configuration, Fable must document the amendment and reason before seeing Rust timings; do not select favorable workloads after the fact. The published Q=1/10/50 duplicate-schema table remains a separate reproduction target.

Twenty repetitions are the historical-comparison minimum, not proof of a small speedup. Confirm a claimed primary win in at least three independent, alternating-order runs with at least 100 measured samples per workload/runtime after a duration pilot. Retain samples and a declared bootstrap/repeatability analysis; the 95% interval for the median speed ratio must stay above 1.0, alongside the 1.10× point-estimate target. Benchmark timeout/cancellation and thermal/contended samples are reported, not silently discarded.

If the complete pinned Python model cannot execute on MPS, record the failure, use Python CPU goldens for correctness with justified cross-device tolerances, and continue mandatory Rust Metal functionality testing. Report Metal-versus-Python-CPU only as a **cross-backend comparison**. Apple accelerator-to-accelerator speedup and this primary performance gate remain unverified until a defensible MPS baseline or explicitly approved alternate comparison exists; CPU fallback does not satisfy the Metal requirement.

### Quality and parity artifacts

Export JSON/NPZ/safetensors test artifacts containing original requests, serialized strings, all five input tensors, token/marker counts, option/action logits, unrounded probabilities, temperatures, native rounded answers and usage. Cover English, non-Latin scripts, JSON, long/truncated inputs and budget boundaries. Read actual tensor formats; never embed pickle or execute untrusted model code at runtime.

Require exact tokens/masks/positions/order. Before inspecting Rust deltas, freeze separate tolerances for CPU FP32, Apple acceleration and CUDA mixed precision based on Python reproducibility. Compare max/mean/RMSE logits, probability differences, first-argmax agreement, expected score, action probabilities and rounded outputs. Quantization gets a separate quality gate and version, not a relaxed baseline tolerance after failure.

Do not advertise generalized calibration or absence of semantic errors. Base checkpoints score roughly 0.35 on typed-decisions; the reported 0.766 requires the specialized fine-tune on that benchmark's training split. English confidence can remain high on unreadable scripts. Proper-scoring-rule training does not guarantee calibration on new domains.

## 5. Rust backend strategy and optimization order

### Start with evidence, not a backend promise

Run two bounded feasibility experiments: **native Rust Metal on Apple Silicon** (first candidate: Candle with its Metal backend) and a portable full-network CPU path (candidate: ONNX export plus Rust `ort`, or Candle CPU if that reduces semantic/package complexity). Verify current crate/runtime versions and real APIs; pin Cargo.lock and runtime/export versions. No upstream ONNX export or TensorRT engine was found in the reviewed repository.

- Apple is the primary accelerated target: execute the entire ModernBERT/mmBERT backbone and custom heads through verified Metal kernels. Candle support for this exact architecture must be established, not assumed. Implement missing model layers only after checking existing upstream model/kernel implementations.
- If Candle cannot provide correct or performant coverage, investigate direct Metal/MPS/MPSGraph integration using maintained Rust/Objective-C bindings as an explicit reviewed alternative. This increases implementation scope; do not improvise custom attention kernels before profiling and parity tests.
- CoreML (including ONNX Runtime's CoreML execution provider) is a distinct possible Apple backend. It may use CPU, GPU or Neural Engine; label it `coreml`, disclose actual placement/compute-unit configuration, and do not treat it as satisfying the native `metal` target merely because it runs on a Mac.
- CPU: ONNX Runtime CPU or the verified CPU implementation of the chosen Rust runtime provides the portable correctness baseline. ONNX must export **the entire backbone plus custom heads**, not a generic classifier.
- NVIDIA: ONNX Runtime CUDA is a later T4-comparison candidate; TensorRT is a later optimized profile, not necessary for local Apple delivery.
- Rust source language does not by itself accelerate GPU operations. Neither complete architecture/operator coverage nor a Metal speedup has been proven at planning time.

Before committing to the backend, prove one complete checkpoint on CPU and native Apple Metal, verify both backbone families, then test packaging and all three checkpoints. A CPU-only implementation does not complete the required Metal gate. Check dynamic shapes, local/global attention, rotary position behavior, normalization, SDPA equivalence, marker gathers, masking, action head and mixed precision. An export success without numerical parity is not a passed spike.

Python may be used for development/export, but **must not be required by end users**. Publish pinned, hashed optimized assets if Rust cannot directly load the upstream safetensors. Asset builds must be reproducible from pinned upstream weights with license notices; startup must not silently execute an export script or fetch an unpinned encoder.

### Apple M-series / Metal acceptance

Target arm64 Apple Silicon (M1 and later as a desired release family, with actual minimum macOS/GPU-family requirements determined from the selected kernels). First hardware gate is this machine's M3 Pro; do not claim M1/M2/M4 validation without running it. Use system Metal frameworks and precompiled/embedded Metal libraries as supported by the chosen runtime; end users must not need Python, Xcode, developer tools or Homebrew. If any first-run kernel compilation occurs through system APIs, disclose and separate it from warm inference latency.

`laya --serve --backend metal` must select a real Metal device, keep model tensors resident, and report backend/device identity to stderr and optional diagnostics. On an unsupported machine or missing required operation it must fail explicitly; no silent CPU fallback. `--backend auto` may choose a documented verified backend, but must disclose the actual choice and fallback reason. Do not ship `metal` as an alias for an unverified CoreML placement policy.

Port and test embeddings, positional/rotary behavior, local/global attention, normalization, linear/activation layers, typed head, marker gather/masking, scorer and action head. Inspect native implementations for hidden CPU operations and unnecessary per-layer synchronization. Unified memory avoids a discrete-device PCIe boundary but does not eliminate copies, command-buffer overhead or resource synchronization.

Preserve a CPU FP32 correctness reference. Benchmark Python MPS and Rust Metal at equivalent precision/sequence shapes; FP16 and any BF16 support are distinct measured configurations with predeclared parity tolerances, never automatic downcasts advertised as same-precision speedups. Validate finite outputs, action logits and calibration/rounding after the dtype change. Synchronize device completion before ending model timing; enqueue latency is not inference latency.

Retain model allocations/compiled kernels across HTTP requests; preallocate bounded work buffers, investigate fused attention and command-buffer overhead only after profiling. Record peak process memory and reported GPU/unified-memory use without adding overlapping accounting as if separate VRAM. Exercise memory-pressure/load failure, concurrent request admission and graceful shutdown on the real device.

### Optimization ladder, with separate measurements

1. Cache state tokenization and repeated schema/option tokenization; preserve per-question budget application. Bound caches by bytes/items and avoid storing user content indefinitely.
2. Reuse host/device buffers, preallocate, reduce allocations and redundant copies; keep loaded sessions warm.
3. Graph fusion and optimized attention/linear kernels with identical semantics and verified precision.
4. Length/option-count bucketing for heterogeneous requests. Quantify queue waiting time as well as throughput.
5. Reduce synchronization/transfers and move appropriate postprocessing onto the device, retaining rounding/calibration semantics.
6. Optional duplicate-question elimination, explicitly reported. It is especially favorable to the published two-schema benchmark and must not be sold as general distinct-question acceleration. Define logical versus physically executed token usage separately if this changes accounting.
7. Only after baseline parity: optional INT8/other quantized profiles with their own output/quality/calibration reports. No inference-equivalence claim across precision changes.

Do not remove the trained action head merely because the Jev adapter omits its fields. Offer reduced-head execution only as a separately evaluated optimization profile, with native parity limitations stated.

## 6. Public workspace and API

Proposed workspace (finalize during the feasibility gate, not by speculative abstraction):

```text
crates/laya-core/       # ordered request types, formatting, calibration, adapters, metrics
crates/laya-runtime/    # asset registry/cache, tokenizer, model/session/backend ownership
crates/laya-server/     # HTTP routing, auth, bounded admission and model workers
crates/laya-cli/        # executable laya, inline/JSONL/server entry points
python/                # locked baseline, export and golden-generation tools
benchmarks/            # owned fixtures, runners, immutable raw results and manifests
manifests/             # pinned source weights/tokenizers/exports and digests
schemas/               # native and Jev-compatible requests/responses/errors
compat/                # pinned SDK client tests, no live commercial endpoint requirement
docs/plans/            # this contract and reviewed amendments
```

Keep core usable without the heavy runtime. Use Rust `tokenizers` only after exact preprocessing/token IDs are verified. Prefer stable typed errors and owned outputs; backend contexts/sessions live on dedicated workers as required by actual thread-safety APIs. Do not add unsafe Send/Sync to bypass ownership. Keep one-shot and HTTP adapters on the same inference implementation.

Public concepts: `ModelProfile`, validated `Request { state, questions }`, insertion-ordered `Question::{Choice,Score,Noul}`, `PreparedBatch`, `NativeResult`, `JevResult`, `Runtime::load/evaluate`, `AssetManifest`, `EvaluationOptions`, `TimingReport`. Preserve unknown top-level metadata only where the compatibility contract explicitly permits it; never let it alter the model silently.

## 7. CLI and resident-server contract

### One-shot and streaming

Proposed grammar:

```sh
# One JSON request; stdout is one JSON response.
laya --request request.json
cat request.json | laya
laya --json '{"state":"Charged twice","questions":{"refund":{"type":"noul","instructions":"Is a refund requested?"}}}'

# Friendly sugar with semantic IDs, same internal request/response.
laya decide --state 'Charged twice' --question 'Which queue?' \
  --option billing='Payments and refunds' --option support='Other issues'

# Persistent model, one request/response per line, no per-line model load.
laya run --input requests.jsonl

# The requested primary deployment experience.
laya --serve
laya --serve --model multilingual --host 127.0.0.1 --port 8080

laya models list
laya models pull multilingual
laya bench --fixture published-latency --backend cpu
```

`laya --serve` defaults to port 8080 on **127.0.0.1**, checkpoint `english`, Jev-compatible output and automatic best **verified** backend for the release target. English is the source API default, not necessarily the best quality/speed choice. `--model multilingual|typed-decisions` is explicit; optional `--model auto` requires a published routing policy and residency budget. Do not silently load all three by default.

CLI options include `--pretty`, `--quiet`, `--format jev|laya-native`, `--backend auto|cpu|metal|coreml|cuda` as actually supported, `--cache-dir`, `--offline`, thread/batch/admission limits, `--diagnostics` and server auth flags. Unsupported backend names must fail clearly, not pretend acceleration. Decide final resource defaults from pilot measurements and record them.

Default decision JSON is lean: native Jev-style envelope/answers, not raw logits, hashes, timings or large model metadata. Optional diagnostics use a separate envelope/sidecar or stderr; HTTP uses explicit extension mode/headers. Logs/progress go only to stderr. `--quiet` suppresses routine native logs but not safety/errors. Pretty printing is opt-in; JSONL stays one object per line. Exit 0 success, 2 invalid input, 1 runtime failure; handle broken pipes without panic. Preserve input order and flush each completed JSONL response.

No network call for already cached verified assets in offline mode. First-run downloads report progress and size to stderr; verification failure prevents readiness. Support `laya models pull` for preprovisioning and `--offline --serve` for air-gapped operation.

### Warm HTTP service

Implement with a small maintained HTTP stack (candidate Axum/Tokio; confirm versions). The service owns a resident runtime and a bounded request queue. Async handlers must not run blocking inference on the executor. Start with a single inference owner per loaded model; add controlled concurrency/microbatching only after measuring safety, throughput and latency.

- Load/verify model once at startup, perform a disclosed warmup, and declare readiness only afterward. No per-request model construction, weight hashing, graph export or device transfers of all weights.
- Small process liveness endpoint `GET /healthz`; `GET /readyz` returns 503 until the model is ready and 200 afterward. During startup serve only health/readiness or bind the inference listener after readiness; choose and test one explicit behavior.
- `POST /v1/systemone` is the primary compatibility route; `GET /v1/models` lists actual local model identities/capabilities.
- `Content-Type: application/json`; body, depth, question, option and total-token limits are finite. Invalid/oversized inputs produce structured errors, not worker crashes or implicit truncation beyond the selected documented policy.
- Bounded queue and configurable deadline; 429/503 plus Retry-After on overload, appropriate 4xx on validation/auth, 5xx on backend failure. Cancellation stops queued work; do not claim a noninterruptible GPU kernel was cancelled.
- Graceful SIGINT/SIGTERM: stop admission, drain within configured grace, join workers and release sessions. Recover or fail readiness after worker errors; never return stale outputs.
- No prompts/responses or secrets in normal logs. No wildcard CORS by default. Binding a non-loopback interface requires explicit authentication configuration or an explicit documented unsafe override; TLS belongs in a reverse proxy unless later implemented.
- Optional static bearer authentication via environment/file/flag, with secret-safe logging. Loopback default may accept the SDK's dummy bearer token without verification; do not forward it to TypeSafe or the model hub.
- Model residency/router preload policy must be explicit. Multiple workers or profiles must not unexpectedly duplicate gigabytes of weights. Batching across callers must preserve question ownership and cancellation boundaries.

## 8. Jev compatibility: wire format, not equal intelligence

### Verified native contract

The pinned official SDK posts to `POST /v1/systemone`, with:

```json
{
  "model": "jev-latest",
  "state": {"body": "I was charged twice. Please refund."},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"billing": "Payments and refunds", "support": "Other support"}
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["low", "medium", "high"]
    },
    "refund": {"type": "noul", "instructions": "Is a refund requested?"}
  }
}
```

Illustrative response shape, not a measured prediction:

```json
{
  "model": "laya-english@<verified-revision>",
  "answers": {
    "department": {
      "type": "choice", "choice": "billing", "confidence": 0.53,
      "probabilities": {"billing": 0.9, "support": 0.1}
    },
    "urgency": {
      "type": "score", "score": 1.2, "confidence": 0.18,
      "legend": {"0": "low", "1": "medium", "2": "high"},
      "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}
    },
    "refund": {"type": "noul", "noul": 0.95}
  },
  "usage": {"input_tokens": 123, "output_tokens": 0}
}
```

Request fields support string/object/array/null state and instructions; instructions may be omitted/null. Criterion descriptions can be structured/null, Noul true/false descriptions are optional, and Score criteria are ordered arrays. Map missing instructions to a documented adapter default (e.g. empty string) and test it; upstream Laya currently directly indexes `instructions`, so native acceptance cannot be assumed.

Choice probabilities are keyed by option label, Score probabilities/legend by stringified level index, Noul uses `noul` rather than `p_yes`/`probability`. Keep arbitrary question IDs and semantic labels. No `choice_index`, option-ID arrays or diagnostic blobs in the default compatibility answers. Native Laya action/confidence extras may be omitted in the Jev adapter while remaining computed/tested in native mode.

`GET /v1/models` returns `{ "models": [{"name":"...","description":"...","release_date":"..."}] }` per the pinned SDK. Report project release dates/actual checkpoint identity; do not invent a TypeSafe model version.

### Compatibility levels and intentional differences

1. **Required:** raw HTTP request/response shape and official JS SDK 0.6.0 against localhost.
2. **Required:** pinned `@ai-sdk/typesafe-ai` provider through its configurable base URL, including conversion of SDK `boolean` to native `noul`.
3. **Best effort after verification:** official Python client; retrieve and pin its real API before claiming support.
4. **Not promised:** intercepting hard-coded TypeSafe URLs, emulating billing, account services or Vercel AI Gateway's proprietary transport. Gateway users must select the direct TypeSafe provider with a local base URL; changing only a model string is not sufficient.

Native SDK: base URL is the root, e.g. `http://127.0.0.1:8080`; it appends `/v1/systemone`. Its API-key environment variable is `TYPESAFE_API_KEY`, and `TYPESAFE_BASE_URL` can override the root. It requires a nonempty key even for localhost; a documented dummy local value is acceptable when local auth is disabled.

AI SDK provider: base URL is `http://127.0.0.1:8080/v1`; it appends `/systemone`. The provider's default key variable is `TYPESAFE_AI_API_KEY`, distinct from the native SDK. Verify provider factory signature/options at the pinned revision before writing examples. Never require a real paid Jev key for compatibility tests.

Accept `jev-latest` as an explicitly documented compatibility alias for the configured Laya profile, so existing requests work after a base-URL switch. Also accept actual Laya profile names. Unknown models return a structured error; do not wildcard arbitrary `jev-*` versions. Response `model` and `X-Laya-*` headers identify the real implementation and artifact, never impersonate a Jev version.

Default Jev-wire rounding: two decimal places for probabilities/scores, matching the pinned provider's `rounding` declaration; native Laya mode retains four-decimal upstream output. Compute winners/confidence/expectation from the same unrounded distribution before display rounding. Test SDK tolerance for sums/expected scores after rounding; do not silently force rounded sums to one. Treat these as adapter differences, not native numerical parity failures.

**Confidence meaning is not interchangeable:** keep Laya's actual statistic, explain it in docs/capability headers, and never advertise Jev's operational calibration. A schema-compatible local substitute can make different decisions and has shorter context limits. The 1–255 Choice / 2–10 Score Jev limits are an API capability comparison, not a claim that Laya handles 255 options well. Respect model marker/head budgets and return explicit unsupported-capability errors where necessary; recommend under ~20 choices.

Document usage as the actual summed attention-mask tokens across repeated question sequences, output_tokens=0. It is not TypeSafe billing or a unique-state-token count. Any later deduplication optimization must make its logical/physical usage distinction explicit.

Error bodies must be accepted by both pinned clients. Start from their real error parsers (`message`, `detail`, `error`, `error_type` as applicable), reproduce status/code/Retry-After behavior where verified, and record differences. Do not promise a guessed full TypeSafe error taxonomy.

## 9. Downloadable releases and startup reliability

Primary acceptance: a native CPU-capable release executable usable without Python or an installed ONNX Runtime. Investigate statically linking the chosen runtime or using a native Rust runtime during the feasibility spike—not at the end. If a platform legally/technically requires colocated runtime libraries, document the platform archive and seek explicit approval before claiming the requested single-binary experience is met.

Initial release matrix: macOS arm64 with verified native Metal plus CPU, and Linux x86_64 CPU; add Linux CUDA and Windows only when actually built and tested. Native Apple Metal is a required deliverable; CPU-only or CoreML-only success must be reported as partial rather than silently satisfying that requirement. Publish clear CPU/GPU requirements; NVIDIA drivers cannot be bundled away.

GitHub Releases should include binaries/platform archives, SHA256SUMS, licenses/notices, source revision, artifact/export manifest and provenance/SBOM where tooling supports it. Use CI release builds with explicit minimum OS/libc/CPU instruction targets, not accidental host-specific AVX requirements. Sign/notarize macOS binaries when credentials exist; otherwise document Gatekeeper handling and the absence of notarization. Do not claim remote CI/release validation merely from writing YAML.

Asset cache precedence: explicit flag, `$LAYA_HOME`, documented platform cache directory. Artifact identity includes immutable model revision, tokenizer, graph/export settings, precision and runtime compatibility. Use process-safe per-artifact locks, atomic downloads/publication, length/hash verification, safe relative paths and canonical cache containment. Reject corrupt content; repair is explicit. Never follow escaping directory symlinks to mutate outside the cache, quarantine local caller files, or redownload verified weights unnecessarily. Validate both registered and custom paths. Concurrent server starts must download once and either reuse or fail cleanly.

First-run download is not “instant readiness.” Print the selected profile, bytes and cache destination to stderr; `/readyz` stays unavailable until verified load/warmup. No telemetry, remote inference, model auto-updates or unpinned code execution by default. A corrupt/missing asset in offline mode must fail with a useful error.

## 10. Milestones and acceptance gates

Every milestone: Fable reviews the contract and evidence; Sol/Luna implement bounded work; run formatting, warnings-as-errors lint, unit/integration checks appropriate to that stage; update docs/PROGRESS.md, docs/RESULTS.md and checklist; commit only the reviewed milestone. Preserve failed/raw evidence instead of rewriting it. Push reviewed commits to the configured repository. No committing weights, caches, credentials or private transcripts.

| Gate | Deliverable and acceptance |
| --- | --- |
| **L0 — baseline and compatibility contract** | Fetch only pinned sources/assets as needed; lock Python environment; verify licenses and source manifests; reconstruct benchmark; run local Python CPU/MPS pilot and baseline before Rust inference; capture exact tokens/logits/native outputs; pin SDK request/response/error fixtures. Publish timing scope and estimated remaining workload. |
| **L1 — backend/export/distribution feasibility** | Prove complete English network plus action head on CPU and native Apple Metal, then multilingual operator coverage; compare with L0 goldens under frozen tolerances. Demonstrate Python-free runtime loading and identify packaging requirements. Choose backend and packaging with explicit fallback/blocked decision, not a stub promising future Metal/export support. |
| **L2 — Rust core and inline CLI** | Ordered request types, exact serializers/tokenizer/marker/truncation/calibration/rounding, native and Jev adapters; all three checkpoint profiles supported or explicit blocker. `laya` stdin/--json/--request and friendly decide output JSON only; JSONL retains a loaded runtime and streams. CPU parity and schema tests pass; action head retained. |
| **L3 — warm HTTP and client compatibility** | `laya --serve`, preload-once proof, health/readiness/auth/admission/body-limit/cancellation/shutdown tests; POST systemone and models route. Official pinned JS client plus AI SDK integration tests against localhost, all primitives, structured/null/missing fields, errors, aliases, rounding and capability boundaries. No real Jev key needed. |
| **L4 — same-machine benchmark and optimization** | Run identical pinned Python/Rust CPU and verified Apple-backend workloads, original duplicate schemas and distinct/ragged cases; report raw samples, p50/p95, load/model/e2e/HTTP split, quality parity and memory. Profile and implement one justified optimization at a time with rollback on regressions. T4 comparison only if matching hardware is available. No speedup claim absent evidence. |
| **L5 — releases and documentation** | GitHub release binary/platform assets actually built, checksummed and tested on clean target. Download → `laya --serve` → ready → SDK call → restart/offline cache reuse works with no Python. CI/macOS+Linux tests, license/notice/security docs, examples, limits/calibration/non-affiliation, final results and compatibility matrix. |

L0 should not silently expand into training/reproducing every marketing quality dataset. Focus native parity fixtures plus a held-out accuracy subset that can detect semantic damage; keep published vendor quality and our measured accuracy separately labeled.

### Test matrix

- No-model: serializers, option order, truncation budget, masks, confidence/temperature/rounding, adapters/schema, routes/limits/auth/error shape, queue states, CLI parsing/streams, checksum/path/lock/cache repair, metric math.
- Golden: Python-vs-Rust inputs exact; raw option/action logits and probabilities within frozen tolerances; every supported primitive/profile; repetition/determinism and first-tie cases.
- Native integration: opt-in `LAYA_INTEGRATION=1`, offline verified cache by default, selected device explicit. Skipped model tests are “not run,” not “passed.”
- Client contract: real pinned SDKs against local HTTP, not only handcrafted requests. Test native SDK and AI SDK's different base URL conventions.
- Lifecycle: model construction counter stays one over repeated requests; concurrent bounded requests, overload, cancelled queue items, worker failure, signals, corrupt cache, interrupted first download and concurrent startup.
- Release: clean environment without Python/Cargo; binary starts, resolves assets, health readiness, correct JSON, offline restart, checksum/signature instructions.
- Performance: dedicated uncontended runs, reproducible scripts/config/fixtures and immutable samples. Tests and builds must not run concurrently with measured inference.

## 11. Open decisions and stop conditions

Resolve in order, recording each decision with evidence:

1. **Model/export support:** can the complete ModernBERT/mmBERT network and heads execute in the selected runtime with accurate local/global attention? Fallback: evaluate the other named backend; if neither passes, stop with unsupported operations/repros rather than substitute a different model.
2. **Single-binary distribution:** can the runtime be embedded/linked compatibly on target platforms? Fallback: explicitly proposed self-contained platform archive; do not silently require system ONNX/Python.
3. **Apple Metal acceleration:** test actual kernels/operator placement/precision and compare to Python MPS. Try the reviewed alternate Metal integration if needed; if no faithful full-network Metal path is defensible, stop the Metal gate with evidence and request a scope decision. Labeled CPU remains usable but does not satisfy the user's Metal requirement; CoreML is a separate backend, not an undisclosed substitute.
4. **Compatibility edge normalization:** omitted/null instructions/state, one-option Choice, all valid structured criteria, float formatting, token/head limits, rounding and error taxonomy. Prove adapters with clients; publish unsupported cases rather than ignore fields.
5. **Calibration/confidence:** preserve actual checkpoint temperatures/statistics. Native parity first; optional refitting requires a separate held-out experiment and version.
6. **Performance gap:** if Rust fails to beat Python, profile before adding complexity. Do not reclassify amortized throughput as latency or win by reducing context/quality. If no defensible optimization remains, stop with baseline, profiles, attempts, quality checks and next decision needed.
7. **Missing T4/access/credentials:** no paid provisioning, live API calls or signing credentials assumed. Mark relevant gates unrun and ask for the specific input needed; complete independent local work meanwhile.

Between iterations, choose the smallest measured correctness/performance bottleneck, state the hypothesis, change one primary factor, rerun parity plus affected benchmarks, and retain both baseline and candidate evidence. For a blocked gate return a concrete checkpoint: completed items, failing command/artifact, attempted paths, blocker, and next input required. Never answer “done” while only an agent is running.

## 12. Fable-led implementation policy

The implementation session's main agent is **Fable**, responsible for design synthesis, scope, source verification, compatibility adjudication, performance claims, reviews and final commits. Use **Sol and Luna** as separate implementation/test/benchmark workers when the harness supports them. Resolve the actual configured model identifiers; never silently substitute an unavailable requested model.

Every worker receives this plan by path and must read it completely, not work from a compressed summary. Give bounded tasks with exact files and acceptance tests. Keep overlapping writes serialized; use isolated worktrees for independent changes. Avoid repeatedly sending multiple agents to duplicate the same broad investigation. Fable must inspect real diffs and evidence; worker summaries are not proof.

Maintain visible progress. Long benchmarks get a pilot, estimated duration and durable per-case checkpoints. Do not run build/test workers beside latency measurements. An idle main conversation while a worker runs is not a completion claim. If a worker ends without a useful result, inspect its files/processes and resume only unfinished work.

No workflow-tool orchestration, persistent goal mode, commercial services or experimental budgets are assumed; use ordinary subagents unless the user explicitly requests otherwise. Do not change or stop the separate openjev-rs work.

## 13. Primary source links

- Overview: https://laya.convaiinnovations.com/
- Code/sequence/head: https://github.com/NandhaKishorM/laya/blob/6a5819129eb220570792e417e49723d697efd76f/laya/common.py
- Load/inference/calibration: https://github.com/NandhaKishorM/laya/blob/6a5819129eb220570792e417e49723d697efd76f/laya/agent.py
- Routing: https://github.com/NandhaKishorM/laya/blob/6a5819129eb220570792e417e49723d697efd76f/laya/router.py
- Benchmark harness: https://github.com/NandhaKishorM/laya/blob/28d43add7e47ce502489c9433310d55276c64e0f/research/scripts/build_benchmark_nb.py (latency around lines 607–635; throughput around 150–214)
- Recorded benchmark: https://github.com/NandhaKishorM/laya/blob/28d43add7e47ce502489c9433310d55276c64e0f/research/results/t4_colab_benchmark.json
- Hub/card: https://huggingface.co/convaiinnovations/laya/tree/c5d78730f3493e4fe16d61507ef4b78eef7318cf
- Native Jev types: https://github.com/typesafe-ai/typesafe-sdk-js/blob/66880ccded6cb642dc1809620c2b108c33730214/src/types.ts
- Native client/route: https://github.com/typesafe-ai/typesafe-sdk-js/blob/66880ccded6cb642dc1809620c2b108c33730214/src/client.ts
- Native models route: https://github.com/typesafe-ai/typesafe-sdk-js/blob/66880ccded6cb642dc1809620c2b108c33730214/src/resources/models.ts
- AI SDK adapter: https://github.com/vercel/ai/blob/20dd00abba618d5a516e0fee40ccd3e18a2bd1fb/packages/typesafe-ai/src/typesafe-ai-evaluation-model.ts
- AI SDK response/error schema: https://github.com/vercel/ai/blob/20dd00abba618d5a516e0fee40ccd3e18a2bd1fb/packages/typesafe-ai/src/typesafe-ai-evaluation-api.ts
- Provider setup: https://ai-sdk.dev/providers/ai-sdk-providers/typesafe-ai

These pins ground a build plan, not a completed port or independent verification of every upstream marketing claim.
