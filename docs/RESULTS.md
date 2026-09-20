# Results

All numbers here are reproducible from committed fixtures/scripts and raw JSON under `benchmarks/`. Contended or pilot samples are labelled and are **not** baseline evidence. Nothing here compares to the published T4 figures as a speedup; no T4 is available.

## L0 — Python baseline and compatibility contract

### Pinned sources and assets

`manifests/sources.json` (regenerate/verify with `cd python && uv run laya-fetch --verify`).

| Profile | `model.safetensors` SHA-256 | bytes | context / head budget |
| --- | --- | ---: | --- |
| english | `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c` | 842,609,210 | 512 / 192 |
| multilingual | `9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204` | 643,835,514 | 1024 / 256 |
| typed-decisions | `4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e` | 842,609,220 | 1024 / 256 |

Bundle hub `convaiinnovations/laya@c5d78730` subfolders are git-OID-identical to the standalone `laya-multilingual@052592a1` and `laya-typed-decisions@f9ab0b22` repos for weights, encoder config, tokenizer.json and rl_agent_config.json; only the multilingual `tokenizer_config.json` differs (bundle 524 B already has `extra_special_tokens` as a mapping; standalone 502 B). `laya.Agent`'s `_fix_tokenizer_config` made **no byte change** to any bundled tokenizer config on this environment (recorded per profile in the manifest).

Python environment: 3.12.7, torch 2.14.0, transformers 5.17.0 (matches the published harness), tokenizers 0.23.2, safetensors 0.8.0, numpy 2.5.3, laya 0.3.3 @ `6a58191`. CPU and MPS run FP32 with autocast disabled (upstream behaviour).

### Load and smoke (all three profiles × cpu/mps)

`benchmarks/results/l0-smoke/*.json`. Every profile loaded on the requested device with no fallback text and `agent.device.type` equal to the request. Load times of 41–58 s were observed during a contended session (other workers active) and are not baseline evidence; cold-start will be re-measured on an idle host.

### Goldens (oracle fixtures)

`benchmarks/goldens/<profile>/{cpu,mps}/<fixture>.json` (+ `.npz` hidden states), 29 fixtures × 3 profiles × 2 devices, produced by instrumenting the real upstream functions; every instrumented pass reproduces `agent.system_one` output bit-for-bit (`instrumented_matches_native: true`). CPU forwards are bitwise deterministic across repeats; MPS repeats are bitwise identical too.

Upstream behaviours discovered and frozen as oracle facts:

- `head_max_len` is a **soft** budget: `edge-options-exceed` (120 choice options) did not raise. Upstream re-truncates each option to `max(4, (head_max_len-16)//k)` tokens and the head instructions to 8 tokens, producing a 370-token (ModernBERT) / 490-token (mmBERT) sequence that still fits `max_len`; only options that cannot fit `max_len` raise `ValueError("... options exceed head_max_len=...")`.
- Single-option choice raises `RuntimeError: selected index k out of range` upstream (`p.topk(2)` on a one-logit row). Native laya-rs must not fabricate a result under a parity claim; the Jev-adapter policy is an open L3 decision (`docs/COMPAT.md`).
- Truncation paths exercised: `hit_max_len` on english for `ragged-q12/{category,intent20}` and `edge-long-state/*`; on multilingual/typed-decisions for `edge-long-state/*`. Per-option re-truncation on `edge-head-overflow` (english `per=8`, others `per=12`) and `edge-options-exceed` (`per=4`).

### Python MPS vs CPU reproducibility (basis for frozen tolerances)

`benchmarks/goldens/<profile>/mps-vs-cpu.json`; tolerances frozen in `benchmarks/goldens/tolerances.json`.

| Profile | tensors | option logits max / mean abs | act logits max / mean abs | probs max abs | argmax | rounded-field flips |
| --- | --- | --- | --- | --- | --- | --- |
| english | exact | 8.77e-5 / 4.08e-6 | 1.20e-2 / 1.89e-3 | 5.28e-6 | 203/203 | 3 (±1 in 4th decimal) |
| multilingual | exact | 4.39e-5 / 2.67e-6 | 1.71e-2 / 7.28e-4 | 6.08e-6 | 203/203 | 2 |
| typed-decisions | exact | 1.39e-5 / 1.88e-6 | 7.32e-3 / 1.21e-3 | 1.61e-6 | 203/203 | 1 |

`act_probs` differ by 0.0 on every fixture (the action head is saturated on these inputs); the act-logit deltas are therefore the more sensitive diagnostic and get a wider frozen tolerance.

### Client compatibility fixtures

`compat/` — `@typesafe-ai/sdk@0.6.0` and `@ai-sdk/typesafe-ai@3.0.4` (source↔npm correspondence verified) driven against a local mock; 32 node:test cases, deterministic fixtures under `compat/fixtures/{native,ai}/`. Verified facts and open decisions: `docs/COMPAT.md`.

### Benchmark pilot (contended — not evidence)

`benchmarks/results/l0-pilot/` was recorded while the goldens worker was running (`contended: true`). It only validates the harness and yields a duration upper bound (~29 min per MPS profile pass, ~72 min per CPU profile pass for `published+distinct+extended`).

| profile / device | fixture | tokens | e2e p50 / p95 ms | model-only p50 | preprocess p50 |
| --- | --- | ---: | --- | ---: | ---: |
| multilingual / mps | distinct-ml-q1 | 291 | 41.4 / 45.1 | 55.9 | 1.3 |
| multilingual / mps | distinct-ml-q10 | 2824 | 307.5 / 359.6 | 323.0 | 11.2 |
| multilingual / mps | distinct-en-q1 | 331 | 50.8 / 52.7 | 46.4 | 1.3 |
| multilingual / mps | distinct-en-q10 | 3289 | 397.6 / 405.9 | 415.4 | 8.7 |
| multilingual / cpu | distinct-ml-q1 | 291 | 780.1 / 922.1 | 694.0 | 1.4 |

(model-only exceeding e2e on some rows is a contention artefact to be re-checked on an idle host.)

### Idle-host L0 baseline

**Not yet run.** At the time of the L0 commit the host carried a load average of ~11 from unrelated user applications (Chrome helpers, `ctx`, Codenotch). The plan forbids reporting a contended run as the isolated baseline. The exact commands are in `benchmarks/README.md`; results will land under `benchmarks/results/l0-full/` and this section will be updated.
