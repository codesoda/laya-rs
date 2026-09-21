# Results

All numbers here are reproducible from committed fixtures/scripts and raw JSON under `benchmarks/`. Contended or pilot samples are labelled and are **not** baseline evidence. Nothing here compares to the published T4 figures as a speedup; no T4 is available.

## L0 — Python baseline and compatibility contract

### Pinned sources and assets

`manifests/sources.json` (regenerate/verify with `cd benchmarks/baseline && uv run laya-fetch --verify`).

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

### Idle-host L0 baseline (Python, fp32, complete network)

Runs `benchmarks/results/l0-full/` (2026-09-21) and `l0-full-b/` (2026-09-22 rerun of 16 cells whose start snapshot showed a contender — ClearDisk at 40–54% CPU — or unexplained load1 ≥ 9.6 during typed-decisions CPU). Merged clean table: `benchmarks/results/l0-baseline-merged.json`. Every cell: 3 warmups, 20 measured reps, device synchronised before/after, idle gate passed at start (`benchmarks/idle-policy.json`), torch 5 threads, raw samples retained. First-run label bug: the harness initially counted its own interpreter as a contender; `laya-bench-report` re-evaluates stored snapshots with the corrected policy (see `benchmarks/README.md`).

**Primary acceptance baseline (frozen in `benchmarks/manifest.json`)** — multilingual, MPS:

| fixture | tokens | e2e p50 | e2e p95 | model-only p50 | Rust Metal must reach (1.10×, ≤5% p95) |
|---|--:|--:|--:|--:|---|
| distinct-ml-q1 | 291 | **34.0 ms** | 34.1 ms | 32.5 ms | p50 ≤ 30.9 ms, p95 ≤ 35.8 ms |
| distinct-ml-q10 | 2824 | **296.7 ms** | 297.0 ms | 292.4 ms | p50 ≤ 269.7 ms, p95 ≤ 311.9 ms |

**Published-workload reproduction** (same fixtures as the upstream harness; T4 column is different hardware/precision and *not* a speedup comparison):

| profile | nq | M3 Pro MPS p50 | M3 Pro CPU p50 | published T4 fp16 p50 |
|---|--:|--:|--:|--:|
| english | 1 | 59.4 | 176.5 | 39.5 |
| english | 10 | 512.0 | 1304.9 | 158.6 |
| english | 50 | 2540.8 | 6265.2 | 771.3 |
| multilingual | 1 | 22.9 | 60.4 | 32.8 |
| multilingual | 10 | 190.2 | 482.4 | 72.3 |
| multilingual | 50 | 943.1 | 2211.6 | 337.4 |

Observations (evidence for L4 planning, not conclusions about Rust):

- Model time is ≥ 97% of e2e everywhere; preprocessing is 0.1–20 ms. Tokenisation caching cannot move the primary metric materially.
- On MPS, cost is linear in tokens with essentially no batch amortisation: ~91 ms/question english, ~30–38 ms/question multilingual at every Q. The M3 Pro GPU is compute-bound at these shapes, unlike the T4 which amortises 4–5× from Q=1 to Q=10.
- MPS vs CPU: 2.6–3× at Q=1, ~2.5× at Q=10.
- typed-decisions MPS timings equal english (same backbone, same shapes) within noise.
- p95/p50 spread is < 1% on nearly all cells — the runs are stable.

**Cold start** (`cold-<profile>-<device>.json`, 3 iterations each, warm page cache after the first): `Agent()` construction 22.6–25 s for every profile/device (dominated by transformers model construction + safetensors load, not device transfer); first `system_one` 113–145 ms on MPS / 52–138 ms on CPU; second call 22–47 ms MPS; peak RSS 5.5 GB multilingual, 3.6 GB english/typed-decisions.

Full table (96 cells):

| profile | device | fixture | nq | tokens | e2e p50 / p95 ms | model p50 | prep p50 | ms/q (amortized) | run |
|---|---|---|--:|--:|---|--:|--:|--:|---|
| english | cpu | bucket-choice-2 | 1 | 30 | 59.8 / 60.3 | 59.6 | 0.1 | 59.8 | l0-full |
| english | cpu | bucket-choice-20 | 1 | 102 | 117.4 / 118.5 | 116.9 | 0.6 | 117.4 | l0-full |
| english | cpu | distinct-en-q1 | 1 | 318 | 247.4 / 248.2 | 246.9 | 0.6 | 247.4 | l0-full-b |
| english | cpu | distinct-en-q10 | 10 | 3100 | 2349.6 / 2354.8 | 2352.7 | 4.1 | 235.0 | l0-full-b |
| english | cpu | distinct-en-q21 | 21 | 6466 | 4845.4 / 4878.8 | 4837.8 | 8.5 | 230.7 | l0-full-b |
| english | cpu | distinct-en-q50 | 50 | 15127 | 11552.7 / 11578.0 | 11526.8 | 18.6 | 231.1 | l0-full-b |
| english | cpu | distinct-ml-q1 | 1 | 346 | 273.7 / 285.5 | 269.7 | 0.6 | 273.7 | l0-full-b |
| english | cpu | distinct-ml-q10 | 10 | 3229 | 2347.1 / 2350.9 | 2343.3 | 3.9 | 234.7 | l0-full-b |
| english | cpu | edge-head-overflow | 1 | 202 | 177.8 / 179.1 | 176.8 | 1.8 | 177.8 | l0-full |
| english | cpu | edge-long-state | 2 | 1024 | 728.0 / 729.0 | 725.1 | 4.0 | 364.0 | l0-full |
| english | cpu | nonlatin-mixed-q6 | 6 | 2327 | 1591.4 / 1592.4 | 1589.2 | 2.2 | 265.2 | l0-full |
| english | cpu | published-latency-q1 | 1 | 194 | 176.5 / 177.1 | 176.4 | 0.4 | 176.5 | l0-full |
| english | cpu | published-latency-q10 | 10 | 1995 | 1304.9 / 1311.0 | 1303.1 | 2.9 | 130.5 | l0-full |
| english | cpu | published-latency-q5 | 5 | 992 | 662.7 / 663.7 | 661.4 | 1.5 | 132.5 | l0-full |
| english | cpu | published-latency-q50 | 50 | 9975 | 6265.2 / 6269.7 | 6260.2 | 13.4 | 125.3 | l0-full |
| english | cpu | ragged-q12 | 12 | 5914 | 4133.7 / 4136.0 | 4124.1 | 7.4 | 344.5 | l0-full |
| english | mps | bucket-choice-2 | 1 | 30 | 22.6 / 22.8 | 21.2 | 0.1 | 22.6 | l0-full |
| english | mps | bucket-choice-20 | 1 | 102 | 36.4 / 36.5 | 34.8 | 0.6 | 36.4 | l0-full |
| english | mps | distinct-en-q1 | 1 | 318 | 83.8 / 83.9 | 82.1 | 0.6 | 83.8 | l0-full |
| english | mps | distinct-en-q10 | 10 | 3100 | 906.9 / 907.4 | 901.9 | 4.1 | 90.7 | l0-full |
| english | mps | distinct-en-q21 | 21 | 6466 | 1921.0 / 1921.7 | 1912.5 | 9.3 | 91.5 | l0-full |
| english | mps | distinct-en-q50 | 50 | 15127 | 4579.6 / 4581.3 | 4562.0 | 19.2 | 91.6 | l0-full |
| english | mps | distinct-ml-q1 | 1 | 346 | 94.9 / 95.0 | 93.2 | 0.6 | 94.9 | l0-full |
| english | mps | distinct-ml-q10 | 10 | 3229 | 906.7 / 907.0 | 901.7 | 3.9 | 90.7 | l0-full |
| english | mps | edge-head-overflow | 1 | 202 | 60.6 / 60.7 | 58.3 | 1.8 | 60.6 | l0-full |
| english | mps | edge-long-state | 2 | 1024 | 267.7 / 268.0 | 263.0 | 4.1 | 133.9 | l0-full |
| english | mps | nonlatin-mixed-q6 | 6 | 2327 | 613.4 / 613.8 | 609.7 | 2.2 | 102.2 | l0-full |
| english | mps | published-latency-q1 | 1 | 194 | 59.4 / 63.7 | 57.8 | 0.4 | 59.4 | l0-full |
| english | mps | published-latency-q10 | 10 | 1995 | 512.0 / 512.9 | 507.3 | 2.9 | 51.2 | l0-full |
| english | mps | published-latency-q5 | 5 | 992 | 260.6 / 261.9 | 258.0 | 1.5 | 52.1 | l0-full-b |
| english | mps | published-latency-q50 | 50 | 9975 | 2540.8 / 2545.5 | 2528.0 | 15.2 | 50.8 | l0-full |
| english | mps | ragged-q12 | 12 | 5914 | 1578.7 / 1579.7 | 1570.8 | 7.4 | 131.6 | l0-full |
| multilingual | cpu | bucket-choice-2 | 1 | 30 | 21.8 / 22.0 | 21.6 | 0.1 | 21.8 | l0-full |
| multilingual | cpu | bucket-choice-20 | 1 | 112 | 45.0 / 45.4 | 44.6 | 0.5 | 45.0 | l0-full |
| multilingual | cpu | distinct-en-q1 | 1 | 331 | 101.0 / 102.4 | 100.7 | 0.5 | 101.0 | l0-full-b |
| multilingual | cpu | distinct-en-q10 | 10 | 3289 | 955.5 / 956.4 | 952.9 | 3.1 | 95.6 | l0-full |
| multilingual | cpu | distinct-en-q21 | 21 | 6876 | 2015.5 / 2018.6 | 2003.9 | 6.4 | 96.0 | l0-full-b |
| multilingual | cpu | distinct-en-q50 | 50 | 16130 | 4741.6 / 4775.4 | 4762.3 | 14.0 | 94.8 | l0-full |
| multilingual | cpu | distinct-ml-q1 | 1 | 291 | 91.6 / 93.5 | 91.3 | 0.4 | 91.6 | l0-full |
| multilingual | cpu | distinct-ml-q10 | 10 | 2824 | 823.3 / 828.6 | 764.0 | 2.9 | 82.3 | l0-full |
| multilingual | cpu | edge-head-overflow | 1 | 266 | 83.9 / 84.9 | 83.1 | 1.4 | 83.9 | l0-full |
| multilingual | cpu | edge-long-state | 2 | 2048 | 687.8 / 689.1 | 689.8 | 3.0 | 343.9 | l0-full |
| multilingual | cpu | nonlatin-mixed-q6 | 6 | 1380 | 356.4 / 386.7 | 387.6 | 1.4 | 59.4 | l0-full |
| multilingual | cpu | published-latency-q1 | 1 | 181 | 60.4 / 60.6 | 60.1 | 0.3 | 60.4 | l0-full-b |
| multilingual | cpu | published-latency-q10 | 10 | 1870 | 482.4 / 483.3 | 480.3 | 2.2 | 48.2 | l0-full |
| multilingual | cpu | published-latency-q5 | 5 | 929 | 244.5 / 245.4 | 243.4 | 1.2 | 48.9 | l0-full-b |
| multilingual | cpu | published-latency-q50 | 50 | 9350 | 2211.6 / 2213.9 | 2202.5 | 10.5 | 44.2 | l0-full |
| multilingual | cpu | ragged-q12 | 12 | 6812 | 2120.0 / 2127.3 | 2118.5 | 5.8 | 176.7 | l0-full |
| multilingual | mps | bucket-choice-2 | 1 | 30 | 12.3 / 12.5 | 11.1 | 0.1 | 12.3 | l0-full |
| multilingual | mps | bucket-choice-20 | 1 | 112 | 18.0 / 18.2 | 16.5 | 0.5 | 18.0 | l0-full |
| multilingual | mps | distinct-en-q1 | 1 | 331 | 39.5 / 39.6 | 37.9 | 0.4 | 39.5 | l0-full |
| multilingual | mps | distinct-en-q10 | 10 | 3289 | 378.8 / 379.3 | 374.6 | 3.1 | 37.9 | l0-full |
| multilingual | mps | distinct-en-q21 | 21 | 6876 | 802.1 / 802.7 | 793.5 | 6.6 | 38.2 | l0-full |
| multilingual | mps | distinct-en-q50 | 50 | 16130 | 2016.6 / 2025.6 | 1910.5 | 17.2 | 40.3 | l0-full |
| multilingual | mps | distinct-ml-q1 | 1 | 291 | 34.0 / 34.1 | 32.5 | 0.4 | 34.0 | l0-full |
| multilingual | mps | distinct-ml-q10 | 10 | 2824 | 296.7 / 297.0 | 292.4 | 2.9 | 29.7 | l0-full |
| multilingual | mps | edge-head-overflow | 1 | 266 | 31.8 / 31.9 | 29.8 | 1.4 | 31.8 | l0-full |
| multilingual | mps | edge-long-state | 2 | 2048 | 238.9 / 239.2 | 235.0 | 3.0 | 119.4 | l0-full |
| multilingual | mps | nonlatin-mixed-q6 | 6 | 1380 | 139.0 / 139.1 | 136.6 | 1.4 | 23.2 | l0-full |
| multilingual | mps | published-latency-q1 | 1 | 181 | 22.9 / 23.0 | 21.6 | 0.3 | 22.9 | l0-full |
| multilingual | mps | published-latency-q10 | 10 | 1870 | 190.2 / 190.3 | 186.5 | 2.2 | 19.0 | l0-full |
| multilingual | mps | published-latency-q5 | 5 | 929 | 97.4 / 97.6 | 95.1 | 1.2 | 19.5 | l0-full |
| multilingual | mps | published-latency-q50 | 50 | 9350 | 943.1 / 943.6 | 932.1 | 10.5 | 18.9 | l0-full |
| multilingual | mps | ragged-q12 | 12 | 6812 | 799.6 / 807.7 | 793.3 | 5.9 | 66.6 | l0-full |
| typed-decisions | cpu | bucket-choice-2 | 1 | 30 | 59.7 / 61.0 | 59.6 | 0.2 | 59.7 | l0-full-b |
| typed-decisions | cpu | bucket-choice-20 | 1 | 102 | 117.2 / 117.8 | 116.8 | 0.6 | 117.2 | l0-full-b |
| typed-decisions | cpu | distinct-en-q1 | 1 | 318 | 248.6 / 249.0 | 248.1 | 0.7 | 248.6 | l0-full |
| typed-decisions | cpu | distinct-en-q10 | 10 | 3100 | 2342.3 / 2346.2 | 2338.3 | 4.1 | 234.2 | l0-full |
| typed-decisions | cpu | distinct-en-q21 | 21 | 6466 | 4843.4 / 4867.0 | 4833.2 | 8.6 | 230.6 | l0-full |
| typed-decisions | cpu | distinct-en-q50 | 50 | 15127 | 11527.3 / 11549.9 | 11496.7 | 18.8 | 230.5 | l0-full |
| typed-decisions | cpu | distinct-ml-q1 | 1 | 346 | 269.4 / 271.4 | 272.3 | 0.6 | 269.4 | l0-full |
| typed-decisions | cpu | distinct-ml-q10 | 10 | 3229 | 2340.7 / 2344.9 | 2338.2 | 4.0 | 234.1 | l0-full |
| typed-decisions | cpu | edge-head-overflow | 1 | 266 | 221.7 / 222.3 | 220.7 | 1.9 | 221.7 | l0-full-b |
| typed-decisions | cpu | edge-long-state | 2 | 2048 | 1597.4 / 1601.7 | 1595.8 | 4.3 | 798.7 | l0-full-b |
| typed-decisions | cpu | nonlatin-mixed-q6 | 6 | 2327 | 1634.7 / 1639.2 | 1599.8 | 2.2 | 272.5 | l0-full |
| typed-decisions | cpu | published-latency-q1 | 1 | 194 | 176.9 / 177.9 | 176.6 | 0.4 | 176.9 | l0-full |
| typed-decisions | cpu | published-latency-q10 | 10 | 1995 | 1304.2 / 1306.7 | 1301.9 | 2.8 | 130.4 | l0-full |
| typed-decisions | cpu | published-latency-q5 | 5 | 992 | 660.9 / 661.9 | 659.5 | 1.5 | 132.2 | l0-full |
| typed-decisions | cpu | published-latency-q50 | 50 | 9975 | 6265.3 / 6282.6 | 6263.6 | 13.4 | 125.3 | l0-full |
| typed-decisions | cpu | ragged-q12 | 12 | 5957 | 4587.8 / 4591.8 | 4551.8 | 7.4 | 382.3 | l0-full-b |
| typed-decisions | mps | bucket-choice-2 | 1 | 30 | 22.6 / 22.7 | 21.1 | 0.1 | 22.6 | l0-full |
| typed-decisions | mps | bucket-choice-20 | 1 | 102 | 36.3 / 36.5 | 34.7 | 0.6 | 36.3 | l0-full |
| typed-decisions | mps | distinct-en-q1 | 1 | 318 | 83.9 / 84.0 | 82.1 | 0.6 | 83.9 | l0-full |
| typed-decisions | mps | distinct-en-q10 | 10 | 3100 | 906.7 / 907.1 | 901.8 | 4.1 | 90.7 | l0-full |
| typed-decisions | mps | distinct-en-q21 | 21 | 6466 | 1921.0 / 1922.4 | 1912.3 | 9.7 | 91.5 | l0-full |
| typed-decisions | mps | distinct-en-q50 | 50 | 15127 | 4576.9 / 4578.5 | 4559.4 | 20.1 | 91.5 | l0-full |
| typed-decisions | mps | distinct-ml-q1 | 1 | 346 | 94.9 / 95.0 | 93.2 | 0.6 | 94.9 | l0-full |
| typed-decisions | mps | distinct-ml-q10 | 10 | 3229 | 906.6 / 907.3 | 901.6 | 3.9 | 90.7 | l0-full |
| typed-decisions | mps | edge-head-overflow | 1 | 266 | 78.0 / 78.1 | 75.9 | 1.8 | 78.0 | l0-full |
| typed-decisions | mps | edge-long-state | 2 | 2048 | 568.8 / 569.3 | 564.3 | 4.2 | 284.4 | l0-full |
| typed-decisions | mps | nonlatin-mixed-q6 | 6 | 2327 | 613.4 / 613.8 | 609.6 | 2.2 | 102.2 | l0-full |
| typed-decisions | mps | published-latency-q1 | 1 | 194 | 59.3 / 59.4 | 57.8 | 0.4 | 59.3 | l0-full |
| typed-decisions | mps | published-latency-q10 | 10 | 1995 | 512.0 / 512.5 | 507.8 | 2.9 | 51.2 | l0-full |
| typed-decisions | mps | published-latency-q5 | 5 | 992 | 260.3 / 260.8 | 257.8 | 1.5 | 52.1 | l0-full |
| typed-decisions | mps | published-latency-q50 | 50 | 9975 | 2542.0 / 2545.9 | 2527.3 | 14.7 | 50.8 | l0-full |
| typed-decisions | mps | ragged-q12 | 12 | 5957 | 1729.5 / 1730.4 | 1721.8 | 7.5 | 144.1 | l0-full |
