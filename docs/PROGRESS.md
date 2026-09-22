# Progress

Contract: [docs/plans/initial-build.md](plans/initial-build.md). Lead: Fable. Workers: Sol (`openai-codex/gpt-5.6-sol`), Luna (`openai-codex/gpt-5.6-luna`).

Host for all local measurements: Apple M3 Pro (11 CPU cores, 14 GPU cores, Metal 4), 18 GB unified memory, macOS 26.2 (25C56), low power mode off.

## Roadmap (per [amendment 2026-09-20](plans/amendment-2026-09-20-roadmap.md))

| Step | Gate | Status | Notes |
| --- | --- | --- | --- |
| 1 | L0 — baseline + eval set (Python reference) | baseline complete | eval set pending (non-gating); L1 may start |
| 2 | L1 — Metal/CPU feasibility spike | complete — correctness GO on CPU + Metal (Candle and mlx-rs), tokenizer exact; backend decision pending | docs/L1-SPIKE.md, docs/L1-MLX-SPIKE.md, docs/L1-TOKENIZER-SPIKE.md |
| 3 | L2 + L3 — basic implementation (core, CLI, HTTP, SDK compat) | L2 core complete — `crates/laya-core` with MLX (Metal) and Candle (CPU) backends, parity PASS on all profiles for both; CLI/HTTP/SDK compat live in the host (SystemOne) | docs/L2-CORE.md |
| 4 | L4a — benchmark basic implementation; first release (L5) | not started | |
| 5 | L4b — quality/architecture improvements; first Rust eval run | not started | |
| 6 | Improvement loop: perf / quality / recalibration / fine-tuning track | not started | each iteration re-runs parity + eval + benchmark |

## L0 checklist

- [x] Pinned upstream sources fetched; runtime equality between reviewed revisions re-verified (`git diff 6a58191..28d43ad -- laya/` is empty — verified by Fable)
- [x] Locked Python environment (`benchmarks/baseline/uv.lock`)
- [x] Pinned hub assets fetched into project cache; `manifests/sources.json` with SHA-256, sizes, licenses
- [x] Bundle-vs-standalone checkpoint parity recorded
- [x] Host/environment capture (`uv run laya-env`)
- [x] CPU and MPS load/smoke for english, multilingual, typed-decisions with explicit fallback detection — all six succeed without fallback
- [x] Preprocessing + tensor + logits + native-output goldens (CPU FP32 and MPS) for all profiles, 29 fixtures each
- [x] MPS-vs-CPU reproducibility deltas → frozen parity tolerances (`benchmarks/goldens/tolerances.json`)
- [x] Published latency harness reconstructed; distinct/ragged workloads defined (`benchmarks/fixtures/requests/`)
- [x] Harness pilot (contended, harness validation only)
- [x] Idle-host baseline measurements (CPU, MPS) with raw samples — `benchmarks/results/l0-full{,-b}`, merged in `l0-baseline-merged.json`
- [x] Idle-host cold start / load measurements
- [ ] Eval set: pinned held-out labelled subset with licences, split held-out/train-candidate, Python reference accuracy per profile (does not gate L1–L3)
- [x] Benchmark manifest freezing primary acceptance (`benchmarks/manifest.json`: multilingual distinct Q=1/Q=10, Rust Metal vs Python MPS, fp32)
- [x] Pinned SDK request/response/error fixtures (`compat/`, `schemas/`)
- [x] `docs/RESULTS.md`, `docs/COMPAT.md` written; milestone committed and pushed

### L0 decisions recorded by Fable

- Python `head_max_len` is a soft budget (upstream re-truncates rather than erroring); Rust must reproduce this exactly. Only sequences whose options cannot fit `max_len` error.
- Single-option choice is unsupported upstream (`RuntimeError` from `topk(2)`). Native mode will return an explicit unsupported-capability error. Jev-adapter behaviour is decided at L3 with client evidence (`docs/COMPAT.md` open question 1).
- Upstream `_fix_tokenizer_config` made no change to bundled configs; Rust loads the pristine `tokenizer_config.json` and must still treat a list-valued `extra_special_tokens` as loadable if a standalone-repo layout is ever pointed at.
- The primary performance acceptance is frozen in `benchmarks/manifest.json` before any Rust code exists. Amendments require a dated entry.

### Remaining L0 workload estimate

Idle-host baseline: ~30 min per profile on MPS and ~70 min per profile on CPU for `published+distinct+extended` (upper bounds from the contended pilot), plus 3 cold-start iterations per profile/device. Total ≈ 5 h of wall time if all three profiles are run on both devices; the multilingual and english passes are the priority and can be run first (~3.5 h).

## Architecture facts verified by Fable from pinned sources (for L1)

- `laya/` runtime identical between `6a58191` (main) and `28d43ad` (research).
- Encoder configs (`encoder/config.json`): ModernBERT `local_attention` 128, `global_attn_every_n_layers` 3, `norm_bias` false, `hidden_activation` gelu, `attention_bias`/`mlp_bias` false. English: 28 layers, d=1024, 16 heads, FFN 2624, RoPE theta 160000 (global) / 10000 (local), vocab 50368, pad 50283, cls 50281, sep 50282. Multilingual (mmBERT-base): 22 layers, d=768, 12 heads, FFN 1152, RoPE theta 160000 for both layer types, vocab 256000, pad 0.
- Head (`common.py::DecisionModel`): `nn.TransformerEncoderLayer(d, d//64, 4d, dropout=0.1, batch_first=True, norm_first=True)` — activation is PyTorch's default **ReLU**; scorer `LayerNorm → Linear(d,d) → GELU → Linear(d,1)`; `act_head Linear(d+4,256) → GELU → Linear(256,n_act)` with `n_act = len(act_costs)+1 = 2`. Padded option logits masked with `-1e4`. Entropy features use `k = marker_mask.sum().clamp(min=2)`, `p.topk(2)`.
- `rl_agent_config.json`: english `max_len` 512 / `head_max_len` 192, temps `[1.6369, 1.2514, 1.9834]` + per-bucket map; multilingual 1024/256, temps all `1.0`, empty bucket map; typed-decisions 1024/256, temps `[1.0148, 1.0374, 1.0575]` + same bucket map as english. `amp_dtype` is `bf16` in all configs but CPU/MPS run FP32 with autocast disabled (`agent.py`).
- Native output: `{"model": "laya-rl-agent", "answers": ..., "usage": {"input_tokens": sum(attention_mask), "output_tokens": 0}}`; four-decimal rounding; noul true probability is index 1; `action.act_probability` = softmax(act)[0].
- `_to_internal`: choice list criteria → `{c: None}`; non-string instructions → `json.dumps(ins)` (ASCII-escaped); state → `json.dumps(state, ensure_ascii=False)`; criterion structured values → `json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)`.
- `_fix_tokenizer_config` rewrites `tokenizer/tokenizer_config.json` in the model directory at load time.
