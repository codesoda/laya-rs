# Frozen request fixtures

Authored by Fable at L0. Each `requests/*.json` has `id`, `source`, `state`, `questions` and optionally `expect_error`. Only `state` and `questions` are passed to the model; the other keys are metadata. Goldens (`baseline/`) and benchmarks (`benchmarks/`) must both read these files so Python and Rust are measured on byte-identical inputs. Do not edit a fixture after goldens or benchmark results referencing it have been committed; add a new id instead.

| Family | Ids | Purpose |
| --- | --- | --- |
| Published latency | `published-latency-q{1,5,10,50}` | Verbatim reconstruction of the upstream research harness §7 workload (one repeated English ticket, alternating 3-option choice / noul). Historical-comparison target only. |
| Primary distinct (English) | `distinct-en-q{1,10,21,50}` | Ten genuinely distinct questions (4 choice incl. 10-option and structured criteria, 2 score, 4 noul incl. descriptions) over a structured English ticket. |
| Primary distinct (multilingual) | `distinct-ml-q{1,10}` | Same schema family in es-MX with a chat-turn state. **These are the primary L4 acceptance workloads for the multilingual profile** (see `benchmarks/manifest.json`). |
| Non-Latin | `nonlatin-mixed-q6` | Hindi/Korean/Japanese/Arabic/Chinese/emoji state and questions. |
| Ragged | `ragged-q12` | Mixed types, option counts 2/3/5/10/20, long/short instructions, empty and JSON instructions, long state. |
| Serialization edges | `edge-serialization`, `edge-empty-state`, `edge-state-list` | Float/Unicode/escape/key-order state, description variants (`None`, `""`, `0`, `False`, dict, list), non-string instructions (ASCII escaping), literal mask tokens, list criteria. |
| Budget edges | `edge-long-state`, `edge-head-overflow`, `edge-options-exceed`, `edge-single-option` | State right-truncation; per-option re-truncation path (`opt_budget < 16`); options that cannot fit (`expect_error`); single-option choice (record actual upstream behaviour). |
| Temperature buckets | `bucket-choice-{2,3,5,6,10,11,20}`, `bucket-score-{2,6}` | Every `temp_bucket` boundary for choice and score. |

Distinct/ragged text is original to this repository (no dataset licensing). The published-latency text is copied from the Apache-2.0 upstream harness for reproduction fidelity.
