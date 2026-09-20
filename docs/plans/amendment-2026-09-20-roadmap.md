# Amendment 2026-09-20: merged roadmap, eval set, calibration and fine-tuning track

Agreed between the owner and Fable after L0. This amends the sequencing in
[initial-build.md](initial-build.md) §10; it does not change any correctness,
benchmark-discipline or distribution requirement.

## Merged flow

1. **L0 — baseline.** As specified, plus: build a labelled held-out **eval set** and run it once
   through the pinned Python baseline whenever the host is idle. The eval set does not gate L1–L3.
2. **L1 — Metal/CPU feasibility spike.** Bounded go/no-go on Candle Metal + CPU for the complete
   network before any wide surface area is written.
3. **L2 + L3 — basic implementation.** Core, inline CLI, JSONL, warm HTTP, SDK compatibility.
4. **L4a — benchmark the basic implementation** against the L0 baseline under the frozen
   manifest, then cut the **first release (L5)** so the no-Python distribution requirement is
   exercised early.
5. **L4b — quality/architecture improvements**, one measured change at a time, then the first
   **Rust eval run** compared to the Python reference score.
6. **Improvement loop.** Each iteration picks one of: performance review, quality review,
   **temperature recalibration** (held-out refit of the per-type/per-bucket temperatures —
   upstream evidence: ECE 0.47→0.08 english, 0.31→0.11 multilingual), and later a
   **fine-tuning track**. Every iteration re-runs parity goldens, the eval set and the benchmark
   manifest, and may cut a release when all are green.

## Eval set rules

- Labelled questions drawn from datasets upstream marked *held-out* (not its training mix), pinned
  by dataset revision with licences recorded; a few hundred items across choice/score/noul and
  both English and non-English scripts.
- Split into `held-out` (never trained on) and `train-candidate` from day one, so a later
  fine-tuning track cannot contaminate the score.
- Reported as *our measured accuracy on our subset*, separately from vendor-published quality.

## Calibration and fine-tuning are new versions, not port optimisations

- A recalibrated or fine-tuned checkpoint is a **distinct, labelled artifact**
  (e.g. `laya-rs-cal-<n>`, `laya-rs-ft-<n>`) with its own manifest entry, eval report and parity
  tolerances. It is never reported under the pinned upstream name and never becomes the default
  without an explicit decision.
- Native parity against the pinned upstream checkpoints remains a permanent gate for the port.
- Recalibration precedes weight fine-tuning because it is cheaper, reversible and evidenced.
- Training requires the owner's approval for any paid hardware; local M3 Pro runs are acceptable
  when they do not contend with measured benchmarks.
