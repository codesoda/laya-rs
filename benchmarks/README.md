# L0 Python benchmarks

These commands run the pinned Python 3.12 environment (`laya==0.3.3`, PyTorch 2.14.0,
Transformers 5.17.0). Run them from the repository root; `uv run` is invoked from
`benchmarks/baseline/` so that the locked project environment is used.

## Pilot

The pilot is deliberately short and is only for estimating duration or checking the
harness. On an otherwise idle host, use:

```sh
cd benchmarks/baseline
~/.local/bin/uv run laya-bench --profile english --device cpu --workload all --reps 5 --warmup 2 --run-id l0-pilot
~/.local/bin/uv run laya-bench --profile english --device mps --workload all --reps 5 --warmup 2 --run-id l0-pilot
~/.local/bin/uv run laya-bench --profile multilingual --device cpu --workload all --reps 5 --warmup 2 --run-id l0-pilot
~/.local/bin/uv run laya-bench --profile multilingual --device mps --workload all --reps 5 --warmup 2 --run-id l0-pilot
```

When the host is shared, add `--contended "<reason>"`; contended samples must not be
reported as an uncontended speed claim. The requested development pilot was recorded
separately under `benchmarks/results/l0-pilot/`.

## Full L0 baseline

After the pilot, stop other builds, tests, inference workers, and background model
jobs. Run the sets serially for all three profiles and both devices:

```sh
cd benchmarks/baseline
for profile in english multilingual typed-decisions; do
  for device in cpu mps; do
    for set in published distinct extended; do
      ~/.local/bin/uv run laya-bench --profile "$profile" --device "$device" --workload "$set" --reps 20 --warmup 3 --run-id l0-full --label "idle-host L0 baseline"
    done
  done
done
```

The duration estimate is the sum of the observed pilot wall-clock duration scaled by
`20/5` and `3/2` for measured repetitions/warmups, with model-load time included. It is
an upper bound because the development pilot is contended and may have thermal and
other-worker interference; replace the estimate below after that pilot completes:
The contended development pilot took about 94.3 s for four multilingual MPS
fixtures and 57.8 s for one multilingual CPU fixture. Scaling the MPS figure to the
16 fixtures in `published + distinct + extended` and from 3/1 to 20/3 samples gives
about **29 minutes per device**; the conservative CPU extrapolation from its one
fixture is about **72 minutes per device**. These are upper bounds, not acceptance
measurements, because the pilot was contended and profile/load costs vary.

Do not compare any contended run to the Rust acceptance target. The mandatory steady
state is three warmups and twenty samples on an idle host, with CPU and MPS requested
explicitly. A requested MPS fallback is an error, not a CPU result.

## Results and reporting

Each warm run writes one immutable file:
`benchmarks/results/<run-id>/<profile>-<device>-<fixture-id>.json`, plus one
`run.json` containing environment, arguments, thread count, dtype, and interruption
state. Raw timestamped samples and host/thermal snapshots are retained. Cold runs use
`cold-<profile>-<device>.json` in the same directory. Results are never overwritten;
choose a new run id for a repeat.

```sh
cd benchmarks/baseline
~/.local/bin/uv run laya-bench-report --run-id l0-pilot --compare-published
```

The published T4 comparison is historical context only: it is different hardware and
is not comparable as a speedup. The four published fixture values come from the pinned
research result at `research/results/t4_colab_benchmark.json` (commit
`28d43add7e47ce502489c9433310d55276c64e0f`).
