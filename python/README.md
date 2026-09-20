# Pinned Laya Python baseline

This uv project reproduces the pinned upstream Laya runtime used as the laya-rs correctness and timing oracle. It requires uv and uses managed Python 3.12.

```sh
cd python
uv sync
uv run laya-fetch --profile all
uv run laya-fetch --verify
uv run laya-env
uv run laya-smoke --profile english --device cpu
uv run laya-smoke --profile english --device mps
```

Repeat the smoke command for `multilingual` and `typed-decisions`. Each smoke result is saved under `benchmarks/results/l0-smoke/` even when loading or inference fails.

The cache root is `$LAYA_HOME` when set, otherwise `<repository>/.cache/laya`. Hugging Face staging data is kept at `<cache>/hf-staging`; installed profiles are under `<cache>/hub/convaiinnovations--laya/c5d78730f3493e4fe16d61507ef4b78eef7318cf/`. Nothing under `.cache` or any model weight is committed.

All CLI stdout is JSON (or empty on failure). Progress and diagnostics go to stderr.
