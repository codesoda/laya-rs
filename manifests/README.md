# Source manifest

`sources.json` records immutable upstream revisions, exact package pins, model-file sizes and SHA-256 digests. It is generated and updated by the pinned Python baseline fetcher. Model weights and cache files are deliberately excluded from Git.

From `baseline/`:

```sh
uv sync
uv run laya-fetch --profile all
uv run laya-fetch --verify
```

The fetcher requests each required Hugging Face file individually at the recorded revision. It does not use an unpinned snapshot download. It stages Hub objects inside the project cache and installs only the model config, encoder config, tokenizer files, weights, and root model card. Existing files are skipped only after size and SHA-256 verification.

Upstream Laya rewrites `tokenizer/tokenizer_config.json` during `Agent` construction. The fetcher preserves the original as `tokenizer/tokenizer_config.upstream.json`; `laya-smoke` records both hashes and a key-level/unified diff under each profile's `tokenizer_config_patch` entry. Re-run `uv run laya-fetch --verify` after smoke runs to validate the current installed form.
