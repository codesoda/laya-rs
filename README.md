# laya-rs

Planned Rust runtime for [Laya](https://github.com/NandhaKishorM/laya): typed local decisions, a lean JSON CLI, and a resident Jev-compatible HTTP server.

**Status: plan only. No implementation, benchmarks, or release binaries yet.**

## Build contract

Read [docs/plans/initial-build.md](docs/plans/initial-build.md).

- Pinned Python baselines and output goldens **before** the Rust inference port.
- Native **Apple M-series Metal** acceleration plus a portable CPU path.
- Inline JSON requests and `laya --serve`, loading model weights once.
- Verified `POST /v1/systemone` wire compatibility with pinned TypeSafe/AI SDK clients; not a claim of equal model behavior or calibration.
- Same-machine Python/Rust benchmarks before speedup claims.
- Precompiled GitHub binaries, verified first-run asset downloads, and offline cache reuse without Python.

A ready-to-paste prompt for a Fable-led session using Sol/Luna workers is in [docs/plans/implementation-prompt.md](docs/plans/implementation-prompt.md).

## License and affiliation

Planned project license: Apache-2.0. Upstream Laya code and model cards declare Apache-2.0; implementation must audit and preserve all applicable dependency, model and tokenizer notices.

This project is independent and is not affiliated with Convai Innovations or TypeSafe AI. Jev-compatible describes an intended API adapter, not TypeSafe weights, identical decisions, or equivalent calibration.
