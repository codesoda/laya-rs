//! `laya-core`: a Python-free runtime for the open Laya decision models.
//!
//! The crate reproduces the pinned upstream `laya` runtime
//! (`NandhaKishorM/laya` at `6a58191`) end to end: request serialization,
//! tokenization and budget truncation, the complete ModernBERT/mmBERT
//! network with its decision heads, temperature calibration and the native
//! four-decimal answer JSON. Parity with the frozen Python goldens under
//! `benchmarks/goldens/` is enforced by `tests/parity.rs` on every enabled
//! backend.
//!
//! Backends are cargo features: `mlx` (Apple Metal through mlx-rs) and
//! `candle` (portable CPU). The library never reads environment variables or
//! hidden configuration; callers pass model directories and cache paths.

pub mod assets;
pub mod backend;
mod error;
pub mod manifest;
pub mod postprocess;
pub mod preprocess;
pub mod pyjson;
pub mod request;
pub mod runtime;

pub use assets::{AgentConfig, ModelDir, TokenizerConfig, Verification};
pub use backend::{Backend, BackendSpec, DeviceKind, ModelOutput, Precision};
pub use error::{LayaError, Result};
pub use manifest::{PinnedFile, Profile, ProfileManifest, HUB_REPO, HUB_REVISION};
pub use postprocess::{Evaluation, QuestionResult, NATIVE_MODEL_NAME};
pub use preprocess::{Batch, LayaTokenizer, PreparedRequest, PreparedSequence};
pub use request::{Question, QuestionType, Request};
pub use runtime::{LoadOptions, Runtime, RuntimeInfo};

/// Upstream source revision this runtime reproduces.
pub const UPSTREAM_LAYA_SHA: &str = "6a5819129eb220570792e417e49723d697efd76f";
