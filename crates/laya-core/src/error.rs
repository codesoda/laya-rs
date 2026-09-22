use std::path::PathBuf;

use thiserror::Error;

/// Every failure the library can report. Variants map onto the upstream
/// Python behaviour where one exists; the message text is safe to show to a
/// caller (no request state, no file contents).
#[derive(Debug, Error)]
pub enum LayaError {
    /// The request is malformed: unknown type, missing required field, wrong
    /// criteria shape. Upstream raises `KeyError`/`TypeError` here.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// Upstream `ValueError("question %r options exceed head_max_len=%d")`:
    /// after budget re-truncation, option markers were cut by `max_len`.
    #[error("question {question:?} options exceed head_max_len={head_max_len}")]
    OptionsExceedBudget {
        question: String,
        head_max_len: usize,
    },

    /// A Choice with exactly one option. Upstream fails inside the action
    /// head (`RuntimeError: selected index k out of range`); this library
    /// refuses before inference instead of fabricating a second option.
    #[error(
        "question {question:?}: a choice with one option is unsupported by the Laya network (upstream: selected index k out of range)"
    )]
    SingleOptionChoice { question: String },

    /// A model asset is missing, unreadable or does not match its pinned
    /// size/SHA-256.
    #[error("model asset {path}: {message}")]
    Asset { path: PathBuf, message: String },

    /// A checkpoint or config file is structurally wrong for this network.
    #[error("checkpoint: {0}")]
    Checkpoint(String),

    /// The requested backend/device/precision cannot run in this build or on
    /// this machine. Never falls back.
    #[error("backend unavailable: {0}")]
    Unavailable(String),

    /// The backend failed while executing a forward pass.
    #[error("inference failed: {0}")]
    Inference(String),

    /// Tokenizer construction or encoding failed.
    #[error("tokenizer: {0}")]
    Tokenizer(String),
}

impl LayaError {
    /// Short stable code for structured error envelopes.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidRequest(_) => "invalid_request",
            Self::OptionsExceedBudget { .. } => "options_exceed_budget",
            Self::SingleOptionChoice { .. } => "single_option_choice",
            Self::Asset { .. } => "asset",
            Self::Checkpoint(_) => "checkpoint",
            Self::Unavailable(_) => "unavailable",
            Self::Inference(_) => "inference",
            Self::Tokenizer(_) => "tokenizer",
        }
    }

    /// The upstream Python exception class this error corresponds to, when
    /// there is one. Used by the parity harness for error fixtures.
    #[must_use]
    pub const fn upstream_class(&self) -> Option<&'static str> {
        match self {
            Self::OptionsExceedBudget { .. } => Some("ValueError"),
            Self::SingleOptionChoice { .. } => Some("RuntimeError"),
            Self::InvalidRequest(_) => Some("KeyError"),
            _ => None,
        }
    }

    pub(crate) fn asset(path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        Self::Asset {
            path: path.into(),
            message: message.into(),
        }
    }
}

pub type Result<T> = std::result::Result<T, LayaError>;
