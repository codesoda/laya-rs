//! A loaded profile: tokenizer, calibration constants and one backend.

use std::{path::PathBuf, time::Instant};

use serde_json::Value;

use crate::{
    assets::{AgentConfig, ModelDir, Verification},
    backend::{load_backend, Backend, BackendSpec, DeviceKind, Precision},
    manifest::Profile,
    postprocess::{postprocess, Evaluation},
    preprocess::{prepare, LayaTokenizer, PreparedRequest},
    request::{Question, Request},
    Result,
};

/// Everything needed to load one profile. Nothing is read from the
/// environment or from hidden configuration files.
#[derive(Clone, Debug)]
pub struct LoadOptions {
    pub profile: Profile,
    /// Directory holding the pinned profile files.
    pub model_dir: PathBuf,
    pub backend: BackendSpec,
    pub verification: Verification,
}

/// Identity and budgets of a loaded runtime, for capability reporting.
#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeInfo {
    pub profile: Profile,
    pub backend: String,
    pub device: DeviceKind,
    pub precision: Precision,
    /// SHA-256 of `model.safetensors`.
    pub weights_sha256: String,
    pub max_len: usize,
    pub head_max_len: usize,
    pub load_ms: f64,
    /// Milliseconds of the warm-up forward pass, if one ran.
    pub warmup_ms: Option<f64>,
}

pub struct Runtime {
    tokenizer: LayaTokenizer,
    agent: AgentConfig,
    backend: Box<dyn Backend>,
    info: RuntimeInfo,
}

impl Runtime {
    /// Verify the model directory, read its constants and load the backend.
    pub fn load(options: &LoadOptions) -> Result<Self> {
        let started = Instant::now();
        let model_dir = ModelDir::open(options.profile, &options.model_dir, options.verification)?;
        let agent = model_dir.agent_config()?;
        let tokenizer_config = model_dir.tokenizer_config()?;
        let tokenizer = LayaTokenizer::from_files(&model_dir.tokenizer_path(), &tokenizer_config)?;
        let backend = load_backend(&options.backend, &model_dir)?;
        let info = RuntimeInfo {
            profile: options.profile,
            backend: backend.label(),
            device: backend.device(),
            precision: backend.precision(),
            weights_sha256: model_dir.manifest().weights_sha256().to_owned(),
            max_len: agent.max_len,
            head_max_len: agent.head_max_len,
            load_ms: started.elapsed().as_secs_f64() * 1000.0,
            warmup_ms: None,
        };
        Ok(Self {
            tokenizer,
            agent,
            backend,
            info,
        })
    }

    #[must_use]
    pub const fn info(&self) -> &RuntimeInfo {
        &self.info
    }

    #[must_use]
    pub const fn agent_config(&self) -> &AgentConfig {
        &self.agent
    }

    #[must_use]
    pub const fn tokenizer(&self) -> &LayaTokenizer {
        &self.tokenizer
    }

    /// Preprocess only (no inference).
    pub fn prepare(&self, request: &Request) -> Result<PreparedRequest> {
        prepare(
            &self.tokenizer,
            request,
            self.agent.max_len,
            self.agent.head_max_len,
        )
    }

    /// Preprocess, run one batched forward pass, postprocess.
    pub fn evaluate(&mut self, request: &Request) -> Result<Evaluation> {
        let prepared = self.prepare(request)?;
        self.evaluate_prepared(request, &prepared)
    }

    /// Run inference for an already prepared request.
    pub fn evaluate_prepared(
        &mut self,
        request: &Request,
        prepared: &PreparedRequest,
    ) -> Result<Evaluation> {
        let output = self.backend.forward(&prepared.batch)?;
        postprocess(
            request,
            prepared,
            &output,
            &self.agent,
            &self.backend.label(),
        )
    }

    /// Run one small request covering all three question types so kernel
    /// compilation (Metal JIT, once per machine) and lazy allocation happen
    /// before the first caller. Returns the elapsed milliseconds and records
    /// them in [`RuntimeInfo::warmup_ms`].
    pub fn warmup(&mut self) -> Result<f64> {
        let request = warmup_request();
        let started = Instant::now();
        self.evaluate(&request)?;
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        self.info.warmup_ms = Some(elapsed);
        Ok(elapsed)
    }
}

/// A fixed synthetic request with one question of each type.
#[must_use]
pub fn warmup_request() -> Request {
    Request {
        state: Value::String("laya-rs resident warm-up".to_owned()),
        questions: vec![
            (
                "choice".to_owned(),
                Question::Choice {
                    instructions: "Select the first option.".to_owned(),
                    criteria: vec![
                        ("ready".to_owned(), Value::Null),
                        ("not-ready".to_owned(), Value::Null),
                    ],
                },
            ),
            (
                "score".to_owned(),
                Question::Score {
                    instructions: "How ready is this?".to_owned(),
                    levels: vec![
                        Value::from("low"),
                        Value::from("medium"),
                        Value::from("high"),
                    ],
                },
            ),
            (
                "noul".to_owned(),
                Question::Noul {
                    instructions: "Is the runtime warm?".to_owned(),
                    criteria: Value::Null,
                },
            ),
        ],
    }
}
