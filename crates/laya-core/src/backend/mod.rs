//! Inference backends. Each one executes the complete Laya network
//! (encoder, type embedding, head layers, scorer, action head) on one
//! device and returns float32 logits. Calibration, softmax and rounding are
//! done once, on the host, in [`crate::postprocess`].

use std::fmt;

use crate::{assets::ModelDir, preprocess::Batch, Result};

#[cfg(feature = "candle")]
pub mod candle;
pub mod config;
pub mod inventory;
#[cfg(feature = "mlx")]
pub mod mlx;

/// Weight/activation precision. `F32` is the parity-gated configuration;
/// `F16` is a separately measured configuration and must never be reported
/// as the fp32 result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Precision {
    F32,
    F16,
}

impl Precision {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }

    /// Name of the frozen tolerance profile in `benchmarks/goldens/tolerances.json`.
    #[must_use]
    pub const fn tolerance_profile(self, device: DeviceKind) -> &'static str {
        match (self, device) {
            (Self::F32, DeviceKind::Cpu) => "rust_cpu_fp32",
            (Self::F32, DeviceKind::Metal) => "rust_metal_fp32",
            (Self::F16, _) => "rust_metal_fp16_or_bf16",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Metal,
}

/// Which backend to construct. Nothing here is guessed from the
/// environment; a caller states the device and the paths it needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendSpec {
    /// MLX Metal GPU (macOS 14+, Apple Silicon). The embedded residual
    /// metallib is extracted under `metallib_cache_dir/<sha256>/`.
    Mlx {
        precision: Precision,
        metallib_cache_dir: std::path::PathBuf,
    },
    /// Candle on the CPU, fp32 only.
    CandleCpu,
}

impl BackendSpec {
    #[must_use]
    pub const fn device(&self) -> DeviceKind {
        match self {
            Self::Mlx { .. } => DeviceKind::Metal,
            Self::CandleCpu => DeviceKind::Cpu,
        }
    }

    #[must_use]
    pub const fn precision(&self) -> Precision {
        match self {
            Self::Mlx { precision, .. } => *precision,
            Self::CandleCpu => Precision::F32,
        }
    }

    /// Whether this build compiled the code for this backend.
    #[must_use]
    pub const fn compiled(&self) -> bool {
        match self {
            Self::Mlx { .. } => cfg!(feature = "mlx"),
            Self::CandleCpu => cfg!(feature = "candle"),
        }
    }

    /// Short label such as `mlx-metal-f32` or `candle-cpu-f32`.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Mlx { precision, .. } => format!("mlx-metal-{}", precision.as_str()),
            Self::CandleCpu => "candle-cpu-f32".to_owned(),
        }
    }
}

impl fmt::Display for BackendSpec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.label())
    }
}

/// Raw network outputs for one batch, float32, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelOutput {
    pub rows: usize,
    pub marker_slots: usize,
    pub n_act: usize,
    /// `rows × marker_slots` scorer logits before `-1e4` masking.
    pub raw_logits: Vec<f32>,
    /// `rows × n_act` action-head logits.
    pub act_logits: Vec<f32>,
    /// `rows × hidden` first-token hidden state after the head layers;
    /// diagnostic only.
    pub hidden_cls: Vec<f32>,
}

/// A loaded network on one device.
pub trait Backend: Send {
    /// Label such as `mlx-metal-f32`.
    fn label(&self) -> String;
    fn device(&self) -> DeviceKind;
    fn precision(&self) -> Precision;
    /// One synchronized forward pass. Every returned value is materialized
    /// on the host before this returns.
    fn forward(&mut self, batch: &Batch) -> Result<ModelOutput>;
}

/// Construct the backend described by `spec` from a verified model
/// directory. Fails explicitly when the feature is not compiled in.
pub fn load_backend(spec: &BackendSpec, model_dir: &ModelDir) -> Result<Box<dyn Backend>> {
    match spec {
        #[cfg(feature = "mlx")]
        BackendSpec::Mlx {
            precision,
            metallib_cache_dir,
        } => Ok(Box::new(mlx::MlxBackend::load(
            model_dir,
            *precision,
            metallib_cache_dir,
        )?)),
        #[cfg(feature = "candle")]
        BackendSpec::CandleCpu => Ok(Box::new(candle::CandleCpuBackend::load(model_dir)?)),
        #[allow(unreachable_patterns)]
        other => {
            let _ = model_dir;
            Err(crate::LayaError::Unavailable(format!(
            "backend {other} is not compiled into this build of laya-core (enable the `{}` feature)",
            match other {
                BackendSpec::Mlx { .. } => "mlx",
                BackendSpec::CandleCpu => "candle",
            }
        )))
        }
    }
}
