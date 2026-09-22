//! MLX Metal backend (Apple Silicon). Lifted from `spikes/mlx-jit`
//! (`model.rs` and the metallib embedding), which passed the frozen fp32 and
//! fp16 tolerances for all three profiles (`docs/L1-MLX-SPIKE.md`,
//! `docs/L1-MLX-JIT.md`).
//!
//! The linked `libmlx` is a `MLX_METAL_JIT=ON` build; most kernels compile
//! from embedded source on first use, and the nine AOT kernel groups live in
//! the 1.4 MB residual `mlx.metallib` committed under `metal/`. That file is
//! written to `<metallib_cache_dir>/<sha256>/mlx.metallib` before MLX
//! initializes Metal. Nothing here reads environment variables.

use std::{
    collections::HashMap,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
    time::Instant,
};

use mlx_rs::{
    fast, metal, nn,
    ops::{self, indexing::IndexOp},
    transforms, Array, Device, DeviceType, Dtype,
};

use super::{config::EncoderConfig, Backend, DeviceKind, ModelOutput, Precision};
use crate::{
    assets::{sha256_bytes, AgentConfig, ModelDir},
    preprocess::Batch,
    LayaError, Result,
};

/// Residual metallib produced by the pinned mlx-sys 0.6.0 (MLX 0.32.2)
/// JIT build with `MACOSX_DEPLOYMENT_TARGET=14.0`.
pub const EMBEDDED_METALLIB: &[u8] = include_bytes!("../../metal/mlx.metallib");
/// SHA-256 of [`EMBEDDED_METALLIB`], also its cache directory name.
pub const EMBEDDED_METALLIB_SHA256: &str =
    "44eb25db5205fbfc2f5c81f59cce9bb8c3c534d0b6707c03e43b057a497e618b";

static METALLIB_PATH: OnceLock<PathBuf> = OnceLock::new();
static METAL_INIT: Mutex<()> = Mutex::new(());

fn inference(error: impl std::fmt::Display) -> LayaError {
    LayaError::Inference(error.to_string())
}

fn checkpoint(error: impl std::fmt::Display) -> LayaError {
    LayaError::Checkpoint(error.to_string())
}

/// Write the embedded residual metallib under `cache_dir` (if it is not
/// already there byte-for-byte) and point MLX at it. Idempotent for one
/// directory per process; a second directory is an error because MLX's
/// metallib path is process-global.
pub fn configure_metallib(cache_dir: &Path) -> Result<PathBuf> {
    let _guard = METAL_INIT
        .lock()
        .map_err(|_| LayaError::Unavailable("metal initialization lock poisoned".to_owned()))?;
    let directory = cache_dir.join(EMBEDDED_METALLIB_SHA256);
    let output = directory.join("mlx.metallib");
    if let Some(existing) = METALLIB_PATH.get() {
        if existing != &output {
            return Err(LayaError::Unavailable(format!(
                "MLX metallib already configured at {}; cannot switch to {}",
                existing.display(),
                output.display()
            )));
        }
        return Ok(existing.clone());
    }
    debug_assert_eq!(sha256_bytes(EMBEDDED_METALLIB), EMBEDDED_METALLIB_SHA256);
    if !fs::read(&output).is_ok_and(|bytes| bytes == EMBEDDED_METALLIB) {
        fs::create_dir_all(&directory).map_err(|error| {
            LayaError::Unavailable(format!(
                "cannot create metallib cache {}: {error}",
                directory.display()
            ))
        })?;
        let temporary = directory.join(format!("mlx.metallib.{}.tmp", std::process::id()));
        let write = || -> std::io::Result<()> {
            let mut file = File::create(&temporary)?;
            file.write_all(EMBEDDED_METALLIB)?;
            file.sync_all()?;
            fs::rename(&temporary, &output)
        };
        write().map_err(|error| {
            let _ = fs::remove_file(&temporary);
            LayaError::Unavailable(format!(
                "cannot install embedded MLX metallib at {}: {error}",
                output.display()
            ))
        })?;
    }
    metal::set_metallib_path(output.to_string_lossy().as_ref())
        .map_err(|error| LayaError::Unavailable(format!("select MLX metallib: {error}")))?;
    let _ = METALLIB_PATH.set(output.clone());
    Ok(output)
}

/// Require the MLX default device to be the Metal GPU and prove the kernel
/// library loads by evaluating one real operation. No CPU fallback.
pub fn require_metal_gpu() -> Result<String> {
    let device =
        Device::try_default().map_err(|error| LayaError::Unavailable(error.to_string()))?;
    let kind = device
        .get_type()
        .map_err(|error| LayaError::Unavailable(error.to_string()))?;
    if !matches!(kind, DeviceType::Gpu) {
        return Err(LayaError::Unavailable(format!(
            "MLX default device is {device}, not the Metal GPU; refusing silent CPU fallback"
        )));
    }
    let input = Array::from_slice(&[1.0_f32, 2.0], &[2]);
    let output = ops::add(&input, &input).map_err(|error| {
        LayaError::Unavailable(format!("enqueue Metal smoke operation: {error}"))
    })?;
    transforms::eval([&output])
        .map_err(|error| LayaError::Unavailable(format!("Metal kernel library: {error}")))?;
    if output
        .to_vec_cast::<f32>()
        .map_err(|error| LayaError::Unavailable(error.to_string()))?
        != [2.0, 4.0]
    {
        return Err(LayaError::Unavailable(
            "Metal smoke operation returned an unexpected result".to_owned(),
        ));
    }
    Ok(device.to_string())
}

struct Linear {
    weight: Array,
    bias: Option<Array>,
}

impl Linear {
    fn forward(&self, x: &Array) -> Result<Array> {
        let y = ops::matmul(x, self.weight.t()).map_err(inference)?;
        match &self.bias {
            Some(bias) => ops::add(&y, bias).map_err(inference),
            None => Ok(y),
        }
    }
}

struct Norm {
    weight: Array,
    bias: Option<Array>,
    eps: f32,
}

impl Norm {
    fn forward(&self, x: &Array) -> Result<Array> {
        fast::layer_norm(x, Some(&self.weight), self.bias.as_ref(), self.eps).map_err(inference)
    }
}

struct EncoderLayer {
    local: bool,
    rope_base: f32,
    attn_norm: Option<Norm>,
    qkv: Linear,
    out: Linear,
    mlp_norm: Norm,
    mlp_in: Linear,
    mlp_out: Linear,
}

struct HeadLayer {
    norm1: Norm,
    norm2: Norm,
    in_proj: Linear,
    out_proj: Linear,
    linear1: Linear,
    linear2: Linear,
}

struct Loader {
    values: HashMap<String, Array>,
    dtype: Dtype,
}

impl Loader {
    fn new(path: &Path, precision: Precision) -> Result<Self> {
        let values = Array::load_safetensors(path)
            .map_err(|error| LayaError::Checkpoint(format!("load {}: {error}", path.display())))?;
        Ok(Self {
            values,
            dtype: match precision {
                Precision::F32 => Dtype::Float32,
                Precision::F16 => Dtype::Float16,
            },
        })
    }

    fn take(&mut self, name: &str, shape: &[i32]) -> Result<Array> {
        let value = self
            .values
            .remove(name)
            .ok_or_else(|| LayaError::Checkpoint(format!("missing tensor {name}")))?;
        if value.shape() != shape {
            return Err(LayaError::Checkpoint(format!(
                "shape mismatch for {name}: expected {shape:?}, got {:?}",
                value.shape()
            )));
        }
        value.as_dtype(self.dtype).map_err(checkpoint)
    }

    fn linear(&mut self, prefix: &str, input: i32, output: i32, bias: bool) -> Result<Linear> {
        let (weight_name, bias_name) = if prefix.ends_with(".self_attn.in_proj") {
            (format!("{prefix}_weight"), format!("{prefix}_bias"))
        } else {
            (format!("{prefix}.weight"), format!("{prefix}.bias"))
        };
        Ok(Linear {
            weight: self.take(&weight_name, &[output, input])?,
            bias: bias.then(|| self.take(&bias_name, &[output])).transpose()?,
        })
    }

    fn norm(&mut self, prefix: &str, dims: i32, bias: bool, eps: f32) -> Result<Norm> {
        Ok(Norm {
            weight: self.take(&format!("{prefix}.weight"), &[dims])?,
            bias: bias
                .then(|| self.take(&format!("{prefix}.bias"), &[dims]))
                .transpose()?,
            eps,
        })
    }

    fn finish(mut self) -> Result<()> {
        // The registered calibration buffer is checked but not part of
        // forward; runtime temperatures come from rl_agent_config.json.
        let temperature = self
            .values
            .remove("temperature")
            .ok_or_else(|| LayaError::Checkpoint("missing tensor temperature".to_owned()))?;
        if temperature.shape() != [3] {
            return Err(LayaError::Checkpoint(format!(
                "shape mismatch for temperature: {:?}",
                temperature.shape()
            )));
        }
        if !self.values.is_empty() {
            let mut names: Vec<_> = self.values.into_keys().collect();
            names.sort();
            return Err(LayaError::Checkpoint(format!(
                "unexpected tensors: {}",
                names.join(", ")
            )));
        }
        Ok(())
    }
}

/// The complete Laya network on the MLX Metal GPU.
pub struct MlxBackend {
    cfg: EncoderConfig,
    precision: Precision,
    n_act: usize,
    device_name: String,
    embeddings: Array,
    embedding_norm: Norm,
    encoder_layers: Vec<EncoderLayer>,
    final_norm: Norm,
    type_embedding: Array,
    head_layers: Vec<HeadLayer>,
    scorer_norm: Norm,
    scorer_in: Linear,
    scorer_out: Linear,
    action_in: Linear,
    action_out: Linear,
    load_ms: f64,
}

impl MlxBackend {
    pub fn load(
        model_dir: &ModelDir,
        precision: Precision,
        metallib_cache_dir: &Path,
    ) -> Result<Self> {
        configure_metallib(metallib_cache_dir)?;
        let device_name = require_metal_gpu()?;
        let started = Instant::now();
        let cfg = EncoderConfig::load(&model_dir.encoder_config_path())?;
        let agent: AgentConfig = model_dir.agent_config()?;
        let d = cfg.hidden_size as i32;
        let i = cfg.intermediate_size as i32;
        let eps = cfg.norm_eps as f32;
        let mut loader = Loader::new(&model_dir.weights_path(), precision)?;
        let embeddings = loader.take(
            "encoder.embeddings.tok_embeddings.weight",
            &[cfg.vocab_size as i32, d],
        )?;
        let embedding_norm = loader.norm("encoder.embeddings.norm", d, false, eps)?;
        let mut encoder_layers = Vec::with_capacity(cfg.num_hidden_layers);
        for index in 0..cfg.num_hidden_layers {
            let prefix = format!("encoder.layers.{index}");
            encoder_layers.push(EncoderLayer {
                local: cfg.is_local(index),
                rope_base: cfg.rope_theta(index) as f32,
                attn_norm: (index != 0)
                    .then(|| loader.norm(&format!("{prefix}.attn_norm"), d, false, eps))
                    .transpose()?,
                qkv: loader.linear(&format!("{prefix}.attn.Wqkv"), d, 3 * d, false)?,
                out: loader.linear(&format!("{prefix}.attn.Wo"), d, d, false)?,
                mlp_norm: loader.norm(&format!("{prefix}.mlp_norm"), d, false, eps)?,
                mlp_in: loader.linear(&format!("{prefix}.mlp.Wi"), d, 2 * i, false)?,
                mlp_out: loader.linear(&format!("{prefix}.mlp.Wo"), i, d, false)?,
            });
        }
        let final_norm = loader.norm("encoder.final_norm", d, false, eps)?;
        let type_embedding = loader.take("type_emb.weight", &[3, d])?;
        let mut head_layers = Vec::with_capacity(agent.head_layers);
        for index in 0..agent.head_layers {
            let prefix = format!("head.layers.{index}");
            // nn.TransformerEncoderLayer defaults: biased norms, eps 1e-5.
            head_layers.push(HeadLayer {
                norm1: loader.norm(&format!("{prefix}.norm1"), d, true, 1e-5)?,
                norm2: loader.norm(&format!("{prefix}.norm2"), d, true, 1e-5)?,
                in_proj: loader.linear(&format!("{prefix}.self_attn.in_proj"), d, 3 * d, true)?,
                out_proj: loader.linear(&format!("{prefix}.self_attn.out_proj"), d, d, true)?,
                linear1: loader.linear(&format!("{prefix}.linear1"), d, 4 * d, true)?,
                linear2: loader.linear(&format!("{prefix}.linear2"), 4 * d, d, true)?,
            });
        }
        let n_act = agent.n_act();
        let scorer_norm = loader.norm("scorer.0", d, true, 1e-5)?;
        let scorer_in = loader.linear("scorer.1", d, d, true)?;
        let scorer_out = loader.linear("scorer.3", d, 1, true)?;
        let action_in = loader.linear("act_head.0", d + 4, 256, true)?;
        let action_out = loader.linear("act_head.2", 256, n_act as i32, true)?;
        loader.finish()?;
        let backend = Self {
            cfg,
            precision,
            n_act,
            device_name,
            embeddings,
            embedding_norm,
            encoder_layers,
            final_norm,
            type_embedding,
            head_layers,
            scorer_norm,
            scorer_in,
            scorer_out,
            action_in,
            action_out,
            load_ms: 0.0,
        };
        // Materialize the converted weights so load time is honest and the
        // first request does not pay for the dtype casts.
        let mut resident: Vec<&Array> = vec![&backend.embeddings, &backend.type_embedding];
        for layer in &backend.encoder_layers {
            resident.extend([
                &layer.qkv.weight,
                &layer.out.weight,
                &layer.mlp_in.weight,
                &layer.mlp_out.weight,
            ]);
        }
        transforms::eval(resident).map_err(checkpoint)?;
        Ok(Self {
            load_ms: started.elapsed().as_secs_f64() * 1000.0,
            ..backend
        })
    }

    #[must_use]
    pub fn load_ms(&self) -> f64 {
        self.load_ms
    }

    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    fn dtype(&self) -> Dtype {
        match self.precision {
            Precision::F32 => Dtype::Float32,
            Precision::F16 => Dtype::Float16,
        }
    }

    fn attention(
        &self,
        x: &Array,
        qkv: &Linear,
        out: &Linear,
        heads: i32,
        mask: &Array,
        rope_base: Option<f32>,
    ) -> Result<Array> {
        let b = x.dim(0);
        let length = x.dim(1);
        let d = x.dim(2);
        let head_dim = d / heads;
        let projected = qkv
            .forward(x)?
            .reshape(&[b, length, 3, heads, head_dim])
            .map_err(inference)?;
        let parts = projected.split_equal(3, 2).map_err(inference)?;
        let mut q = parts[0]
            .squeeze_axes(&[2])
            .map_err(inference)?
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(inference)?;
        let mut k = parts[1]
            .squeeze_axes(&[2])
            .map_err(inference)?
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(inference)?;
        let v = parts[2]
            .squeeze_axes(&[2])
            .map_err(inference)?
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(inference)?;
        if let Some(base) = rope_base {
            q = fast::rope(&q, head_dim, false, base, 1.0, 0, None).map_err(inference)?;
            k = fast::rope(&k, head_dim, false, base, 1.0, 0, None).map_err(inference)?;
        }
        let attended = fast::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            (head_dim as f32).powf(-0.5),
            mask,
            None,
        )
        .map_err(inference)?;
        let joined = attended
            .transpose_axes(&[0, 2, 1, 3])
            .map_err(inference)?
            .reshape(&[b, length, d])
            .map_err(inference)?;
        out.forward(&joined)
    }

    fn make_masks(&self, batch: &Batch) -> (Array, Array) {
        let b = batch.rows;
        let length = batch.length;
        let valid: Vec<bool> = batch.attention_mask.iter().map(|v| *v != 0).collect();
        let full = Array::from_slice(&valid, &[b as i32, 1, 1, length as i32]);
        let radius = self.cfg.local_half_window() as i64;
        let mut local = vec![false; b * length * length];
        for row in 0..b {
            for query in 0..length {
                let query_valid = valid[row * length + query];
                for key in 0..length {
                    let key_valid = valid[row * length + key];
                    local[(row * length + query) * length + key] =
                        key_valid && (!query_valid || (query as i64 - key as i64).abs() <= radius);
                }
            }
        }
        let local = Array::from_slice(&local, &[b as i32, 1, length as i32, length as i32]);
        (full, local)
    }
}

impl Backend for MlxBackend {
    fn label(&self) -> String {
        format!("mlx-metal-{}", self.precision.as_str())
    }

    fn device(&self) -> DeviceKind {
        DeviceKind::Metal
    }

    fn precision(&self) -> Precision {
        self.precision
    }

    fn forward(&mut self, batch: &Batch) -> Result<ModelOutput> {
        let b = batch.rows as i32;
        let length = batch.length as i32;
        let markers = batch.marker_slots as i32;
        let d = self.cfg.hidden_size as i32;
        if markers < 2 {
            return Err(LayaError::Inference(
                "the action head needs at least two marker slots".to_owned(),
            ));
        }
        let input_ids: Vec<i32> = batch.input_ids.iter().map(|v| *v as i32).collect();
        let qtype: Vec<i32> = batch.qtype.iter().map(|v| i32::from(*v)).collect();
        let input_ids = Array::from_slice(&input_ids, &[b, length]);
        let qtype = Array::from_slice(&qtype, &[b]);
        let (full_mask, local_mask) = self.make_masks(batch);
        let mut h = self.embeddings.index(&input_ids);
        h = self.embedding_norm.forward(&h)?;
        for layer in &self.encoder_layers {
            let normalized = match &layer.attn_norm {
                Some(norm) => norm.forward(&h)?,
                None => h.clone(),
            };
            let mask = if layer.local { &local_mask } else { &full_mask };
            let attention = self.attention(
                &normalized,
                &layer.qkv,
                &layer.out,
                self.cfg.num_attention_heads as i32,
                mask,
                Some(layer.rope_base),
            )?;
            h = ops::add(&h, &attention).map_err(inference)?;
            let normalized = layer.mlp_norm.forward(&h)?;
            let gates = layer
                .mlp_in
                .forward(&normalized)?
                .split_equal(2, -1)
                .map_err(inference)?;
            let value = nn::gelu(&gates[0]).map_err(inference)?;
            let gated = ops::multiply(&value, &gates[1]).map_err(inference)?;
            let mlp = layer.mlp_out.forward(&gated)?;
            h = ops::add(&h, &mlp).map_err(inference)?;
        }
        h = self.final_norm.forward(&h)?;
        let typed = self
            .type_embedding
            .index(&qtype)
            .reshape(&[b, 1, d])
            .map_err(inference)?;
        h = ops::add(&h, &typed).map_err(inference)?;
        for layer in &self.head_layers {
            let normalized = layer.norm1.forward(&h)?;
            let attention = self.attention(
                &normalized,
                &layer.in_proj,
                &layer.out_proj,
                (d / 64).max(1),
                &full_mask,
                None,
            )?;
            h = ops::add(&h, &attention).map_err(inference)?;
            let normalized = layer.norm2.forward(&h)?;
            let ff = layer.linear1.forward(&normalized)?;
            let ff = nn::relu(&ff).map_err(inference)?;
            let ff = layer.linear2.forward(&ff)?;
            h = ops::add(&h, &ff).map_err(inference)?;
        }
        let hidden_cls = h.index((.., 0, ..));
        let flat = h.reshape(&[b * length, d]).map_err(inference)?;
        let gather_indices: Vec<i32> = batch
            .marker_pos
            .iter()
            .enumerate()
            .map(|(index, position)| (index as i32 / markers) * length + *position as i32)
            .collect();
        let gather_indices = Array::from_slice(&gather_indices, &[b * markers]);
        let marker_hidden = flat
            .take_axis(&gather_indices, 0)
            .map_err(inference)?
            .reshape(&[b, markers, d])
            .map_err(inference)?;
        let score = self.scorer_norm.forward(&marker_hidden)?;
        let score = self.scorer_in.forward(&score)?;
        let score = nn::gelu(&score).map_err(inference)?;
        let raw_logits = self
            .scorer_out
            .forward(&score)?
            .squeeze_axes(&[-1])
            .map_err(inference)?
            .as_dtype(Dtype::Float32)
            .map_err(inference)?;
        let marker_mask = Array::from_slice(&batch.marker_mask, &[b, markers]);
        let masked_logits = ops::select(&marker_mask, &raw_logits, Array::from_f32(-10_000.0))
            .map_err(inference)?;
        let probabilities = ops::softmax_axis(&masked_logits, -1, Some(true)).map_err(inference)?;
        let counts: Vec<f32> = (0..batch.rows)
            .map(|row| batch.marker_count(row).max(2) as f32)
            .collect();
        let counts = Array::from_slice(&counts, &[b]);
        let safe_p = ops::maximum(&probabilities, Array::from_f32(1e-9)).map_err(inference)?;
        let entropy_terms = ops::multiply(&probabilities, &ops::log(&safe_p).map_err(inference)?)
            .map_err(inference)?;
        let entropy = ops::negative(&ops::sum_axis(&entropy_terms, -1, None).map_err(inference)?)
            .map_err(inference)?;
        let entropy =
            ops::divide(&entropy, &ops::log(&counts).map_err(inference)?).map_err(inference)?;
        let sorted = ops::sort_axis(&probabilities, -1).map_err(inference)?;
        let second = sorted.index((.., markers - 2));
        let first = sorted.index((.., markers - 1));
        let margin = ops::subtract(&first, &second).map_err(inference)?;
        let normalized_count = ops::divide(&counts, Array::from_f32(255.0)).map_err(inference)?;
        let features =
            ops::stack(&[&first, &margin, &entropy, &normalized_count], -1).map_err(inference)?;
        let pooled = hidden_cls.as_dtype(Dtype::Float32).map_err(inference)?;
        let action_input = ops::concatenate(&[&pooled, &features], -1)
            .map_err(inference)?
            .as_dtype(self.dtype())
            .map_err(inference)?;
        let action = self.action_in.forward(&action_input)?;
        let action = nn::gelu(&action).map_err(inference)?;
        let act_logits = self
            .action_out
            .forward(&action)?
            .as_dtype(Dtype::Float32)
            .map_err(inference)?;
        let hidden_cls = hidden_cls.as_dtype(Dtype::Float32).map_err(inference)?;
        transforms::eval([&raw_logits, &act_logits, &hidden_cls]).map_err(inference)?;
        let to_host = |array: &Array| -> Result<Vec<f32>> {
            array
                .contiguous()
                .map_err(inference)?
                .to_vec_cast::<f32>()
                .map_err(inference)
        };
        Ok(ModelOutput {
            rows: batch.rows,
            marker_slots: batch.marker_slots,
            n_act: self.n_act,
            raw_logits: to_host(&raw_logits)?,
            act_logits: to_host(&act_logits)?,
            hidden_cls: to_host(&hidden_cls)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_metallib_matches_recorded_sha256() {
        assert_eq!(EMBEDDED_METALLIB.len(), 1_455_256);
        assert_eq!(sha256_bytes(EMBEDDED_METALLIB), EMBEDDED_METALLIB_SHA256);
    }
}
