use std::{collections::HashMap, path::Path};

use anyhow::{bail, Context, Result};
use mlx_rs::{
    fast, nn,
    ops::{self, indexing::IndexOp},
    Array, Dtype,
};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, clap::ValueEnum, serde::Serialize)]
pub enum Precision {
    F32,
    F16,
}

impl Precision {
    pub fn dtype(self) -> Dtype {
        match self {
            Self::F32 => Dtype::Float32,
            Self::F16 => Dtype::Float16,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EncoderConfig {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i32,
    #[serde(default = "default_eps")]
    pub norm_eps: f32,
    #[serde(default = "default_local_attention")]
    pub local_attention: i32,
    #[serde(default = "default_global_every")]
    pub global_attn_every_n_layers: usize,
    pub layer_types: Option<Vec<String>>,
    pub rope_parameters: Option<HashMap<String, RopeConfig>>,
}

#[derive(Debug, Deserialize)]
pub struct RopeConfig {
    pub rope_theta: f32,
    #[serde(default)]
    pub rope_type: Option<String>,
}

fn default_eps() -> f32 {
    1e-5
}
fn default_local_attention() -> i32 {
    128
}
fn default_global_every() -> usize {
    3
}

#[derive(Debug, Deserialize)]
pub struct AgentConfig {
    #[serde(default = "default_head_layers")]
    pub head_layers: usize,
}
fn default_head_layers() -> usize {
    2
}

#[derive(Debug)]
struct Linear {
    weight: Array,
    bias: Option<Array>,
}

impl Linear {
    fn forward(&self, x: &Array) -> Result<Array> {
        let y = ops::matmul(x, self.weight.t()).context("linear matmul")?;
        match &self.bias {
            Some(bias) => ops::add(&y, bias).context("linear bias"),
            None => Ok(y),
        }
    }
}

#[derive(Debug)]
struct Norm {
    weight: Array,
    bias: Option<Array>,
    eps: f32,
}

impl Norm {
    fn forward(&self, x: &Array) -> Result<Array> {
        fast::layer_norm(x, Some(&self.weight), self.bias.as_ref(), self.eps)
            .context("layer normalization")
    }
}

#[derive(Debug)]
struct EncoderLayer {
    attention_type: String,
    attn_norm: Option<Norm>,
    qkv: Linear,
    out: Linear,
    mlp_norm: Norm,
    mlp_in: Linear,
    mlp_out: Linear,
}

#[derive(Debug)]
struct HeadLayer {
    norm1: Norm,
    norm2: Norm,
    in_proj: Linear,
    out_proj: Linear,
    linear1: Linear,
    linear2: Linear,
}

#[derive(Debug)]
pub struct Batch {
    pub input_ids: Vec<i32>,
    pub attention_mask: Vec<bool>,
    pub marker_pos: Vec<i32>,
    pub marker_mask: Vec<bool>,
    pub qtype: Vec<i32>,
    pub batch: i32,
    pub length: i32,
    pub markers: i32,
}

#[derive(Debug)]
pub struct ForwardOutput {
    pub raw_logits: Array,
    pub masked_logits: Array,
    pub act_logits: Array,
    pub hidden_cls: Array,
}

#[derive(Debug)]
pub struct DecisionModel {
    cfg: EncoderConfig,
    precision: Precision,
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
}

struct Loader {
    values: HashMap<String, Array>,
    dtype: Dtype,
}

impl Loader {
    fn new(path: &Path, precision: Precision) -> Result<Self> {
        let values = Array::load_safetensors(path)
            .with_context(|| format!("load upstream safetensors {}", path.display()))?;
        Ok(Self {
            values,
            dtype: precision.dtype(),
        })
    }

    fn take(&mut self, name: &str, shape: &[i32]) -> Result<Array> {
        let value = self
            .values
            .remove(name)
            .with_context(|| format!("missing checkpoint tensor {name}"))?;
        if value.shape() != shape {
            bail!(
                "checkpoint shape mismatch for {name}: expected {shape:?}, got {:?}",
                value.shape()
            );
        }
        value
            .as_dtype(self.dtype)
            .with_context(|| format!("cast checkpoint tensor {name}"))
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
        // The registered calibration buffer is loaded and checked but is not part of forward.
        let temperature = self
            .values
            .remove("temperature")
            .context("missing checkpoint tensor temperature")?;
        if temperature.shape() != [3] {
            bail!(
                "checkpoint shape mismatch for temperature: {:?}",
                temperature.shape()
            );
        }
        if !self.values.is_empty() {
            let mut names: Vec<_> = self.values.into_keys().collect();
            names.sort();
            bail!("unexpected checkpoint tensors: {}", names.join(", "));
        }
        Ok(())
    }
}

impl DecisionModel {
    pub fn load(model_dir: &Path, precision: Precision) -> Result<Self> {
        let cfg: EncoderConfig = serde_json::from_slice(
            &std::fs::read(model_dir.join("encoder/config.json")).context("read encoder config")?,
        )
        .context("parse encoder config")?;
        let agent: AgentConfig = serde_json::from_slice(
            &std::fs::read(model_dir.join("rl_agent_config.json")).context("read agent config")?,
        )
        .context("parse agent config")?;
        if cfg.hidden_size % cfg.num_attention_heads != 0
            || (cfg.hidden_size / cfg.num_attention_heads) % 2 != 0
        {
            bail!("invalid ModernBERT head dimensions");
        }
        let mut loader = Loader::new(&model_dir.join("model.safetensors"), precision)?;
        let d = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let embeddings = loader.take(
            "encoder.embeddings.tok_embeddings.weight",
            &[cfg.vocab_size, d],
        )?;
        let embedding_norm = loader.norm("encoder.embeddings.norm", d, false, cfg.norm_eps)?;
        let layer_types = cfg.layer_types.clone().unwrap_or_else(|| {
            (0..cfg.num_hidden_layers)
                .map(|n| {
                    if n % cfg.global_attn_every_n_layers == 0 {
                        "full_attention"
                    } else {
                        "sliding_attention"
                    }
                    .to_owned()
                })
                .collect()
        });
        if layer_types.len() != cfg.num_hidden_layers {
            bail!("encoder layer_types count does not match num_hidden_layers");
        }
        let mut encoder_layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (index, attention_type) in layer_types.into_iter().enumerate() {
            if attention_type != "full_attention" && attention_type != "sliding_attention" {
                bail!("unsupported encoder attention type {attention_type}");
            }
            if let Some(rope) = cfg
                .rope_parameters
                .as_ref()
                .and_then(|values| values.get(&attention_type))
            {
                if rope.rope_type.as_deref().unwrap_or("default") != "default" {
                    bail!("unsupported scaled RoPE in {attention_type}");
                }
            }
            let prefix = format!("encoder.layers.{index}");
            encoder_layers.push(EncoderLayer {
                attention_type,
                attn_norm: (index != 0)
                    .then(|| loader.norm(&format!("{prefix}.attn_norm"), d, false, cfg.norm_eps))
                    .transpose()?,
                qkv: loader.linear(&format!("{prefix}.attn.Wqkv"), d, 3 * d, false)?,
                out: loader.linear(&format!("{prefix}.attn.Wo"), d, d, false)?,
                mlp_norm: loader.norm(&format!("{prefix}.mlp_norm"), d, false, cfg.norm_eps)?,
                mlp_in: loader.linear(&format!("{prefix}.mlp.Wi"), d, 2 * i, false)?,
                mlp_out: loader.linear(&format!("{prefix}.mlp.Wo"), i, d, false)?,
            });
        }
        let final_norm = loader.norm("encoder.final_norm", d, false, cfg.norm_eps)?;
        let type_embedding = loader.take("type_emb.weight", &[3, d])?;
        let mut head_layers = Vec::with_capacity(agent.head_layers);
        for index in 0..agent.head_layers {
            let prefix = format!("head.layers.{index}");
            head_layers.push(HeadLayer {
                norm1: loader.norm(&format!("{prefix}.norm1"), d, true, 1e-5)?,
                norm2: loader.norm(&format!("{prefix}.norm2"), d, true, 1e-5)?,
                in_proj: loader.linear(&format!("{prefix}.self_attn.in_proj"), d, 3 * d, true)?,
                out_proj: loader.linear(&format!("{prefix}.self_attn.out_proj"), d, d, true)?,
                linear1: loader.linear(&format!("{prefix}.linear1"), d, 4 * d, true)?,
                linear2: loader.linear(&format!("{prefix}.linear2"), 4 * d, d, true)?,
            });
        }
        let scorer_norm = loader.norm("scorer.0", d, true, 1e-5)?;
        let scorer_in = loader.linear("scorer.1", d, d, true)?;
        let scorer_out = loader.linear("scorer.3", d, 1, true)?;
        let action_in = loader.linear("act_head.0", d + 4, 256, true)?;
        let action_out = loader.linear("act_head.2", 256, 2, true)?;
        loader.finish()?;
        Ok(Self {
            cfg,
            precision,
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
        })
    }

    fn rope_base(&self, kind: &str) -> f32 {
        self.cfg
            .rope_parameters
            .as_ref()
            .and_then(|values| values.get(kind))
            .map(|value| value.rope_theta)
            .unwrap_or(if kind == "full_attention" {
                160_000.0
            } else {
                10_000.0
            })
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
        let projected = qkv.forward(x)?.reshape(&[b, length, 3, heads, head_dim])?;
        let parts = projected.split_equal(3, 2)?;
        let mut q = parts[0].squeeze_axes(&[2])?.transpose_axes(&[0, 2, 1, 3])?;
        let mut k = parts[1].squeeze_axes(&[2])?.transpose_axes(&[0, 2, 1, 3])?;
        let v = parts[2].squeeze_axes(&[2])?.transpose_axes(&[0, 2, 1, 3])?;
        if let Some(base) = rope_base {
            q = fast::rope(&q, head_dim, false, base, 1.0, 0, None)?;
            k = fast::rope(&k, head_dim, false, base, 1.0, 0, None)?;
        }
        let attended = fast::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            (head_dim as f32).powf(-0.5),
            mask,
            None,
        )?;
        let joined = attended
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, length, d])?;
        out.forward(&joined)
    }

    fn make_masks(&self, batch: &Batch) -> (Array, Array) {
        let b = batch.batch as usize;
        let length = batch.length as usize;
        let full = Array::from_slice(&batch.attention_mask, &[batch.batch, 1, 1, batch.length]);
        let radius = self.cfg.local_attention / 2;
        let mut local = vec![false; b * length * length];
        for row in 0..b {
            for query in 0..length {
                let query_valid = batch.attention_mask[row * length + query];
                for key in 0..length {
                    let key_valid = batch.attention_mask[row * length + key];
                    local[(row * length + query) * length + key] =
                        key_valid && (!query_valid || (query as i32 - key as i32).abs() <= radius);
                }
            }
        }
        let local = Array::from_slice(&local, &[batch.batch, 1, batch.length, batch.length]);
        (full, local)
    }

    pub fn forward(&self, batch: &Batch) -> Result<ForwardOutput> {
        let b = batch.batch;
        let length = batch.length;
        let d = self.cfg.hidden_size;
        let input_ids = Array::from_slice(&batch.input_ids, &[b, length]);
        let qtype = Array::from_slice(&batch.qtype, &[b]);
        let (full_mask, local_mask) = self.make_masks(batch);
        let mut h = self.embeddings.index(&input_ids);
        h = self.embedding_norm.forward(&h)?;
        for layer in &self.encoder_layers {
            let normalized = match &layer.attn_norm {
                Some(norm) => norm.forward(&h)?,
                None => h.clone(),
            };
            let mask = if layer.attention_type == "full_attention" {
                &full_mask
            } else {
                &local_mask
            };
            let attention = self.attention(
                &normalized,
                &layer.qkv,
                &layer.out,
                self.cfg.num_attention_heads,
                mask,
                Some(self.rope_base(&layer.attention_type)),
            )?;
            h = ops::add(&h, &attention)?;
            let normalized = layer.mlp_norm.forward(&h)?;
            let gates = layer.mlp_in.forward(&normalized)?.split_equal(2, -1)?;
            let value = nn::gelu(&gates[0])?;
            let gated = ops::multiply(&value, &gates[1])?;
            let mlp = layer.mlp_out.forward(&gated)?;
            h = ops::add(&h, &mlp)?;
        }
        h = self.final_norm.forward(&h)?;
        let typed = self.type_embedding.index(&qtype).reshape(&[b, 1, d])?;
        h = ops::add(&h, &typed)?;
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
            h = ops::add(&h, &attention)?;
            let normalized = layer.norm2.forward(&h)?;
            let ff = layer.linear1.forward(&normalized)?;
            let ff = nn::relu(&ff)?;
            let ff = layer.linear2.forward(&ff)?;
            h = ops::add(&h, &ff)?;
        }
        let hidden_cls = h.index((.., 0, ..));
        let flat = h.reshape(&[b * length, d])?;
        let gather_indices: Vec<i32> = batch
            .marker_pos
            .iter()
            .enumerate()
            .map(|(index, position)| (index as i32 / batch.markers) * length + position.max(&0))
            .collect();
        let gather_indices = Array::from_slice(&gather_indices, &[b * batch.markers]);
        let markers = flat
            .take_axis(&gather_indices, 0)?
            .reshape(&[b, batch.markers, d])?;
        let score = self.scorer_norm.forward(&markers)?;
        let score = self.scorer_in.forward(&score)?;
        let score = nn::gelu(&score)?;
        let raw_logits = self
            .scorer_out
            .forward(&score)?
            .squeeze_axes(&[-1])?
            .as_dtype(Dtype::Float32)?;
        let marker_mask = Array::from_slice(&batch.marker_mask, &[b, batch.markers]);
        let masked_logits = ops::select(&marker_mask, &raw_logits, Array::from_f32(-10_000.0))?;
        let probabilities = ops::softmax_axis(&masked_logits, -1, Some(true))?;
        let counts: Vec<f32> = batch
            .marker_mask
            .chunks(batch.markers as usize)
            .map(|row| row.iter().filter(|value| **value).count().max(2) as f32)
            .collect();
        let counts = Array::from_slice(&counts, &[b]);
        let safe_p = ops::maximum(&probabilities, Array::from_f32(1e-9))?;
        let entropy_terms = ops::multiply(&probabilities, &ops::log(&safe_p)?)?;
        let entropy = ops::negative(&ops::sum_axis(&entropy_terms, -1, None)?)?;
        let entropy = ops::divide(&entropy, &ops::log(&counts)?)?;
        let sorted = ops::sort_axis(&probabilities, -1)?;
        let second = sorted.index((.., batch.markers - 2));
        let first = sorted.index((.., batch.markers - 1));
        let margin = ops::subtract(&first, &second)?;
        let normalized_count = ops::divide(&counts, Array::from_f32(255.0))?;
        let features = ops::stack(&[&first, &margin, &entropy, &normalized_count], -1)?;
        let pooled = hidden_cls.as_dtype(Dtype::Float32)?;
        let action_input =
            ops::concatenate(&[&pooled, &features], -1)?.as_dtype(self.precision.dtype())?;
        let action = self.action_in.forward(&action_input)?;
        let action = nn::gelu(&action)?;
        let act_logits = self.action_out.forward(&action)?.as_dtype(Dtype::Float32)?;
        Ok(ForwardOutput {
            raw_logits,
            masked_logits,
            act_logits,
            hidden_cls: hidden_cls.as_dtype(Dtype::Float32)?,
        })
    }
}
