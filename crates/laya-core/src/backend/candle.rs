//! Candle CPU backend (fp32). Lifted from `spikes/candle-parity`
//! (`model.rs`), which passed the frozen `rust_cpu_fp32` tolerances for all
//! three profiles (`docs/L1-SPIKE.md`). Candle's Metal path and half
//! precision are deliberately not exposed: Metal was 4–5× slower than the
//! MLX path and fp16/bf16 failed the fp16 tolerance profile.
//!
//! The ModernBERT backbone starts from candle-transformers at commit
//! `1bbda281093b29a62ab2331b239894778b3aefb2` (Apache-2.0/MIT), adapted to
//! the checkpoint's `rope_parameters`/`layer_types` config and Transformers
//! 5.17 semantics (inclusive local window, `f32::MIN` padding mask,
//! bias-free LayerNorm, erf-GELU).

use std::{collections::BTreeMap, sync::Arc, time::Instant};

use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{
    embedding, layer_norm, layer_norm_no_bias, linear, linear_no_bias, ops::softmax, Embedding,
    LayerNorm, Linear, VarBuilder,
};

use super::{
    config::EncoderConfig,
    inventory::{expected_shapes, inventory_errors},
    Backend, DeviceKind, ModelOutput, Precision,
};
use crate::{
    assets::{AgentConfig, ModelDir},
    preprocess::Batch,
    LayaError, Result,
};

type CResult<T> = candle_core::Result<T>;

fn inference(error: candle_core::Error) -> LayaError {
    LayaError::Inference(error.to_string())
}

fn checkpoint(error: candle_core::Error) -> LayaError {
    LayaError::Checkpoint(error.to_string())
}

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(config: &EncoderConfig, theta: f64, device: &Device) -> CResult<Self> {
        let dim = config.head_dim();
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|index| 1f32 / theta.powf(index as f64 / dim as f64) as f32)
            .collect::<Vec<_>>();
        // Transformers builds RoPE frequencies, sin and cos in FP32.
        let inv_freq = Tensor::from_vec(inv_freq, (1, dim / 2), device)?;
        let positions = Tensor::arange(0u32, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?;
        let frequencies = positions.matmul(&inv_freq)?;
        Ok(Self {
            sin: frequencies.sin()?,
            cos: frequencies.cos()?,
        })
    }

    fn apply(&self, query: &Tensor, key: &Tensor) -> CResult<(Tensor, Tensor)> {
        // Non-interleaved rotate-half convention, as Transformers uses.
        let length = query.dim(2)?;
        let cos = self.cos.narrow(0, 0, length)?;
        let sin = self.sin.narrow(0, 0, length)?;
        Ok((
            candle_nn::rotary_emb::rope_slow(&query.contiguous()?, &cos, &sin)?,
            candle_nn::rotary_emb::rope_slow(&key.contiguous()?, &cos, &sin)?,
        ))
    }
}

struct Attention {
    qkv: Linear,
    output: Linear,
    heads: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
}

impl Attention {
    fn load(
        vb: VarBuilder<'_>,
        config: &EncoderConfig,
        rotary: Arc<RotaryEmbedding>,
    ) -> CResult<Self> {
        Ok(Self {
            qkv: linear_no_bias(config.hidden_size, config.hidden_size * 3, vb.pp("Wqkv"))?,
            output: linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("Wo"))?,
            heads: config.num_attention_heads,
            head_dim: config.head_dim(),
            rotary,
        })
    }

    fn forward(&self, hidden: &Tensor, mask: &Tensor) -> CResult<Tensor> {
        let (batch, sequence, width) = hidden.dims3()?;
        let qkv = hidden
            .apply(&self.qkv)?
            .reshape((batch, sequence, 3, self.heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let query = qkv.get(0)?;
        let key = qkv.get(1)?;
        let value = qkv.get(2)?;
        let (query, key) = self.rotary.apply(&query, &key)?;
        let scores = (query * (self.head_dim as f64).powf(-0.5))?
            .matmul(&key.transpose(D::Minus2, D::Minus1)?)?
            .broadcast_add(mask)?;
        let probabilities = softmax(&scores, D::Minus1)?;
        probabilities
            .matmul(&value)?
            .transpose(1, 2)?
            .reshape((batch, sequence, width))?
            .apply(&self.output)
    }
}

struct Mlp {
    input: Linear,
    output: Linear,
}

impl Mlp {
    fn load(vb: VarBuilder<'_>, config: &EncoderConfig) -> CResult<Self> {
        Ok(Self {
            input: linear_no_bias(
                config.hidden_size,
                config.intermediate_size * 2,
                vb.pp("Wi"),
            )?,
            output: linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("Wo"))?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, hidden: &Tensor) -> CResult<Tensor> {
        let halves = hidden.apply(&self.input)?.chunk(2, D::Minus1)?;
        (&halves[0].gelu_erf()? * &halves[1])?.apply(&self.output)
    }
}

struct EncoderLayer {
    attention: Attention,
    mlp: Mlp,
    attention_norm: Option<LayerNorm>,
    mlp_norm: LayerNorm,
    local: bool,
}

impl EncoderLayer {
    fn load(
        vb: VarBuilder<'_>,
        config: &EncoderConfig,
        index: usize,
        rotary: Arc<RotaryEmbedding>,
    ) -> CResult<Self> {
        let attention_norm = if index == 0 {
            None
        } else {
            Some(layer_norm_no_bias(
                config.hidden_size,
                config.norm_eps,
                vb.pp("attn_norm"),
            )?)
        };
        Ok(Self {
            attention: Attention::load(vb.pp("attn"), config, rotary)?,
            mlp: Mlp::load(vb.pp("mlp"), config)?,
            attention_norm,
            mlp_norm: layer_norm_no_bias(config.hidden_size, config.norm_eps, vb.pp("mlp_norm"))?,
            local: config.is_local(index),
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        padding_mask: &Tensor,
        local_mask: &Tensor,
    ) -> CResult<Tensor> {
        let normalized = match &self.attention_norm {
            Some(norm) => hidden.apply(norm)?,
            None => hidden.clone(),
        };
        let owned;
        let mask = if self.local {
            owned = padding_mask.broadcast_add(local_mask)?;
            &owned
        } else {
            padding_mask
        };
        let hidden = (hidden + self.attention.forward(&normalized, mask)?)?;
        let mlp = hidden.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        hidden + mlp
    }
}

struct Encoder {
    embeddings: Embedding,
    embedding_norm: LayerNorm,
    layers: Vec<EncoderLayer>,
    final_norm: LayerNorm,
    local_half_window: usize,
}

impl Encoder {
    fn load(vb: VarBuilder<'_>, config: &EncoderConfig) -> CResult<Self> {
        let embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embeddings.tok_embeddings"),
        )?;
        let embedding_norm = layer_norm_no_bias(
            config.hidden_size,
            config.norm_eps,
            vb.pp("embeddings.norm"),
        )?;
        let global_rotary = Arc::new(RotaryEmbedding::new(
            config,
            config.rope_parameters.full_attention.rope_theta,
            vb.device(),
        )?);
        let local_rotary = Arc::new(RotaryEmbedding::new(
            config,
            config.rope_parameters.sliding_attention.rope_theta,
            vb.device(),
        )?);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            layers.push(EncoderLayer::load(
                vb.pp(format!("layers.{index}")),
                config,
                index,
                if config.is_local(index) {
                    local_rotary.clone()
                } else {
                    global_rotary.clone()
                },
            )?);
        }
        Ok(Self {
            embeddings,
            embedding_norm,
            layers,
            final_norm: layer_norm_no_bias(
                config.hidden_size,
                config.norm_eps,
                vb.pp("final_norm"),
            )?,
            local_half_window: config.local_half_window(),
        })
    }

    fn forward(&self, input_ids: &Tensor, padding_mask: &Tensor) -> CResult<Tensor> {
        let sequence = input_ids.dim(1)?;
        let local_mask =
            local_attention_mask(sequence, self.local_half_window, input_ids.device())?;
        let mut hidden = input_ids
            .apply(&self.embeddings)?
            .apply(&self.embedding_norm)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, padding_mask, &local_mask)?;
        }
        hidden.apply(&self.final_norm)
    }
}

struct HeadAttention {
    input: Linear,
    output: Linear,
    heads: usize,
    head_dim: usize,
}

impl HeadAttention {
    fn load(vb: VarBuilder<'_>, width: usize) -> CResult<Self> {
        let heads = (width / 64).max(1);
        Ok(Self {
            input: Linear::new(
                vb.get((width * 3, width), "in_proj_weight")?,
                Some(vb.get(width * 3, "in_proj_bias")?),
            ),
            output: linear(width, width, vb.pp("out_proj"))?,
            heads,
            head_dim: width / heads,
        })
    }

    fn forward(&self, hidden: &Tensor, padding_mask: &Tensor) -> CResult<Tensor> {
        let (batch, sequence, width) = hidden.dims3()?;
        let qkv = hidden
            .apply(&self.input)?
            .reshape((batch, sequence, 3, self.heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let scores = (qkv.get(0)? * (self.head_dim as f64).powf(-0.5))?
            .matmul(&qkv.get(1)?.transpose(D::Minus2, D::Minus1)?)?
            .broadcast_add(padding_mask)?;
        softmax(&scores, D::Minus1)?
            .matmul(&qkv.get(2)?)?
            .transpose(1, 2)?
            .reshape((batch, sequence, width))?
            .apply(&self.output)
    }
}

struct HeadLayer {
    attention: HeadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl HeadLayer {
    fn load(vb: VarBuilder<'_>, width: usize) -> CResult<Self> {
        Ok(Self {
            attention: HeadAttention::load(vb.pp("self_attn"), width)?,
            linear1: linear(width, width * 4, vb.pp("linear1"))?,
            linear2: linear(width * 4, width, vb.pp("linear2"))?,
            norm1: layer_norm(width, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(width, 1e-5, vb.pp("norm2"))?,
        })
    }

    fn forward(&self, hidden: &Tensor, padding_mask: &Tensor) -> CResult<Tensor> {
        // PyTorch TransformerEncoderLayer(norm_first=True), default ReLU.
        // The biased norms use the documented slow formulation, which is
        // what the L1 parity evidence was recorded with.
        let attended = self
            .attention
            .forward(&apply_layer_norm_slow(hidden, &self.norm1)?, padding_mask)?;
        let hidden = (hidden + attended)?;
        let feed_forward = apply_layer_norm_slow(&hidden, &self.norm2)?
            .apply(&self.linear1)?
            .relu()?
            .apply(&self.linear2)?;
        hidden + feed_forward
    }
}

struct Scorer {
    norm: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl Scorer {
    fn load(vb: VarBuilder<'_>, width: usize) -> CResult<Self> {
        Ok(Self {
            norm: layer_norm(width, 1e-5, vb.pp("0"))?,
            linear1: linear(width, width, vb.pp("1"))?,
            linear2: linear(width, 1, vb.pp("3"))?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> CResult<Tensor> {
        apply_layer_norm_slow(hidden, &self.norm)?
            .apply(&self.linear1)?
            .gelu_erf()?
            .apply(&self.linear2)
    }
}

fn apply_layer_norm_slow(hidden: &Tensor, norm: &LayerNorm) -> CResult<Tensor> {
    let bias = norm
        .bias()
        .ok_or_else(|| candle_core::Error::Msg("expected biased layer norm".to_owned()))?;
    candle_nn::ops::layer_norm_slow(hidden, norm.weight(), bias, norm.eps() as f32)
}

struct ActionHead {
    linear1: Linear,
    linear2: Linear,
}

impl ActionHead {
    fn load(vb: VarBuilder<'_>, width: usize, outputs: usize) -> CResult<Self> {
        Ok(Self {
            linear1: linear(width + 4, 256, vb.pp("0"))?,
            linear2: linear(256, outputs, vb.pp("2"))?,
        })
    }

    fn forward(&self, features: &Tensor) -> CResult<Tensor> {
        features
            .apply(&self.linear1)?
            .gelu_erf()?
            .apply(&self.linear2)
    }
}

fn local_attention_mask(sequence: usize, half_window: usize, device: &Device) -> CResult<Tensor> {
    let values = (0..sequence)
        .flat_map(|query| {
            (0..sequence).map(move |key| {
                if query.abs_diff(key) > half_window {
                    f32::MIN
                } else {
                    0f32
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (1, 1, sequence, sequence), device)
}

/// The complete Laya network on the CPU through Candle, fp32.
pub struct CandleCpuBackend {
    device: Device,
    encoder: Encoder,
    type_embedding: Embedding,
    head: Vec<HeadLayer>,
    scorer: Scorer,
    action_head: ActionHead,
    n_act: usize,
    load_ms: f64,
}

impl CandleCpuBackend {
    pub fn load(model_dir: &ModelDir) -> Result<Self> {
        let started = Instant::now();
        let device = Device::Cpu;
        let encoder_config = EncoderConfig::load(&model_dir.encoder_config_path())?;
        let agent: AgentConfig = model_dir.agent_config()?;
        let weights = model_dir.weights_path();
        let tensors = candle_core::safetensors::load(&weights, &device).map_err(|error| {
            LayaError::Checkpoint(format!("load {}: {error}", weights.display()))
        })?;
        let actual: BTreeMap<String, Vec<usize>> = tensors
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.dims().to_vec()))
            .collect();
        let errors = inventory_errors(&expected_shapes(&encoder_config, &agent), &actual);
        if !errors.is_empty() {
            return Err(LayaError::Checkpoint(errors.join("; ")));
        }
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let build = || -> CResult<(Encoder, Embedding, Vec<HeadLayer>, Scorer, ActionHead)> {
            let encoder = Encoder::load(vb.pp("encoder"), &encoder_config)?;
            let type_embedding = embedding(3, encoder_config.hidden_size, vb.pp("type_emb"))?;
            let mut head = Vec::with_capacity(agent.head_layers);
            for index in 0..agent.head_layers {
                head.push(HeadLayer::load(
                    vb.pp(format!("head.layers.{index}")),
                    encoder_config.hidden_size,
                )?);
            }
            let scorer = Scorer::load(vb.pp("scorer"), encoder_config.hidden_size)?;
            let action_head =
                ActionHead::load(vb.pp("act_head"), encoder_config.hidden_size, agent.n_act())?;
            // Strict-load the calibration buffer even though runtime
            // temperatures come from rl_agent_config.json.
            vb.get(3, "temperature")?;
            Ok((encoder, type_embedding, head, scorer, action_head))
        };
        let (encoder, type_embedding, head, scorer, action_head) = build().map_err(checkpoint)?;
        Ok(Self {
            device,
            encoder,
            type_embedding,
            head,
            scorer,
            action_head,
            n_act: agent.n_act(),
            load_ms: started.elapsed().as_secs_f64() * 1000.0,
        })
    }

    #[must_use]
    pub fn load_ms(&self) -> f64 {
        self.load_ms
    }

    fn run(&self, batch: &Batch) -> CResult<(Tensor, Tensor, Tensor)> {
        let rows = batch.rows;
        let input_ids = Tensor::from_slice(&batch.input_ids, (rows, batch.length), &self.device)?;
        let attention_mask: Vec<u32> = batch.attention_mask.iter().map(|v| u32::from(*v)).collect();
        let attention_mask =
            Tensor::from_slice(&attention_mask, (rows, batch.length), &self.device)?;
        let marker_pos =
            Tensor::from_slice(&batch.marker_pos, (rows, batch.marker_slots), &self.device)?;
        let marker_mask: Vec<u8> = batch.marker_mask.iter().map(|v| u8::from(*v)).collect();
        let marker_mask =
            Tensor::from_slice(&marker_mask, (rows, batch.marker_slots), &self.device)?;
        let qtype: Vec<u32> = batch.qtype.iter().map(|v| u32::from(*v)).collect();
        let qtype = Tensor::from_slice(&qtype, rows, &self.device)?;

        let padding_mask = attention_mask
            .eq(0u32)?
            .to_dtype(DType::F32)?
            .affine(f64::from(f32::MIN), 0.)?
            .unsqueeze(1)?
            .unsqueeze(2)?;
        let mut hidden = self.encoder.forward(&input_ids, &padding_mask)?;
        hidden = hidden.broadcast_add(&qtype.apply(&self.type_embedding)?.unsqueeze(1)?)?;
        for layer in &self.head {
            hidden = layer.forward(&hidden, &padding_mask)?;
        }

        let (rows, sequence, width) = hidden.dims3()?;
        let options = marker_pos.dim(1)?;
        let offsets = Tensor::arange(0u32, rows as u32, &self.device)?
            .affine(sequence as f64, 0.)?
            .unsqueeze(1)?;
        let indices = marker_pos
            .broadcast_add(&offsets)?
            .flatten_all()?
            .to_dtype(DType::U32)?;
        let markers = hidden
            .reshape((rows * sequence, width))?
            .index_select(&indices, 0)?
            .reshape((rows, options, width))?;
        let raw_logits = self.scorer.forward(&markers)?.squeeze(2)?;
        let masked_fill = Tensor::full(-1e4f32, raw_logits.dims(), &self.device)?;
        let logits = marker_mask.where_cond(&raw_logits, &masked_fill)?;

        let probabilities = softmax(&logits, D::Minus1)?;
        let count = marker_mask
            .to_dtype(DType::F32)?
            .sum(1)?
            .clamp(2f32, f32::MAX)?;
        let entropy = (&probabilities * probabilities.clamp(1e-9f32, 1f32)?.log()?)?
            .sum(1)?
            .neg()?
            .broadcast_div(&count.log()?)?;
        let (sorted, _) = probabilities.contiguous()?.sort_last_dim(false)?;
        let maximum = sorted.i((.., 0))?;
        let margin = (&maximum - sorted.i((.., 1))?)?;
        let features = Tensor::stack(&[maximum, margin, entropy, (&count / 255f64)?], 1)?;
        let hidden_cls = hidden.i((.., 0, ..))?.contiguous()?;
        let action_logits = self
            .action_head
            .forward(&Tensor::cat(&[&hidden_cls, &features], 1)?)?;
        Ok((raw_logits, action_logits, hidden_cls))
    }
}

impl Backend for CandleCpuBackend {
    fn label(&self) -> String {
        "candle-cpu-f32".to_owned()
    }

    fn device(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn precision(&self) -> Precision {
        Precision::F32
    }

    fn forward(&mut self, batch: &Batch) -> Result<ModelOutput> {
        if batch.marker_slots < 2 {
            return Err(LayaError::Inference(
                "the action head needs at least two marker slots".to_owned(),
            ));
        }
        let (raw_logits, action_logits, hidden_cls) = self.run(batch).map_err(inference)?;
        let flatten = |tensor: Tensor| -> Result<Vec<f32>> {
            tensor
                .flatten_all()
                .map_err(inference)?
                .to_vec1::<f32>()
                .map_err(inference)
        };
        Ok(ModelOutput {
            rows: batch.rows,
            marker_slots: batch.marker_slots,
            n_act: self.n_act,
            raw_logits: flatten(raw_logits)?,
            act_logits: flatten(action_logits)?,
            hidden_cls: flatten(hidden_cls)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_mask_is_inclusive_at_half_window() -> CResult<()> {
        let mask = local_attention_mask(5, 1, &Device::Cpu)?
            .reshape((5, 5))?
            .to_vec2::<f32>()?;
        assert_eq!(mask[2], [f32::MIN, 0., 0., 0., f32::MIN]);
        assert_eq!(mask[0], [0., 0., f32::MIN, f32::MIN, f32::MIN]);
        Ok(())
    }
}
