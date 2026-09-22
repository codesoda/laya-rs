//! Complete Laya network for the L1 feasibility spike.
//!
//! The ModernBERT backbone starts from Candle's upstream implementation at
//! huggingface/candle commit 1bbda281093b29a62ab2331b239894778b3aefb2
//! (`candle-transformers/src/models/modernbert.rs`, Apache-2.0/MIT), adapted for
//! Laya's checkpoint prefix and Transformers 5.17 semantics.

use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    embedding, layer_norm, layer_norm_no_bias, linear, linear_no_bias, ops::softmax, Embedding,
    LayerNorm, Linear, Module, VarBuilder,
};

use crate::config::{AgentConfig, EncoderConfig};

#[derive(Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, config: &EncoderConfig, theta: f64, device: &Device) -> Result<Self> {
        let dim = config.hidden_size / config.num_attention_heads;
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|index| 1f32 / theta.powf(index as f64 / dim as f64) as f32)
            .collect::<Vec<_>>();
        // Transformers forces RoPE frequency construction, matmul, sin and cos to FP32,
        // then casts the result to the model dtype.
        let inv_freq = Tensor::from_vec(inv_freq, (1, dim / 2), device)?;
        let positions = Tensor::arange(0u32, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?;
        let frequencies = positions.matmul(&inv_freq)?;
        Ok(Self {
            sin: frequencies.sin()?.to_dtype(dtype)?,
            cos: frequencies.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, query: &Tensor, key: &Tensor) -> Result<(Tensor, Tensor)> {
        // `rope_slow` is the non-interleaved rotate-half convention used by
        // Transformers. Candle 0.11's fused non-interleaved `rope` has no Metal
        // implementation, while this tensor-op formulation stays entirely on Metal.
        Ok((
            candle_nn::rotary_emb::rope_slow(&query.contiguous()?, &self.cos, &self.sin)?,
            candle_nn::rotary_emb::rope_slow(&key.contiguous()?, &self.cos, &self.sin)?,
        ))
    }
}

#[derive(Clone)]
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
    ) -> Result<Self> {
        Ok(Self {
            qkv: linear_no_bias(config.hidden_size, config.hidden_size * 3, vb.pp("Wqkv"))?,
            output: linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("Wo"))?,
            heads: config.num_attention_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
            rotary,
        })
    }

    fn forward(&self, hidden: &Tensor, mask: &Tensor) -> Result<Tensor> {
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
        // HF eager explicitly softmaxes in FP32. Torch SDPA has the same FP32
        // accumulation contract for this FP32 acceptance path.
        let probabilities =
            softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(hidden.dtype())?;
        probabilities
            .matmul(&value)?
            .transpose(1, 2)?
            .reshape((batch, sequence, width))?
            .apply(&self.output)
    }
}

#[derive(Clone)]
struct Mlp {
    input: Linear,
    output: Linear,
}

impl Mlp {
    fn load(vb: VarBuilder<'_>, config: &EncoderConfig) -> Result<Self> {
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
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let halves = hidden.apply(&self.input)?.chunk(2, D::Minus1)?;
        (&halves[0].gelu_erf()? * &halves[1])?.apply(&self.output)
    }
}

#[derive(Clone)]
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
    ) -> Result<Self> {
        let attention_norm = if index == 0 {
            None
        } else {
            Some(layer_norm_no_bias(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("attn_norm"),
            )?)
        };
        Ok(Self {
            attention: Attention::load(vb.pp("attn"), config, rotary)?,
            mlp: Mlp::load(vb.pp("mlp"), config)?,
            attention_norm,
            mlp_norm: layer_norm_no_bias(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("mlp_norm"),
            )?,
            local: config.layer_types[index] == "sliding_attention",
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        padding_mask: &Tensor,
        local_mask: &Tensor,
    ) -> Result<Tensor> {
        let normalized = match &self.attention_norm {
            Some(norm) => hidden.apply(norm)?,
            None => hidden.clone(),
        };
        let owned_mask;
        let mask = if self.local {
            owned_mask = padding_mask.broadcast_add(local_mask)?;
            &owned_mask
        } else {
            padding_mask
        };
        let hidden = (hidden + self.attention.forward(&normalized, mask)?)?;
        let mlp = hidden.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        hidden + mlp
    }
}

#[derive(Clone)]
struct Encoder {
    embeddings: Embedding,
    embedding_norm: LayerNorm,
    layers: Vec<EncoderLayer>,
    final_norm: LayerNorm,
    local_half_window: usize,
}

impl Encoder {
    fn load(vb: VarBuilder<'_>, config: &EncoderConfig) -> Result<Self> {
        let embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embeddings.tok_embeddings"),
        )?;
        let embedding_norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("embeddings.norm"),
        )?;
        let global_rotary = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            config,
            config.rope_parameters.full_attention.rope_theta,
            vb.device(),
        )?);
        let local_rotary = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            config,
            config.rope_parameters.sliding_attention.rope_theta,
            vb.device(),
        )?);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let local = config.layer_types[index] == "sliding_attention";
            layers.push(EncoderLayer::load(
                vb.pp(format!("layers.{index}")),
                config,
                index,
                if local {
                    local_rotary.clone()
                } else {
                    global_rotary.clone()
                },
            )?);
        }
        let final_norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("final_norm"),
        )?;
        Ok(Self {
            embeddings,
            embedding_norm,
            layers,
            final_norm,
            local_half_window: config.local_attention / 2,
        })
    }

    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let sequence = input_ids.dim(1)?;
        let dtype = self.embedding_norm.weight().dtype();
        let masked_value = dtype_min(dtype);
        let padding_mask = attention_mask
            .eq(0u32)?
            .to_dtype(dtype)?
            .affine(masked_value, 0.)?
            .unsqueeze(1)?
            .unsqueeze(2)?;
        let local_mask =
            local_attention_mask(sequence, self.local_half_window, dtype, input_ids.device())?;
        let mut hidden = input_ids
            .apply(&self.embeddings)?
            .apply(&self.embedding_norm)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &padding_mask, &local_mask)?;
        }
        hidden.apply(&self.final_norm)
    }
}

#[derive(Clone)]
struct HeadAttention {
    input: Linear,
    output: Linear,
    heads: usize,
    head_dim: usize,
}

impl HeadAttention {
    fn load(vb: VarBuilder<'_>, width: usize) -> Result<Self> {
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

    fn forward(&self, hidden: &Tensor, padding_mask: &Tensor) -> Result<Tensor> {
        let (batch, sequence, width) = hidden.dims3()?;
        let qkv = hidden
            .apply(&self.input)?
            .reshape((batch, sequence, 3, self.heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let query = qkv.get(0)?;
        let key = qkv.get(1)?;
        let value = qkv.get(2)?;
        let scores = (query * (self.head_dim as f64).powf(-0.5))?
            .matmul(&key.transpose(D::Minus2, D::Minus1)?)?
            .broadcast_add(padding_mask)?;
        let probabilities =
            softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(hidden.dtype())?;
        probabilities
            .matmul(&value)?
            .transpose(1, 2)?
            .reshape((batch, sequence, width))?
            .apply(&self.output)
    }
}

#[derive(Clone)]
struct HeadLayer {
    attention: HeadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl HeadLayer {
    fn load(vb: VarBuilder<'_>, width: usize) -> Result<Self> {
        Ok(Self {
            attention: HeadAttention::load(vb.pp("self_attn"), width)?,
            linear1: linear(width, width * 4, vb.pp("linear1"))?,
            linear2: linear(width * 4, width, vb.pp("linear2"))?,
            norm1: layer_norm(width, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(width, 1e-5, vb.pp("norm2"))?,
        })
    }

    fn forward(&self, hidden: &Tensor, padding_mask: &Tensor) -> Result<Tensor> {
        // PyTorch TransformerEncoderLayer(norm_first=true), activation default ReLU.
        // Candle 0.11's fused layer_norm custom op has no Metal kernel. The
        // documented slow formulation uses ordinary Metal tensor kernels.
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

#[derive(Clone)]
struct Scorer {
    norm: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl Scorer {
    fn load(vb: VarBuilder<'_>, width: usize) -> Result<Self> {
        Ok(Self {
            norm: layer_norm(width, 1e-5, vb.pp("0"))?,
            linear1: linear(width, width, vb.pp("1"))?,
            linear2: linear(width, 1, vb.pp("3"))?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        apply_layer_norm_slow(hidden, &self.norm)?
            .apply(&self.linear1)?
            .gelu_erf()?
            .apply(&self.linear2)
    }
}

#[derive(Clone)]
struct ActionHead {
    linear1: Linear,
    linear2: Linear,
}

impl ActionHead {
    fn load(vb: VarBuilder<'_>, width: usize, outputs: usize) -> Result<Self> {
        Ok(Self {
            linear1: linear(width + 4, 256, vb.pp("0"))?,
            linear2: linear(256, outputs, vb.pp("2"))?,
        })
    }

    fn forward(&self, features: &Tensor) -> Result<Tensor> {
        features
            .apply(&self.linear1)?
            .gelu_erf()?
            .apply(&self.linear2)
    }
}

#[derive(Clone)]
pub struct LayaModel {
    encoder: Encoder,
    type_embedding: Embedding,
    head: Vec<HeadLayer>,
    scorer: Scorer,
    action_head: ActionHead,
    // Strict-load checkpoint buffer; runtime calibration uses rl_agent_config.json.
    _temperature: Tensor,
    dtype: DType,
}

pub struct ForwardOutput {
    pub logits: Tensor,
    pub action_logits: Tensor,
    pub action_probs: Tensor,
    pub hidden_cls: Tensor,
}

impl LayaModel {
    pub fn load(
        vb: VarBuilder<'_>,
        encoder_config: &EncoderConfig,
        agent_config: &AgentConfig,
    ) -> Result<Self> {
        let encoder = Encoder::load(vb.pp("encoder"), encoder_config)?;
        let type_embedding = embedding(3, encoder_config.hidden_size, vb.pp("type_emb"))?;
        let mut head = Vec::with_capacity(agent_config.head_layers);
        for index in 0..agent_config.head_layers {
            head.push(HeadLayer::load(
                vb.pp(format!("head.layers.{index}")),
                encoder_config.hidden_size,
            )?);
        }
        Ok(Self {
            encoder,
            type_embedding,
            head,
            scorer: Scorer::load(vb.pp("scorer"), encoder_config.hidden_size)?,
            action_head: ActionHead::load(
                vb.pp("act_head"),
                encoder_config.hidden_size,
                agent_config.n_act(),
            )?,
            _temperature: vb.get(3, "temperature")?,
            dtype: vb.dtype(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        marker_pos: &Tensor,
        marker_mask: &Tensor,
        qtype: &Tensor,
    ) -> Result<ForwardOutput> {
        let mut hidden = self.encoder.forward(input_ids, attention_mask)?;
        hidden = hidden.broadcast_add(&qtype.apply(&self.type_embedding)?.unsqueeze(1)?)?;

        let dtype = hidden.dtype();
        let padding_mask = attention_mask
            .eq(0u32)?
            .to_dtype(dtype)?
            .affine(dtype_min(dtype), 0.)?
            .unsqueeze(1)?
            .unsqueeze(2)?;
        for layer in &self.head {
            hidden = layer.forward(&hidden, &padding_mask)?;
        }

        let (batch, sequence, width) = hidden.dims3()?;
        let options = marker_pos.dim(1)?;
        let offsets = Tensor::arange(0u32, batch as u32, hidden.device())?
            .affine(sequence as f64, 0.)?
            .unsqueeze(1)?;
        let indices = marker_pos
            .broadcast_add(&offsets)?
            .flatten_all()?
            .to_dtype(DType::U32)?;
        let markers = hidden
            .reshape((batch * sequence, width))?
            .index_select(&indices, 0)?
            .reshape((batch, options, width))?;
        let raw_logits = self
            .scorer
            .forward(&markers)?
            .squeeze(2)?
            .to_dtype(DType::F32)?;
        let masked = Tensor::full(-1e4f32, raw_logits.dims(), raw_logits.device())?;
        let logits = marker_mask.where_cond(&raw_logits, &masked)?;

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
        let features = Tensor::stack(&[maximum, margin, entropy, (&count / 255f64)?], 1)?
            .to_dtype(self.dtype)?;
        let hidden_cls = hidden.i((.., 0, ..))?.contiguous()?;
        let action_input = Tensor::cat(&[&hidden_cls, &features], 1)?;
        let action_logits = self
            .action_head
            .forward(&action_input)?
            .to_dtype(DType::F32)?;
        let action_probs = softmax(&action_logits, D::Minus1)?;

        Ok(ForwardOutput {
            logits,
            action_logits,
            action_probs,
            hidden_cls: hidden_cls.to_dtype(DType::F32)?,
        })
    }
}

fn apply_layer_norm_slow(hidden: &Tensor, norm: &LayerNorm) -> Result<Tensor> {
    let bias = norm
        .bias()
        .ok_or_else(|| candle_core::Error::Msg("expected biased layer norm".to_owned()))?;
    candle_nn::ops::layer_norm_slow(hidden, norm.weight(), bias, norm.eps() as f32)
}

fn dtype_min(dtype: DType) -> f64 {
    match dtype {
        DType::F16 => -65_504.,
        DType::BF16 => f32::MIN as f64,
        _ => f32::MIN as f64,
    }
}

pub fn local_attention_mask(
    sequence: usize,
    half_window: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let minimum = dtype_min(dtype) as f32;
    let values = (0..sequence)
        .flat_map(|query| {
            (0..sequence).map(move |key| {
                if query.abs_diff(key) > half_window {
                    minimum
                } else {
                    0f32
                }
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (1, 1, sequence, sequence), device)?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_mask_is_inclusive_at_half_window() -> Result<()> {
        let mask = local_attention_mask(5, 1, DType::F32, &Device::Cpu)?
            .reshape((5, 5))?
            .to_vec2::<f32>()?;
        assert_eq!(mask[2], [f32::MIN, 0., 0., 0., f32::MIN]);
        assert_eq!(mask[0], [0., 0., f32::MIN, f32::MIN, f32::MIN]);
        Ok(())
    }
}
