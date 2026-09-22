//! `encoder/config.json` as the network needs it, with the ModernBERT
//! assumptions this port makes checked explicitly rather than assumed.

use std::path::Path;

use serde::Deserialize;

use crate::{LayaError, Result};

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct RopeParameter {
    pub rope_theta: f64,
    pub rope_type: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct RopeParameters {
    pub full_attention: RopeParameter,
    pub sliding_attention: RopeParameter,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct EncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    /// Transformers ModernBERT reads `norm_eps` for every LayerNorm.
    pub norm_eps: f64,
    pub pad_token_id: u32,
    pub global_attn_every_n_layers: usize,
    pub local_attention: usize,
    pub layer_types: Vec<String>,
    pub rope_parameters: RopeParameters,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub norm_bias: bool,
    pub hidden_activation: String,
}

impl EncoderConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(path).map_err(|error| LayaError::asset(path, error.to_string()))?;
        let config: Self = serde_json::from_slice(&bytes)
            .map_err(|error| LayaError::asset(path, format!("parse: {error}")))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        let fail = |message: String| Err(LayaError::Checkpoint(message));
        if self.num_attention_heads == 0 || !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return fail("hidden size is not divisible by attention heads".to_owned());
        }
        if !(self.hidden_size / self.num_attention_heads).is_multiple_of(2) {
            return fail("head dimension must be even for RoPE".to_owned());
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return fail(format!(
                "layer_types has {} entries, expected {}",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        for (index, layer_type) in self.layer_types.iter().enumerate() {
            let expected = if index % self.global_attn_every_n_layers == 0 {
                "full_attention"
            } else {
                "sliding_attention"
            };
            if layer_type != expected {
                return fail(format!(
                    "layer_types[{index}] is {layer_type:?}, but the every-{} rule requires {expected:?}",
                    self.global_attn_every_n_layers
                ));
            }
        }
        if self.attention_bias || self.mlp_bias || self.norm_bias {
            return fail(format!(
                "this port implements the bias-free ModernBERT backbone (attention_bias={}, mlp_bias={}, norm_bias={})",
                self.attention_bias, self.mlp_bias, self.norm_bias
            ));
        }
        if self.hidden_activation != "gelu" {
            return fail(format!(
                "unsupported backbone activation {:?}",
                self.hidden_activation
            ));
        }
        if self.rope_parameters.full_attention.rope_type != "default"
            || self.rope_parameters.sliding_attention.rope_type != "default"
        {
            return fail("only default (unscaled) RoPE is supported".to_owned());
        }
        if !self.local_attention.is_multiple_of(2) {
            return fail("local_attention must be even".to_owned());
        }
        Ok(())
    }

    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Inclusive half window: `|i - j| > local_attention / 2` is masked.
    #[must_use]
    pub fn local_half_window(&self) -> usize {
        self.local_attention / 2
    }

    #[must_use]
    pub fn is_local(&self, layer: usize) -> bool {
        self.layer_types[layer] == "sliding_attention"
    }

    #[must_use]
    pub fn rope_theta(&self, layer: usize) -> f64 {
        if self.is_local(layer) {
            self.rope_parameters.sliding_attention.rope_theta
        } else {
            self.rope_parameters.full_attention.rope_theta
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(layers: &[&str]) -> EncoderConfig {
        EncoderConfig {
            vocab_size: 10,
            hidden_size: 64,
            num_hidden_layers: layers.len(),
            num_attention_heads: 2,
            intermediate_size: 8,
            max_position_embeddings: 16,
            norm_eps: 1e-5,
            pad_token_id: 0,
            global_attn_every_n_layers: 3,
            local_attention: 128,
            layer_types: layers.iter().map(|s| (*s).to_owned()).collect(),
            rope_parameters: RopeParameters {
                full_attention: RopeParameter {
                    rope_theta: 1.0,
                    rope_type: "default".into(),
                },
                sliding_attention: RopeParameter {
                    rope_theta: 2.0,
                    rope_type: "default".into(),
                },
            },
            attention_bias: false,
            mlp_bias: false,
            norm_bias: false,
            hidden_activation: "gelu".into(),
        }
    }

    #[test]
    fn validates_layer_pattern_and_biases() {
        let good = config(&["full_attention", "sliding_attention", "sliding_attention"]);
        assert!(good.validate().is_ok());
        assert!(good.is_local(1) && !good.is_local(0));
        assert_eq!(good.rope_theta(1), 2.0);
        assert!(config(&["sliding_attention"]).validate().is_err());
        let mut biased = good.clone();
        biased.norm_bias = true;
        assert!(biased.validate().is_err());
    }
}
