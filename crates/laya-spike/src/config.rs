use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Profile {
    English,
    Multilingual,
    TypedDecisions,
}

impl Profile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::English => "english",
            Self::Multilingual => "multilingual",
            Self::TypedDecisions => "typed-decisions",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum RequestedDevice {
    Cpu,
    Metal,
}

impl RequestedDevice {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum RequestedDType {
    F32,
    F16,
    Bf16,
}

impl RequestedDType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    pub fn candle(self) -> candle_core::DType {
        match self {
            Self::F32 => candle_core::DType::F32,
            Self::F16 => candle_core::DType::F16,
            Self::Bf16 => candle_core::DType::BF16,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeParameter {
    pub rope_theta: f64,
    pub rope_type: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeParameters {
    pub full_attention: RopeParameter,
    pub sliding_attention: RopeParameter,
}

#[derive(Clone, Debug, Deserialize)]
pub struct EncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub pad_token_id: u32,
    pub global_attn_every_n_layers: usize,
    pub local_attention: usize,
    pub layer_types: Vec<String>,
    pub rope_parameters: RopeParameters,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub norm_bias: bool,
    pub hidden_activation: String,
    pub position_embedding_type: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ConfigAdapterOutput {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub pad_token_id: u32,
    pub global_attn_every_n_layers: usize,
    pub local_attention: usize,
    pub global_rope_theta: f64,
    pub local_rope_theta: f64,
    pub position_embedding_type: String,
    pub layer_types_every_n_rule_verified: bool,
}

impl EncoderConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let config: Self = serde_json::from_slice(
            &fs::read(path).with_context(|| format!("read encoder config {}", path.display()))?,
        )
        .with_context(|| format!("parse encoder config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            bail!("hidden size is not divisible by attention heads")
        }
        if self.layer_types.len() != self.num_hidden_layers {
            bail!(
                "layer_types has {} entries, expected {}",
                self.layer_types.len(),
                self.num_hidden_layers
            )
        }
        for (index, layer_type) in self.layer_types.iter().enumerate() {
            let expected = if index % self.global_attn_every_n_layers == 0 {
                "full_attention"
            } else {
                "sliding_attention"
            };
            if layer_type != expected {
                bail!(
                    "layer_types[{index}] is {layer_type:?}, but every-{} rule requires {expected:?}",
                    self.global_attn_every_n_layers
                )
            }
        }
        if self.attention_bias || self.mlp_bias || self.norm_bias {
            bail!(
                "spike expects bias-free ModernBERT backbone (attention_bias={}, mlp_bias={}, norm_bias={})",
                self.attention_bias,
                self.mlp_bias,
                self.norm_bias
            )
        }
        if self.hidden_activation != "gelu" {
            bail!(
                "unsupported backbone activation {:?}",
                self.hidden_activation
            )
        }
        if self.rope_parameters.full_attention.rope_type != "default"
            || self.rope_parameters.sliding_attention.rope_type != "default"
        {
            bail!("only default RoPE is supported by the spike")
        }
        Ok(())
    }

    pub fn adapter_output(&self) -> ConfigAdapterOutput {
        ConfigAdapterOutput {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            intermediate_size: self.intermediate_size,
            max_position_embeddings: self.max_position_embeddings,
            layer_norm_eps: self.layer_norm_eps,
            pad_token_id: self.pad_token_id,
            global_attn_every_n_layers: self.global_attn_every_n_layers,
            local_attention: self.local_attention,
            global_rope_theta: self.rope_parameters.full_attention.rope_theta,
            local_rope_theta: self.rope_parameters.sliding_attention.rope_theta,
            position_embedding_type: self.position_embedding_type.clone(),
            layer_types_every_n_rule_verified: true,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct AgentConfig {
    pub head_layers: usize,
    pub act_costs: serde_json::Map<String, serde_json::Value>,
}

impl AgentConfig {
    pub fn load(path: &Path) -> Result<Self> {
        serde_json::from_slice(
            &fs::read(path).with_context(|| format!("read agent config {}", path.display()))?,
        )
        .with_context(|| format!("parse agent config {}", path.display()))
    }

    pub fn n_act(&self) -> usize {
        self.act_costs.len() + 1
    }
}

pub fn repo_root() -> Result<PathBuf> {
    let start = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    start
        .ancestors()
        .find(|path| path.join("benchmarks/goldens/tolerances.json").is_file())
        .map(Path::to_path_buf)
        .context("cannot locate repository root from CARGO_MANIFEST_DIR")
}

pub fn profile_dir(profile: Profile) -> Result<PathBuf> {
    if let Some(root) = std::env::var_os("LAYA_MODEL_ROOT") {
        let candidate = PathBuf::from(root).join(profile.as_str());
        if candidate.join("model.safetensors").is_file() {
            return Ok(candidate);
        }
        bail!(
            "LAYA_MODEL_ROOT profile is missing model.safetensors: {}",
            candidate.display()
        )
    }

    let relative = Path::new(".cache/laya/hub/convaiinnovations--laya")
        .join("c5d78730f3493e4fe16d61507ef4b78eef7318cf")
        .join(profile.as_str());
    let local = repo_root()?.join(&relative);
    if local.join("model.safetensors").is_file() {
        return Ok(local);
    }

    let output = std::process::Command::new("git")
        .args(["rev-parse", "--git-common-dir"])
        .current_dir(repo_root()?)
        .output()
        .context("run git rev-parse --git-common-dir while locating model cache")?;
    if output.status.success() {
        let common = String::from_utf8_lossy(&output.stdout);
        let common = PathBuf::from(common.trim());
        let common = if common.is_absolute() {
            common
        } else {
            repo_root()?.join(common)
        };
        if let Some(main_root) = common.parent() {
            let candidate = main_root.join(relative);
            if candidate.join("model.safetensors").is_file() {
                return Ok(candidate);
            }
        }
    }

    bail!(
        "cannot locate {} checkpoint; set LAYA_MODEL_ROOT to the directory containing profile subdirectories",
        profile.as_str()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_in_configs_follow_every_three_rule() -> Result<()> {
        for profile in [
            Profile::English,
            Profile::Multilingual,
            Profile::TypedDecisions,
        ] {
            let dir = profile_dir(profile)?;
            EncoderConfig::load(&dir.join("encoder/config.json"))?;
        }
        Ok(())
    }
}
