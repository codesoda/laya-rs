use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use safetensors::SafeTensors;
use serde::Serialize;

use crate::config::{AgentConfig, ConfigAdapterOutput, EncoderConfig};

#[derive(Debug, Serialize)]
pub struct TensorSummary {
    pub tensor_count: usize,
    pub total_bytes: usize,
    pub dtypes: BTreeMap<String, usize>,
    pub tensors: Vec<TensorInfo>,
    pub missing: Vec<String>,
    pub unexpected: Vec<String>,
    pub shape_mismatches: Vec<String>,
    pub config_adapter: ConfigAdapterOutput,
    pub observations: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: usize,
}

pub fn inspect_checkpoint(
    weights_path: &Path,
    encoder: &EncoderConfig,
    agent: &AgentConfig,
) -> Result<TensorSummary> {
    let bytes = fs::read(weights_path)
        .with_context(|| format!("read safetensors header/data {}", weights_path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parse safetensors {}", weights_path.display()))?;
    let expected = expected_shapes(encoder, agent);
    let actual_names = tensors
        .names()
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    let expected_names = expected.keys().cloned().collect::<BTreeSet<_>>();
    let missing = expected_names.difference(&actual_names).cloned().collect();
    let unexpected = actual_names.difference(&expected_names).cloned().collect();
    let mut shape_mismatches = Vec::new();
    let mut infos = Vec::with_capacity(tensors.len());
    let mut dtypes = BTreeMap::new();
    let mut total_bytes = 0;
    let mut names = tensors.names();
    names.sort_unstable();
    for name in names {
        let view = tensors.tensor(name)?;
        let dtype = format!("{:?}", view.dtype());
        *dtypes.entry(dtype.clone()).or_insert(0) += 1;
        total_bytes += view.data().len();
        if let Some(expected_shape) = expected.get(name) {
            if view.shape() != expected_shape {
                shape_mismatches.push(format!(
                    "{name}: actual {:?}, expected {expected_shape:?}",
                    view.shape()
                ));
            }
        }
        infos.push(TensorInfo {
            name: name.to_owned(),
            dtype,
            shape: view.shape().to_vec(),
            bytes: view.data().len(),
        });
    }
    let mut observations = Vec::new();
    if dtypes.get("F16").copied().unwrap_or(0) + dtypes.get("F32").copied().unwrap_or(0)
        == tensors.len()
        && dtypes.get("F32") == Some(&1)
    {
        observations.push(
            "checkpoint stores every trained parameter as F16 except temperature (F32); the Python FP32 oracle copies these values into FP32 parameters, so rust_cpu_fp32 loads/converts every parameter to F32"
                .to_owned(),
        );
    } else if dtypes.get("F16") == Some(&tensors.len()) {
        observations.push(
            "checkpoint stores every tensor, including temperature, as F16; FP32 parity explicitly converts all tensors to F32 while loading"
                .to_owned(),
        );
    }
    if encoder.position_embedding_type == "sans_pos" {
        observations.push(
            "position_embedding_type=sans_pos is a no-op for ModernBertModel; RoPE remains active"
                .to_owned(),
        );
    }
    Ok(TensorSummary {
        tensor_count: tensors.len(),
        total_bytes,
        dtypes,
        tensors: infos,
        missing,
        unexpected,
        shape_mismatches,
        config_adapter: encoder.adapter_output(),
        observations,
    })
}

pub fn validate_checkpoint(summary: &TensorSummary) -> Result<()> {
    if !summary.missing.is_empty()
        || !summary.unexpected.is_empty()
        || !summary.shape_mismatches.is_empty()
    {
        bail!(
            "checkpoint inventory failed: missing={:?}, unexpected={:?}, shape_mismatches={:?}",
            summary.missing,
            summary.unexpected,
            summary.shape_mismatches
        )
    }
    Ok(())
}

fn expected_shapes(encoder: &EncoderConfig, agent: &AgentConfig) -> BTreeMap<String, Vec<usize>> {
    let mut shapes = BTreeMap::new();
    let d = encoder.hidden_size;
    let i = encoder.intermediate_size;
    shapes.insert(
        "encoder.embeddings.tok_embeddings.weight".to_owned(),
        vec![encoder.vocab_size, d],
    );
    shapes.insert("encoder.embeddings.norm.weight".to_owned(), vec![d]);
    shapes.insert("encoder.final_norm.weight".to_owned(), vec![d]);
    for layer in 0..encoder.num_hidden_layers {
        let prefix = format!("encoder.layers.{layer}");
        shapes.insert(format!("{prefix}.attn.Wqkv.weight"), vec![3 * d, d]);
        shapes.insert(format!("{prefix}.attn.Wo.weight"), vec![d, d]);
        if layer != 0 {
            shapes.insert(format!("{prefix}.attn_norm.weight"), vec![d]);
        }
        shapes.insert(format!("{prefix}.mlp.Wi.weight"), vec![2 * i, d]);
        shapes.insert(format!("{prefix}.mlp.Wo.weight"), vec![d, i]);
        shapes.insert(format!("{prefix}.mlp_norm.weight"), vec![d]);
    }
    shapes.insert("type_emb.weight".to_owned(), vec![3, d]);
    for layer in 0..agent.head_layers {
        let prefix = format!("head.layers.{layer}");
        shapes.insert(format!("{prefix}.self_attn.in_proj_weight"), vec![3 * d, d]);
        shapes.insert(format!("{prefix}.self_attn.in_proj_bias"), vec![3 * d]);
        shapes.insert(format!("{prefix}.self_attn.out_proj.weight"), vec![d, d]);
        shapes.insert(format!("{prefix}.self_attn.out_proj.bias"), vec![d]);
        shapes.insert(format!("{prefix}.linear1.weight"), vec![4 * d, d]);
        shapes.insert(format!("{prefix}.linear1.bias"), vec![4 * d]);
        shapes.insert(format!("{prefix}.linear2.weight"), vec![d, 4 * d]);
        shapes.insert(format!("{prefix}.linear2.bias"), vec![d]);
        for norm in ["norm1", "norm2"] {
            shapes.insert(format!("{prefix}.{norm}.weight"), vec![d]);
            shapes.insert(format!("{prefix}.{norm}.bias"), vec![d]);
        }
    }
    shapes.insert("scorer.0.weight".to_owned(), vec![d]);
    shapes.insert("scorer.0.bias".to_owned(), vec![d]);
    shapes.insert("scorer.1.weight".to_owned(), vec![d, d]);
    shapes.insert("scorer.1.bias".to_owned(), vec![d]);
    shapes.insert("scorer.3.weight".to_owned(), vec![1, d]);
    shapes.insert("scorer.3.bias".to_owned(), vec![1]);
    shapes.insert("act_head.0.weight".to_owned(), vec![256, d + 4]);
    shapes.insert("act_head.0.bias".to_owned(), vec![256]);
    shapes.insert("act_head.2.weight".to_owned(), vec![agent.n_act(), 256]);
    shapes.insert("act_head.2.bias".to_owned(), vec![agent.n_act()]);
    shapes.insert("temperature".to_owned(), vec![3]);
    shapes
}
