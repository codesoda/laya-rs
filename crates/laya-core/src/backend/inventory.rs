//! Expected checkpoint tensor names and shapes. Every backend loads strictly:
//! a missing, unexpected or misshaped tensor is a checkpoint error, never a
//! randomly initialized substitute.

use std::collections::BTreeMap;

use super::config::EncoderConfig;
use crate::assets::AgentConfig;

#[must_use]
pub fn expected_shapes(
    encoder: &EncoderConfig,
    agent: &AgentConfig,
) -> BTreeMap<String, Vec<usize>> {
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

/// Compare an actual `{name: shape}` inventory with the expectation and
/// describe every difference.
#[must_use]
pub fn inventory_errors(
    expected: &BTreeMap<String, Vec<usize>>,
    actual: &BTreeMap<String, Vec<usize>>,
) -> Vec<String> {
    let mut errors = Vec::new();
    for (name, shape) in expected {
        match actual.get(name) {
            None => errors.push(format!("missing tensor {name}")),
            Some(found) if found != shape => errors.push(format!(
                "shape mismatch for {name}: expected {shape:?}, got {found:?}"
            )),
            Some(_) => {}
        }
    }
    for name in actual.keys() {
        if !expected.contains_key(name) {
            errors.push(format!("unexpected tensor {name}"));
        }
    }
    errors
}
