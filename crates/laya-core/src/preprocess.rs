//! Exact reproduction of upstream `build_sequence` and `collate_items`.
//! Lifted from `spikes/tokenizer-parity` (87/87 goldens bit-exact); the
//! semantics are unchanged, only panics became errors and the tokenizer's
//! special tokens now come from `tokenizer_config.json`.
//!
//! Sequence layout per question:
//! `[CLS] <type> question: <ins> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]`

use std::path::Path;

use serde_json::{Map, Value};
use tokenizers::{EncodeInput, Tokenizer};

use crate::{
    assets::TokenizerConfig,
    pyjson::serialize_state,
    request::{Question, QuestionType, Request},
    LayaError, Result,
};

/// Upstream per-option cap before budget allocation.
pub const OPTION_TOKEN_CAP: usize = 48;

/// The upstream tokenizer plus the four special tokens the sequence builder
/// needs, resolved from `tokenizer_config.json` strings.
pub struct LayaTokenizer {
    tokenizer: Tokenizer,
    cls_id: u32,
    sep_id: u32,
    mask_id: u32,
    pad_id: u32,
    mask_token: String,
}

impl LayaTokenizer {
    pub fn from_files(tokenizer_json: &Path, config: &TokenizerConfig) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_json)
            .map_err(|error| LayaError::Tokenizer(error.to_string()))?;
        let id = |token: &str| {
            tokenizer.token_to_id(token).ok_or_else(|| {
                LayaError::Tokenizer(format!(
                    "tokenizer vocabulary has no special token {token:?}"
                ))
            })
        };
        Ok(Self {
            cls_id: id(&config.cls_token)?,
            sep_id: id(&config.sep_token)?,
            mask_id: id(&config.mask_token)?,
            pad_id: id(&config.pad_token)?,
            mask_token: config.mask_token.clone(),
            tokenizer,
        })
    }

    #[must_use]
    pub const fn pad_id(&self) -> u32 {
        self.pad_id
    }

    #[must_use]
    pub fn mask_token(&self) -> &str {
        &self.mask_token
    }

    #[must_use]
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    /// `tok(text, add_special_tokens=False)["input_ids"]`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(EncodeInput::Single(text.to_owned().into()), false)
            .map_err(|error| LayaError::Tokenizer(error.to_string()))?
            .get_ids()
            .to_vec())
    }

    /// Upstream replaces literal mask tokens in user text with a space.
    #[must_use]
    pub fn scrub(&self, text: &str) -> String {
        text.replace(&self.mask_token, " ")
    }
}

/// Everything `build_sequence` computes for one question, including the
/// intermediate values the goldens record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedSequence {
    pub head_ids_full: Vec<u32>,
    pub head_ids_kept: Vec<u32>,
    pub option_ids_full: Vec<Vec<u32>>,
    pub option_ids_kept: Vec<Vec<u32>>,
    pub opt_budget_initial: i64,
    pub retruncated: bool,
    pub per: Option<usize>,
    pub state_ids_full_len: usize,
    pub state_ids_kept_len: usize,
    pub room: usize,
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub truncated_state_tokens: usize,
    pub sequence_len: usize,
    pub hit_max_len: bool,
}

impl PreparedSequence {
    /// The golden `tokens.<qid>` record.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut object = Map::new();
        object.insert("head_ids_full".to_owned(), u32_array(&self.head_ids_full));
        object.insert("head_ids_kept".to_owned(), u32_array(&self.head_ids_kept));
        object.insert(
            "option_ids_full".to_owned(),
            nested_u32_array(&self.option_ids_full),
        );
        object.insert(
            "option_ids_kept".to_owned(),
            nested_u32_array(&self.option_ids_kept),
        );
        object.insert(
            "opt_budget_initial".to_owned(),
            Value::from(self.opt_budget_initial),
        );
        object.insert("retruncated".to_owned(), Value::Bool(self.retruncated));
        object.insert(
            "per".to_owned(),
            self.per
                .map_or(Value::Null, |value| Value::from(value as u64)),
        );
        object.insert(
            "state_ids_full_len".to_owned(),
            Value::from(self.state_ids_full_len as u64),
        );
        object.insert(
            "state_ids_kept_len".to_owned(),
            Value::from(self.state_ids_kept_len as u64),
        );
        object.insert("room".to_owned(), Value::from(self.room as u64));
        object.insert("ids".to_owned(), u32_array(&self.ids));
        object.insert("markers".to_owned(), usize_array(&self.markers));
        object.insert(
            "truncated_state_tokens".to_owned(),
            Value::from(self.truncated_state_tokens as u64),
        );
        object.insert(
            "sequence_len".to_owned(),
            Value::from(self.sequence_len as u64),
        );
        object.insert("hit_max_len".to_owned(), Value::Bool(self.hit_max_len));
        Value::Object(object)
    }
}

fn u32_array(values: &[u32]) -> Value {
    Value::Array(values.iter().map(|value| Value::from(*value)).collect())
}
fn usize_array(values: &[usize]) -> Value {
    Value::Array(
        values
            .iter()
            .map(|value| Value::from(*value as u64))
            .collect(),
    )
}
fn nested_u32_array(values: &[Vec<u32>]) -> Value {
    Value::Array(values.iter().map(|value| u32_array(value)).collect())
}

/// Upstream `build_sequence(tok, state, q, max_len, head_max_len)` with the
/// default right-truncation policy. `state_text` is the already scrubbed
/// serialized state so it is tokenized once per question, as upstream does.
pub fn build_sequence(
    tokenizer: &LayaTokenizer,
    state_text: &str,
    question: &Question,
    max_len: usize,
    head_max_len: usize,
) -> Result<PreparedSequence> {
    let options = question.render_options();
    let head_text = format!(
        "{} question: {}",
        question.question_type().as_str(),
        tokenizer.scrub(question.instructions())
    );
    let head_ids_full = tokenizer.encode(&head_text)?;
    let mut option_ids_full = Vec::with_capacity(options.len());
    let mut option_ids_kept = Vec::with_capacity(options.len());
    for option in &options {
        let mut ids = tokenizer.encode(&format!(" {}", tokenizer.scrub(option)))?;
        ids.truncate(OPTION_TOKEN_CAP);
        option_ids_full.push(ids.clone());
        let mut kept = vec![tokenizer.mask_id];
        kept.extend(ids);
        option_ids_kept.push(kept);
    }
    let opt_budget_initial =
        head_max_len as i64 - option_ids_kept.iter().map(Vec::len).sum::<usize>() as i64;
    let (option_ids_kept, retruncated, per) = if opt_budget_initial < 16 {
        let per = std::cmp::max(
            4,
            head_max_len.saturating_sub(16) / std::cmp::max(1, option_ids_kept.len()),
        );
        let kept = option_ids_kept
            .iter()
            .map(|option| option[..std::cmp::min(option.len(), per)].to_vec())
            .collect();
        (kept, true, Some(per))
    } else {
        (option_ids_kept, false, None)
    };
    let opt_budget =
        head_max_len as i64 - option_ids_kept.iter().map(Vec::len).sum::<usize>() as i64;
    let head_keep = std::cmp::max(8, opt_budget.max(0) as usize);
    let head_ids_kept = head_ids_full[..std::cmp::min(head_ids_full.len(), head_keep)].to_vec();
    let mut ids = vec![tokenizer.cls_id];
    ids.extend(&head_ids_kept);
    ids.push(tokenizer.sep_id);
    let mut markers = Vec::with_capacity(option_ids_kept.len());
    for option in &option_ids_kept {
        markers.push(ids.len());
        ids.extend(option);
    }
    ids.push(tokenizer.sep_id);
    let room = max_len.saturating_sub(ids.len() + 1);
    let state_ids_full = tokenizer.encode(state_text)?;
    let state_ids_kept = state_ids_full[..std::cmp::min(state_ids_full.len(), room)].to_vec();
    let truncated_state_tokens = state_ids_full.len().saturating_sub(state_ids_kept.len());
    ids.extend(&state_ids_kept);
    ids.push(tokenizer.sep_id);
    let sequence_len = ids.len().min(max_len);
    ids.truncate(max_len);
    let markers = markers
        .into_iter()
        .filter(|marker| *marker < max_len)
        .collect();
    Ok(PreparedSequence {
        head_ids_full,
        head_ids_kept,
        option_ids_full,
        option_ids_kept,
        opt_budget_initial,
        retruncated,
        per,
        state_ids_full_len: state_ids_full.len(),
        state_ids_kept_len: state_ids_kept.len(),
        room,
        ids,
        markers,
        truncated_state_tokens,
        sequence_len,
        hit_max_len: sequence_len == max_len,
    })
}

/// The five padded tensors upstream `collate_items` produces, row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Batch {
    pub rows: usize,
    pub length: usize,
    pub marker_slots: usize,
    /// `rows × length`, right-padded with `pad_id`.
    pub input_ids: Vec<u32>,
    /// `rows × length`, 1 for real tokens.
    pub attention_mask: Vec<u8>,
    /// `rows × marker_slots`, zero where `marker_mask` is false.
    pub marker_pos: Vec<u32>,
    pub marker_mask: Vec<bool>,
    /// Question-type index per row.
    pub qtype: Vec<u8>,
    pub pad_id: u32,
    /// `attention_mask.sum()`: the usage token count.
    pub n_tokens: u64,
}

impl Batch {
    #[must_use]
    pub fn marker_count(&self, row: usize) -> usize {
        self.marker_mask[row * self.marker_slots..(row + 1) * self.marker_slots]
            .iter()
            .filter(|value| **value)
            .count()
    }

    /// The golden `batch` record.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let rows2 =
            |flat: &[u32], width: usize| Value::Array(flat.chunks(width).map(u32_array).collect());
        let mut object = Map::new();
        object.insert("input_ids".to_owned(), rows2(&self.input_ids, self.length));
        object.insert(
            "attention_mask".to_owned(),
            Value::Array(
                self.attention_mask
                    .chunks(self.length)
                    .map(|row| Value::Array(row.iter().map(|v| Value::from(*v)).collect()))
                    .collect(),
            ),
        );
        object.insert(
            "marker_pos".to_owned(),
            rows2(&self.marker_pos, self.marker_slots),
        );
        object.insert(
            "marker_mask".to_owned(),
            Value::Array(
                self.marker_mask
                    .chunks(self.marker_slots)
                    .map(|row| Value::Array(row.iter().map(|v| Value::Bool(*v)).collect()))
                    .collect(),
            ),
        );
        object.insert(
            "qtype".to_owned(),
            Value::Array(self.qtype.iter().map(|v| Value::from(*v)).collect()),
        );
        object.insert("pad_id".to_owned(), Value::from(self.pad_id));
        object.insert("n_tokens".to_owned(), Value::from(self.n_tokens));
        Value::Object(object)
    }
}

/// Upstream `collate_items([items], pad_id)`.
pub fn collate(items: &[PreparedSequence], qtypes: &[QuestionType], pad_id: u32) -> Result<Batch> {
    if items.is_empty() || items.len() != qtypes.len() {
        return Err(LayaError::InvalidRequest(
            "collate requires at least one sequence and one qtype per sequence".to_owned(),
        ));
    }
    let length = items.iter().map(|item| item.ids.len()).max().unwrap_or(0);
    let marker_slots = items
        .iter()
        .map(|item| item.markers.len())
        .max()
        .unwrap_or(0);
    let rows = items.len();
    let mut input_ids = vec![pad_id; rows * length];
    let mut attention_mask = vec![0_u8; rows * length];
    let mut marker_pos = vec![0_u32; rows * marker_slots];
    let mut marker_mask = vec![false; rows * marker_slots];
    let mut n_tokens = 0_u64;
    for (row, item) in items.iter().enumerate() {
        let start = row * length;
        input_ids[start..start + item.ids.len()].copy_from_slice(&item.ids);
        attention_mask[start..start + item.ids.len()].fill(1);
        n_tokens += item.ids.len() as u64;
        let mstart = row * marker_slots;
        for (index, marker) in item.markers.iter().enumerate() {
            marker_pos[mstart + index] = *marker as u32;
            marker_mask[mstart + index] = true;
        }
    }
    Ok(Batch {
        rows,
        length,
        marker_slots,
        input_ids,
        attention_mask,
        marker_pos,
        marker_mask,
        qtype: qtypes.iter().map(|kind| kind.index() as u8).collect(),
        pad_id,
        n_tokens,
    })
}

/// A request after preprocessing: sequences, the collated batch and the
/// per-question facts postprocessing needs.
#[derive(Clone, Debug)]
pub struct PreparedRequest {
    /// Scrubbed `serialize_state(state)`.
    pub state_text: String,
    pub sequences: Vec<PreparedSequence>,
    pub batch: Batch,
    /// Per question: number of scored options (`len(markers)`).
    pub option_counts: Vec<usize>,
}

impl PreparedRequest {
    /// Total state tokens dropped by right truncation across all rows.
    #[must_use]
    pub fn truncated_state_tokens(&self) -> usize {
        self.sequences
            .iter()
            .map(|sequence| sequence.truncated_state_tokens)
            .sum()
    }
}

/// Upstream `Agent.system_one` preprocessing half: serialize, build every
/// sequence, apply the two upstream error rules, collate.
pub fn prepare(
    tokenizer: &LayaTokenizer,
    request: &Request,
    max_len: usize,
    head_max_len: usize,
) -> Result<PreparedRequest> {
    let state_text = tokenizer.scrub(&serialize_state(&request.state));
    let mut sequences = Vec::with_capacity(request.questions.len());
    let mut qtypes = Vec::with_capacity(request.questions.len());
    let mut option_counts = Vec::with_capacity(request.questions.len());
    for (id, question) in &request.questions {
        if matches!(question, Question::Choice { criteria, .. } if criteria.len() < 2) {
            // Upstream fails later, inside topk(2) on the action features.
            // Refusing here keeps the error deterministic and model-free.
            return Err(LayaError::SingleOptionChoice {
                question: id.clone(),
            });
        }
        let sequence = build_sequence(tokenizer, &state_text, question, max_len, head_max_len)?;
        if sequence.markers.len() != question.option_count() {
            return Err(LayaError::OptionsExceedBudget {
                question: id.clone(),
                head_max_len,
            });
        }
        option_counts.push(sequence.markers.len());
        qtypes.push(question.question_type());
        sequences.push(sequence);
    }
    let batch = collate(&sequences, &qtypes, tokenizer.pad_id())?;
    Ok(PreparedRequest {
        state_text,
        sequences,
        batch,
        option_counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sequence(ids: Vec<u32>, markers: Vec<usize>) -> PreparedSequence {
        PreparedSequence {
            head_ids_full: vec![],
            head_ids_kept: vec![],
            option_ids_full: vec![],
            option_ids_kept: vec![],
            opt_budget_initial: 0,
            retruncated: false,
            per: None,
            state_ids_full_len: 0,
            state_ids_kept_len: 0,
            room: 0,
            sequence_len: ids.len(),
            hit_max_len: false,
            truncated_state_tokens: 0,
            ids,
            markers,
        }
    }

    #[test]
    fn collate_pads_right_and_leaves_padded_markers_at_zero() {
        let batch = collate(
            &[
                sequence(vec![1, 2, 3], vec![1]),
                sequence(vec![4, 5], vec![0, 1]),
            ],
            &[QuestionType::Noul, QuestionType::Choice],
            9,
        )
        .unwrap();
        assert_eq!(batch.input_ids, [1, 2, 3, 4, 5, 9]);
        assert_eq!(batch.attention_mask, [1, 1, 1, 1, 1, 0]);
        assert_eq!(batch.marker_pos, [1, 0, 0, 1]);
        assert_eq!(batch.marker_mask, [true, false, true, true]);
        assert_eq!(batch.qtype, [2, 0]);
        assert_eq!(batch.n_tokens, 5);
        assert_eq!(batch.marker_count(0), 1);
        assert_eq!(batch.to_value()["input_ids"][1][2], 9);
    }
}
