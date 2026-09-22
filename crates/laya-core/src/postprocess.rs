//! Upstream `Agent.system_one` postprocessing: temperature selection,
//! calibrated softmax, normalized-entropy confidence, expected Score, the
//! action probability, and the four-decimal native answer JSON.
//!
//! Arithmetic follows the NumPy dtypes upstream ends up with: logits and
//! probabilities are float32, the confidence statistic is float32, the
//! expected score is float64 (`np.arange(k) * p` promotes to float64), and
//! rounding is Python `round(x, 4)` (correctly rounded, ties to even).

use serde_json::{Map, Value};

use crate::{
    assets::AgentConfig,
    backend::ModelOutput,
    preprocess::PreparedRequest,
    request::{Question, QuestionType, Request},
    LayaError, Result,
};

/// Upstream `model` field of every native response.
pub const NATIVE_MODEL_NAME: &str = "laya-rl-agent";
/// Minimum temperature denominator (`max(1e-3, t)`).
pub const MIN_TEMPERATURE: f64 = 1e-3;
/// Decimal places of the native answer JSON.
pub const NATIVE_DECIMALS: usize = 4;

/// Upstream `temp_bucket(qtype, k)`.
#[must_use]
pub fn temp_bucket(kind: QuestionType, k: usize) -> String {
    let size = if k <= 2 {
        "2"
    } else if k <= 5 {
        "3-5"
    } else if k <= 10 {
        "6-10"
    } else {
        "11+"
    };
    format!("{}:{size}", kind.as_str())
}

/// `temperature_by_options.get(bucket, temperature[qtype])`.
#[must_use]
pub fn select_temperature(config: &AgentConfig, kind: QuestionType, k: usize) -> (f64, String) {
    let bucket = temp_bucket(kind, k);
    let temperature = config
        .temperature_by_options
        .get(&bucket)
        .copied()
        .unwrap_or(config.temperature[kind.index()]);
    (temperature, bucket)
}

/// float32 stable softmax of `logits / max(1e-3, temperature)`.
#[must_use]
pub fn calibrated_softmax(logits: &[f32], temperature: f64) -> Vec<f32> {
    let scale = temperature.max(MIN_TEMPERATURE) as f32;
    let scaled: Vec<f32> = logits.iter().map(|value| value / scale).collect();
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = scaled.iter().map(|value| (value - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    for value in &mut out {
        *value /= sum;
    }
    out
}

/// Upstream `confidence_from_probs(p, k)`: `1 - H(p) / log(k)`, clipped.
#[must_use]
pub fn normalized_entropy_confidence(probabilities: &[f32]) -> f32 {
    let k = probabilities.len();
    if k < 2 {
        return 1.0;
    }
    let entropy: f32 = probabilities
        .iter()
        .map(|p| -(p * p.clamp(1e-12, 1.0).ln()))
        .sum();
    (1.0 - entropy / (k as f32).ln()).clamp(0.0, 1.0)
}

/// `float((np.arange(k) * p).sum())`.
#[must_use]
pub fn expected_score(probabilities: &[f32]) -> f64 {
    probabilities
        .iter()
        .enumerate()
        .map(|(index, p)| index as f64 * f64::from(*p))
        .sum()
}

/// First index of the maximum (`np.argmax`).
#[must_use]
pub fn first_argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

/// Python `round(value, decimals)` for finite inputs.
#[must_use]
pub fn round_decimals(value: f64, decimals: usize) -> f64 {
    format!("{value:.decimals$}").parse().unwrap_or(value)
}

/// Everything computed for one question, before rounding.
#[derive(Clone, Debug, PartialEq)]
pub struct QuestionResult {
    pub id: String,
    pub question_type: QuestionType,
    /// `k`: options actually scored.
    pub option_count: usize,
    pub temperature: f64,
    pub temp_bucket: String,
    pub raw_logits: Vec<f32>,
    pub probabilities: Vec<f32>,
    /// Normalized-entropy statistic (upstream `conf_score`), for every type.
    pub confidence: f32,
    /// Score only.
    pub expected_score: Option<f64>,
    pub act_logits: Vec<f32>,
    pub act_probs: Vec<f32>,
    pub argmax: usize,
}

impl QuestionResult {
    /// Upstream `act_probability`: probability of the first action.
    #[must_use]
    pub fn act_probability(&self) -> f32 {
        self.act_probs.first().copied().unwrap_or(f32::NAN)
    }

    /// Noul: `p[1]` is the probability that the statement holds.
    #[must_use]
    pub fn noul_probability(&self) -> Option<f32> {
        (self.question_type == QuestionType::Noul).then(|| self.probabilities[1])
    }
}

/// A complete evaluation of one request.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    pub results: Vec<QuestionResult>,
    /// `attention_mask.sum()` across all rows.
    pub input_tokens: u64,
    /// State tokens dropped by right truncation, summed over rows.
    pub truncated_state_tokens: usize,
    /// Rows whose sequence reached `max_len`.
    pub rows_at_max_len: usize,
    /// Backend label, e.g. `mlx-metal-f32`.
    pub backend: String,
    /// `[rows * hidden_size]` post-head CLS states. Diagnostic only; not
    /// part of the answer and not gated by tolerances.
    pub hidden_cls: Vec<f32>,
}

impl Evaluation {
    /// The upstream `Agent.system_one` return value, including its
    /// four-decimal rounding and `action` extension.
    #[must_use]
    pub fn native_json(&self, request: &Request) -> Value {
        let mut answers = Map::new();
        for (result, (id, question)) in self.results.iter().zip(&request.questions) {
            answers.insert(id.clone(), native_answer(result, question));
        }
        let mut usage = Map::new();
        usage.insert("input_tokens".to_owned(), Value::from(self.input_tokens));
        usage.insert("output_tokens".to_owned(), Value::from(0));
        let mut out = Map::new();
        out.insert(
            "model".to_owned(),
            Value::String(NATIVE_MODEL_NAME.to_owned()),
        );
        out.insert("answers".to_owned(), Value::Object(answers));
        out.insert("usage".to_owned(), Value::Object(usage));
        Value::Object(out)
    }
}

fn rounded(value: f64) -> Value {
    Value::from(round_decimals(value, NATIVE_DECIMALS))
}

fn native_answer(result: &QuestionResult, question: &Question) -> Value {
    let mut answer = Map::new();
    let kind = result.question_type.as_str();
    answer.insert("type".to_owned(), Value::String(kind.to_owned()));
    let confidence = rounded(f64::from(result.confidence));
    match question {
        Question::Choice { criteria, .. } => {
            let keys: Vec<&String> = criteria.iter().map(|(key, _)| key).collect();
            answer.insert(
                "choice".to_owned(),
                Value::String(keys[result.argmax].clone()),
            );
            let mut probabilities = Map::new();
            for (key, p) in keys.iter().zip(&result.probabilities) {
                probabilities.insert((*key).clone(), rounded(f64::from(*p)));
            }
            answer.insert("probabilities".to_owned(), Value::Object(probabilities));
            answer.insert("confidence".to_owned(), confidence);
        }
        Question::Score { levels, .. } => {
            answer.insert(
                "score".to_owned(),
                rounded(result.expected_score.unwrap_or(f64::NAN)),
            );
            let mut legend = Map::new();
            for (index, level) in levels.iter().enumerate() {
                legend.insert(index.to_string(), level.clone());
            }
            answer.insert("legend".to_owned(), Value::Object(legend));
            let mut probabilities = Map::new();
            for (index, p) in result.probabilities.iter().enumerate() {
                probabilities.insert(index.to_string(), rounded(f64::from(*p)));
            }
            answer.insert("probabilities".to_owned(), Value::Object(probabilities));
            answer.insert("confidence".to_owned(), confidence);
        }
        Question::Noul { .. } => {
            let p_true = f64::from(result.probabilities[1]);
            answer.insert("noul".to_owned(), rounded(p_true));
            answer.insert("confidence".to_owned(), rounded(p_true.max(1.0 - p_true)));
        }
    }
    let mut action = Map::new();
    action.insert(
        "act_probability".to_owned(),
        rounded(f64::from(result.act_probability())),
    );
    answer.insert("action".to_owned(), Value::Object(action));
    Value::Object(answer)
}

/// Turn a forward pass into per-question results.
pub fn postprocess(
    request: &Request,
    prepared: &PreparedRequest,
    output: &ModelOutput,
    config: &AgentConfig,
    backend: &str,
) -> Result<Evaluation> {
    let rows = prepared.batch.rows;
    if output.rows != rows || output.marker_slots != prepared.batch.marker_slots {
        return Err(LayaError::Inference(format!(
            "backend returned {}×{} logits for a {}×{} batch",
            output.rows, output.marker_slots, rows, prepared.batch.marker_slots
        )));
    }
    if output.act_logits.len() != rows * output.n_act {
        return Err(LayaError::Inference(
            "backend returned a malformed action-logit block".to_owned(),
        ));
    }
    let mut results = Vec::with_capacity(rows);
    for (row, (id, question)) in request.questions.iter().enumerate() {
        let k = prepared.option_counts[row];
        let start = row * output.marker_slots;
        let raw_logits = output.raw_logits[start..start + k].to_vec();
        if raw_logits.iter().any(|value| !value.is_finite()) {
            return Err(LayaError::Inference(format!(
                "non-finite option logits for question {id:?}"
            )));
        }
        let kind = question.question_type();
        let (temperature, temp_bucket) = select_temperature(config, kind, k);
        let probabilities = calibrated_softmax(&raw_logits, temperature);
        let confidence = normalized_entropy_confidence(&probabilities);
        let act_logits = output.act_logits[row * output.n_act..(row + 1) * output.n_act].to_vec();
        let act_probs = calibrated_softmax(&act_logits, 1.0);
        if probabilities
            .iter()
            .chain(&act_probs)
            .any(|value| !value.is_finite())
        {
            return Err(LayaError::Inference(format!(
                "non-finite probabilities for question {id:?}"
            )));
        }
        let argmax = first_argmax(&probabilities);
        results.push(QuestionResult {
            id: id.clone(),
            question_type: kind,
            option_count: k,
            temperature,
            temp_bucket,
            expected_score: (kind == QuestionType::Score).then(|| expected_score(&probabilities)),
            raw_logits,
            probabilities,
            confidence,
            act_logits,
            act_probs,
            argmax,
        });
    }
    Ok(Evaluation {
        results,
        input_tokens: prepared.batch.n_tokens,
        truncated_state_tokens: prepared.truncated_state_tokens(),
        rows_at_max_len: prepared
            .sequences
            .iter()
            .filter(|sequence| sequence.hit_max_len)
            .count(),
        backend: backend.to_owned(),
        hidden_cls: output.hidden_cls.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_and_temperature_selection_follow_upstream() {
        assert_eq!(temp_bucket(QuestionType::Choice, 2), "choice:2");
        assert_eq!(temp_bucket(QuestionType::Choice, 5), "choice:3-5");
        assert_eq!(temp_bucket(QuestionType::Score, 6), "score:6-10");
        assert_eq!(temp_bucket(QuestionType::Noul, 11), "noul:11+");
        let config: AgentConfig = serde_json::from_str(
            r#"{"head_layers":2,"max_len":512,"head_max_len":192,"temperature":[1.5,2.0,3.0],"temperature_by_options":{"choice:3-5":0.7}}"#,
        )
        .unwrap();
        assert_eq!(
            select_temperature(&config, QuestionType::Choice, 4),
            (0.7, "choice:3-5".to_owned())
        );
        assert_eq!(select_temperature(&config, QuestionType::Noul, 2).0, 3.0);
    }

    #[test]
    fn softmax_confidence_and_score_match_numpy_semantics() {
        let p = calibrated_softmax(&[1000.0, 999.0], 1.0);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[0] > p[1]);
        let uniform = calibrated_softmax(&[0.0, 0.0, 0.0], 0.0);
        assert!((normalized_entropy_confidence(&uniform)).abs() < 1e-6);
        assert_eq!(normalized_entropy_confidence(&[1.0]), 1.0);
        assert!((expected_score(&[0.25, 0.5, 0.25]) - 1.0).abs() < 1e-7);
        assert_eq!(first_argmax(&[0.5, 0.5]), 0);
    }

    #[test]
    fn rounding_is_python_round_half_even_on_exact_ties() {
        assert_eq!(round_decimals(0.949_761_927_127_838_1, 4), 0.9498);
        assert_eq!(round_decimals(0.000_05, 4), 0.0001); // 0.00005 is slightly above the tie in binary
        assert_eq!(round_decimals(0.5, 0), 0.0);
        assert_eq!(round_decimals(1.5, 0), 2.0);
        assert_eq!(round_decimals(0.712_592_720_985_412_6, 4), 0.7126);
    }
}
