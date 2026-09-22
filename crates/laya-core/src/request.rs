//! Native request types and the upstream `Agent._to_internal` /
//! `render_options` semantics.
//!
//! The wire form (`{"type","instructions","criteria"}` per question) is the
//! same one upstream accepts. `Question::from_wire` applies exactly the
//! upstream conversions: list-form choice criteria become `{label: null}`;
//! non-string instructions are ASCII JSON text; a missing `instructions`
//! key is an error, as upstream's `qdef["instructions"]` is.

use serde_json::{Map, Value};

use crate::{
    pyjson::{json_text, render_criterion},
    LayaError, Result,
};

/// Upstream `QTYPES` indices; also the row of the type embedding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuestionType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl QuestionType {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }

    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// One typed question. `instructions` is already the upstream `ins` text.
#[derive(Clone, Debug, PartialEq)]
pub enum Question {
    /// Labels in insertion order with their (possibly null) descriptions.
    Choice {
        instructions: String,
        criteria: Vec<(String, Value)>,
    },
    /// Ordered rubric levels.
    Score {
        instructions: String,
        levels: Vec<Value>,
    },
    /// `criteria` is the caller's object (or null) unchanged; the `true` and
    /// `false` keys describe the two fixed options.
    Noul {
        instructions: String,
        criteria: Value,
    },
}

impl Question {
    #[must_use]
    pub const fn question_type(&self) -> QuestionType {
        match self {
            Self::Choice { .. } => QuestionType::Choice,
            Self::Score { .. } => QuestionType::Score,
            Self::Noul { .. } => QuestionType::Noul,
        }
    }

    #[must_use]
    pub fn instructions(&self) -> &str {
        match self {
            Self::Choice { instructions, .. }
            | Self::Score { instructions, .. }
            | Self::Noul { instructions, .. } => instructions,
        }
    }

    /// Number of options the network scores for this question.
    #[must_use]
    pub fn option_count(&self) -> usize {
        match self {
            Self::Choice { criteria, .. } => criteria.len(),
            Self::Score { levels, .. } => levels.len(),
            Self::Noul { .. } => 2,
        }
    }

    /// Upstream `Agent._to_internal` applied to one wire question object.
    pub fn from_wire(id: &str, qdef: &Value) -> Result<Self> {
        let object = qdef.as_object().ok_or_else(|| {
            LayaError::InvalidRequest(format!("question {id:?} must be an object"))
        })?;
        let kind = object.get("type").and_then(Value::as_str).ok_or_else(|| {
            LayaError::InvalidRequest(format!("question {id:?}: type must be a string"))
        })?;
        let instructions = match object.get("instructions") {
            None => {
                return Err(LayaError::InvalidRequest(format!(
                    "question {id:?}: instructions is required (upstream KeyError)"
                )))
            }
            Some(Value::String(text)) => text.clone(),
            Some(other) => json_text(other, true),
        };
        let criteria = object.get("criteria").unwrap_or(&Value::Null);
        match kind {
            "choice" => {
                let criteria = match criteria {
                    Value::Array(values) => {
                        let mut pairs = Vec::with_capacity(values.len());
                        for value in values {
                            let label = value.as_str().ok_or_else(|| {
                                LayaError::InvalidRequest(format!(
                                    "question {id:?}: list-form choice criteria must be strings"
                                ))
                            })?;
                            pairs.push((label.to_owned(), Value::Null));
                        }
                        dedupe_labels(id, pairs)?
                    }
                    Value::Object(map) => map
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect(),
                    _ => {
                        return Err(LayaError::InvalidRequest(format!(
                            "question {id:?}: choice criteria must be an object or a list"
                        )))
                    }
                };
                Ok(Self::Choice {
                    instructions,
                    criteria,
                })
            }
            "score" => {
                let levels = criteria
                    .as_array()
                    .ok_or_else(|| {
                        LayaError::InvalidRequest(format!(
                            "question {id:?}: score criteria must be an array"
                        ))
                    })?
                    .clone();
                Ok(Self::Score {
                    instructions,
                    levels,
                })
            }
            "noul" => {
                if !(criteria.is_null() || criteria.is_object()) {
                    return Err(LayaError::InvalidRequest(format!(
                        "question {id:?}: noul criteria must be an object or null"
                    )));
                }
                Ok(Self::Noul {
                    instructions,
                    criteria: criteria.clone(),
                })
            }
            other => Err(LayaError::InvalidRequest(format!(
                "question {id:?}: unknown type {other:?}"
            ))),
        }
    }

    /// Upstream `render_options`: option texts in label order; Noul is
    /// always `[false, true]`.
    #[must_use]
    pub fn render_options(&self) -> Vec<String> {
        match self {
            Self::Choice { criteria, .. } => criteria
                .iter()
                .map(|(key, value)| {
                    if is_empty_description(value) {
                        key.clone()
                    } else {
                        format!("{}: {}", key, render_criterion(value))
                    }
                })
                .collect(),
            Self::Score { levels, .. } => levels
                .iter()
                .enumerate()
                .map(|(i, value)| format!("level {}: {}", i, render_criterion(value)))
                .collect(),
            Self::Noul { criteria, .. } => {
                let text = |key: &str, default: &str| match criteria.get(key) {
                    Some(value) if !is_empty_description(value) => render_criterion(value),
                    _ => default.to_owned(),
                };
                vec![
                    format!(
                        "false: {}",
                        text("false", "no, the statement does not hold")
                    ),
                    format!("true: {}", text("true", "yes, the statement holds")),
                ]
            }
        }
    }

    /// The upstream internal dict `{"t", "ins", "crit"}`, for golden
    /// comparison and diagnostics.
    #[must_use]
    pub fn internal_value(&self) -> Value {
        let crit = match self {
            Self::Choice { criteria, .. } => Value::Object(
                criteria
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
            ),
            Self::Score { levels, .. } => Value::Array(levels.clone()),
            Self::Noul { criteria, .. } => criteria.clone(),
        };
        let mut internal = Map::new();
        internal.insert(
            "t".to_owned(),
            Value::String(self.question_type().as_str().to_owned()),
        );
        internal.insert(
            "ins".to_owned(),
            Value::String(self.instructions().to_owned()),
        );
        internal.insert("crit".to_owned(), crit);
        Value::Object(internal)
    }
}

/// Python: `v is None or v == ""`. `0`, `false`, `[]` are real descriptions.
fn is_empty_description(value: &Value) -> bool {
    value.is_null() || value.as_str() == Some("")
}

fn dedupe_labels(id: &str, pairs: Vec<(String, Value)>) -> Result<Vec<(String, Value)>> {
    // Python `{c: None for c in crit}` collapses duplicates silently; this
    // library reports them so a caller never loses an option unknowingly.
    let mut seen = std::collections::HashSet::with_capacity(pairs.len());
    for (label, _) in &pairs {
        if !seen.insert(label.as_str()) {
            return Err(LayaError::InvalidRequest(format!(
                "question {id:?}: duplicate choice label {label:?}"
            )));
        }
    }
    Ok(pairs)
}

/// A complete native request: one state shared by every question.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub state: Value,
    pub questions: Vec<(String, Question)>,
}

impl Request {
    pub fn new(state: Value, questions: Vec<(String, Question)>) -> Result<Self> {
        if questions.is_empty() {
            return Err(LayaError::InvalidRequest(
                "questions must contain at least one question".to_owned(),
            ));
        }
        let mut seen = std::collections::HashSet::with_capacity(questions.len());
        for (id, _) in &questions {
            if !seen.insert(id.as_str()) {
                return Err(LayaError::InvalidRequest(format!(
                    "duplicate question id {id:?}"
                )));
            }
        }
        Ok(Self { state, questions })
    }

    /// Parse the upstream wire form `{"state": ..., "questions": {...}}`.
    pub fn from_wire(value: &Value) -> Result<Self> {
        let object = value
            .as_object()
            .ok_or_else(|| LayaError::InvalidRequest("request must be an object".to_owned()))?;
        let state = object.get("state").cloned().unwrap_or(Value::Null);
        let questions = object
            .get("questions")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                LayaError::InvalidRequest("request.questions must be an object".to_owned())
            })?;
        let mut parsed = Vec::with_capacity(questions.len());
        for (id, qdef) in questions {
            parsed.push((id.clone(), Question::from_wire(id, qdef)?));
        }
        Self::new(state, parsed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(text: &str) -> Value {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn ascii_instruction_uses_surrogate_pairs() {
        let question = Question::from_wire(
            "q",
            &value(r#"{"type":"noul","instructions":{"q":"café 🙂"}}"#),
        )
        .unwrap();
        assert_eq!(
            question.instructions(),
            r#"{"q": "caf\u00e9 \ud83d\ude42"}"#
        );
        assert_eq!(question.internal_value()["crit"], Value::Null);
    }

    #[test]
    fn option_rendering_preserves_falsey_values() {
        let question = Question::from_wire(
            "q",
            &value(
                r#"{"type":"choice","instructions":"x","criteria":{"a":null,"b":"","c":0,"d":false}}"#,
            ),
        )
        .unwrap();
        assert_eq!(question.render_options(), ["a", "b", "c: 0", "d: false"]);
    }

    #[test]
    fn list_criteria_become_null_descriptions_and_noul_defaults_apply() {
        let choice = Question::from_wire(
            "q",
            &value(r#"{"type":"choice","instructions":"x","criteria":["b","a"]}"#),
        )
        .unwrap();
        assert_eq!(choice.render_options(), ["b", "a"]);
        assert_eq!(
            choice.internal_value()["crit"],
            value(r#"{"b":null,"a":null}"#)
        );
        let noul =
            Question::from_wire("n", &value(r#"{"type":"noul","instructions":"x"}"#)).unwrap();
        assert_eq!(
            noul.render_options(),
            [
                "false: no, the statement does not hold",
                "true: yes, the statement holds"
            ]
        );
        let described = Question::from_wire(
            "n",
            &value(
                r#"{"type":"noul","instructions":"x","criteria":{"true":"yes","false":{"k":1}}}"#,
            ),
        )
        .unwrap();
        assert_eq!(
            described.render_options(),
            ["false: {\"k\": 1}", "true: yes"]
        );
    }

    #[test]
    fn missing_instructions_and_bad_shapes_are_errors() {
        assert!(matches!(
            Question::from_wire("q", &value(r#"{"type":"noul"}"#)),
            Err(LayaError::InvalidRequest(_))
        ));
        assert!(Question::from_wire(
            "q",
            &value(r#"{"type":"score","instructions":"x","criteria":{}}"#)
        )
        .is_err());
        assert!(
            Question::from_wire("q", &value(r#"{"type":"other","instructions":"x"}"#)).is_err()
        );
        assert!(Question::from_wire(
            "q",
            &value(r#"{"type":"choice","instructions":"x","criteria":["a","a"]}"#)
        )
        .is_err());
        assert!(Request::from_wire(&value(r#"{"state":"s","questions":{}}"#)).is_err());
    }
}
