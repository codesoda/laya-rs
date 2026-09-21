use std::fmt::Write;
use std::path::Path;

use serde_json::{Map, Number, Value};
use tokenizers::{EncodeInput, Tokenizer};

#[derive(Debug, Clone)]
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

fn escape_json_string(value: &str, ascii_only: bool) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                write!(out, "\\u{:04x}", c as u32).expect("writing to String cannot fail");
            }
            c if ascii_only && (c as u32) > 0x7f => {
                let cp = c as u32;
                if cp <= 0xffff {
                    write!(out, "\\u{:04x}", cp).expect("writing to String cannot fail");
                } else {
                    let n = cp - 0x1_0000;
                    let high = 0xd800 + (n >> 10);
                    let low = 0xdc00 + (n & 0x3ff);
                    write!(out, "\\u{:04x}\\u{:04x}", high, low)
                        .expect("writing to String cannot fail");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn exponent_parts(s: &str) -> Option<(&str, i32)> {
    let (mantissa, exponent) = s.split_once(['e', 'E'])?;
    Some((mantissa, exponent.parse().ok()?))
}

fn digits_and_decimal(mantissa: &str) -> (bool, String, i32) {
    let negative = mantissa.starts_with('-');
    let body = mantissa.strip_prefix('-').unwrap_or(mantissa);
    let (whole, fraction) = body.split_once('.').unwrap_or((body, ""));
    let decimal = whole.len() as i32;
    (negative, format!("{}{}", whole, fraction), decimal)
}

fn fixed_from_parts(negative: bool, digits: &str, decimal: i32) -> String {
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if decimal <= 0 {
        out.push_str("0.");
        for _ in 0..-decimal {
            out.push('0');
        }
        out.push_str(digits);
    } else if decimal as usize >= digits.len() {
        out.push_str(digits);
        for _ in 0..(decimal as usize - digits.len()) {
            out.push('0');
        }
        out.push_str(".0");
    } else {
        let at = decimal as usize;
        out.push_str(&digits[..at]);
        out.push('.');
        out.push_str(&digits[at..]);
    }
    out
}

fn scientific_from_parts(negative: bool, digits: &str, decimal: i32) -> String {
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    let (first, rest) = digits.split_at(1);
    out.push_str(first);
    if !rest.is_empty() {
        out.push('.');
        out.push_str(rest);
    }
    let exponent = decimal - 1;
    if exponent < 0 {
        if exponent > -10 {
            write!(out, "e-0{}", -exponent).expect("writing to String cannot fail");
        } else {
            write!(out, "e{}", exponent).expect("writing to String cannot fail");
        }
    } else {
        write!(out, "e+{}", exponent).expect("writing to String cannot fail");
    }
    out
}

/// Format an f64 with Python's json/repr spelling, given ryu's shortest-roundtrip digits.
/// Python uses fixed notation for 1e-4 <= abs(x) < 1e16 and exponent notation otherwise.
pub fn python_float(value: f64) -> String {
    assert!(value.is_finite(), "JSON cannot represent non-finite floats");
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .to_owned();
    }
    let mut buffer = ryu::Buffer::new();
    let raw = buffer.format_finite(value);
    let negative = raw.starts_with('-');
    let (raw_mantissa, raw_exp) = match exponent_parts(raw) {
        Some((m, e)) => (m, Some(e)),
        None => (raw, None),
    };
    let (negative_mantissa, mut digits, mut decimal) = digits_and_decimal(raw_mantissa);
    debug_assert_eq!(negative, negative_mantissa);
    if let Some(exp) = raw_exp {
        decimal += exp;
    }
    let leading_zeroes = digits.bytes().take_while(|byte| *byte == b'0').count();
    if leading_zeroes > 0 {
        digits.drain(..leading_zeroes);
        decimal -= leading_zeroes as i32;
    }
    if digits.is_empty() {
        digits.push('0');
    }
    let abs = value.abs();
    let exponent_notation = abs != 0.0 && !(1e-4..1e16).contains(&abs);
    if exponent_notation {
        scientific_from_parts(negative, &digits, decimal)
    } else {
        fixed_from_parts(negative, &digits, decimal)
    }
}

fn number_text(number: &Number) -> String {
    if number.is_i64() || number.is_u64() {
        number.to_string()
    } else {
        python_float(
            number
                .as_f64()
                .expect("serde JSON number must be representable as f64"),
        )
    }
}

fn json_text(value: &Value, ascii_only: bool, separators: bool) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(v) => v.to_string(),
        Value::Number(v) => number_text(v),
        Value::String(v) => escape_json_string(v, ascii_only),
        Value::Array(values) => {
            let joiner = if separators { ", " } else { "," };
            let mut out = String::from("[");
            for (i, value) in values.iter().enumerate() {
                if i > 0 {
                    out.push_str(joiner);
                }
                out.push_str(&json_text(value, ascii_only, separators));
            }
            out.push(']');
            out
        }
        Value::Object(values) => {
            let joiner = if separators { ", " } else { "," };
            let colon = if separators { ": " } else { ":" };
            let mut out = String::from("{");
            for (i, (key, value)) in values.iter().enumerate() {
                if i > 0 {
                    out.push_str(joiner);
                }
                out.push_str(&escape_json_string(key, ascii_only));
                out.push_str(colon);
                out.push_str(&json_text(value, ascii_only, separators));
            }
            out.push('}');
            out
        }
    }
}

/// Python: json.dumps(state, ensure_ascii=False), preserving insertion order and spacing.
pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(value) => value.clone(),
        _ => json_text(state, false, true),
    }
}

/// Python: json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str).
pub fn render_criterion(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        _ => json_text(value, false, true),
    }
}

/// Python Agent._to_internal. The non-string instruction path deliberately uses ASCII JSON.
pub fn to_internal(qdef: &Value) -> Value {
    let object = qdef
        .as_object()
        .expect("question definition must be an object");
    let t = object
        .get("type")
        .and_then(Value::as_str)
        .expect("question type must be a string");
    let criteria = object.get("criteria").cloned().unwrap_or(Value::Null);
    let criteria = if t == "choice" {
        if let Value::Array(values) = criteria {
            let mut map = Map::new();
            for value in values {
                let key = value
                    .as_str()
                    .expect("choice list criteria must be strings");
                map.insert(key.to_owned(), Value::Null);
            }
            Value::Object(map)
        } else {
            criteria
        }
    } else {
        criteria
    };
    let instructions = object
        .get("instructions")
        .expect("question requires instructions");
    let instructions = match instructions {
        Value::String(value) => Value::String(value.clone()),
        value => Value::String(json_text(value, true, true)),
    };
    let mut internal = Map::new();
    internal.insert("t".to_owned(), Value::String(t.to_owned()));
    internal.insert("ins".to_owned(), instructions);
    internal.insert("crit".to_owned(), criteria);
    Value::Object(internal)
}

pub fn render_options(internal: &Value) -> Vec<String> {
    let object = internal
        .as_object()
        .expect("internal question must be an object");
    let t = object.get("t").and_then(Value::as_str).unwrap();
    let criteria = object.get("crit").unwrap_or(&Value::Null);
    match t {
        "choice" => criteria
            .as_object()
            .expect("choice criteria must be an object")
            .iter()
            .map(|(key, value)| {
                if value.is_null() || value.as_str() == Some("") {
                    key.clone()
                } else {
                    format!("{}: {}", key, render_criterion(value))
                }
            })
            .collect(),
        "score" => criteria
            .as_array()
            .expect("score criteria must be an array")
            .iter()
            .enumerate()
            .map(|(i, value)| format!("level {}: {}", i, render_criterion(value)))
            .collect(),
        "noul" => {
            let criteria = criteria.as_object();
            let false_value = criteria.and_then(|v| v.get("false"));
            let true_value = criteria.and_then(|v| v.get("true"));
            let false_text = match false_value {
                Some(value) if !value.is_null() && value.as_str() != Some("") => {
                    render_criterion(value)
                }
                _ => "no, the statement does not hold".to_owned(),
            };
            let true_text = match true_value {
                Some(value) if !value.is_null() && value.as_str() != Some("") => {
                    render_criterion(value)
                }
                _ => "yes, the statement holds".to_owned(),
            };
            vec![
                format!("false: {}", false_text),
                format!("true: {}", true_text),
            ]
        }
        _ => panic!("unknown question type: {t}"),
    }
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    tokenizer
        .encode(EncodeInput::Single(text.to_owned().into()), false)
        .expect("tokenizer encoding failed")
        .get_ids()
        .to_vec()
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> u32 {
    tokenizer
        .token_to_id(token)
        .unwrap_or_else(|| panic!("tokenizer has no token {token:?}"))
}

pub fn build_sequence(
    tokenizer: &Tokenizer,
    state: &Value,
    internal: &Value,
    max_len: usize,
    head_max_len: usize,
) -> PreparedSequence {
    let mask_token = ["[MASK]", "<mask>"]
        .iter()
        .find(|candidate| tokenizer.token_to_id(candidate).is_some())
        .copied()
        .expect("could not identify tokenizer mask token");
    let mask_id = token_id(tokenizer, mask_token);
    let cls_id = token_id(
        tokenizer,
        if tokenizer.token_to_id("[CLS]").is_some() {
            "[CLS]"
        } else {
            "<bos>"
        },
    );
    let sep_id = token_id(
        tokenizer,
        if tokenizer.token_to_id("[SEP]").is_some() {
            "[SEP]"
        } else {
            "<eos>"
        },
    );
    let object = internal.as_object().unwrap();
    let t = object.get("t").and_then(Value::as_str).unwrap();
    let ins = object.get("ins").and_then(Value::as_str).unwrap();
    let options = render_options(internal);
    let head_text = format!("{} question: {}", t, ins.replace(mask_token, " "));
    let head_ids_full = encode(tokenizer, &head_text);
    let mut option_ids_full = Vec::with_capacity(options.len());
    let mut option_ids_kept = Vec::with_capacity(options.len());
    for option in &options {
        let mut ids = encode(tokenizer, &format!(" {}", option.replace(mask_token, " ")));
        ids.truncate(48);
        option_ids_full.push(ids.clone());
        let mut kept = vec![mask_id];
        kept.extend(ids);
        option_ids_kept.push(kept);
    }
    let opt_budget_initial =
        head_max_len as i64 - option_ids_kept.iter().map(Vec::len).sum::<usize>() as i64;
    let (option_ids_kept, retruncated, per) = if opt_budget_initial < 16 {
        let per = std::cmp::max(
            4,
            (head_max_len.saturating_sub(16)) / std::cmp::max(1, option_ids_kept.len()),
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
    let mut ids = vec![cls_id];
    ids.extend(&head_ids_kept);
    ids.push(sep_id);
    let mut markers = Vec::with_capacity(option_ids_kept.len());
    for option in &option_ids_kept {
        markers.push(ids.len());
        ids.extend(option);
    }
    ids.push(sep_id);
    let room = max_len.saturating_sub(ids.len() + 1);
    let state_text = serialize_state(state).replace(mask_token, " ");
    let state_ids_full = encode(tokenizer, &state_text);
    let state_ids_kept = state_ids_full[..std::cmp::min(state_ids_full.len(), room)].to_vec();
    let truncated_state_tokens = state_ids_full.len().saturating_sub(state_ids_kept.len());
    ids.extend(&state_ids_kept);
    ids.push(sep_id);
    let sequence_len = ids.len().min(max_len);
    ids.truncate(max_len);
    let markers = markers
        .into_iter()
        .filter(|marker| *marker < max_len)
        .collect();
    PreparedSequence {
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
    }
}

pub fn prepared_to_value(prepared: &PreparedSequence) -> Value {
    let mut object = Map::new();
    object.insert(
        "head_ids_full".to_owned(),
        u32_array(&prepared.head_ids_full),
    );
    object.insert(
        "head_ids_kept".to_owned(),
        u32_array(&prepared.head_ids_kept),
    );
    object.insert(
        "option_ids_full".to_owned(),
        nested_u32_array(&prepared.option_ids_full),
    );
    object.insert(
        "option_ids_kept".to_owned(),
        nested_u32_array(&prepared.option_ids_kept),
    );
    object.insert(
        "opt_budget_initial".to_owned(),
        Value::from(prepared.opt_budget_initial),
    );
    object.insert("retruncated".to_owned(), Value::Bool(prepared.retruncated));
    object.insert(
        "per".to_owned(),
        prepared
            .per
            .map_or(Value::Null, |value| Value::from(value as u64)),
    );
    object.insert(
        "state_ids_full_len".to_owned(),
        Value::from(prepared.state_ids_full_len as u64),
    );
    object.insert(
        "state_ids_kept_len".to_owned(),
        Value::from(prepared.state_ids_kept_len as u64),
    );
    object.insert("room".to_owned(), Value::from(prepared.room as u64));
    object.insert("ids".to_owned(), u32_array(&prepared.ids));
    object.insert("markers".to_owned(), usize_array(&prepared.markers));
    object.insert(
        "truncated_state_tokens".to_owned(),
        Value::from(prepared.truncated_state_tokens as u64),
    );
    object.insert(
        "sequence_len".to_owned(),
        Value::from(prepared.sequence_len as u64),
    );
    object.insert("hit_max_len".to_owned(), Value::Bool(prepared.hit_max_len));
    Value::Object(object)
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

pub fn collate(items: &[PreparedSequence], qtypes: &[u32], pad_id: u32) -> Value {
    if items.is_empty() {
        return Value::Null;
    }
    let max_len = items.iter().map(|item| item.ids.len()).max().unwrap();
    let max_markers = items.iter().map(|item| item.markers.len()).max().unwrap();
    let mut input_ids = Vec::new();
    let mut attention_mask = Vec::new();
    let mut marker_pos = Vec::new();
    let mut marker_mask = Vec::new();
    for item in items {
        let mut ids = vec![pad_id; max_len];
        ids[..item.ids.len()].copy_from_slice(&item.ids);
        input_ids.push(u32_array(&ids));
        let mut attention = vec![0_u32; max_len];
        attention[..item.ids.len()].fill(1);
        attention_mask.push(u32_array(&attention));
        let mut positions = vec![0_u32; max_markers];
        positions[..item.markers.len()].copy_from_slice(
            &item
                .markers
                .iter()
                .map(|value| *value as u32)
                .collect::<Vec<_>>(),
        );
        marker_pos.push(u32_array(&positions));
        let mut mask = vec![false; max_markers];
        mask[..item.markers.len()].fill(true);
        marker_mask.push(Value::Array(mask.into_iter().map(Value::Bool).collect()));
    }
    let mut object = Map::new();
    object.insert("input_ids".to_owned(), Value::Array(input_ids));
    object.insert("attention_mask".to_owned(), Value::Array(attention_mask));
    object.insert("marker_pos".to_owned(), Value::Array(marker_pos));
    object.insert("marker_mask".to_owned(), Value::Array(marker_mask));
    object.insert("qtype".to_owned(), u32_array(qtypes));
    object.insert("pad_id".to_owned(), Value::from(pad_id));
    object.insert(
        "n_tokens".to_owned(),
        Value::from(items.iter().map(|item| item.ids.len()).sum::<usize>() as u64),
    );
    Value::Object(object)
}

pub fn tokenizer_from_file(
    path: impl AsRef<Path>,
) -> Result<Tokenizer, Box<dyn std::error::Error>> {
    Ok(Tokenizer::from_file(path).map_err(|error| error.to_string())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(text: &str) -> Value {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn serialization_edges_match_python() {
        let cases = [
            ("1.0", "1.0"),
            ("1e-5", "1e-05"),
            ("-0.0", "-0.0"),
            ("1e21", "1e+21"),
            ("0.30000000000000004", "0.30000000000000004"),
            ("7", "7"),
            ("1e-4", "0.0001"),
            ("1e16", "1e+16"),
            ("1.2345e-4", "0.00012345"),
            ("-1.2e20", "-1.2e+20"),
        ];
        for (input, expected) in cases {
            assert_eq!(serialize_state(&value(input)), expected, "{input}");
        }
        assert_eq!(
            serialize_state(&value(r#"{"z": 1.0, "u": "café 🙂", "x": [true, null]}"#)),
            r#"{"z": 1.0, "u": "café 🙂", "x": [true, null]}"#
        );
    }

    #[test]
    fn python_float_fixture_matches_all_cases() {
        let cases: Value =
            serde_json::from_str(include_str!("../tests/fixtures/python-floats.json")).unwrap();
        for case in cases.as_array().unwrap() {
            let value = case.get("value").unwrap();
            let expected = case.get("expected").unwrap().as_str().unwrap();
            assert_eq!(serialize_state(value), expected, "value={value}");
        }
    }

    #[test]
    fn ascii_instruction_uses_surrogate_pairs() {
        assert_eq!(
            to_internal(&value(r#"{"type":"noul","instructions":{"q":"café 🙂"}}"#))["ins"],
            r#"{"q": "caf\u00e9 \ud83d\ude42"}"#
        );
    }

    #[test]
    fn option_rendering_preserves_falsey_values() {
        let internal = to_internal(&value(
            r#"{"type":"choice","instructions":"x","criteria":{"a":null,"b":"","c":0,"d":false}}"#,
        ));
        assert_eq!(render_options(&internal), ["a", "b", "c: 0", "d: false"]);
    }
}
