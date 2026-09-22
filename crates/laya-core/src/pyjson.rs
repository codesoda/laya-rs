//! Byte-exact reproduction of the Python `json.dumps` spellings the upstream
//! runtime uses when it turns request values into text. Lifted from the L1
//! tokenizer spike (`spikes/tokenizer-parity`); semantics unchanged.
//!
//! Two flavours exist upstream and must not be confused:
//! - state and criterion values: `ensure_ascii=False`, default `", "`/`": "`
//!   separators, insertion order;
//! - non-string instructions: `json.dumps(ins)` with default ASCII escaping.

use std::fmt::Write;

use serde_json::{Number, Value};

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
                    write!(out, "\\u{cp:04x}").expect("writing to String cannot fail");
                } else {
                    let n = cp - 0x1_0000;
                    let high = 0xd800 + (n >> 10);
                    let low = 0xdc00 + (n & 0x3ff);
                    write!(out, "\\u{high:04x}\\u{low:04x}")
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
    (negative, format!("{whole}{fraction}"), decimal)
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
            write!(out, "e{exponent}").expect("writing to String cannot fail");
        }
    } else {
        write!(out, "e+{exponent}").expect("writing to String cannot fail");
    }
    out
}

/// Format an f64 with Python's json/repr spelling, given ryu's shortest
/// round-trip digits. Python uses fixed notation for 1e-4 <= abs(x) < 1e16
/// and exponent notation otherwise.
#[must_use]
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

/// `json.dumps(value)` with Python's default separators. `ascii_only`
/// selects `ensure_ascii=True`.
#[must_use]
pub fn json_text(value: &Value, ascii_only: bool) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(v) => v.to_string(),
        Value::Number(v) => number_text(v),
        Value::String(v) => escape_json_string(v, ascii_only),
        Value::Array(values) => {
            let mut out = String::from("[");
            for (i, value) in values.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&json_text(value, ascii_only));
            }
            out.push(']');
            out
        }
        Value::Object(values) => {
            let mut out = String::from("{");
            for (i, (key, value)) in values.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&escape_json_string(key, ascii_only));
                out.push_str(": ");
                out.push_str(&json_text(value, ascii_only));
            }
            out.push('}');
            out
        }
    }
}

/// Python: `state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)`.
#[must_use]
pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(value) => value.clone(),
        _ => json_text(state, false),
    }
}

/// Python: `json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)`.
#[must_use]
pub fn render_criterion(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        _ => json_text(value, false),
    }
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
        let cases = cases.as_array().unwrap();
        assert!(cases.len() >= 1000);
        for case in cases {
            let value = case.get("value").unwrap();
            let expected = case.get("expected").unwrap().as_str().unwrap();
            assert_eq!(serialize_state(value), expected, "value={value}");
        }
    }

    #[test]
    fn ascii_mode_uses_surrogate_pairs() {
        assert_eq!(
            json_text(&value(r#"{"q":"café 🙂"}"#), true),
            r#"{"q": "caf\u00e9 \ud83d\ude42"}"#
        );
    }
}
