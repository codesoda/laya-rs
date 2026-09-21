use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use laya_tokenizer_parity::{
    build_sequence, collate, prepared_to_value, render_options, serialize_state, to_internal,
    tokenizer_from_file,
};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

#[derive(Default)]
struct Counts {
    fixtures: usize,
    exact: usize,
    mismatches: usize,
}

fn load(path: &Path) -> Value {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn first_diff(expected: &Value, actual: &Value, path: &str) -> Option<(String, Value, Value)> {
    if expected == actual {
        return None;
    }
    match (expected, actual) {
        (Value::Object(a), Value::Object(b)) => {
            for (key, value) in a {
                let child = format!("{path}.{key}");
                match b.get(key) {
                    Some(other) => {
                        if let Some(diff) = first_diff(value, other, &child) {
                            return Some(diff);
                        }
                    }
                    None => return Some((child, value.clone(), Value::Null)),
                }
            }
            b.keys()
                .find(|key| !a.contains_key(*key))
                .map(|key| (format!("{path}.{key}"), Value::Null, b[key].clone()))
        }
        (Value::Array(a), Value::Array(b)) => {
            let common = a.len().min(b.len());
            for index in 0..common {
                if let Some(diff) = first_diff(&a[index], &b[index], &format!("{path}[{index}]")) {
                    return Some(diff);
                }
            }
            if a.len() != b.len() {
                Some((
                    format!("{path}[{common}]"),
                    a.get(common).cloned().unwrap_or(Value::Null),
                    b.get(common).cloned().unwrap_or(Value::Null),
                ))
            } else {
                None
            }
        }
        _ => Some((path.to_owned(), expected.clone(), actual.clone())),
    }
}

fn token_context(tokenizer: &Tokenizer, value: &Value) -> String {
    let Some(id) = value.as_u64() else {
        return value.to_string();
    };
    let piece = tokenizer
        .id_to_token(id as u32)
        .unwrap_or_else(|| "<unknown>".to_owned());
    format!("{id} ({piece:?})")
}

fn report(
    counts: &mut Counts,
    tokenizer: &Tokenizer,
    profile: &str,
    fixture: &str,
    field: &str,
    expected: &Value,
    actual: &Value,
) {
    if let Some((path, expected_value, actual_value)) = first_diff(expected, actual, field) {
        counts.mismatches += 1;
        eprintln!(
            "MISMATCH profile={profile} fixture={fixture} field={field} first={path}: golden={} rust={}",
            token_context(tokenizer, &expected_value),
            token_context(tokenizer, &actual_value)
        );
    }
}

fn map_value(entries: impl IntoIterator<Item = (String, Value)>) -> Value {
    let mut map = Map::new();
    for (key, value) in entries {
        map.insert(key, value);
    }
    Value::Object(map)
}

fn profile_config(profile: &str) -> (&'static str, usize, usize, &'static str, &'static str) {
    match profile {
        "english" => (
            "english/tokenizer/tokenizer.json",
            512,
            192,
            "[PAD]",
            "english",
        ),
        "multilingual" => (
            "multilingual/tokenizer/tokenizer.json",
            1024,
            256,
            "<pad>",
            "multilingual",
        ),
        "typed-decisions" => (
            "typed-decisions/tokenizer/tokenizer.json",
            1024,
            256,
            "[PAD]",
            "typed-decisions",
        ),
        _ => panic!("unknown profile {profile}"),
    }
}

fn run_profile(root: &Path, profile: &str) -> Counts {
    let (tokenizer_rel, max_len, head_max_len, pad_token, _) = profile_config(profile);
    let mask_token = if profile == "multilingual" {
        "<mask>"
    } else {
        "[MASK]"
    };
    let tokenizer_path = root
        .join(".cache/laya/hub/convaiinnovations--laya/c5d78730f3493e4fe16d61507ef4b78eef7318cf")
        .join(tokenizer_rel);
    let tokenizer = tokenizer_from_file(&tokenizer_path).unwrap();
    let pad_id = tokenizer.token_to_id(pad_token).unwrap();
    let fixture_dir = root.join("benchmarks/fixtures/requests");
    let golden_dir = root.join("benchmarks/goldens").join(profile).join("cpu");
    let mut golden_paths: Vec<PathBuf> = fs::read_dir(&golden_dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    golden_paths.sort();

    let mut counts = Counts::default();
    for golden_path in golden_paths {
        let fixture = golden_path.file_stem().unwrap().to_str().unwrap();
        let golden = load(&golden_path);
        let request = load(&fixture_dir.join(format!("{fixture}.json")));
        counts.fixtures += 1;
        let state = request.get("state").unwrap();
        let questions = request.get("questions").unwrap().as_object().unwrap();
        let state_text = serialize_state(state).replace(mask_token, " ");
        let mut mismatched = false;
        let golden_state = golden.get("state_text").unwrap();
        if golden_state != &Value::String(state_text.clone()) {
            report(
                &mut counts,
                &tokenizer,
                profile,
                fixture,
                "state_text",
                golden_state,
                &Value::String(state_text.clone()),
            );
            mismatched = true;
        }

        let mut serialized = Map::new();
        let mut tokens = Map::new();
        let mut sequences = Vec::new();
        let mut qtypes = Vec::new();
        for (qid, qdef) in questions {
            let internal = to_internal(qdef);
            let options = render_options(&internal);
            let internal_object = internal.as_object().unwrap();
            let ins = internal_object.get("ins").unwrap().as_str().unwrap();
            let qtype = internal_object.get("t").unwrap().as_str().unwrap();
            let head_text = format!("{qtype} question: {}", ins.replace(mask_token, " "));
            let option_texts: Vec<String> = options
                .iter()
                .map(|option| {
                    format!(
                        " {}",
                        option.replace(
                            if profile == "multilingual" {
                                "<mask>"
                            } else {
                                "[MASK]"
                            },
                            " ",
                        )
                    )
                })
                .collect();
            serialized.insert(
                qid.clone(),
                map_value([
                    ("internal".to_owned(), internal.clone()),
                    (
                        "rendered_options".to_owned(),
                        Value::Array(options.iter().cloned().map(Value::String).collect()),
                    ),
                    ("head_text".to_owned(), Value::String(head_text)),
                    (
                        "option_texts".to_owned(),
                        Value::Array(option_texts.into_iter().map(Value::String).collect()),
                    ),
                    (
                        "state_text_ref".to_owned(),
                        Value::String("$.state_text".to_owned()),
                    ),
                ]),
            );
            let prepared = build_sequence(&tokenizer, state, &internal, max_len, head_max_len);
            tokens.insert(qid.clone(), prepared_to_value(&prepared));
            sequences.push(prepared);
            qtypes.push(match qtype {
                "choice" => 0,
                "score" => 1,
                "noul" => 2,
                _ => panic!("unknown qtype {qtype}"),
            });
        }
        let batch = collate(&sequences, &qtypes, pad_id);
        let generated_serialized = Value::Object(serialized);
        let generated_tokens = Value::Object(tokens);
        for field in [
            "internal",
            "rendered_options",
            "head_text",
            "option_texts",
            "state_text_ref",
        ] {
            for (qid, expected_row) in golden.get("serialized").unwrap().as_object().unwrap() {
                let expected = expected_row.get(field).unwrap();
                let actual = generated_serialized.get(qid).unwrap().get(field).unwrap();
                if expected != actual {
                    report(
                        &mut counts,
                        &tokenizer,
                        profile,
                        fixture,
                        &format!("serialized.{qid}.{field}"),
                        expected,
                        actual,
                    );
                    mismatched = true;
                }
            }
        }
        for (qid, expected_row) in golden.get("tokens").unwrap().as_object().unwrap() {
            for (field, expected) in expected_row.as_object().unwrap() {
                let actual = generated_tokens.get(qid).unwrap().get(field).unwrap();
                if expected != actual {
                    report(
                        &mut counts,
                        &tokenizer,
                        profile,
                        fixture,
                        &format!("tokens.{qid}.{field}"),
                        expected,
                        actual,
                    );
                    mismatched = true;
                }
            }
        }
        let expected_batch = golden.get("batch").unwrap();
        for (field, expected) in expected_batch.as_object().unwrap() {
            let actual = batch.get(field).unwrap();
            if expected != actual {
                report(
                    &mut counts,
                    &tokenizer,
                    profile,
                    fixture,
                    &format!("batch.{field}"),
                    expected,
                    actual,
                );
                mismatched = true;
            }
        }
        if !mismatched {
            counts.exact += 1;
        }
    }
    println!(
        "{profile}: {}/{} fixtures exact",
        counts.exact, counts.fixtures
    );
    counts
}

fn main() {
    let root = env::var_os("LAYA_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."));
    let mut total = Counts::default();
    for profile in ["english", "multilingual", "typed-decisions"] {
        let counts = run_profile(&root, profile);
        total.fixtures += counts.fixtures;
        total.exact += counts.exact;
        total.mismatches += counts.mismatches;
    }
    println!("total: {}/{} fixtures exact", total.exact, total.fixtures);
    if total.mismatches > 0 {
        std::process::exit(1);
    }
}
