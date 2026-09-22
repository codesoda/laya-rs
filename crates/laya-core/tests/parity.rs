//! Permanent parity gate: every frozen golden under
//! `benchmarks/goldens/<profile>/cpu/` against every enabled backend, judged
//! by `benchmarks/goldens/tolerances.json`.
//!
//! Exact checks: serialized text, token IDs, markers, batch tensors,
//! `usage.input_tokens`, error class for upstream-error fixtures, legend and
//! key order of the native answer JSON. Numeric checks: per-fixture
//! `max_abs`, per-profile population `mean_abs` (tolerances.json amendment
//! 2026-09-22 — the reading used is printed), argmax with the near-tie rule,
//! and four-decimal native field flips (≤ 1e-4 each, ≤ 2% of questions).
//!
//! Model files are located through `LAYA_MODEL_ROOT` (test-only) or the
//! repository's `.cache/laya/hub/...` directory. A missing profile is a loud
//! SKIP, never a pass; set `LAYA_REQUIRE_PARITY=1` to turn skips into
//! failures.

use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Mutex,
    time::Instant,
};

use laya_core::{
    BackendSpec, LayaError, LoadOptions, Profile, Request, Runtime, Verification, HUB_REVISION,
};
use ndarray::ArrayD;
use ndarray_npy::NpzReader;
use serde::Serialize;
use serde_json::Value;

static SERIAL: Mutex<()> = Mutex::new(());

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn model_root() -> PathBuf {
    std::env::var_os("LAYA_MODEL_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            repo_root().join(format!(
                ".cache/laya/hub/convaiinnovations--laya/{HUB_REVISION}"
            ))
        })
}

fn load_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[derive(Clone, Debug)]
struct Limits {
    option_logits_raw_max: f64,
    option_logits_raw_mean: Option<f64>,
    act_logits_max: Option<f64>,
    act_logits_mean: Option<f64>,
    probs_max: f64,
    expected_score_max: f64,
    confidence_max: Option<f64>,
    act_probs_max: f64,
    /// fp32: 100% argmax agreement outside near-ties (gap < 1e-5).
    /// fp16: ≥ 99% excluding near-ties (gap < 5e-3).
    near_tie_gap: f64,
    min_argmax_agreement: f64,
    gate_native_rounded: bool,
    /// Fraction of questions per profile allowed a last-digit flip.
    rounded_flip_fraction: f64,
}

fn limits(tolerances: &Value, profile_name: &str) -> Limits {
    let profile = &tolerances["numeric_profiles"][profile_name];
    assert!(
        profile.is_object(),
        "tolerances.json has no numeric profile {profile_name}"
    );
    let max = |field: &str| profile[field]["max_abs"].as_f64();
    let mean = |field: &str| profile[field]["mean_abs"].as_f64();
    let fp16 = profile_name == "rust_metal_fp16_or_bf16";
    Limits {
        option_logits_raw_max: max("option_logits_raw").expect("option_logits_raw.max_abs"),
        option_logits_raw_mean: mean("option_logits_raw"),
        act_logits_max: max("act_logits"),
        act_logits_mean: mean("act_logits"),
        probs_max: max("probs_unrounded").expect("probs_unrounded.max_abs"),
        expected_score_max: max("expected_score_unrounded").expect("expected_score_unrounded"),
        confidence_max: max("confidence_unrounded"),
        act_probs_max: max("act_probs").expect("act_probs.max_abs"),
        near_tie_gap: if fp16 { 5e-3 } else { 1e-5 },
        min_argmax_agreement: if fp16 { 0.99 } else { 1.0 },
        gate_native_rounded: !fp16,
        // rust_cpu_fp32: 2%. rust_metal_fp32: its own gate version, 3%
        // (tolerances.json amendment 2026-09-23). Read from the rule text
        // so a silent edit of the file cannot go unnoticed.
        rounded_flip_fraction: rounded_flip_fraction(profile["native_rounded_fields"].as_str()),
    }
}

fn rounded_flip_fraction(rule: Option<&str>) -> f64 {
    let rule = rule.expect("native_rounded_fields rule text");
    let rule = if rule == "as rust_cpu_fp32" {
        "must not exceed 2% of questions"
    } else {
        rule
    };
    let percent = rule
        .split("must not exceed ")
        .nth(1)
        .and_then(|rest| rest.split('%').next())
        .and_then(|digits| digits.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("cannot read the flip percentage from rule: {rule}"));
    percent / 100.0
}

#[derive(Clone, Debug, Default, Serialize)]
struct Metric {
    count: usize,
    max_abs: f64,
    mean_abs: f64,
    #[serde(skip)]
    sum_abs: f64,
}

impl Metric {
    fn add(&mut self, actual: f64, expected: f64) {
        let diff = (actual - expected).abs();
        self.count += 1;
        self.max_abs = self.max_abs.max(diff);
        self.sum_abs += diff;
        self.mean_abs = self.sum_abs / self.count as f64;
    }

    fn add_all(&mut self, actual: &[f32], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "comparison length mismatch");
        for (a, e) in actual.iter().zip(expected) {
            self.add(f64::from(*a), *e);
        }
    }

    fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.max_abs = self.max_abs.max(other.max_abs);
        self.sum_abs += other.sum_abs;
        if self.count > 0 {
            self.mean_abs = self.sum_abs / self.count as f64;
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
struct Fields {
    option_logits_raw: Metric,
    act_logits: Metric,
    act_probs: Metric,
    probs_unrounded: Metric,
    confidence_unrounded: Metric,
    expected_score_unrounded: Metric,
    hidden_cls: Metric,
}

impl Fields {
    fn merge(&mut self, other: &Self) {
        self.option_logits_raw.merge(&other.option_logits_raw);
        self.act_logits.merge(&other.act_logits);
        self.act_probs.merge(&other.act_probs);
        self.probs_unrounded.merge(&other.probs_unrounded);
        self.confidence_unrounded.merge(&other.confidence_unrounded);
        self.expected_score_unrounded
            .merge(&other.expected_score_unrounded);
        self.hidden_cls.merge(&other.hidden_cls);
    }
}

#[derive(Debug, Serialize)]
struct FixtureReport {
    fixture: String,
    questions: usize,
    exact: bool,
    fields: Fields,
    argmax_agree: usize,
    near_ties: Vec<String>,
    /// Every flipped field, `native_result.answers.<qid>...: rust x golden y`.
    rounded_flips: Vec<String>,
    /// Questions with at least one flipped field (the unit the 2% rule uses).
    flipped_questions: Vec<String>,
    elapsed_ms: f64,
    failures: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ProfileReport {
    profile: String,
    backend: String,
    tolerance_profile: String,
    mean_abs_reading: &'static str,
    model_dir: String,
    load_ms: f64,
    warmup_ms: f64,
    fixtures: Vec<FixtureReport>,
    error_fixtures: Vec<String>,
    aggregate: Fields,
    questions: usize,
    argmax_agree: usize,
    /// Flipped fields (informational) and flipped questions (gated at 2%).
    rounded_flip_fields: usize,
    rounded_flip_questions: usize,
    aggregate_failures: Vec<String>,
    pass: bool,
}

const MEAN_ABS_READING: &str = "population aggregate per profile (all questions in all fixtures); max_abs per element/fixture — tolerances.json amendment 2026-09-22";

fn numbers(value: &Value, field: &str) -> Vec<f64> {
    value[field]
        .as_array()
        .unwrap_or_else(|| panic!("golden field {field} is not an array"))
        .iter()
        .map(|v| v.as_f64().expect("numeric golden"))
        .collect()
}

fn first_diff(expected: &Value, actual: &Value, path: &str) -> Option<String> {
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
                    None => return Some(format!("{child}: missing in rust output")),
                }
            }
            b.keys()
                .find(|key| !a.contains_key(*key))
                .map(|key| format!("{path}.{key}: extra in rust output"))
        }
        (Value::Array(a), Value::Array(b)) => {
            for (index, (x, y)) in a.iter().zip(b).enumerate() {
                if let Some(diff) = first_diff(x, y, &format!("{path}[{index}]")) {
                    return Some(diff);
                }
            }
            (a.len() != b.len()).then(|| format!("{path}: length {} vs {}", a.len(), b.len()))
        }
        _ => Some(format!("{path}: golden={expected} rust={actual}")),
    }
}

fn load_hidden_cls(path: &Path) -> Vec<f64> {
    let mut archive = NpzReader::new(File::open(path).unwrap()).unwrap();
    let hidden: ArrayD<f32> = archive.by_name("hidden_cls.npy").unwrap();
    hidden.iter().map(|v| f64::from(*v)).collect()
}

/// Compare the rounded native answer JSON field by field. Non-numeric
/// fields must be identical; numeric fields may differ by one last digit.
fn compare_native(
    expected: &Value,
    actual: &Value,
    path: &str,
    flips: &mut Vec<String>,
    failures: &mut Vec<String>,
) {
    match (expected, actual) {
        (Value::Object(a), Value::Object(b)) => {
            if a.keys().ne(b.keys()) {
                failures.push(format!(
                    "{path}: key order/set differs: {:?} vs {:?}",
                    a.keys().collect::<Vec<_>>(),
                    b.keys().collect::<Vec<_>>()
                ));
                return;
            }
            for (key, value) in a {
                compare_native(value, &b[key], &format!("{path}.{key}"), flips, failures);
            }
        }
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            let diff = (a - b).abs();
            if diff > 1e-4 + 1e-9 {
                failures.push(format!(
                    "{path}: rounded {b} vs golden {a} (|diff| {diff:.3e} > 1e-4)"
                ));
            } else if diff > 1e-9 {
                flips.push(format!("{path}: rust {b} golden {a}"));
            }
        }
        _ => {
            if expected != actual {
                failures.push(format!("{path}: rust {actual} vs golden {expected}"));
            }
        }
    }
}

fn evaluate_fixture(
    runtime: &mut Runtime,
    fixture: &str,
    golden: &Value,
    golden_path: &Path,
    request: &Request,
    limits: &Limits,
) -> FixtureReport {
    let mut failures = Vec::new();
    let prepared = runtime
        .prepare(request)
        .unwrap_or_else(|e| panic!("{fixture}: prepare: {e}"));
    let tokenizer = runtime.tokenizer();

    // Exact preprocessing.
    let mut exact = true;
    let mut check_exact = |field: &str, expected: &Value, actual: &Value| {
        if let Some(diff) = first_diff(expected, actual, field) {
            exact = false;
            failures.push(format!("exact mismatch {diff}"));
        }
    };
    check_exact(
        "state_text",
        &golden["state_text"],
        &Value::String(prepared.state_text.clone()),
    );
    for ((qid, question), sequence) in request.questions.iter().zip(&prepared.sequences) {
        let expected = &golden["serialized"][qid];
        check_exact(
            &format!("serialized.{qid}.internal"),
            &expected["internal"],
            &question.internal_value(),
        );
        let options = question.render_options();
        check_exact(
            &format!("serialized.{qid}.rendered_options"),
            &expected["rendered_options"],
            &Value::Array(options.iter().cloned().map(Value::String).collect()),
        );
        let head_text = format!(
            "{} question: {}",
            question.question_type().as_str(),
            tokenizer.scrub(question.instructions())
        );
        check_exact(
            &format!("serialized.{qid}.head_text"),
            &expected["head_text"],
            &Value::String(head_text),
        );
        let option_texts: Vec<Value> = options
            .iter()
            .map(|option| Value::String(format!(" {}", tokenizer.scrub(option))))
            .collect();
        check_exact(
            &format!("serialized.{qid}.option_texts"),
            &expected["option_texts"],
            &Value::Array(option_texts),
        );
        check_exact(
            &format!("tokens.{qid}"),
            &golden["tokens"][qid],
            &sequence.to_value(),
        );
    }
    check_exact("batch", &golden["batch"], &prepared.batch.to_value());

    // Inference.
    let started = Instant::now();
    let evaluation = runtime
        .evaluate_prepared(request, &prepared)
        .unwrap_or_else(|e| panic!("{fixture}: evaluate: {e}"));
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

    let mut fields = Fields::default();
    let mut argmax_agree = 0;
    let mut near_ties = Vec::new();
    let expected_model = golden["model"].as_object().expect("model object");
    assert_eq!(
        expected_model.len(),
        evaluation.results.len(),
        "{fixture}: question count"
    );
    for ((qid, expected), result) in expected_model.iter().zip(&evaluation.results) {
        assert_eq!(qid, &result.id);
        fields
            .option_logits_raw
            .add_all(&result.raw_logits, &numbers(expected, "option_logits_raw"));
        fields
            .act_logits
            .add_all(&result.act_logits, &numbers(expected, "act_logits"));
        fields
            .act_probs
            .add_all(&result.act_probs, &numbers(expected, "act_probs"));
        fields
            .probs_unrounded
            .add_all(&result.probabilities, &numbers(expected, "probs_unrounded"));
        fields.confidence_unrounded.add(
            f64::from(result.confidence),
            expected["confidence_unrounded"]
                .as_f64()
                .expect("confidence"),
        );
        if let Some(score) = result.expected_score {
            fields.expected_score_unrounded.add(
                score,
                expected["expected_score_unrounded"]
                    .as_f64()
                    .expect("expected_score_unrounded"),
            );
        }
        if expected["temperature"].as_f64() != Some(result.temperature)
            || expected["temp_bucket"].as_str() != Some(result.temp_bucket.as_str())
        {
            failures.push(format!(
                "{qid}: temperature/bucket {} {} vs golden {} {}",
                result.temperature,
                result.temp_bucket,
                expected["temperature"],
                expected["temp_bucket"]
            ));
        }
        let expected_argmax = expected["argmax_index"].as_u64().expect("argmax_index") as usize;
        if result.argmax == expected_argmax {
            argmax_agree += 1;
        } else {
            let mut ordered = numbers(expected, "probs_unrounded");
            ordered.sort_by(|a, b| b.total_cmp(a));
            let gap = ordered[0] - ordered[1];
            let note = format!(
                "{qid}: argmax rust={} golden={expected_argmax} golden top-two gap={gap:.3e}",
                result.argmax
            );
            if gap < limits.near_tie_gap {
                near_ties.push(note);
            } else {
                failures.push(note);
            }
        }
    }
    // Diagnostic only: post-head CLS states are reported, not gated.
    let sidecar = golden["hidden_states_npz"]
        .as_str()
        .expect("hidden_states_npz");
    let expected_hidden = load_hidden_cls(&golden_path.with_file_name(sidecar));
    fields
        .hidden_cls
        .add_all(&evaluation.hidden_cls, &expected_hidden);

    // Native rounded JSON.
    let native = evaluation.native_json(request);
    let mut rounded_flips = Vec::new();
    compare_native(
        &golden["native_result"],
        &native,
        "native_result",
        &mut rounded_flips,
        &mut failures,
    );
    if golden["native_result"]["usage"]["input_tokens"].as_u64() != Some(evaluation.input_tokens) {
        failures.push(format!(
            "usage.input_tokens {} vs golden {}",
            evaluation.input_tokens, golden["native_result"]["usage"]["input_tokens"]
        ));
    }

    // Per-fixture max_abs gates.
    let mut check_max = |name: &str, metric: &Metric, limit: Option<f64>| {
        if let Some(limit) = limit {
            if metric.count > 0 && metric.max_abs > limit {
                failures.push(format!(
                    "{name} max_abs {:.6e} > {limit:.6e}",
                    metric.max_abs
                ));
            }
        }
    };
    check_max(
        "option_logits_raw",
        &fields.option_logits_raw,
        Some(limits.option_logits_raw_max),
    );
    check_max("act_logits", &fields.act_logits, limits.act_logits_max);
    check_max(
        "probs_unrounded",
        &fields.probs_unrounded,
        Some(limits.probs_max),
    );
    check_max(
        "expected_score_unrounded",
        &fields.expected_score_unrounded,
        Some(limits.expected_score_max),
    );
    check_max(
        "confidence_unrounded",
        &fields.confidence_unrounded,
        limits.confidence_max,
    );
    check_max("act_probs", &fields.act_probs, Some(limits.act_probs_max));

    // The tolerance counts flips per question: "must not exceed 2% of
    // questions per profile". One question's probability map may hold two
    // flipped fields (its entries sum to one), so group fields by question.
    let mut flipped_questions: Vec<String> = rounded_flips
        .iter()
        .filter_map(|flip| {
            flip.strip_prefix("native_result.answers.")
                .and_then(|rest| rest.split('.').next())
                .map(str::to_owned)
        })
        .collect();
    flipped_questions.sort();
    flipped_questions.dedup();
    FixtureReport {
        fixture: fixture.to_owned(),
        questions: evaluation.results.len(),
        exact,
        fields,
        argmax_agree,
        near_ties,
        rounded_flips,
        flipped_questions,
        elapsed_ms,
        failures,
    }
}

fn run_profile(profile: Profile, spec: &BackendSpec) -> Option<ProfileReport> {
    let root = repo_root();
    let model_dir = model_root().join(profile.as_str());
    if !model_dir.join("model.safetensors").is_file() {
        let message = format!(
            "SKIP (not run, not passed): {profile} model files missing at {}; set LAYA_MODEL_ROOT",
            model_dir.display()
        );
        assert!(
            std::env::var_os("LAYA_REQUIRE_PARITY").is_none(),
            "{message}"
        );
        eprintln!("{message}");
        return None;
    }
    let tolerances = load_json(&root.join("benchmarks/goldens/tolerances.json"));
    let tolerance_profile = spec.precision().tolerance_profile(spec.device());
    let limits = limits(&tolerances, tolerance_profile);
    println!(
        "parity: profile={profile} backend={spec} tolerance_profile={tolerance_profile}\n  mean_abs reading: {MEAN_ABS_READING}"
    );

    let mut runtime = Runtime::load(&LoadOptions {
        profile,
        model_dir: model_dir.clone(),
        backend: spec.clone(),
        verification: Verification::Full,
    })
    .unwrap_or_else(|e| panic!("load {profile} on {spec}: {e}"));
    let warmup_ms = runtime.warmup().unwrap();
    println!(
        "  loaded in {:.0} ms (SHA-256 verified), warm-up {:.0} ms",
        runtime.info().load_ms,
        warmup_ms
    );

    let golden_dir = root
        .join("benchmarks/goldens")
        .join(profile.as_str())
        .join("cpu");
    let mut paths: Vec<PathBuf> = fs::read_dir(&golden_dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    paths.sort();
    assert!(
        !paths.is_empty(),
        "no goldens under {}",
        golden_dir.display()
    );

    let mut fixtures = Vec::new();
    let mut error_fixtures = Vec::new();
    let mut aggregate = Fields::default();
    let mut questions = 0;
    let mut argmax_agree = 0;
    let mut rounded_flip_fields = 0;
    let mut rounded_flip_questions = 0;
    let mut aggregate_failures = Vec::new();
    for path in &paths {
        let fixture = path.file_stem().unwrap().to_str().unwrap();
        let golden = load_json(path);
        let request_value =
            load_json(&root.join(format!("benchmarks/fixtures/requests/{fixture}.json")));
        let request =
            Request::from_wire(&request_value).unwrap_or_else(|e| panic!("{fixture}: {e}"));
        if !golden["error"].is_null() {
            let expected_class = golden["error"]["type"].as_str().unwrap();
            match runtime.prepare(&request) {
                Err(error) => {
                    if error.upstream_class() != Some(expected_class) {
                        aggregate_failures.push(format!(
                            "{fixture}: rust error {} ({error}) does not map to upstream {expected_class}",
                            error.code()
                        ));
                    }
                    println!(
                        "  {fixture}: upstream {expected_class} -> rust {} ({error})",
                        error.code()
                    );
                    if matches!(error, LayaError::SingleOptionChoice { .. }) {
                        assert!(
                            runtime.evaluate(&request).is_err(),
                            "evaluate must also refuse"
                        );
                    }
                }
                Ok(_) => aggregate_failures.push(format!(
                    "{fixture}: upstream raised {expected_class} but rust accepted the request"
                )),
            }
            error_fixtures.push(fixture.to_owned());
            continue;
        }
        let report = evaluate_fixture(&mut runtime, fixture, &golden, path, &request, &limits);
        println!(
            "  {fixture}: {} exact={} raw max={:.3e} act max={:.3e} probs max={:.3e} argmax={}/{} flips={} {:.1}ms",
            if report.failures.is_empty() { "PASS" } else { "FAIL" },
            report.exact,
            report.fields.option_logits_raw.max_abs,
            report.fields.act_logits.max_abs,
            report.fields.probs_unrounded.max_abs,
            report.argmax_agree,
            report.questions,
            report.rounded_flips.len(),
            report.elapsed_ms
        );
        for failure in &report.failures {
            println!("    FAIL {failure}");
        }
        for tie in &report.near_ties {
            println!("    near-tie {tie}");
        }
        for flip in &report.rounded_flips {
            println!("    flip {flip}");
        }
        aggregate.merge(&report.fields);
        questions += report.questions;
        argmax_agree += report.argmax_agree;
        rounded_flip_fields += report.rounded_flips.len();
        rounded_flip_questions += report.flipped_questions.len();
        fixtures.push(report);
    }
    assert!(questions > 0, "no successful fixtures evaluated");

    // Aggregate (population) gates.
    if let Some(limit) = limits.option_logits_raw_mean {
        if aggregate.option_logits_raw.mean_abs > limit {
            aggregate_failures.push(format!(
                "option_logits_raw population mean_abs {:.6e} > {limit:.6e}",
                aggregate.option_logits_raw.mean_abs
            ));
        }
    }
    if let Some(limit) = limits.act_logits_mean {
        if aggregate.act_logits.mean_abs > limit {
            aggregate_failures.push(format!(
                "act_logits population mean_abs {:.6e} > {limit:.6e}",
                aggregate.act_logits.mean_abs
            ));
        }
    }
    let near_ties: usize = fixtures.iter().map(|f| f.near_ties.len()).sum();
    let agreement = (argmax_agree + near_ties) as f64 / questions as f64;
    if agreement < limits.min_argmax_agreement {
        aggregate_failures.push(format!(
            "argmax agreement {argmax_agree}/{questions} (+{near_ties} near-ties) below {:.0}%",
            limits.min_argmax_agreement * 100.0
        ));
    }
    if limits.gate_native_rounded
        && rounded_flip_questions as f64 > limits.rounded_flip_fraction * questions as f64
    {
        aggregate_failures.push(format!(
            "questions with a native rounded-field flip: {rounded_flip_questions} ({rounded_flip_fields} fields) exceed {:.0}% of {questions} questions",
            limits.rounded_flip_fraction * 100.0
        ));
    }
    let pass = fixtures.iter().all(|f| f.failures.is_empty()) && aggregate_failures.is_empty();
    println!(
        "  aggregate: {} fixtures={} questions={questions} raw max={:.3e} mean={:.3e} act max={:.3e} mean={:.3e} probs max={:.3e} conf max={:.3e} hidden_cls(diag) max={:.3e} argmax={argmax_agree}/{questions} near-ties={near_ties} flips={rounded_flip_questions} questions/{rounded_flip_fields} fields (limit {:.0}% of questions) error-fixtures={error_fixtures:?}",
        if pass { "PASS" } else { "FAIL" },
        fixtures.len(),
        aggregate.option_logits_raw.max_abs,
        aggregate.option_logits_raw.mean_abs,
        aggregate.act_logits.max_abs,
        aggregate.act_logits.mean_abs,
        aggregate.probs_unrounded.max_abs,
        aggregate.confidence_unrounded.max_abs,
        aggregate.hidden_cls.max_abs,
        limits.rounded_flip_fraction * 100.0,
    );
    for failure in &aggregate_failures {
        println!("    FAIL {failure}");
    }
    let report = ProfileReport {
        profile: profile.to_string(),
        backend: spec.label(),
        tolerance_profile: tolerance_profile.to_owned(),
        mean_abs_reading: MEAN_ABS_READING,
        model_dir: model_dir.display().to_string(),
        load_ms: runtime.info().load_ms,
        warmup_ms,
        fixtures,
        error_fixtures,
        aggregate,
        questions,
        argmax_agree,
        rounded_flip_fields,
        rounded_flip_questions,
        aggregate_failures,
        pass,
    };
    let out_dir = root.join("target/laya-parity");
    fs::create_dir_all(&out_dir).unwrap();
    let out = out_dir.join(format!("{profile}-{}.json", spec.label()));
    fs::write(&out, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!("  report: {}", out.display());
    Some(report)
}

fn run_all(spec: &BackendSpec) {
    let _guard = SERIAL.lock().unwrap_or_else(|poison| poison.into_inner());
    let mut ran = 0;
    let mut failed = Vec::new();
    for profile in Profile::ALL {
        if let Some(report) = run_profile(profile, spec) {
            ran += 1;
            if !report.pass {
                failed.push(report.profile);
            }
        }
    }
    if ran == 0 {
        eprintln!("SKIP: no profile model directories found; parity NOT RUN for {spec}");
    }
    assert!(failed.is_empty(), "parity failed for {spec}: {failed:?}");
}

#[cfg(feature = "mlx")]
use laya_core::Precision;

#[cfg(feature = "mlx")]
#[test]
fn parity_mlx_metal_f32() {
    run_all(&BackendSpec::Mlx {
        precision: Precision::F32,
        metallib_cache_dir: repo_root().join("target/laya-parity/metallib-cache"),
    });
}

/// Separate measured configuration; opt in with `--ignored`.
#[cfg(feature = "mlx")]
#[test]
#[ignore = "fp16 is a separately reported configuration; run with --ignored"]
fn parity_mlx_metal_f16() {
    run_all(&BackendSpec::Mlx {
        precision: Precision::F16,
        metallib_cache_dir: repo_root().join("target/laya-parity/metallib-cache"),
    });
}

#[cfg(feature = "candle")]
#[test]
fn parity_candle_cpu_f32() {
    run_all(&BackendSpec::CandleCpu);
}

#[cfg(not(any(feature = "mlx", feature = "candle")))]
#[test]
fn parity_requires_a_backend_feature() {
    eprintln!("SKIP: no backend feature enabled; parity NOT RUN. Use --features mlx and/or candle");
}

#[test]
fn error_fixtures_have_no_batch_dependency() {
    // The single-option fixture must be refused before any model loads.
    let root = repo_root();
    let request = Request::from_wire(&load_json(
        &root.join("benchmarks/fixtures/requests/edge-single-option.json"),
    ))
    .unwrap();
    let golden = load_json(&root.join("benchmarks/goldens/english/cpu/edge-single-option.json"));
    assert_eq!(golden["error"]["type"], "RuntimeError");
    assert!(matches!(
        request.questions[0].1,
        laya_core::Question::Choice { ref criteria, .. } if criteria.len() == 1
    ));
}
