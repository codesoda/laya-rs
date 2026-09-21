use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{bail, Context, Result};
use mlx_rs::{transforms, Array};
use ndarray::ArrayD;
use ndarray_npy::NpzReader;
use serde::Serialize;
use serde_json::Value;

use crate::model::{Batch, DecisionModel, Precision};

#[derive(Debug, Clone, Default, Serialize)]
pub struct Metrics {
    pub count: usize,
    pub max_abs: f64,
    pub mean_abs: f64,
    pub rmse: f64,
    #[serde(skip)]
    abs_sum: f64,
    #[serde(skip)]
    square_sum: f64,
}

impl Metrics {
    fn add(&mut self, actual: f32, expected: f64) {
        let diff = (f64::from(actual) - expected).abs();
        self.count += 1;
        self.max_abs = self.max_abs.max(diff);
        self.abs_sum += diff;
        self.square_sum += diff * diff;
        self.finish();
    }

    fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.max_abs = self.max_abs.max(other.max_abs);
        self.abs_sum += other.abs_sum;
        self.square_sum += other.square_sum;
        self.finish();
    }

    fn finish(&mut self) {
        if self.count != 0 {
            self.mean_abs = self.abs_sum / self.count as f64;
            self.rmse = (self.square_sum / self.count as f64).sqrt();
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Fields {
    pub option_logits_masked_full: Metrics,
    pub option_logits_raw: Metrics,
    pub act_logits: Metrics,
    pub act_probs: Metrics,
    pub probabilities: Metrics,
    pub confidence: Metrics,
    pub expected_score: Metrics,
    pub hidden_cls: Metrics,
}

impl Fields {
    fn merge(&mut self, other: &Self) {
        self.option_logits_masked_full
            .merge(&other.option_logits_masked_full);
        self.option_logits_raw.merge(&other.option_logits_raw);
        self.act_logits.merge(&other.act_logits);
        self.act_probs.merge(&other.act_probs);
        self.probabilities.merge(&other.probabilities);
        self.confidence.merge(&other.confidence);
        self.expected_score.merge(&other.expected_score);
        self.hidden_cls.merge(&other.hidden_cls);
    }
}

#[derive(Debug, Serialize)]
pub struct FixtureReport {
    pub fixture: String,
    pub questions: usize,
    pub sequence_shape: [i32; 2],
    pub marker_slots: i32,
    pub elapsed_ms: f64,
    pub fields: Fields,
    pub argmax_agree: usize,
    pub argmax_total: usize,
    pub finite: bool,
    pub pass: bool,
    pub failures: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ParityReport {
    pub schema_version: u32,
    pub backend: &'static str,
    pub mlx_rs: &'static str,
    pub mlx_native: &'static str,
    pub profile: String,
    pub dtype: &'static str,
    pub tolerance_profile: &'static str,
    pub model_dir: String,
    pub golden_dir: String,
    pub skipped_upstream_errors: Vec<String>,
    pub fixtures: Vec<FixtureReport>,
    pub aggregate: Fields,
    pub aggregate_failures: Vec<String>,
    pub argmax_agree: usize,
    pub argmax_total: usize,
    pub pass: bool,
}

fn nested_i32(value: &Value, name: &str) -> Result<(Vec<i32>, i32, i32)> {
    let rows = value
        .as_array()
        .with_context(|| format!("{name} is not an array"))?;
    let columns = rows
        .first()
        .and_then(Value::as_array)
        .with_context(|| format!("{name} is empty or not rank 2"))?
        .len();
    let mut flat = Vec::with_capacity(rows.len() * columns);
    for row in rows {
        let row = row
            .as_array()
            .with_context(|| format!("{name} row is not an array"))?;
        if row.len() != columns {
            bail!("{name} is ragged");
        }
        for value in row {
            flat.push(
                value
                    .as_i64()
                    .with_context(|| format!("{name} is not integer"))? as i32,
            );
        }
    }
    Ok((flat, rows.len() as i32, columns as i32))
}

fn nested_bool(value: &Value, name: &str) -> Result<(Vec<bool>, i32, i32)> {
    let rows = value
        .as_array()
        .with_context(|| format!("{name} is not an array"))?;
    let columns = rows
        .first()
        .and_then(Value::as_array)
        .with_context(|| format!("{name} is empty or not rank 2"))?
        .len();
    let mut flat = Vec::with_capacity(rows.len() * columns);
    for row in rows {
        let row = row
            .as_array()
            .with_context(|| format!("{name} row is not an array"))?;
        if row.len() != columns {
            bail!("{name} is ragged");
        }
        for value in row {
            flat.push(
                value
                    .as_bool()
                    .with_context(|| format!("{name} is not boolean"))?,
            );
        }
    }
    Ok((flat, rows.len() as i32, columns as i32))
}

fn load_batch(golden: &Value) -> Result<Batch> {
    let source = golden.get("batch").context("golden missing batch")?;
    let (input_ids, batch, length) = nested_i32(&source["input_ids"], "input_ids")?;
    let (attention_mask, att_batch, att_length) =
        nested_i32(&source["attention_mask"], "attention_mask")?;
    let attention_mask = attention_mask.into_iter().map(|v| v != 0).collect();
    let (marker_pos, marker_batch, markers) = nested_i32(&source["marker_pos"], "marker_pos")?;
    let (marker_mask, mask_batch, mask_markers) =
        nested_bool(&source["marker_mask"], "marker_mask")?;
    if (batch, length) != (att_batch, att_length)
        || batch != marker_batch
        || batch != mask_batch
        || markers != mask_markers
    {
        bail!("inconsistent golden batch tensor shapes");
    }
    let qtype = source["qtype"]
        .as_array()
        .context("qtype is not an array")?
        .iter()
        .map(|value| {
            value
                .as_i64()
                .context("qtype is not integer")
                .map(|v| v as i32)
        })
        .collect::<Result<Vec<_>>>()?;
    if qtype.len() != batch as usize {
        bail!("qtype length does not match batch");
    }
    Ok(Batch {
        input_ids,
        attention_mask,
        marker_pos,
        marker_mask,
        qtype,
        batch,
        length,
        markers,
    })
}

fn values(array: &Array) -> Result<Vec<f32>> {
    let contiguous = array
        .contiguous()
        .context("materialize contiguous result")?;
    contiguous.to_vec_cast::<f32>().context("copy MLX result")
}

fn json_numbers(value: &Value, field: &str) -> Result<Vec<f64>> {
    value
        .get(field)
        .with_context(|| format!("golden model row missing {field}"))?
        .as_array()
        .with_context(|| format!("golden {field} is not array"))?
        .iter()
        .map(|v| {
            v.as_f64()
                .with_context(|| format!("golden {field} is not numeric"))
        })
        .collect()
}

fn stable_softmax(values: &[f32], scale: f64) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut result: Vec<f32> = values
        .iter()
        .map(|value| ((f64::from(*value - max) / scale).exp()) as f32)
        .collect();
    let sum: f32 = result.iter().sum();
    for value in &mut result {
        *value /= sum;
    }
    result
}

fn compare_slice(metric: &mut Metrics, actual: &[f32], expected: &[f64]) -> Result<()> {
    if actual.len() != expected.len() {
        bail!(
            "comparison length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );
    }
    for (&actual, &expected) in actual.iter().zip(expected) {
        metric.add(actual, expected);
    }
    Ok(())
}

fn confidence(probabilities: &[f32]) -> f32 {
    if probabilities.len() < 2 {
        return 1.0;
    }
    let entropy: f32 = probabilities.iter().map(|p| -*p * p.max(1e-12).ln()).sum();
    (1.0 - entropy / (probabilities.len() as f32).ln()).clamp(0.0, 1.0)
}

fn load_hidden(path: &Path) -> Result<Vec<f32>> {
    let mut archive = NpzReader::new(File::open(path)?)?;
    let hidden: ArrayD<f32> = archive.by_name("hidden_cls.npy")?;
    Ok(hidden.into_raw_vec_and_offset().0)
}

fn evaluate_fixture(
    model: &DecisionModel,
    precision: Precision,
    fixture: &str,
    path: &Path,
) -> Result<FixtureReport> {
    let golden: Value = serde_json::from_slice(&fs::read(path)?)?;
    let batch = load_batch(&golden)?;
    let started = Instant::now();
    let output = model.forward(&batch)?;
    transforms::eval([
        &output.raw_logits,
        &output.masked_logits,
        &output.act_logits,
        &output.hidden_cls,
    ])?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let raw = values(&output.raw_logits)?;
    let masked = values(&output.masked_logits)?;
    let act = values(&output.act_logits)?;
    let hidden = values(&output.hidden_cls)?;
    let expected_model = golden["model"].as_object().context("model is not object")?;
    if expected_model.len() != batch.batch as usize {
        bail!("model question count differs from batch");
    }
    let mut fields = Fields::default();
    let mut argmax_agree = 0;
    let mut argmax_total = 0;
    let mut finite = raw
        .iter()
        .chain(&masked)
        .chain(&act)
        .chain(&hidden)
        .all(|v| v.is_finite());
    for (row, (_qid, expected)) in expected_model.iter().enumerate() {
        let start = row * batch.markers as usize;
        let end = start + batch.markers as usize;
        let expected_masked = json_numbers(expected, "option_logits_masked_full")?;
        compare_slice(
            &mut fields.option_logits_masked_full,
            &masked[start..end],
            &expected_masked,
        )?;
        let expected_raw = json_numbers(expected, "option_logits_raw")?;
        compare_slice(
            &mut fields.option_logits_raw,
            &raw[start..start + expected_raw.len()],
            &expected_raw,
        )?;
        let expected_act = json_numbers(expected, "act_logits")?;
        compare_slice(
            &mut fields.act_logits,
            &act[row * 2..row * 2 + 2],
            &expected_act,
        )?;
        let actual_act_probs = stable_softmax(&act[row * 2..row * 2 + 2], 1.0);
        let expected_act_probs = json_numbers(expected, "act_probs")?;
        compare_slice(
            &mut fields.act_probs,
            &actual_act_probs,
            &expected_act_probs,
        )?;
        let temperature = expected["temperature"]
            .as_f64()
            .context("temperature is not numeric")?
            .max(1e-3);
        let probabilities = stable_softmax(&raw[start..start + expected_raw.len()], temperature);
        let expected_probabilities = json_numbers(expected, "probs_unrounded")?;
        compare_slice(
            &mut fields.probabilities,
            &probabilities,
            &expected_probabilities,
        )?;
        let kind = batch.qtype[row];
        // The instrumented golden's model.confidence_unrounded records the shared
        // normalized-entropy statistic before native noul presentation overrides it.
        let actual_confidence = confidence(&probabilities);
        fields.confidence.add(
            actual_confidence,
            expected["confidence_unrounded"]
                .as_f64()
                .context("confidence_unrounded is not numeric")?,
        );
        if kind == 1 {
            let score: f32 = probabilities
                .iter()
                .enumerate()
                .map(|(i, p)| i as f32 * p)
                .sum();
            fields.expected_score.add(
                score,
                expected["expected_score_unrounded"]
                    .as_f64()
                    .context("expected score is not numeric")?,
            );
        }
        let actual_argmax = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .unwrap_or(0);
        let expected_argmax = expected["argmax_index"]
            .as_u64()
            .context("argmax_index is not integer")? as usize;
        argmax_total += 1;
        argmax_agree += usize::from(actual_argmax == expected_argmax);
        finite &= probabilities
            .iter()
            .chain(&actual_act_probs)
            .all(|v| v.is_finite());
    }
    let sidecar = golden["hidden_states_npz"]
        .as_str()
        .context("hidden_states_npz missing")?;
    let expected_hidden = load_hidden(&path.with_file_name(sidecar))?;
    let expected_hidden: Vec<f64> = expected_hidden.into_iter().map(f64::from).collect();
    compare_slice(&mut fields.hidden_cls, &hidden, &expected_hidden)?;

    let mut failures = Vec::new();
    match precision {
        Precision::F32 => {
            check_max(
                &mut failures,
                "option_logits_raw",
                &fields.option_logits_raw,
                4e-4,
            );
            check_max(&mut failures, "act_logits", &fields.act_logits, 5e-2);
            check_max(&mut failures, "probabilities", &fields.probabilities, 4e-5);
            check_max(
                &mut failures,
                "expected_score",
                &fields.expected_score,
                1e-4,
            );
            check_max(&mut failures, "confidence", &fields.confidence, 1e-4);
            check_max(&mut failures, "act_probs", &fields.act_probs, 1e-3);
        }
        Precision::F16 => {
            check_max(
                &mut failures,
                "option_logits_raw",
                &fields.option_logits_raw,
                5e-2,
            );
            check_max(&mut failures, "probabilities", &fields.probabilities, 5e-3);
            check_max(
                &mut failures,
                "expected_score",
                &fields.expected_score,
                1e-2,
            );
            check_max(&mut failures, "act_probs", &fields.act_probs, 1e-2);
        }
    }
    if !finite {
        failures.push("non-finite model output".to_owned());
    }
    if argmax_agree != argmax_total {
        failures.push(format!("argmax agreement {argmax_agree}/{argmax_total}"));
    }
    Ok(FixtureReport {
        fixture: fixture.to_owned(),
        questions: batch.batch as usize,
        sequence_shape: [batch.batch, batch.length],
        marker_slots: batch.markers,
        elapsed_ms,
        fields,
        argmax_agree,
        argmax_total,
        finite,
        pass: failures.is_empty(),
        failures,
    })
}

fn check_max(failures: &mut Vec<String>, name: &str, metric: &Metrics, limit: f64) {
    if metric.count != 0 && metric.max_abs > limit {
        failures.push(format!(
            "{name} max_abs {:.6e} > {limit:.6e}",
            metric.max_abs
        ));
    }
}

pub fn run(
    root: &Path,
    model_dir: &Path,
    profile: &str,
    fixture: &str,
    precision: Precision,
    report_dir: Option<&Path>,
) -> Result<(ParityReport, PathBuf)> {
    let golden_dir = root.join("benchmarks/goldens").join(profile).join("cpu");
    let mut paths = if fixture == "all" {
        fs::read_dir(&golden_dir)
            .with_context(|| format!("read golden directory {}", golden_dir.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().and_then(|v| v.to_str()) == Some("json"))
            .collect::<Vec<_>>()
    } else {
        vec![golden_dir.join(format!("{fixture}.json"))]
    };
    paths.sort();
    let model = DecisionModel::load(model_dir, precision)?;
    let mut fixtures = Vec::new();
    let mut skipped_upstream_errors = Vec::new();
    for path in paths {
        let id = path
            .file_stem()
            .and_then(|v| v.to_str())
            .context("non-UTF8 fixture filename")?;
        let value: Value = serde_json::from_slice(&fs::read(&path)?)?;
        if !value.get("error").unwrap_or(&Value::Null).is_null() {
            skipped_upstream_errors.push(id.to_owned());
            continue;
        }
        let result = evaluate_fixture(&model, precision, id, &path)
            .with_context(|| format!("evaluate fixture {id}"))?;
        println!(
            "{}: {} raw max={:.3e} mean={:.3e} act max={:.3e} probs max={:.3e} argmax={}/{} {:.1}ms",
            id,
            if result.pass { "PASS" } else { "FAIL" },
            result.fields.option_logits_raw.max_abs,
            result.fields.option_logits_raw.mean_abs,
            result.fields.act_logits.max_abs,
            result.fields.probabilities.max_abs,
            result.argmax_agree,
            result.argmax_total,
            result.elapsed_ms,
        );
        for failure in &result.failures {
            println!("  {failure}");
        }
        fixtures.push(result);
    }
    let mut aggregate = Fields::default();
    let mut argmax_agree = 0;
    let mut argmax_total = 0;
    let mut pass = !fixtures.is_empty();
    for fixture in &fixtures {
        aggregate.merge(&fixture.fields);
        argmax_agree += fixture.argmax_agree;
        argmax_total += fixture.argmax_total;
        pass &= fixture.pass;
    }
    let mut aggregate_failures = Vec::new();
    // Frozen mean-error limits describe the complete profile aggregate. Applying
    // them to a single selected fixture would make `--fixture` disagree with the
    // same fixture's result inside `--fixture all`.
    if fixture == "all" && matches!(precision, Precision::F32) {
        if aggregate.option_logits_raw.mean_abs > 4e-5 {
            aggregate_failures.push(format!(
                "option_logits_raw aggregate mean_abs {:.6e} > 4.000000e-5",
                aggregate.option_logits_raw.mean_abs
            ));
        }
        if aggregate.act_logits.mean_abs > 5e-3 {
            aggregate_failures.push(format!(
                "act_logits aggregate mean_abs {:.6e} > 5.000000e-3",
                aggregate.act_logits.mean_abs
            ));
        }
    }
    pass &= aggregate_failures.is_empty();
    let report = ParityReport {
        schema_version: 1,
        backend: "mlx-rs/Metal",
        mlx_rs: "0.32.0",
        mlx_native: "0.32.2",
        profile: profile.to_owned(),
        dtype: precision.label(),
        tolerance_profile: match precision {
            Precision::F32 => "rust_metal_fp32",
            Precision::F16 => "rust_metal_fp16_or_bf16",
        },
        model_dir: model_dir.display().to_string(),
        golden_dir: golden_dir.display().to_string(),
        skipped_upstream_errors,
        fixtures,
        aggregate,
        aggregate_failures,
        argmax_agree,
        argmax_total,
        pass,
    };
    let output_dir = report_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(|| root.join("benchmarks/results/l1-spike-mlx"));
    fs::create_dir_all(&output_dir)?;
    let output_path = output_dir.join(format!("{profile}-{}.json", precision.label()));
    fs::write(&output_path, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "aggregate: {} raw max={:.3e} mean={:.3e} act max={:.3e} probs max={:.3e} argmax={}/{}",
        if pass { "PASS" } else { "FAIL" },
        report.aggregate.option_logits_raw.max_abs,
        report.aggregate.option_logits_raw.mean_abs,
        report.aggregate.act_logits.max_abs,
        report.aggregate.probabilities.max_abs,
        argmax_agree,
        argmax_total,
    );
    for failure in &report.aggregate_failures {
        println!("  {failure}");
    }
    println!("report: {}", output_path.display());
    Ok((report, output_path))
}

pub fn timing(
    root: &Path,
    model_dir: &Path,
    profile: &str,
    fixture: &str,
    precision: Precision,
    reps: usize,
) -> Result<()> {
    let path = root
        .join("benchmarks/goldens")
        .join(profile)
        .join("cpu")
        .join(format!("{fixture}.json"));
    let golden: Value = serde_json::from_slice(&fs::read(&path)?)?;
    if !golden.get("error").unwrap_or(&Value::Null).is_null() {
        bail!("cannot time upstream-error fixture {fixture}");
    }
    let batch = load_batch(&golden)?;
    let model = DecisionModel::load(model_dir, precision)?;
    mlx_rs::memory::reset_peak_memory()?;
    let mut samples = Vec::with_capacity(reps + 1);
    for _ in 0..=reps {
        let started = Instant::now();
        let output = model.forward(&batch)?;
        transforms::eval([
            &output.raw_logits,
            &output.masked_logits,
            &output.act_logits,
            &output.hidden_cls,
        ])?;
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    let first = samples.remove(0);
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let median = sorted[sorted.len() / 2];
    println!("label: spike, not acceptance");
    println!(
        "profile={profile} fixture={fixture} dtype={} reps={reps}",
        precision.label()
    );
    println!("first_call_ms={first:.3}");
    println!("warm_p50_ms={median:.3} warm_mean_ms={mean:.3}");
    println!(
        "mlx_active_bytes={} mlx_peak_bytes={}",
        mlx_rs::memory::active_memory()?,
        mlx_rs::memory::peak_memory()?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_are_weighted_when_merged() {
        let mut first = Metrics::default();
        first.add(1.0, 0.0);
        let mut second = Metrics::default();
        second.add(2.0, 0.0);
        second.add(2.0, 0.0);
        first.merge(&second);
        assert_eq!(first.count, 3);
        assert!((first.mean_abs - 5.0 / 3.0).abs() < 1e-12);
        assert_eq!(first.max_abs, 2.0);
    }

    #[test]
    fn stable_softmax_is_normalized() {
        let values = stable_softmax(&[1000.0, 999.0], 1.0);
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(values[0] > values[1]);
    }
}
