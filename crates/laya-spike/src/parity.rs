use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use ndarray::{ArrayD, IxDyn};
use ndarray_npy::NpzReader;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{repo_root, Profile, RequestedDType, RequestedDevice};
use crate::model::{ForwardOutput, LayaModel};

#[derive(Debug, Deserialize)]
struct GoldenRecord {
    batch: Option<GoldenBatch>,
    model: Option<serde_json::Map<String, Value>>,
    error: Option<Value>,
    hidden_states_npz: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GoldenBatch {
    input_ids: Vec<Vec<u32>>,
    attention_mask: Vec<Vec<u32>>,
    marker_pos: Vec<Vec<u32>>,
    marker_mask: Vec<Vec<bool>>,
    qtype: Vec<u32>,
}

pub(crate) struct DeviceBatch {
    input_ids: Tensor,
    attention_mask: Tensor,
    marker_pos: Tensor,
    marker_mask: Tensor,
    qtype: Tensor,
}

impl DeviceBatch {
    fn from_golden(batch: &GoldenBatch, device: &Device) -> Result<Self> {
        Ok(Self {
            input_ids: Tensor::new(batch.input_ids.clone(), device)?,
            attention_mask: Tensor::new(batch.attention_mask.clone(), device)?,
            marker_pos: Tensor::new(batch.marker_pos.clone(), device)?,
            marker_mask: Tensor::new(
                batch
                    .marker_mask
                    .iter()
                    .map(|row| row.iter().map(|value| u8::from(*value)).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                device,
            )?,
            qtype: Tensor::new(batch.qtype.clone(), device)?,
        })
    }

    pub(crate) fn forward(&self, model: &LayaModel) -> candle_core::Result<ForwardOutput> {
        model.forward(
            &self.input_ids,
            &self.attention_mask,
            &self.marker_pos,
            &self.marker_mask,
            &self.qtype,
        )
    }
}

pub(crate) fn prepare_fixture(path: &Path, device: &Device) -> Result<DeviceBatch> {
    let record: GoldenRecord = serde_json::from_slice(&fs::read(path)?)
        .with_context(|| format!("parse golden {}", path.display()))?;
    if record.error.is_some() {
        bail!(
            "timing fixture {} is an upstream error fixture",
            path.display()
        )
    }
    DeviceBatch::from_golden(
        record
            .batch
            .as_ref()
            .context("successful fixture lacks batch")?,
        device,
    )
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Metrics {
    pub count: usize,
    pub max_abs: f64,
    pub mean_abs: f64,
    pub rmse: f64,
    #[serde(skip)]
    sum_abs: f64,
    #[serde(skip)]
    sum_squared: f64,
}

impl Metrics {
    fn add(&mut self, actual: &[f32], expected: &[f32]) -> Result<()> {
        if actual.len() != expected.len() {
            bail!(
                "metric vector length mismatch: actual {}, expected {}",
                actual.len(),
                expected.len()
            )
        }
        for (actual, expected) in actual.iter().zip(expected) {
            let difference = f64::from((*actual - *expected).abs());
            self.count += 1;
            self.max_abs = self.max_abs.max(difference);
            self.sum_abs += difference;
            self.sum_squared += difference * difference;
        }
        self.finish();
        Ok(())
    }

    fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.max_abs = self.max_abs.max(other.max_abs);
        self.sum_abs += other.sum_abs;
        self.sum_squared += other.sum_squared;
        self.finish();
    }

    fn finish(&mut self) {
        if self.count != 0 {
            self.mean_abs = self.sum_abs / self.count as f64;
            self.rmse = (self.sum_squared / self.count as f64).sqrt();
        }
    }
}

#[derive(Debug, Serialize)]
pub struct FixtureReport {
    pub fixture: String,
    pub metrics: BTreeMap<String, Metrics>,
    pub argmax_agree: usize,
    pub questions: usize,
    pub near_tie_disagreements: Vec<String>,
    pub failures: Vec<String>,
    pub finite_outputs: bool,
    pub outputs_on_requested_device: bool,
    pub pass: bool,
}

#[derive(Debug, Serialize)]
pub struct ParityReport {
    pub schema_version: u32,
    pub generated_unix_seconds: u64,
    pub profile: String,
    pub device: String,
    pub device_name: String,
    pub dtype: String,
    pub tolerance_profile: String,
    pub candle_versions: BTreeMap<String, String>,
    pub checkpoint: String,
    pub load_mmap_to_model_ready_ms: f64,
    pub peak_rss_bytes: u64,
    pub metal_allocation_bytes: Option<u64>,
    pub metal_allocation_note: String,
    pub all_outputs_finite: bool,
    pub all_model_outputs_resident_on_requested_device: bool,
    pub inference_ops_run_on_cpu: Vec<String>,
    pub fixtures_skipped_upstream_error: Vec<String>,
    pub fixtures: Vec<FixtureReport>,
    pub aggregate: BTreeMap<String, Metrics>,
    pub argmax_agree: usize,
    pub questions: usize,
    pub failures: Vec<String>,
    pub pass: bool,
}

pub struct ParityContext<'a> {
    pub profile: Profile,
    pub requested_device: RequestedDevice,
    pub requested_dtype: RequestedDType,
    pub device: &'a Device,
    pub device_name: String,
    pub model: &'a LayaModel,
    pub checkpoint: &'a Path,
    pub load_ms: f64,
}

pub fn run_parity(context: ParityContext<'_>, fixture: &str) -> Result<PathBuf> {
    let golden_dir = repo_root()?
        .join("benchmarks/goldens")
        .join(context.profile.as_str())
        .join("cpu");
    let paths = fixture_paths(&golden_dir, fixture)?;
    let mut aggregate = BTreeMap::<String, Metrics>::new();
    let mut fixture_reports = Vec::new();
    let mut skipped = Vec::new();
    let mut total_argmax = 0;
    let mut total_questions = 0;
    let mut all_resident = true;

    for path in paths {
        let fixture_id = path
            .file_stem()
            .and_then(|value| value.to_str())
            .context("golden fixture has non-UTF8 filename")?
            .to_owned();
        let record: GoldenRecord = serde_json::from_slice(&fs::read(&path)?)
            .with_context(|| format!("parse golden {}", path.display()))?;
        if record.error.is_some() {
            skipped.push(fixture_id);
            continue;
        }
        let batch = record
            .batch
            .as_ref()
            .with_context(|| format!("successful fixture {fixture_id} lacks batch"))?;
        let expected_model = record
            .model
            .as_ref()
            .with_context(|| format!("successful fixture {fixture_id} lacks model"))?;
        let device_batch = DeviceBatch::from_golden(batch, context.device)?;
        let output = context.model.forward(
            &device_batch.input_ids,
            &device_batch.attention_mask,
            &device_batch.marker_pos,
            &device_batch.marker_mask,
            &device_batch.qtype,
        )?;
        context.device.synchronize()?;
        let mut report = compare_fixture(
            &fixture_id,
            &path,
            batch,
            expected_model,
            record.hidden_states_npz.as_deref(),
            &output,
            &context,
        )?;
        apply_tolerances(
            &mut report,
            context.requested_device,
            context.requested_dtype,
        );
        for (field, metrics) in &report.metrics {
            aggregate.entry(field.clone()).or_default().merge(metrics);
        }
        total_argmax += report.argmax_agree;
        total_questions += report.questions;
        all_resident &= report.outputs_on_requested_device;
        print_fixture(&report);
        fixture_reports.push(report);
    }

    let mut failures = Vec::new();
    apply_aggregate_tolerances(
        &aggregate,
        total_argmax,
        total_questions,
        context.requested_device,
        context.requested_dtype,
        &mut failures,
    );
    if fixture_reports.iter().any(|report| !report.pass) {
        failures.push("one or more fixtures failed per-fixture tolerance checks".to_owned());
    }
    if !all_resident {
        failures.push(
            "one or more output tensors were not resident on the requested device".to_owned(),
        );
    }
    let tolerance_profile = tolerance_profile(context.requested_device, context.requested_dtype);
    let report = ParityReport {
        schema_version: 1,
        generated_unix_seconds: unix_seconds(),
        profile: context.profile.as_str().to_owned(),
        device: context.requested_device.as_str().to_owned(),
        device_name: context.device_name,
        dtype: context.requested_dtype.as_str().to_owned(),
        tolerance_profile: tolerance_profile.to_owned(),
        candle_versions: BTreeMap::from([
            ("candle-core".to_owned(), "0.11.0".to_owned()),
            ("candle-nn".to_owned(), "0.11.0".to_owned()),
        ]),
        checkpoint: context.checkpoint.display().to_string(),
        load_mmap_to_model_ready_ms: context.load_ms,
        peak_rss_bytes: peak_rss_bytes(),
        metal_allocation_bytes: None,
        metal_allocation_note: "Candle 0.11.0 exposes synchronization and device residency but no public current/peak Metal allocation counter".to_owned(),
        all_outputs_finite: true,
        all_model_outputs_resident_on_requested_device: all_resident,
        inference_ops_run_on_cpu: Vec::new(),
        fixtures_skipped_upstream_error: skipped,
        fixtures: fixture_reports,
        aggregate,
        argmax_agree: total_argmax,
        questions: total_questions,
        pass: failures.is_empty(),
        failures,
    };
    print_aggregate(&report);
    let output_path = write_report(&report)?;
    println!("report={}", output_path.display());
    if !report.pass {
        bail!("parity gate failed; see {}", output_path.display())
    }
    Ok(output_path)
}

fn compare_fixture(
    fixture: &str,
    golden_path: &Path,
    batch: &GoldenBatch,
    expected_model: &serde_json::Map<String, Value>,
    npz_name: Option<&str>,
    output: &ForwardOutput,
    context: &ParityContext<'_>,
) -> Result<FixtureReport> {
    let logits = output.logits.to_vec2::<f32>()?;
    let action_logits = output.action_logits.to_vec2::<f32>()?;
    let action_probs = output.action_probs.to_vec2::<f32>()?;
    let hidden_cls = output.hidden_cls.to_vec2::<f32>()?;
    let expected_hidden = load_hidden_cls(
        &golden_path.with_file_name(npz_name.context("successful fixture lacks NPZ sidecar name")?),
    )?;
    let expected_hidden = expected_hidden
        .as_slice()
        .context("hidden_cls NPZ array is not contiguous")?;
    let actual_hidden = hidden_cls.iter().flatten().copied().collect::<Vec<_>>();
    ensure_finite(
        fixture,
        "option_logits_masked_full",
        logits.iter().flatten().copied(),
    )?;
    ensure_finite(
        fixture,
        "act_logits",
        action_logits.iter().flatten().copied(),
    )?;
    ensure_finite(fixture, "act_probs", action_probs.iter().flatten().copied())?;
    ensure_finite(fixture, "hidden_cls", actual_hidden.iter().copied())?;

    let mut metrics = BTreeMap::new();
    metric(&mut metrics, "hidden_cls", &actual_hidden, expected_hidden)?;
    let mut argmax_agree = 0;
    let mut near_tie_disagreements = Vec::new();
    let mut failures = Vec::new();

    if expected_model.len() != logits.len() {
        bail!(
            "fixture {fixture}: golden question count {} != batch {}",
            expected_model.len(),
            logits.len()
        )
    }
    for (row, (qid, expected)) in expected_model.iter().enumerate() {
        let expected = expected
            .as_object()
            .with_context(|| format!("model.{qid} is not an object"))?;
        let marker_count = batch.marker_mask[row]
            .iter()
            .filter(|value| **value)
            .count();
        let expected_raw = float_array(expected, "option_logits_raw")?;
        let expected_masked = float_array(expected, "option_logits_masked_full")?;
        let expected_action = float_array(expected, "act_logits")?;
        let expected_action_probs = float_array(expected, "act_probs")?;
        metric(
            &mut metrics,
            "option_logits_raw",
            &logits[row][..marker_count],
            &expected_raw,
        )?;
        metric(
            &mut metrics,
            "option_logits_masked_full",
            &logits[row],
            &expected_masked,
        )?;
        metric(
            &mut metrics,
            "act_logits",
            &action_logits[row],
            &expected_action,
        )?;
        metric(
            &mut metrics,
            "act_probs",
            &action_probs[row],
            &expected_action_probs,
        )?;

        let temperature = expected
            .get("temperature")
            .and_then(Value::as_f64)
            .with_context(|| format!("model.{qid}.temperature missing"))?
            .max(1e-3);
        let probabilities = stable_softmax(
            &logits[row][..marker_count]
                .iter()
                .map(|value| f64::from(*value) / temperature)
                .collect::<Vec<_>>(),
        );
        let probabilities_f32 = probabilities
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        ensure_finite(
            fixture,
            &format!("model.{qid}.probs_unrounded"),
            probabilities_f32.iter().copied(),
        )?;
        let expected_probabilities = float_array(expected, "probs_unrounded")?;
        metric(
            &mut metrics,
            "probs_unrounded",
            &probabilities_f32,
            &expected_probabilities,
        )?;
        let confidence = normalized_entropy_confidence(&probabilities) as f32;
        let expected_confidence = expected
            .get("confidence_unrounded")
            .and_then(Value::as_f64)
            .with_context(|| format!("model.{qid}.confidence_unrounded missing"))?
            as f32;
        metric(
            &mut metrics,
            "confidence_unrounded",
            &[confidence],
            &[expected_confidence],
        )?;
        if let Some(expected_score) = expected
            .get("expected_score_unrounded")
            .and_then(Value::as_f64)
        {
            let score = probabilities
                .iter()
                .enumerate()
                .map(|(index, probability)| index as f64 * probability)
                .sum::<f64>() as f32;
            metric(
                &mut metrics,
                "expected_score_unrounded",
                &[score],
                &[expected_score as f32],
            )?;
        }
        let actual_argmax = first_argmax(&probabilities);
        let expected_argmax = expected
            .get("argmax_index")
            .and_then(Value::as_u64)
            .with_context(|| format!("model.{qid}.argmax_index missing"))?
            as usize;
        if actual_argmax == expected_argmax {
            argmax_agree += 1;
        } else {
            let mut ordered = expected_probabilities.clone();
            ordered.sort_by(|left, right| right.total_cmp(left));
            let threshold = if context.requested_dtype == RequestedDType::F32 {
                1e-5
            } else {
                5e-3
            };
            let gap = f64::from(ordered[0] - ordered[1]);
            if gap.abs() < threshold {
                near_tie_disagreements.push(format!(
                    "{qid}: actual={actual_argmax}, expected={expected_argmax}, golden top-two gap={gap:.9e}"
                ));
            } else {
                failures.push(format!(
                    "{qid}: argmax actual={actual_argmax}, expected={expected_argmax}, golden top-two gap={gap:.9e}"
                ));
            }
        }
    }

    let outputs_on_requested_device = [
        &output.logits,
        &output.action_logits,
        &output.action_probs,
        &output.hidden_cls,
    ]
    .iter()
    .all(|tensor| tensor.device().same_device(context.device));
    Ok(FixtureReport {
        fixture: fixture.to_owned(),
        metrics,
        argmax_agree,
        questions: expected_model.len(),
        near_tie_disagreements,
        failures,
        finite_outputs: true,
        outputs_on_requested_device,
        pass: true,
    })
}

fn apply_tolerances(report: &mut FixtureReport, device: RequestedDevice, dtype: RequestedDType) {
    let limits = limits(device, dtype);
    // Frozen max-absolute limits apply to each fixture. Frozen mean limits are
    // profile aggregates (matching the L0 evidence aggregation), and are applied
    // after every fixture contributes its values.
    for (field, (max_limit, _mean_limit)) in limits {
        if let Some(metrics) = report.metrics.get(field) {
            if metrics.max_abs > max_limit {
                report.failures.push(format!(
                    "{field}.max_abs {:.9e} > {:.9e}",
                    metrics.max_abs, max_limit
                ));
            }
        }
    }
    report.pass = report.failures.is_empty();
}

fn apply_aggregate_tolerances(
    aggregate: &BTreeMap<String, Metrics>,
    argmax_agree: usize,
    questions: usize,
    device: RequestedDevice,
    dtype: RequestedDType,
    failures: &mut Vec<String>,
) {
    for (field, (max_limit, mean_limit)) in limits(device, dtype) {
        if let Some(metrics) = aggregate.get(field) {
            if metrics.max_abs > max_limit {
                failures.push(format!(
                    "aggregate {field}.max_abs {:.9e} > {:.9e}",
                    metrics.max_abs, max_limit
                ));
            }
            if let Some(mean_limit) = mean_limit {
                if metrics.mean_abs > mean_limit {
                    failures.push(format!(
                        "aggregate {field}.mean_abs {:.9e} > {:.9e}",
                        metrics.mean_abs, mean_limit
                    ));
                }
            }
        }
    }
    if dtype == RequestedDType::F32 && argmax_agree != questions {
        failures.push(format!(
            "argmax agreement {argmax_agree}/{questions}, expected 100%"
        ));
    }
    if dtype != RequestedDType::F32 && questions != 0 {
        let agreement = argmax_agree as f64 / questions as f64;
        if agreement < 0.99 {
            failures.push(format!("argmax agreement {:.3}% < 99%", agreement * 100.));
        }
    }
}

fn limits(
    device: RequestedDevice,
    dtype: RequestedDType,
) -> Vec<(&'static str, (f64, Option<f64>))> {
    if dtype != RequestedDType::F32 {
        return vec![
            ("option_logits_raw", (5e-2, None)),
            ("probs_unrounded", (5e-3, None)),
            ("expected_score_unrounded", (1e-2, None)),
            ("act_probs", (1e-2, None)),
        ];
    }
    let (option_max, option_mean, probability_max, score_max, confidence_max) = match device {
        RequestedDevice::Cpu => (2e-4, 2e-5, 2e-5, 5e-5, 5e-5),
        RequestedDevice::Metal => (4e-4, 4e-5, 4e-5, 1e-4, 1e-4),
    };
    vec![
        ("option_logits_raw", (option_max, Some(option_mean))),
        ("act_logits", (5e-2, Some(5e-3))),
        ("probs_unrounded", (probability_max, None)),
        ("expected_score_unrounded", (score_max, None)),
        ("confidence_unrounded", (confidence_max, None)),
        ("act_probs", (1e-3, None)),
    ]
}

fn tolerance_profile(device: RequestedDevice, dtype: RequestedDType) -> &'static str {
    match (device, dtype) {
        (RequestedDevice::Cpu, RequestedDType::F32) => "rust_cpu_fp32",
        (RequestedDevice::Metal, RequestedDType::F32) => "rust_metal_fp32",
        (_, _) => "rust_metal_fp16_or_bf16",
    }
}

fn metric(
    metrics: &mut BTreeMap<String, Metrics>,
    name: &str,
    actual: &[f32],
    expected: &[f32],
) -> Result<()> {
    metrics
        .entry(name.to_owned())
        .or_default()
        .add(actual, expected)
}

fn float_array(object: &serde_json::Map<String, Value>, name: &str) -> Result<Vec<f32>> {
    object
        .get(name)
        .and_then(Value::as_array)
        .with_context(|| format!("missing array field {name}"))?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .with_context(|| format!("{name} includes non-number"))
        })
        .collect()
}

fn load_hidden_cls(path: &Path) -> Result<ArrayD<f32>> {
    let mut reader =
        NpzReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?)?;
    reader
        .by_name::<ndarray::OwnedRepr<f32>, IxDyn>("hidden_cls")
        .with_context(|| format!("read hidden_cls from {}", path.display()))
}

fn ensure_finite(fixture: &str, field: &str, values: impl IntoIterator<Item = f32>) -> Result<()> {
    if let Some((index, value)) = values
        .into_iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        bail!("fixture {fixture}: non-finite {field}[{index}]={value}")
    }
    Ok(())
}

fn stable_softmax(values: &[f64]) -> Vec<f64> {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut output = values
        .iter()
        .map(|value| (value - maximum).exp())
        .collect::<Vec<_>>();
    let total = output.iter().sum::<f64>();
    for value in &mut output {
        *value /= total;
    }
    output
}

fn normalized_entropy_confidence(probabilities: &[f64]) -> f64 {
    if probabilities.len() < 2 {
        return 1.;
    }
    let entropy = -probabilities
        .iter()
        .map(|probability| probability * probability.max(1e-12).ln())
        .sum::<f64>();
    (1. - entropy / (probabilities.len() as f64).ln()).clamp(0., 1.)
}

fn first_argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| right_index.cmp(left_index))
        })
        .map_or(0, |(index, _)| index)
}

fn fixture_paths(directory: &Path, fixture: &str) -> Result<Vec<PathBuf>> {
    if fixture != "all" {
        let path = directory.join(format!("{fixture}.json"));
        if !path.is_file() {
            bail!("golden fixture does not exist: {}", path.display())
        }
        return Ok(vec![path]);
    }
    let mut paths = fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.retain(|path| {
        path.extension()
            .is_some_and(|extension| extension == "json")
    });
    paths.sort();
    Ok(paths)
}

fn print_fixture(report: &FixtureReport) {
    println!(
        "fixture={} status={} argmax={}/{} finite={} resident={}",
        report.fixture,
        if report.pass { "PASS" } else { "FAIL" },
        report.argmax_agree,
        report.questions,
        report.finite_outputs,
        report.outputs_on_requested_device
    );
    for (field, metrics) in &report.metrics {
        println!(
            "  {field}: max={:.9e} mean={:.9e} rmse={:.9e} n={}",
            metrics.max_abs, metrics.mean_abs, metrics.rmse, metrics.count
        );
    }
    for failure in &report.failures {
        println!("  failure: {failure}");
    }
}

fn print_aggregate(report: &ParityReport) {
    println!(
        "aggregate status={} profile={} device={} dtype={} argmax={}/{} rss={} load_ms={:.3}",
        if report.pass { "PASS" } else { "FAIL" },
        report.profile,
        report.device,
        report.dtype,
        report.argmax_agree,
        report.questions,
        report.peak_rss_bytes,
        report.load_mmap_to_model_ready_ms
    );
    for (field, metrics) in &report.aggregate {
        println!(
            "  {field}: max={:.9e} mean={:.9e} rmse={:.9e} n={}",
            metrics.max_abs, metrics.mean_abs, metrics.rmse, metrics.count
        );
    }
    for failure in &report.failures {
        println!("  failure: {failure}");
    }
}

fn write_report(report: &ParityReport) -> Result<PathBuf> {
    let directory = repo_root()?.join("benchmarks/results/l1-spike");
    fs::create_dir_all(&directory)?;
    let stem = format!("{}-{}-{}", report.profile, report.device, report.dtype);
    let mut path = directory.join(format!("{stem}.json"));
    if path.exists() {
        path = directory.join(format!("{stem}-{}.json", unix_seconds()));
        let mut sequence = 1;
        while path.exists() {
            path = directory.join(format!("{stem}-{}-{sequence}.json", unix_seconds()));
            sequence += 1;
        }
    }
    fs::write(&path, serde_json::to_vec_pretty(report)?)?;
    Ok(path)
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage initializes the provided rusage on success.
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return 0;
    }
    // macOS reports ru_maxrss in bytes (Linux reports KiB; this L1 target is macOS).
    unsafe { usage.assume_init() }.ru_maxrss as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_is_stable_and_first_argmax_wins_ties() {
        let probabilities = stable_softmax(&[10_000., 10_000.]);
        assert_eq!(probabilities, [0.5, 0.5]);
        assert_eq!(first_argmax(&probabilities), 0);
    }

    #[test]
    fn metrics_are_exact() -> Result<()> {
        let mut metrics = Metrics::default();
        metrics.add(&[1., 3.], &[0., 1.])?;
        assert_eq!(metrics.max_abs, 2.);
        assert_eq!(metrics.mean_abs, 1.5);
        assert!((metrics.rmse - 2.5f64.sqrt()).abs() < 1e-12);
        Ok(())
    }
}
