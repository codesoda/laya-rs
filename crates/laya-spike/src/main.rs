mod config;
mod inventory;
mod model;
mod parity;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::Device;
use candle_nn::VarBuilder;
use clap::{Parser, Subcommand};

use config::{
    profile_dir, repo_root, AgentConfig, EncoderConfig, Profile, RequestedDType, RequestedDevice,
};
use inventory::{inspect_checkpoint, validate_checkpoint};
use model::LayaModel;
use parity::{peak_rss_bytes, prepare_fixture, run_parity, ParityContext};

#[derive(Debug, Parser)]
#[command(
    name = "laya-spike",
    about = "L1 Candle CPU/Metal full-network feasibility spike"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print checkpoint tensors and the ModernBERT config adapter.
    Inspect {
        #[arg(long, value_enum)]
        profile: Profile,
    },
    /// Run complete-network parity against frozen Python CPU goldens.
    Parity {
        #[arg(long, value_enum)]
        profile: Profile,
        #[arg(long, value_enum)]
        device: RequestedDevice,
        #[arg(long, default_value = "all")]
        fixture: String,
        #[arg(long, value_enum, default_value = "f32")]
        dtype: RequestedDType,
    },
    /// Rough synchronized model-only timing. Spike evidence, not an acceptance benchmark.
    Timing {
        #[arg(long, value_enum)]
        profile: Profile,
        #[arg(long, value_enum)]
        device: RequestedDevice,
        #[arg(long)]
        fixture: String,
        #[arg(long)]
        reps: usize,
    },
}

struct LoadedModel {
    model: LayaModel,
    device: Device,
    weights: PathBuf,
    load_ms: f64,
    device_name: String,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Inspect { profile } => inspect(profile),
        Command::Parity {
            profile,
            device,
            fixture,
            dtype,
        } => {
            let loaded = load_model(profile, device, dtype)?;
            run_parity(
                ParityContext {
                    profile,
                    requested_device: device,
                    requested_dtype: dtype,
                    device: &loaded.device,
                    device_name: loaded.device_name,
                    model: &loaded.model,
                    checkpoint: &loaded.weights,
                    load_ms: loaded.load_ms,
                },
                &fixture,
            )?;
            Ok(())
        }
        Command::Timing {
            profile,
            device,
            fixture,
            reps,
        } => timing(profile, device, &fixture, reps),
    }
}

fn inspect(profile: Profile) -> Result<()> {
    let directory = profile_dir(profile)?;
    let encoder = EncoderConfig::load(&directory.join("encoder/config.json"))?;
    let agent = AgentConfig::load(&directory.join("rl_agent_config.json"))?;
    let summary = inspect_checkpoint(&directory.join("model.safetensors"), &encoder, &agent)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    validate_checkpoint(&summary)
}

fn load_model(
    profile: Profile,
    requested_device: RequestedDevice,
    dtype: RequestedDType,
) -> Result<LoadedModel> {
    let directory = profile_dir(profile)?;
    let weights = directory.join("model.safetensors");
    let encoder = EncoderConfig::load(&directory.join("encoder/config.json"))?;
    let agent = AgentConfig::load(&directory.join("rl_agent_config.json"))?;
    let summary = inspect_checkpoint(&weights, &encoder, &agent)?;
    validate_checkpoint(&summary)?;

    let device = match requested_device {
        RequestedDevice::Cpu => Device::Cpu,
        RequestedDevice::Metal => Device::new_metal(0)
            .context("construct native Candle Metal device 0 (no CPU fallback is permitted)")?,
    };
    let device_name = device_name(&device);
    eprintln!(
        "loading profile={} device={} ({}) dtype={} checkpoint={}",
        profile.as_str(),
        requested_device.as_str(),
        device_name,
        dtype.as_str(),
        weights.display()
    );
    let start = Instant::now();
    // SAFETY: this spike only maps the pinned, hash-recorded local safetensors checkpoint.
    // SafeTensors validates offsets and shapes before Candle exposes the mapped tensors.
    let builder = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights.as_path()], dtype.candle(), &device)
    }
    .with_context(|| format!("mmap checkpoint {}", weights.display()))?;
    let model = LayaModel::load(builder, &encoder, &agent)
        .with_context(|| format!("load complete Laya model from {}", weights.display()))?;
    device.synchronize()?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    eprintln!(
        "model-ready load_ms={load_ms:.3} peak_rss_bytes={} device_location={:?}",
        peak_rss_bytes(),
        device.location()
    );
    Ok(LoadedModel {
        model,
        device,
        weights,
        load_ms,
        device_name,
    })
}

fn timing(
    profile: Profile,
    requested_device: RequestedDevice,
    fixture: &str,
    reps: usize,
) -> Result<()> {
    if reps == 0 {
        bail!("--reps must be greater than zero")
    }
    let loaded = load_model(profile, requested_device, RequestedDType::F32)?;
    let golden = repo_root()?
        .join("benchmarks/goldens")
        .join(profile.as_str())
        .join("cpu")
        .join(format!("{fixture}.json"));
    let batch = prepare_fixture(&golden, &loaded.device)?;

    loaded.device.synchronize()?;
    let first_start = Instant::now();
    let first_output = batch.forward(&loaded.model).with_context(|| {
        format!(
            "first complete-network forward on {} failed (no CPU fallback): {}",
            requested_device.as_str(),
            golden.display()
        )
    })?;
    loaded.device.synchronize()?;
    let first_ms = first_start.elapsed().as_secs_f64() * 1000.;
    let first_resident = [
        &first_output.logits,
        &first_output.action_logits,
        &first_output.action_probs,
        &first_output.hidden_cls,
    ]
    .iter()
    .all(|tensor| tensor.device().same_device(&loaded.device));

    let mut warm_ms = Vec::with_capacity(reps);
    for _ in 0..reps {
        loaded.device.synchronize()?;
        let start = Instant::now();
        let output = batch.forward(&loaded.model).with_context(|| {
            format!(
                "warm complete-network forward on {} failed (no CPU fallback)",
                requested_device.as_str()
            )
        })?;
        loaded.device.synchronize()?;
        warm_ms.push(start.elapsed().as_secs_f64() * 1000.);
        if ![
            &output.logits,
            &output.action_logits,
            &output.action_probs,
            &output.hidden_cls,
        ]
        .iter()
        .all(|tensor| tensor.device().same_device(&loaded.device))
        {
            bail!("timing output escaped requested device")
        }
    }
    let mean = warm_ms.iter().sum::<f64>() / warm_ms.len() as f64;
    let minimum = warm_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = warm_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!("SPIKE TIMING — NOT AN ACCEPTANCE BENCHMARK");
    println!(
        "profile={} device={} name={} dtype=f32 fixture={} load_ms={:.3} first_ms={first_ms:.3} warm_mean_ms={mean:.3} warm_min_ms={minimum:.3} warm_max_ms={maximum:.3} reps={} outputs_resident={} peak_rss_bytes={} metal_allocation=unavailable_in_candle_0.11.0",
        profile.as_str(),
        requested_device.as_str(),
        loaded.device_name,
        fixture,
        loaded.load_ms,
        reps,
        first_resident,
        peak_rss_bytes()
    );
    Ok(())
}

fn device_name(device: &Device) -> String {
    if device.is_metal() {
        if let Ok(output) = std::process::Command::new("system_profiler")
            .arg("SPDisplaysDataType")
            .output()
        {
            let text = String::from_utf8_lossy(&output.stdout);
            if let Some(name) = text.lines().find_map(|line| {
                line.trim()
                    .strip_prefix("Chipset Model:")
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
            }) {
                return name.to_owned();
            }
        }
        return format!("{:?}", device.location());
    }
    command_line("sysctl", &["-n", "machdep.cpu.brand_string"]).unwrap_or_else(|| "CPU".to_owned())
}

fn command_line(command: &str, arguments: &[&str]) -> Option<String> {
    let output = std::process::Command::new(command)
        .args(arguments)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    (!value.is_empty()).then_some(value)
}
