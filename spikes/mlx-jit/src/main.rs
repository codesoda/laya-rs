mod model;
mod parity;

use std::{
    env,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use mlx_rs::{metal, ops, transforms, Array, Device, DeviceType};
use model::Precision;

const ASSET_RELATIVE: &str =
    ".cache/laya/hub/convaiinnovations--laya/c5d78730f3493e4fe16d61507ef4b78eef7318cf";
const METALLIB_SHA256: &str = "44eb25db5205fbfc2f5c81f59cce9bb8c3c534d0b6707c03e43b057a497e618b";
const EMBEDDED_METALLIB: &[u8] =
    include_bytes!("../../../benchmarks/results/l1-spike-mlx/jit/native/mlx.metallib");

#[derive(Debug, Parser)]
#[command(about = "Laya full-network mlx-rs parity spike")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print and enforce the native Metal GPU device.
    Device,
    /// Compare the complete network with frozen Python CPU goldens.
    Parity {
        #[arg(long)]
        profile: String,
        #[arg(long, default_value = "all")]
        fixture: String,
        #[arg(long, value_enum, default_value = "f32")]
        dtype: Precision,
        /// Directory receiving <profile>-<dtype>.json.
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Synchronized model-only spike timing (not acceptance evidence).
    Timing {
        #[arg(long)]
        profile: String,
        #[arg(long)]
        fixture: String,
        #[arg(long, default_value_t = 20)]
        reps: usize,
        #[arg(long, value_enum, default_value = "f32")]
        dtype: Precision,
    },
}

fn configure_metallib() -> Result<()> {
    let executable = env::current_exe().context("resolve current executable")?;
    let sibling = executable
        .parent()
        .context("executable has no parent directory")?
        .join("mlx.metallib");
    let metallib = if sibling.is_file() {
        sibling
    } else {
        extract_embedded_metallib()?
    };
    metal::set_metallib_path(metallib.to_string_lossy().as_ref()).context("select MLX metallib")?;
    Ok(())
}

fn extract_embedded_metallib() -> Result<PathBuf> {
    let cache_root = if let Some(root) = env::var_os("XDG_CACHE_HOME") {
        PathBuf::from(root)
    } else {
        PathBuf::from(env::var_os("HOME").context("HOME is not set")?).join("Library/Caches")
    };
    let directory = cache_root.join("laya-rs").join(METALLIB_SHA256);
    let output = directory.join("mlx.metallib");
    if fs::read(&output).is_ok_and(|bytes| bytes == EMBEDDED_METALLIB) {
        return Ok(output);
    }

    fs::create_dir_all(&directory)
        .with_context(|| format!("create metallib cache {}", directory.display()))?;
    let temporary = directory.join(format!("mlx.metallib.{}.tmp", std::process::id()));
    let mut file =
        File::create(&temporary).with_context(|| format!("create {}", temporary.display()))?;
    file.write_all(EMBEDDED_METALLIB)
        .context("write embedded MLX metallib")?;
    file.sync_all().context("sync embedded MLX metallib")?;
    fs::rename(&temporary, &output).with_context(|| {
        format!(
            "install embedded metallib {} -> {}",
            temporary.display(),
            output.display()
        )
    })?;
    Ok(output)
}

fn device() -> Result<()> {
    configure_metallib()?;
    let device = Device::try_default()?;
    let kind = device.get_type()?;
    println!("MLX default device: {device}");
    if !matches!(kind, DeviceType::Gpu) {
        bail!("MLX default device is not GPU; refusing silent CPU fallback");
    }
    // Device discovery alone does not load the Metal kernel library. Evaluate a
    // real operation so `device` also catches incomplete release archives.
    let input = Array::from_slice(&[1.0_f32, 2.0], &[2]);
    let output = ops::add(&input, &input).context("enqueue Metal smoke operation")?;
    transforms::eval([&output]).context("evaluate Metal smoke operation")?;
    if output.to_vec_cast::<f32>()? != [2.0, 4.0] {
        bail!("Metal smoke operation returned an unexpected result");
    }
    println!("MLX metallib: {}", metal::metallib_path()?);
    println!("backend: Metal GPU (kernel smoke PASS)");
    Ok(())
}

fn repo_root() -> Result<PathBuf> {
    if let Some(root) = env::var_os("LAYA_RS_ROOT") {
        let root = PathBuf::from(root);
        if root.join("benchmarks/goldens").is_dir() {
            return Ok(root);
        }
        bail!(
            "LAYA_RS_ROOT does not contain benchmarks/goldens: {}",
            root.display()
        );
    }
    let current = env::current_dir()?;
    for candidate in current.ancestors() {
        if candidate.join("benchmarks/goldens").is_dir() {
            return Ok(candidate.to_path_buf());
        }
    }
    let canonical = PathBuf::from("/Users/chrisraethke/projects/laya-rs");
    if canonical.join("benchmarks/goldens").is_dir() {
        return Ok(canonical);
    }
    bail!("cannot locate laya-rs root; set LAYA_RS_ROOT")
}

fn model_dir(root: &Path, profile: &str) -> Result<PathBuf> {
    if !matches!(profile, "english" | "multilingual" | "typed-decisions") {
        bail!("unknown profile {profile:?}");
    }
    let local = root.join(ASSET_RELATIVE).join(profile);
    if local.join("model.safetensors").is_file() {
        return Ok(local);
    }
    let canonical = PathBuf::from("/Users/chrisraethke/projects/laya-rs")
        .join(ASSET_RELATIVE)
        .join(profile);
    if canonical.join("model.safetensors").is_file() {
        return Ok(canonical);
    }
    bail!(
        "missing pinned upstream checkpoint for {profile}; checked {} and {}",
        local.display(),
        canonical.display()
    )
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Device => device(),
        Command::Parity {
            profile,
            fixture,
            dtype,
            report,
        } => {
            device()?;
            let root = repo_root()?;
            let model_dir = model_dir(&root, &profile)?;
            let (report, _) = parity::run(
                &root,
                &model_dir,
                &profile,
                &fixture,
                dtype,
                report.as_deref(),
            )?;
            if !report.pass {
                bail!("parity failed for {profile} {}", dtype.label());
            }
            Ok(())
        }
        Command::Timing {
            profile,
            fixture,
            reps,
            dtype,
        } => {
            device()?;
            if reps == 0 {
                bail!("--reps must be positive");
            }
            let root = repo_root().context("resolve repository root")?;
            let model_dir = model_dir(&root, &profile)?;
            parity::timing(&root, &model_dir, &profile, &fixture, dtype, reps)
        }
    }
}
