//! Model-directory resolution and verification.
//!
//! The library never guesses where weights live: a caller passes the profile
//! directory (the layout published in the pinned Hub revision) and this
//! module checks every required file against the embedded manifest before
//! anything is parsed. Runtime constants (`max_len`, `head_max_len`,
//! temperatures, special tokens) are read from those files, never hard-coded.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{
    manifest::{Profile, ProfileManifest},
    LayaError, Result,
};

/// Calibration and budget constants from `rl_agent_config.json`.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AgentConfig {
    pub head_layers: usize,
    pub max_len: usize,
    pub head_max_len: usize,
    /// Per-type temperature, indexed by [`crate::QuestionType::index`].
    pub temperature: [f64; 3],
    /// Per-`temp_bucket` override, keys like `"choice:3-5"`.
    #[serde(default)]
    pub temperature_by_options: BTreeMap<String, f64>,
    #[serde(default)]
    pub act_costs: BTreeMap<String, f64>,
    #[serde(default)]
    pub model_name: Option<String>,
    #[serde(default)]
    pub encoder: Option<String>,
}

impl AgentConfig {
    /// Action-head width: one logit per cost entry plus the null action.
    #[must_use]
    pub fn n_act(&self) -> usize {
        self.act_costs.len() + 1
    }
}

/// Special-token strings from `tokenizer_config.json`. IDs are looked up in
/// the tokenizer vocabulary at load.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct TokenizerConfig {
    pub cls_token: String,
    pub sep_token: String,
    pub mask_token: String,
    pub pad_token: String,
}

/// A verified profile directory.
#[derive(Clone, Debug)]
pub struct ModelDir {
    profile: Profile,
    root: PathBuf,
    manifest: ProfileManifest,
}

/// How thoroughly to check files before loading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verification {
    /// Size and SHA-256 of every required file (the default; ~1–2 s for the
    /// weights).
    Full,
    /// Presence and size only. For repeated loads of a directory this
    /// process already hashed.
    SizeOnly,
}

impl ModelDir {
    /// Check that `root` holds the pinned files for `profile`.
    pub fn open(profile: Profile, root: impl Into<PathBuf>, verify: Verification) -> Result<Self> {
        let root = root.into();
        let manifest = ProfileManifest::embedded(profile)?;
        for file in &manifest.files {
            let path = root.join(&file.path);
            let metadata = fs::metadata(&path)
                .map_err(|error| LayaError::asset(&path, format!("cannot stat: {error}")))?;
            if !metadata.is_file() {
                return Err(LayaError::asset(&path, "is not a regular file"));
            }
            if metadata.len() != file.bytes {
                return Err(LayaError::asset(
                    &path,
                    format!(
                        "size {} does not match pinned {} bytes for {profile}",
                        metadata.len(),
                        file.bytes
                    ),
                ));
            }
            if verify == Verification::Full {
                let actual = sha256_file(&path)?;
                if actual != file.sha256 {
                    return Err(LayaError::asset(
                        &path,
                        format!(
                            "SHA-256 {actual} does not match pinned {} for {profile}",
                            file.sha256
                        ),
                    ));
                }
            }
        }
        Ok(Self {
            profile,
            root,
            manifest,
        })
    }

    #[must_use]
    pub const fn profile(&self) -> Profile {
        self.profile
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[must_use]
    pub const fn manifest(&self) -> &ProfileManifest {
        &self.manifest
    }

    #[must_use]
    pub fn weights_path(&self) -> PathBuf {
        self.root.join("model.safetensors")
    }

    #[must_use]
    pub fn encoder_config_path(&self) -> PathBuf {
        self.root.join("encoder/config.json")
    }

    #[must_use]
    pub fn tokenizer_path(&self) -> PathBuf {
        self.root.join("tokenizer/tokenizer.json")
    }

    pub fn agent_config(&self) -> Result<AgentConfig> {
        read_json(&self.root.join("rl_agent_config.json"))
    }

    pub fn tokenizer_config(&self) -> Result<TokenizerConfig> {
        read_json(&self.root.join("tokenizer/tokenizer_config.json"))
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).map_err(|error| LayaError::asset(path, error.to_string()))?;
    serde_json::from_slice(&bytes).map_err(|error| LayaError::asset(path, error.to_string()))
}

/// Streaming SHA-256 of a file, lowercase hex.
pub fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path).map_err(|error| LayaError::asset(path, error.to_string()))?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| LayaError::asset(path, error.to_string()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

/// SHA-256 of an in-memory buffer, lowercase hex.
#[must_use]
pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_reports_missing_and_mismatched_files() {
        let dir = tempfile::tempdir().unwrap();
        let error = ModelDir::open(Profile::English, dir.path(), Verification::Full).unwrap_err();
        assert!(matches!(error, LayaError::Asset { .. }), "{error}");
        assert!(error.to_string().contains("encoder/config.json"));

        fs::create_dir_all(dir.path().join("encoder")).unwrap();
        fs::write(dir.path().join("encoder/config.json"), b"{}").unwrap();
        let error = ModelDir::open(Profile::English, dir.path(), Verification::Full).unwrap_err();
        assert!(
            error.to_string().contains("size 2 does not match"),
            "{error}"
        );
    }

    #[test]
    fn sha256_helpers_agree() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x");
        fs::write(&path, b"abc").unwrap();
        assert_eq!(
            sha256_file(&path).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(sha256_bytes(b"abc"), sha256_file(&path).unwrap());
    }

    #[test]
    fn agent_config_parses_pinned_shape() {
        let config: AgentConfig = serde_json::from_str(
            r#"{"encoder":"x","head_layers":2,"max_len":512,"head_max_len":192,"act_costs":{"escalate":0.5},"temperature":[1.0,2.0,3.0],"temperature_by_options":{"choice:2":1.5}}"#,
        )
        .unwrap();
        assert_eq!(config.n_act(), 2);
        assert_eq!(config.temperature[2], 3.0);
    }
}
