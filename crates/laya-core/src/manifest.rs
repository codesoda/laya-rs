//! Pinned upstream assets. The manifest is `manifests/sources.json`, embedded
//! at compile time so a consumer of this crate has the same pins as the
//! repository that generated the goldens.

use std::{collections::BTreeMap, fmt, str::FromStr};

use serde::Deserialize;

use crate::{LayaError, Result};

const SOURCES_JSON: &str = include_str!("../../../manifests/sources.json");

/// Hugging Face repository the bundled checkpoints come from.
pub const HUB_REPO: &str = "convaiinnovations/laya";
/// Immutable Hub revision every profile is pinned to.
pub const HUB_REVISION: &str = "c5d78730f3493e4fe16d61507ef4b78eef7318cf";

/// One of the three published Laya checkpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Profile {
    English,
    Multilingual,
    TypedDecisions,
}

impl Profile {
    pub const ALL: [Self; 3] = [Self::English, Self::Multilingual, Self::TypedDecisions];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::English => "english",
            Self::Multilingual => "multilingual",
            Self::TypedDecisions => "typed-decisions",
        }
    }
}

impl fmt::Display for Profile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for Profile {
    type Err = LayaError;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "english" => Ok(Self::English),
            "multilingual" => Ok(Self::Multilingual),
            "typed-decisions" => Ok(Self::TypedDecisions),
            other => Err(LayaError::InvalidRequest(format!(
                "unknown Laya profile {other:?}; expected english, multilingual or typed-decisions"
            ))),
        }
    }
}

/// A pinned file inside a profile directory.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct PinnedFile {
    /// Path relative to the profile directory (`model.safetensors`,
    /// `tokenizer/tokenizer.json`, ...).
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    /// Path inside the Hub repository at [`HUB_REVISION`].
    pub source_path: String,
}

impl PinnedFile {
    /// Direct download URL for this exact file at the pinned revision.
    #[must_use]
    pub fn hub_url(&self) -> String {
        format!(
            "https://huggingface.co/{HUB_REPO}/resolve/{HUB_REVISION}/{}",
            self.source_path
        )
    }
}

/// Files that must be present and hash-verified before a profile loads.
pub const REQUIRED_FILES: [&str; 5] = [
    "encoder/config.json",
    "model.safetensors",
    "rl_agent_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
];

#[derive(Debug, Deserialize)]
struct ProfileEntry {
    files: BTreeMap<String, PinnedFile>,
}

#[derive(Debug, Deserialize)]
struct Sources {
    profiles: BTreeMap<String, ProfileEntry>,
}

/// Pinned files for one profile.
#[derive(Clone, Debug)]
pub struct ProfileManifest {
    pub profile: Profile,
    pub files: Vec<PinnedFile>,
}

impl ProfileManifest {
    /// Load the embedded manifest entry for `profile`. Only the five required
    /// runtime files are kept; `tokenizer_config.upstream.json` is a baseline
    /// bookkeeping copy and is not part of the runtime contract.
    pub fn embedded(profile: Profile) -> Result<Self> {
        let sources: Sources = serde_json::from_str(SOURCES_JSON)
            .map_err(|error| LayaError::Checkpoint(format!("embedded manifest: {error}")))?;
        let entry = sources.profiles.get(profile.as_str()).ok_or_else(|| {
            LayaError::Checkpoint(format!("embedded manifest has no profile {profile}"))
        })?;
        let mut files = Vec::with_capacity(REQUIRED_FILES.len());
        for name in REQUIRED_FILES {
            let file = entry.files.get(name).ok_or_else(|| {
                LayaError::Checkpoint(format!("embedded manifest for {profile} lacks {name}"))
            })?;
            files.push(file.clone());
        }
        Ok(Self { profile, files })
    }

    #[must_use]
    pub fn file(&self, path: &str) -> Option<&PinnedFile> {
        self.files.iter().find(|file| file.path == path)
    }

    /// SHA-256 of `model.safetensors`; part of the public model identity.
    #[must_use]
    pub fn weights_sha256(&self) -> &str {
        self.file("model.safetensors")
            .map(|file| file.sha256.as_str())
            .unwrap_or_default()
    }

    /// Total bytes of the five runtime files.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|file| file.bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_profile_has_five_pinned_files_with_hashes() {
        for profile in Profile::ALL {
            let manifest = ProfileManifest::embedded(profile).unwrap();
            assert_eq!(manifest.files.len(), 5);
            for file in &manifest.files {
                assert_eq!(file.sha256.len(), 64, "{profile} {}", file.path);
                assert!(file.bytes > 0);
                assert!(file.hub_url().contains(HUB_REVISION));
            }
            assert!(manifest.weights_sha256().len() == 64);
        }
        assert_eq!(
            ProfileManifest::embedded(Profile::Multilingual)
                .unwrap()
                .weights_sha256(),
            "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204"
        );
    }

    #[test]
    fn profile_names_round_trip() {
        for profile in Profile::ALL {
            assert_eq!(profile.as_str().parse::<Profile>().unwrap(), profile);
        }
        assert!("jev".parse::<Profile>().is_err());
    }
}
