"""Fetch and verify the exact model files used by the Python baseline."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from .paths import (
    HUB_REPO_ID,
    HUB_REVISION,
    PROFILES,
    REPO_ROOT,
    cache_root,
    manifest_path,
    profile_dir,
    revision_root,
)

REQUIRED_FILES = (
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "model.safetensors",
)
SOURCE_PREFIX = {
    "english": "",
    "multilingual": "multilingual/",
    "typed-decisions": "typed-decisions/",
}
EXPECTED_MODEL_BYTES = {
    "english": 842_609_210,
    "multilingual": 643_835_514,
    "typed-decisions": 842_609_220,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, relative: str, source_path: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if source_path is not None:
        record["source_path"] = source_path
    return record


def _base_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": _now(),
        "sources": {
            "laya_code": {
                "url": "https://github.com/NandhaKishorM/laya",
                "revision": "6a5819129eb220570792e417e49723d697efd76f",
                "license": "Apache-2.0",
                "package": {"name": "laya", "version": "0.3.3"},
                "files": {},
            },
            "laya_research": {
                "url": "https://github.com/NandhaKishorM/laya",
                "revision": "28d43add7e47ce502489c9433310d55276c64e0f",
                "license": "Apache-2.0",
                "runtime_equal_to_laya_code_revision": True,
                "files": {},
            },
            "bundled_hub": {
                "url": "https://huggingface.co/convaiinnovations/laya",
                "revision": HUB_REVISION,
                "license": "Apache-2.0",
                "files": {},
            },
            "standalone_multilingual": {
                "url": "https://huggingface.co/convaiinnovations/laya-multilingual",
                "revision": "052592a15d198d9ad47da779604259b10b47b7aa",
                "license": "Apache-2.0",
                "files": {},
            },
            "standalone_typed_decisions": {
                "url": "https://huggingface.co/convaiinnovations/laya-typed-decisions",
                "revision": "f9ab0b228f0fc0f14d873dbc99038f135c2da1b2",
                "license": "Apache-2.0",
                "files": {},
            },
            "typesafe_sdk_js": {
                "url": "https://github.com/typesafe-ai/typesafe-sdk-js",
                "revision": "66880ccded6cb642dc1809620c2b108c33730214",
                "license": "MIT",
                "package": {"name": "@typesafe-ai/sdk", "version": "0.6.0"},
                "files": {},
            },
            "vercel_ai": {
                "url": "https://github.com/vercel/ai",
                "revision": "20dd00abba618d5a516e0fee40ccd3e18a2bd1fb",
                "license": "Apache-2.0",
                "package": {"name": "@ai-sdk/typesafe-ai", "version": "3.0.4"},
                "files": {},
            },
        },
        "python_environment": {
            "requires_python": "==3.12.*",
            "pins": {
                "torch": "2.14.0",
                "transformers": "5.17.0",
                "tokenizers": "0.23.2",
                "safetensors": "0.8.0",
                "numpy": "2.5.3",
                "huggingface_hub": "1.32.0",
                "laya": "git+https://github.com/NandhaKishorM/laya@6a5819129eb220570792e417e49723d697efd76f",
            },
            "reason": "Transformers matches the published T4 harness; other packages are exact current pins. Torch 2.14.0 is the newest release with a CPython 3.12 macOS arm64 wheel resolved by uv on this host.",
        },
        "profiles": {},
        "bundle_parity": {},
    }


def load_manifest() -> dict[str, Any]:
    path = manifest_path()
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    return _base_manifest()


def save_manifest(manifest: dict[str, Any]) -> None:
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["generated_at"] = _now()
    payload = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def _install(source: Path, target: Path, *, allow_link: bool = True) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    temporary.unlink(missing_ok=True)
    if allow_link:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
    else:
        shutil.copy2(source, temporary)
    temporary.replace(target)


def _is_expected(path: Path, record: dict[str, Any] | None) -> bool:
    if not path.is_file() or not record:
        return False
    return path.stat().st_size == record.get("bytes") and sha256_file(path) == record.get("sha256")


def _source_filename(profile: str, relative: str) -> str:
    return SOURCE_PREFIX[profile] + relative


def _fetch_one(profile: str, relative: str, manifest: dict[str, Any], offline: bool) -> None:
    source_path = _source_filename(profile, relative)
    hub_files = manifest["sources"]["bundled_hub"]["files"]
    expected = hub_files.get(source_path)
    destination = profile_dir(profile) / relative
    pristine = destination.with_name("tokenizer_config.upstream.json") if relative == "tokenizer/tokenizer_config.json" else destination

    if _is_expected(pristine, expected):
        print(f"verified, skipping {profile}/{relative}", file=sys.stderr)
        downloaded = pristine
    else:
        print(f"fetching {source_path}", file=sys.stderr)
        downloaded = Path(
            hf_hub_download(
                repo_id=HUB_REPO_ID,
                filename=source_path,
                revision=HUB_REVISION,
                cache_dir=cache_root() / "hf-staging",
                local_files_only=offline,
            )
        )
        _install(downloaded, pristine)

    upstream_record = file_record(pristine, relative=source_path)
    hub_files[source_path] = upstream_record

    if relative == "tokenizer/tokenizer_config.json":
        patch = manifest.get("profiles", {}).get(profile, {}).get("tokenizer_config_patch", {})
        active_matches_patch = _is_expected(destination, patch.get("patched"))
        active_matches_upstream = _is_expected(destination, upstream_record)
        shares_pristine_inode = destination.exists() and os.path.samefile(pristine, destination)
        if not active_matches_patch and not active_matches_upstream or shares_pristine_inode:
            # Agent may rewrite the active config in place. It must never share an inode
            # with the pristine upstream copy (or the Hub staging cache).
            _install(pristine, destination, allow_link=False)
    elif destination != pristine and not _is_expected(destination, upstream_record):
        _install(pristine, destination)


def _fetch_readme(manifest: dict[str, Any], offline: bool) -> None:
    destination = revision_root() / "README.md"
    expected = manifest["sources"]["bundled_hub"]["files"].get("README.md")
    if not _is_expected(destination, expected):
        print("fetching README.md", file=sys.stderr)
        downloaded = Path(
            hf_hub_download(
                repo_id=HUB_REPO_ID,
                filename="README.md",
                revision=HUB_REVISION,
                cache_dir=cache_root() / "hf-staging",
                local_files_only=offline,
            )
        )
        _install(downloaded, destination)
    else:
        print("verified, skipping README.md", file=sys.stderr)
    manifest["sources"]["bundled_hub"]["files"]["README.md"] = file_record(
        destination, relative="README.md"
    )


def _update_profile(profile: str, manifest: dict[str, Any]) -> None:
    directory = profile_dir(profile)
    files: dict[str, Any] = {}
    for relative in REQUIRED_FILES:
        local = directory / relative
        files[relative] = file_record(local, relative=relative, source_path=_source_filename(profile, relative))
        if relative == "tokenizer/tokenizer_config.json":
            pristine_relative = "tokenizer/tokenizer_config.upstream.json"
            files[pristine_relative] = file_record(
                directory / pristine_relative,
                relative=pristine_relative,
                source_path=_source_filename(profile, relative),
            )
    with (directory / "rl_agent_config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    previous_patch = manifest.get("profiles", {}).get(profile, {}).get("tokenizer_config_patch")
    profile_record: dict[str, Any] = {
        "source_subfolder": SOURCE_PREFIX[profile].rstrip("/") or None,
        "directory": str(directory.relative_to(REPO_ROOT)),
        "files": files,
        "rl_agent_config": config,
        "facts": {
            "max_len": config.get("max_len"),
            "head_max_len": config.get("head_max_len"),
            "temperature": config.get("temperature"),
            "temperature_by_options": config.get("temperature_by_options"),
        },
    }
    if previous_patch:
        profile_record["tokenizer_config_patch"] = previous_patch
    manifest["profiles"][profile] = profile_record


def fetch_profiles(profiles: tuple[str, ...], offline: bool) -> None:
    os.environ["HF_HUB_CACHE"] = str(cache_root() / "hf-staging")
    manifest = load_manifest()
    _fetch_readme(manifest, offline)
    for profile in profiles:
        print(f"profile {profile}", file=sys.stderr)
        for relative in REQUIRED_FILES:
            _fetch_one(profile, relative, manifest, offline)
        _update_profile(profile, manifest)
        model_bytes = manifest["profiles"][profile]["files"]["model.safetensors"]["bytes"]
        if model_bytes != EXPECTED_MODEL_BYTES[profile]:
            raise RuntimeError(
                f"{profile} model.safetensors has {model_bytes} bytes; expected {EXPECTED_MODEL_BYTES[profile]}"
            )
        save_manifest(manifest)
    print(f"manifest: {manifest_path()}", file=sys.stderr)


def record_tokenizer_patch(profile: str) -> dict[str, Any]:
    """Record the tokenizer config mutation made by laya.Agent after its first load."""
    manifest = load_manifest()
    directory = profile_dir(profile)
    upstream_path = directory / "tokenizer/tokenizer_config.upstream.json"
    patched_path = directory / "tokenizer/tokenizer_config.json"
    upstream_text = upstream_path.read_text(encoding="utf-8")
    patched_text = patched_path.read_text(encoding="utf-8")
    upstream_json = json.loads(upstream_text)
    patched_json = json.loads(patched_text)
    removed = sorted(set(upstream_json) - set(patched_json))
    added = sorted(set(patched_json) - set(upstream_json))
    changed = sorted(k for k in set(upstream_json) & set(patched_json) if upstream_json[k] != patched_json[k])
    unified = "".join(
        difflib.unified_diff(
            upstream_text.splitlines(keepends=True),
            patched_text.splitlines(keepends=True),
            fromfile="tokenizer_config.upstream.json",
            tofile="tokenizer_config.json",
        )
    )
    patch = {
        "observed_after_agent_load": True,
        "observed_at": _now(),
        "upstream": file_record(upstream_path, relative="tokenizer/tokenizer_config.upstream.json"),
        "patched": file_record(patched_path, relative="tokenizer/tokenizer_config.json"),
        "changed": upstream_text != patched_text,
        "summary": {"added_keys": added, "removed_keys": removed, "changed_keys": changed},
        "unified_diff": unified,
    }
    manifest["profiles"][profile]["tokenizer_config_patch"] = patch
    manifest["profiles"][profile]["files"]["tokenizer/tokenizer_config.json"] = patch["patched"] | {
        "source_path": _source_filename(profile, "tokenizer/tokenizer_config.json")
    }
    save_manifest(manifest)
    return patch


def verify(profiles: tuple[str, ...]) -> bool:
    manifest = load_manifest()
    failures: list[str] = []
    for profile in profiles:
        profile_record = manifest.get("profiles", {}).get(profile)
        if not profile_record:
            failures.append(f"profile missing from manifest: {profile}")
            continue
        directory = profile_dir(profile)
        for relative, expected in profile_record["files"].items():
            path = directory / relative
            if not path.is_file():
                failures.append(f"missing: {path}")
                continue
            actual_size = path.stat().st_size
            actual_hash = sha256_file(path)
            if actual_size != expected["bytes"] or actual_hash != expected["sha256"]:
                failures.append(
                    f"mismatch: {path} expected {expected['bytes']} {expected['sha256']}, "
                    f"got {actual_size} {actual_hash}"
                )
    readme = revision_root() / "README.md"
    readme_record = manifest.get("sources", {}).get("bundled_hub", {}).get("files", {}).get("README.md")
    if readme_record:
        if not _is_expected(readme, readme_record):
            failures.append(f"mismatch or missing: {readme}")
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return False
    print(f"verified {', '.join(profiles)}", file=sys.stderr)
    return True


def _profiles_from_argument(value: str) -> tuple[str, ...]:
    return PROFILES if value == "all" else (value,)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=(*PROFILES, "all"), default="all")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verify", action="store_true", help="rehash local files without downloading")
    args = parser.parse_args()
    profiles = _profiles_from_argument(args.profile)
    try:
        if args.verify:
            raise SystemExit(0 if verify(profiles) else 1)
        fetch_profiles(profiles, args.offline)
    except Exception as error:
        print(f"laya-fetch: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
