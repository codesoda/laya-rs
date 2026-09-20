"""Emit a machine-readable description of the Python baseline environment."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

import torch

LAYA_GIT_SHA = "6a5819129eb220570792e417e49723d697efd76f"


def _run(*command: str) -> str | None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout or result.stderr).strip()
    return text or None


def _sysctl(name: str) -> str | None:
    value = _run("sysctl", "-n", name)
    return value.splitlines()[0] if value else None


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _laya_git_sha() -> str | None:
    try:
        direct_url = metadata.distribution("laya").read_text("direct_url.json")
        if direct_url:
            return json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    except (metadata.PackageNotFoundError, json.JSONDecodeError, AttributeError):
        pass
    return None


def _mps_device_name() -> str | None:
    raw = _run("system_profiler", "SPDisplaysDataType", "-json")
    if not raw:
        return None
    try:
        displays = json.loads(raw).get("SPDisplaysDataType", [])
        if not displays:
            return None
        return displays[0].get("sppci_model") or displays[0].get("_name")
    except (json.JSONDecodeError, AttributeError, IndexError):
        return None


def _power_mode() -> dict[str, Any]:
    raw = _run("pmset", "-g")
    matching = [line.strip() for line in raw.splitlines() if "lowpowermode" in line.lower()] if raw else []
    value: int | None = None
    if matching:
        try:
            value = int(matching[-1].split()[-1])
        except (ValueError, IndexError):
            pass
    return {"lowpowermode": value, "matching_lines": matching, "raw_available": raw is not None}


def environment_summary() -> dict[str, Any]:
    import huggingface_hub
    import laya
    import numpy
    import safetensors
    import tokenizers
    import transformers

    ram_bytes = _sysctl("hw.memsize")
    physical = _sysctl("hw.physicalcpu")
    logical = _sysctl("hw.logicalcpu")
    mps = getattr(torch.backends, "mps", None)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "host": {
            "os": platform.system(),
            "macos_version": platform.mac_ver()[0],
            "machine": platform.machine(),
            "chip": _sysctl("machdep.cpu.brand_string") or platform.processor(),
            "ram_bytes": int(ram_bytes) if ram_bytes and ram_bytes.isdigit() else None,
            "cores": {
                "physical": int(physical) if physical and physical.isdigit() else None,
                "logical": int(logical) if logical and logical.isdigit() else os.cpu_count(),
                "performance": _as_int(_sysctl("hw.perflevel0.physicalcpu")),
                "efficiency": _as_int(_sysctl("hw.perflevel1.physicalcpu")),
            },
            "power_mode": _power_mode(),
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "safetensors": safetensors.__version__,
            "numpy": numpy.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "laya": getattr(laya, "__version__", _package_version("laya")),
        },
        "laya": {
            "version": getattr(laya, "__version__", None),
            "git_sha": _laya_git_sha(),
            "expected_git_sha": LAYA_GIT_SHA,
        },
        "torch": {
            "thread_count": torch.get_num_threads(),
            "interop_thread_count": torch.get_num_interop_threads(),
            "mps": {
                "is_available": bool(mps and mps.is_available()),
                "is_built": bool(mps and mps.is_built()),
                "device_name": _mps_device_name() if mps and mps.is_built() else None,
            },
        },
        "executable": sys.executable,
    }


def _as_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def main() -> None:
    json.dump(environment_summary(), sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
