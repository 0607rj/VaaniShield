"""Environment introspection used by scripts/environment_check.py.

Every collector must succeed on a machine without torch, CUDA, ffmpeg or psutil: missing pieces are
reported as ``None``/``"not installed"``, never raised. Pure parsing helpers are unit-tested.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Any

# Packages whose versions are recorded in every env report.
AUDIO_PACKAGES = ("soundfile", "librosa", "soxr", "scipy", "numpy")
ML_PACKAGES = (
    "torch",
    "transformers",
    "huggingface-hub",
    "safetensors",
    "scikit-learn",
    "pandas",
    "pydantic",
)
# Deliberately unused but reported, so an accidental install is visible (DEC-14).
WATCH_PACKAGES = ("torchaudio",)

# ffmpeg encoders needed by the codec augmentation / simulated channel conditions (TD-13).
REQUIRED_FFMPEG_ENCODERS = ("libopus", "libmp3lame")
OPTIONAL_FFMPEG_ENCODERS = ("libopencore_amrnb", "libvo_amrwbenc", "libspeex")

_GIB = 1024**3


def bytes_to_gib(n: int | float | None) -> float | None:
    """Bytes -> GiB rounded to 2 decimals; ``None`` passes through."""
    if n is None:
        return None
    return round(float(n) / _GIB, 2)


def package_versions(names: tuple[str, ...] | list[str]) -> dict[str, str | None]:
    """Installed distribution versions; ``None`` when a package is not installed."""
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def python_info() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "in_virtualenv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
    }


def os_info() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }


def _cpu_name() -> str:
    system = platform.system()
    if system == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(name).strip()
        except OSError:
            pass
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or "unknown"


def cpu_info() -> dict[str, Any]:
    logical = os.cpu_count()
    physical = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    return {"name": _cpu_name(), "physical_cores": physical, "logical_cores": logical}


def ram_info() -> dict[str, Any]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {"total_gib": bytes_to_gib(vm.total), "available_gib": bytes_to_gib(vm.available)}
    except ImportError:
        return {"total_gib": None, "available_gib": None, "note": "psutil not installed"}


def torch_info() -> dict[str, Any]:
    """PyTorch / CUDA / GPU details. Imports torch lazily; never raises."""
    try:
        import torch
    except ImportError:
        return {"installed": False}
    info: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version_built": torch.version.cuda,  # None on CPU-only builds
        "cudnn_version": None,
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "num_threads": torch.get_num_threads(),
        "gpus": [],
    }
    if info["cuda_available"]:
        info["cudnn_version"] = torch.backends.cudnn.version()
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append(
                {
                    "index": i,
                    "name": props.name,
                    "vram_gib": bytes_to_gib(props.total_memory),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    return info


def parse_ffmpeg_version(first_line: str) -> str | None:
    """'ffmpeg version 7.1-full_build-www.gyan.dev Copyright ...' -> '7.1-full_build-www.gyan.dev'."""
    m = re.match(r"^\s*ffmpeg version (\S+)", first_line)
    return m.group(1) if m else None


def parse_ffmpeg_encoders(text: str) -> set[str]:
    """Parse ``ffmpeg -hide_banner -encoders`` output into the set of encoder names.

    Encoder lines look like ' A....D libopus              libopus Opus' (6-char flag field, then name).
    """
    names: set[str] = set()
    started = False
    for line in text.splitlines():
        if line.strip().startswith("------"):
            started = True
            continue
        if not started:
            continue
        m = re.match(r"^\s*[VASFXBD.]{6}\s+(\S+)", line)
        if m:
            names.add(m.group(1))
    return names


def check_encoders(available: set[str]) -> dict[str, Any]:
    missing_required = [e for e in REQUIRED_FFMPEG_ENCODERS if e not in available]
    return {
        "required": {e: e in available for e in REQUIRED_FFMPEG_ENCODERS},
        "optional": {e: e in available for e in OPTIONAL_FFMPEG_ENCODERS},
        "ok": not missing_required,
        "missing_required": missing_required,
    }


def ffmpeg_info() -> dict[str, Any]:
    exe = shutil.which("ffmpeg")
    if exe is None:
        return {"found": False, "path": None, "version": None, "encoders": check_encoders(set())}
    try:
        ver = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=15)
        enc = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "found": True,
            "path": exe,
            "version": None,
            "error": str(exc),
            "encoders": check_encoders(set()),
        }
    first = ver.stdout.splitlines()[0] if ver.stdout else ""
    return {
        "found": True,
        "path": exe,
        "version": parse_ffmpeg_version(first),
        "encoders": check_encoders(parse_ffmpeg_encoders(enc.stdout)),
    }


def system_gpus() -> list[dict[str, Any]]:
    """All display adapters (incl. non-CUDA), to explain *why* CUDA is unavailable. Best effort."""
    system = platform.system()
    try:
        if system == "Windows":
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.AdapterRAM }",
            ]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
            gpus = []
            for line in out.splitlines():
                if "|" in line:
                    name, ram = line.rsplit("|", 1)
                    # AdapterRAM is a 32-bit field and saturates at 4 GiB; report as a lower bound.
                    gpus.append(
                        {
                            "name": name.strip(),
                            "adapter_ram_gib_lower_bound": bytes_to_gib(int(ram))
                            if ram.strip().isdigit()
                            else None,
                        }
                    )
            return gpus
        if shutil.which("nvidia-smi"):
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout
            return [
                {"name": n.strip(), "vram_gib": round(float(m) / 1024, 2)}
                for n, m in (line.rsplit(",", 1) for line in out.splitlines() if "," in line)
            ]
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return []


def collect_environment() -> dict[str, Any]:
    """Full environment report (JSON-serialisable)."""
    return {
        "python": python_info(),
        "os": os_info(),
        "cpu": cpu_info(),
        "ram": ram_info(),
        "torch": torch_info(),
        "system_gpus": system_gpus(),
        "ffmpeg": ffmpeg_info(),
        "packages": {
            "audio": package_versions(AUDIO_PACKAGES),
            "ml": package_versions(ML_PACKAGES),
            "watch_should_be_absent": package_versions(WATCH_PACKAGES),
        },
    }


def summarize_issues(env: dict[str, Any], expected_python: str = "3.12") -> list[str]:
    """Human-readable problems that block or degrade the planned workflow. Pure function (tested)."""
    issues: list[str] = []
    if not env["python"]["version"].startswith(expected_python + "."):
        issues.append(f"Python {env['python']['version']} != expected {expected_python}.x (DEC-6)")
    if not env["python"].get("in_virtualenv"):
        issues.append("Not running inside a virtual environment")
    t = env["torch"]
    if not t.get("installed"):
        issues.append("PyTorch not installed")
    elif not t.get("cuda_available"):
        issues.append(
            "CUDA not available: CPU-only here; train on Colab/Kaggle (T4) per docs/reproducibility.md"
        )
    ff = env["ffmpeg"]
    if not ff["found"]:
        issues.append(
            "ffmpeg not found on PATH: needed for decoding non-WAV input and codec augmentation (P1.3)"
        )
    elif not ff["encoders"]["ok"]:
        issues.append(f"ffmpeg missing required encoders: {ff['encoders']['missing_required']}")
    elif not ff["encoders"]["optional"].get("libopencore_amrnb"):
        issues.append("ffmpeg lacks libopencore_amrnb: AMR-NB will fall back to labelled 'narrowband_sim'")
    for pkg, ver in env["packages"]["audio"].items():
        if ver is None:
            issues.append(f"Audio package missing: {pkg}")
    if env["packages"]["watch_should_be_absent"].get("torchaudio"):
        issues.append(
            "torchaudio is installed but unused by design (DEC-14); check it does not shadow I/O choices"
        )
    return issues
