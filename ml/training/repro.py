"""Reproducibility primitives (docs/architecture.md M02, docs/reproducibility.md §3)."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

_CHUNK = 1 << 20


def seed_everything(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Seed python/numpy/torch. Returns what was configured (stored in the run dir)."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative int")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    applied: dict[str, Any] = {"seed": seed, "torch": False, "deterministic": deterministic}
    try:
        import torch
    except ImportError:
        return applied
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required by some CUDA matmul kernels for determinism; harmless on CPU.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    applied["torch"] = True
    return applied


def item_seed(base_seed: int, epoch: int, clip_id: str) -> int:
    """Stable per-item seed (independent of PYTHONHASHSEED, worker count and process)."""
    digest = hashlib.sha256(f"{base_seed}|{epoch}|{clip_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def item_rng(base_seed: int, epoch: int, clip_id: str) -> np.random.Generator:
    return np.random.default_rng(item_seed(base_seed, epoch, clip_id))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_manifest_bytes(path: str | Path, key: str = "clip_id") -> bytes:
    """CSV canonicalised: columns sorted by name, rows sorted by `key`, LF line endings, UTF-8."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or key not in reader.fieldnames:
            raise ValueError(f"{path}: manifest must have a {key!r} column")
        cols = sorted(reader.fieldnames)
        rows = sorted(reader, key=lambda r: r[key])
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(cols)
    for r in rows:
        writer.writerow([r[c] for c in cols])
    return buf.getvalue().encode("utf-8")


def sha256_manifest(path: str | Path, key: str = "clip_id") -> str:
    """Hash that is invariant to row and column order (docs/architecture.md §4.2)."""
    return sha256_bytes(canonical_manifest_bytes(path, key))


def git_state(repo_dir: str | Path = ".") -> dict[str, Any]:
    """Commit / dirty flag / branch. Works (returns available=False) outside a git repo."""

    def _git(*args: str) -> str | None:
        try:
            r = subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    if _git("rev-parse", "--is-inside-work-tree") != "true":
        return {"available": False, "commit": None, "dirty": None, "branch": None}
    commit = _git("rev-parse", "HEAD")  # None before the first commit
    status = _git("status", "--porcelain")
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD") if commit else None,
    }


def resolve_device(name: str = "auto") -> str:
    """'auto' -> 'cuda' if a CUDA GPU is visible, else 'cpu'. Asking for 'cuda' without a GPU is an error, not a fallback."""
    import torch

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but torch.cuda.is_available() is False")
    if name not in ("cpu", "cuda"):
        raise ValueError(f"unknown device {name!r} (use auto, cpu or cuda)")
    return name


def resolve_path(value: str | None) -> Path | None:
    """Expand ~ and environment variables; relative paths stay relative to the current working directory (repo root)."""
    if value is None:
        return None
    return Path(os.path.expandvars(os.path.expanduser(value)))

