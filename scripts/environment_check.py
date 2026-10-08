#!/usr/bin/env python
"""Report the compute environment for VaaniShield and save it as JSON.

Usage:
    python scripts/environment_check.py                 # print report, save environment/reports/env_<UTC>.json
    python scripts/environment_check.py --no-save       # print only
    python scripts/environment_check.py --strict        # exit 1 if any issue is found (for CI / Colab gating)

Works with or without torch/ffmpeg/psutil installed and without installing the package: the collector
module is loaded directly from scripts/env_info.py. No network access, no downloads.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_env_info():
    path = Path(__file__).resolve().parent / "env_info.py"
    spec = importlib.util.spec_from_file_location("vaanishield_env_info", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _fmt(v) -> str:
    return "-" if v is None else str(v)


def render(env: dict, issues: list[str]) -> str:
    t, ff = env["torch"], env["ffmpeg"]
    lines = [
        "VaaniShield environment check",
        "=" * 60,
        f"Python        : {env['python']['version']} ({env['python']['implementation']})  venv={env['python']['in_virtualenv']}",
        f"Executable    : {env['python']['executable']}",
        f"OS            : {env['os']['system']} {env['os']['release']} ({env['os']['version']}) {env['os']['machine']}",
        f"CPU           : {env['cpu']['name']}  cores={_fmt(env['cpu']['physical_cores'])} physical / {_fmt(env['cpu']['logical_cores'])} logical",
        f"RAM           : {_fmt(env['ram'].get('total_gib'))} GiB total, {_fmt(env['ram'].get('available_gib'))} GiB available",
    ]
    if t.get("installed"):
        lines += [
            f"PyTorch       : {t['version']}  threads={t['num_threads']}",
            f"CUDA available: {t['cuda_available']}",
            f"CUDA (build)  : {_fmt(t['cuda_version_built'])}   cuDNN: {_fmt(t['cudnn_version'])}   MPS: {t['mps_available']}",
        ]
        if t["gpus"]:
            for g in t["gpus"]:
                lines.append(
                    f"CUDA GPU {g['index']}    : {g['name']}  VRAM={g['vram_gib']} GiB  cc={g['compute_capability']}"
                )
        else:
            lines.append("CUDA GPU      : none")
    else:
        lines.append("PyTorch       : not installed")
    for g in env["system_gpus"]:
        ram = g.get("vram_gib", g.get("adapter_ram_gib_lower_bound"))
        lines.append(f"System GPU    : {g['name']}  (reported memory {_fmt(ram)} GiB)")
    enc = ff["encoders"]
    lines += [
        f"ffmpeg        : {'found ' + _fmt(ff['version']) + ' at ' + ff['path'] if ff['found'] else 'NOT FOUND'}",
        f"  encoders    : required {enc['required']}  optional {enc['optional']}",
        "Audio packages: " + ", ".join(f"{k}={_fmt(v)}" for k, v in env["packages"]["audio"].items()),
        "ML packages   : " + ", ".join(f"{k}={_fmt(v)}" for k, v in env["packages"]["ml"].items()),
        "Unused-by-design (should be absent): "
        + ", ".join(f"{k}={_fmt(v)}" for k, v in env["packages"]["watch_should_be_absent"].items()),
        "-" * 60,
    ]
    lines += ["Issues:"] + [f"  - {i}" for i in issues] if issues else ["Issues: none"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="VaaniShield environment check")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--out-dir", default=str(REPO / "environment" / "reports"))
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)

    mod = _load_env_info()
    env = mod.collect_environment()
    issues = mod.summarize_issues(env)
    now = datetime.now(UTC)
    report = {"generated_at": now.isoformat(), "environment": env, "issues": issues}
    print(render(env, issues))
    if not args.no_save:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"env_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nSaved: {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")
    return 1 if (args.strict and issues) else 0


if __name__ == "__main__":
    sys.exit(main())
