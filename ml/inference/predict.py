"""Score one audio file with a trained checkpoint.

Usage:
    python -m ml.inference.predict --checkpoint runs/B0_lfcc_cnn_pilot/best.pt --audio path/to/file.wav

The file goes through exactly the training preprocessing (mono, 16 kHz, silence trim). Long files are split into
overlapping 4 s windows; the clip score is the mean window spoof-logit. The label uses the dev-chosen threshold
stored in the checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.data.audio import preprocess, windows
from ml.models.lfcc_cnn import spoof_logit
from ml.models import build_model, load_weights

DISCLAIMER = "Model estimate of synthetic-speech artefacts, not forensic proof. It can be wrong, especially on unseen languages, generators and channels."


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Checkpoints hold CPU tensors; the model is moved to ``device`` ('cpu', 'cuda' or 'auto') after loading."""
    from ml.training.repro import resolve_device

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["model_kind"], ckpt["model_config"])
    load_weights(model, ckpt["model_state"])
    model.to(resolve_device(device)).eval()
    return model, ckpt


@torch.no_grad()
def clip_logit(model: torch.nn.Module, wave: np.ndarray, crop: int, batch: int = 16) -> tuple[float, torch.Tensor]:
    """Mean spoof-logit over overlapping windows, plus the per-window logits."""
    wins = windows(wave, crop, crop // 2)
    device = next(model.parameters()).device
    parts = []
    for i in range(0, len(wins), batch):
        x = torch.from_numpy(np.stack(wins[i : i + batch]).astype(np.float32)).to(device)
        parts.append(spoof_logit(model(x)).float().cpu())
    per_window = torch.cat(parts)
    return float(per_window.mean()), per_window


@torch.no_grad()
def predict_file(model: torch.nn.Module, ckpt: dict, audio_path: str | Path, batch: int = 16) -> dict:
    wave = preprocess(audio_path)
    logit, win_logits = clip_logit(model, wave, ckpt["crop_samples"], batch)
    p_spoof = float(torch.sigmoid(torch.tensor(logit)))
    thr = ckpt["threshold_p_spoof"]
    return {
        "file": str(audio_path),
        "label": "SPOOF" if p_spoof >= thr else "REAL",
        "p_spoof": round(p_spoof, 4),
        "p_real": round(1 - p_spoof, 4),
        "threshold_p_spoof": round(thr, 4),
        "duration_s_after_trim": round(wave.size / ckpt["sample_rate"], 2),
        "n_windows": len(win_logits),
        "window_p_spoof": [round(float(v), 4) for v in torch.sigmoid(win_logits)][:20],
        "model": ckpt["model_kind"],
        "trained_on": ckpt["manifest"],
        "note": DISCLAIMER,
    }


@torch.no_grad()
def score_rows(model: torch.nn.Module, ckpt: dict, rows, data_root: str | Path) -> pd.DataFrame:
    """Score manifest rows through the inference path: clip_id, label (0/1), spoof_logit, p_spoof."""
    model.eval()
    out = []
    for r in rows:
        logit, _ = clip_logit(model, preprocess(Path(data_root) / r.path), ckpt["crop_samples"])
        out.append({"clip_id": r.clip_id, "label": 1 if r.label == "spoof" else 0, "spoof_logit": logit,
                    "p_spoof": float(torch.sigmoid(torch.tensor(logit)))})
    return pd.DataFrame(out)


def benchmark(model: torch.nn.Module, ckpt: dict, audio_path: str | Path, runs: int = 30, warmup: int = 3) -> dict:
    """Wall-clock latency of the full predict_file path (load + preprocess + model) and of one model forward pass."""
    import time

    for _ in range(warmup):
        predict_file(model, ckpt, audio_path)
    file_ms, fwd_ms = [], []
    device = next(model.parameters()).device
    x = torch.from_numpy(windows(preprocess(audio_path), ckpt["crop_samples"])[0][None].astype(np.float32)).to(device)
    for _ in range(runs):
        t0 = time.perf_counter()
        predict_file(model, ckpt, audio_path)
        file_ms.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fwd_ms.append((time.perf_counter() - t0) * 1000)
    q = lambda v, p: round(float(np.percentile(v, p)), 1)  # noqa: E731
    return {"runs": runs, "torch_threads": torch.get_num_threads(), "device": str(device),
            "file_ms": {"median": q(file_ms, 50), "p95": q(file_ms, 95)},
            "single_window_forward_ms": {"median": q(fwd_ms, 50), "p95": q(fwd_ms, 95)},
            "params": sum(p.numel() for p in model.parameters())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score one audio file (REAL/SPOOF probability).")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    ap.add_argument("--benchmark", type=int, default=0, metavar="N", help="also time N runs (latency on --device)")
    args = ap.parse_args(argv)
    model, ckpt = load_checkpoint(args.checkpoint, args.device)
    result = predict_file(model, ckpt, args.audio)
    if args.benchmark:
        result["latency"] = benchmark(model, ckpt, args.audio, args.benchmark)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
