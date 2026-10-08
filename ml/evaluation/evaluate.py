"""Evaluate a checkpoint on a manifest split, or compute metrics from an existing score CSV.

    python -m ml.evaluation.evaluate --config ml/configs/smoke_cpu.yaml [--split test]
    python -m ml.evaluation.evaluate --checkpoint runs/B0_lfcc_cnn_pilot/best.pt --split test
    python -m ml.evaluation.evaluate --scores <csv> [--threshold <spoof_logit threshold chosen on dev>]

Checkpoint mode scores every clip through the inference path (ml.inference.predict), applies the threshold that was
chosen on DEV and stored in the checkpoint, and writes <split>_scores.csv + <split>_metrics.json next to it.
No threshold or model choice is ever made on the evaluated split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from ml.evaluation.metrics import brier, ece, eer, nll, rates_at_threshold, roc_auc

_LABEL_MAP = {"bonafide": 0, "spoof": 1, 0: 0, 1: 1, "0": 0, "1": 1}


def evaluate_frame(df: pd.DataFrame, threshold: float | None = None) -> dict:
    missing = {"clip_id", "label", "spoof_logit"} - set(df.columns)
    if missing:
        raise ValueError(f"score file missing columns: {sorted(missing)}")
    if df["clip_id"].duplicated().any():
        raise ValueError("duplicate clip_id in score file")
    labels = df["label"].map(_LABEL_MAP)
    if labels.isna().any():
        raise ValueError("labels must be bonafide/spoof or 0/1")
    y = labels.to_numpy(dtype=int)
    s = df["spoof_logit"].to_numpy(dtype=float)
    e, thr_eer = eer(s, y)
    out: dict = {
        "n": int(len(df)),
        "n_spoof": int(y.sum()),
        "eer": e,
        "eer_threshold": thr_eer,
        "auc": roc_auc(s, y),
    }
    if threshold is not None:
        out["at_threshold"] = rates_at_threshold(s, y, threshold)
    if "p_spoof" in df.columns:
        p = df["p_spoof"].to_numpy(dtype=float)
        out["calibration"] = {
            "ece_equal_mass_15": ece(p, y, 15, "equal_mass"),
            "ece_equal_width_15": ece(p, y, 15, "equal_width"),
            "nll": nll(p, y),
            "brier": brier(p, y),
        }
    return out


def evaluate_checkpoint(checkpoint: str | Path, split: str, manifest: str | None = None, cfg=None,
                        device: str = "cpu", out_dir: str | Path | None = None, data_root: str | None = None) -> dict:
    """Score `split` through the inference path with the DEV threshold stored in the checkpoint.

    Data selection (sources, subset size, seed) comes from `cfg`, or else from the config saved in the checkpoint,
    so evaluation always uses the same selection rules as training.
    """
    from ml.inference.predict import load_checkpoint, score_rows
    from ml.training.config import ExperimentCfg
    from ml.training.repro import resolve_path
    from ml.training.train import data_root_of, rows_for

    model, ckpt = load_checkpoint(checkpoint, device)
    cfg = cfg or ExperimentCfg.model_validate(ckpt["config"])
    manifest = manifest or ckpt["manifest"]
    root = resolve_path(data_root) or data_root_of(cfg)
    df = score_rows(model, ckpt, rows_for(cfg, split, manifest), root)
    thr_p = ckpt["threshold_p_spoof"]
    if 0 < thr_p < 1:
        thr_logit = math.log(thr_p / (1 - thr_p))
    else:
        thr_logit = math.inf if thr_p >= 1 else -math.inf
    metrics = {
        "checkpoint": str(checkpoint),
        "manifest": manifest,
        "split": split,
        "n_scored": int(len(df)),
        "device": str(next(model.parameters()).device),
        "threshold_from": "dev (stored in checkpoint)",
        "threshold_p_spoof": thr_p,
        **evaluate_frame(df, thr_logit),
    }
    out = Path(out_dir) if out_dir else Path(checkpoint).parent
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{split}_scores.csv", index=False)
    (out / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    from ml.training.train import add_override_args, overrides_from

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="evaluate <output.checkpoint_dir>/<exp_id>/best.pt; results to <output.results_dir>/<exp_id>/")
    src.add_argument("--checkpoint", help="evaluate this checkpoint; results next to it")
    src.add_argument("--scores")
    ap.add_argument("--split", default="test", help="manifest split to evaluate")
    ap.add_argument("--manifest", default=None, help="default: manifest stored in the checkpoint")
    ap.add_argument("--threshold", type=float, default=None, help="with --scores: dev-chosen spoof_logit threshold")
    add_override_args(ap)
    args = ap.parse_args(argv)
    if args.config:
        from ml.training.config import load_config

        cfg = load_config(args.config, overrides_from(args))
        ckpt_path = Path(cfg.output.checkpoint_dir) / cfg.exp_id / "best.pt"
        result = evaluate_checkpoint(ckpt_path, args.split, args.manifest, cfg, cfg.train.device,
                                     Path(cfg.output.results_dir) / cfg.exp_id)
    elif args.checkpoint:
        result = evaluate_checkpoint(args.checkpoint, args.split, args.manifest, device=args.device or "cpu",
                                     data_root=args.data_root)
    else:
        result = evaluate_frame(pd.read_csv(args.scores), args.threshold)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
