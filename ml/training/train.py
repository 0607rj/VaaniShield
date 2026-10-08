"""Train a detector and save the best checkpoint (selected on DEV EER only).

Usage:
    python -m ml.training.train --config ml/configs/smoke_cpu.yaml
    python -m ml.training.train --config ml/configs/full_gpu_aasist.yaml [--device cuda] [--data-root D:/vaani-data]
    python -m ml.training.train --config <same yaml> --resume        # continue from <checkpoint_dir>/<exp_id>/last.pt

Everything machine-specific comes from the YAML config (data.*, train.*, output.*); a few CLI flags override it.
Outputs:
    <output.checkpoint_dir>/<exp_id>/best.pt                    (weights saved as CPU tensors: loads on any machine)
    <output.checkpoint_dir>/<exp_id>/last.pt                    (end of every epoch: model + optimizer + RNG, for --resume)
    <output.results_dir>/<exp_id>/{history.json, dev_scores.csv, run_info.json}
The decision threshold is chosen on dev (at the dev EER point) and stored in the checkpoint; test data is never used here.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.data.dataset import AudioDataset, select_rows
from ml.data.provenance import DataPaths
from ml.evaluation.metrics import eer
from ml.models import build_model, checkpoint_state, load_weights, model_params
from ml.models.lfcc_cnn import spoof_logit
from ml.training.config import ExperimentCfg, config_hash, load_config
from ml.training.repro import git_state, resolve_device, resolve_path, seed_everything, sha256_manifest


def add_override_args(ap: argparse.ArgumentParser) -> None:
    """CLI flags that override the YAML (shared by train and evaluate)."""
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None, help="override train.device")
    ap.add_argument("--data-root", default=None, help="override data.data_root (audio root)")
    ap.add_argument("--num-workers", type=int, default=None, help="override data.num_workers")
    ap.add_argument("--checkpoint-dir", default=None, help="override output.checkpoint_dir")
    ap.add_argument("--results-dir", default=None, help="override output.results_dir")


def overrides_from(args) -> dict:
    return {"train.device": args.device, "data.data_root": args.data_root, "data.num_workers": args.num_workers,
            "output.checkpoint_dir": args.checkpoint_dir, "output.results_dir": args.results_dir}


def data_root_of(cfg: ExperimentCfg) -> Path:
    return resolve_path(cfg.data.data_root) or DataPaths.resolve().data_root


def rows_for(cfg: ExperimentCfg, split: str, manifest: str | None = None):
    return select_rows(manifest or cfg.data.train_manifests[0], split, cfg.data.include_sources,
                       cfg.data.max_clips_per_split.get(split), cfg.train.seed)


@torch.no_grad()
def score_dataset(model: nn.Module, ds: AudioDataset, device: str, batch_size: int = 16) -> pd.DataFrame:
    """clip_id, label, spoof_logit, p_spoof for every item (eval mode, deterministic crops)."""
    model.eval()
    out = []
    for wave, label, cid in DataLoader(ds, batch_size=batch_size, shuffle=False):
        s = spoof_logit(model(wave.to(device))).float().cpu()
        out += [{"clip_id": c, "label": int(y), "spoof_logit": float(v), "p_spoof": float(torch.sigmoid(v))}
                for c, y, v in zip(cid, label, s, strict=True)]
    return pd.DataFrame(out)


def resume_hash(cfg: ExperimentCfg) -> str:
    """Config identity for --resume. Ignores what may legitimately change between interruptions or machines:
    train.epochs (extend a run), train.device, data.num_workers, data.data_root, data.preload_audio, output dirs."""
    neutral = cfg.model_copy(update={
        "train": cfg.train.model_copy(update={"epochs": 1, "device": "auto"}),
        "data": cfg.data.model_copy(update={"num_workers": 0, "data_root": None, "preload_audio": True}),
        "output": cfg.output.model_copy(update={"checkpoint_dir": "", "results_dir": ""})})
    return config_hash(neutral)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train a VaaniShield detector.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    ap.add_argument("--out-dir", default=None, help="legacy: sets both checkpoint_dir and results_dir")
    ap.add_argument("--resume", action="store_true", help="continue from <checkpoint_dir>/<exp_id>/last.pt")
    add_override_args(ap)
    args = ap.parse_args(argv)

    ov = overrides_from(args) | {"train.epochs": args.epochs}
    if args.out_dir:
        ov |= {"output.checkpoint_dir": args.out_dir, "output.results_dir": args.out_dir}
    cfg = load_config(args.config, ov)
    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    use_amp = cfg.train.amp and device == "cuda"
    data_root = data_root_of(cfg)
    manifest = cfg.data.train_manifests[0]
    train_rows, dev_rows = rows_for(cfg, "train"), rows_for(cfg, "dev")
    train_ds = AudioDataset(train_rows, data_root, train=True, seed=cfg.train.seed, crop=cfg.data.crop_samples, preload=cfg.data.preload_audio)
    dev_ds = AudioDataset(dev_rows, data_root, train=False, crop=cfg.data.crop_samples, preload=cfg.data.preload_audio)
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else None
    print(f"device {device}{f' ({gpu})' if gpu else ''}  amp {use_amp}  workers {cfg.data.num_workers}  data_root {data_root}")
    print(f"train {len(train_ds)} clips, dev {len(dev_ds)} clips (manifest {manifest}, sources {cfg.data.include_sources or 'all'})")

    model = build_model(cfg.model.kind, model_params(cfg.model)).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model {cfg.model.kind}: {sum(p.numel() for p in model.parameters()):,} params, {n_trainable:,} trainable")
    labels = torch.tensor([1 if r.label == "spoof" else 0 for r in train_rows])
    counts = torch.bincount(labels, minlength=2).float()
    weights = torch.tensor(cfg.train.class_weights) if cfg.train.class_weights else counts.sum() / (2 * counts)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    if hasattr(model, "param_groups"):  # e.g. XLS-R: head lr vs encoder lr
        opt = torch.optim.AdamW(model.param_groups(cfg.train.lr_head, cfg.train.lr_encoder, cfg.train.weight_decay))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr_head, weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gen = torch.Generator().manual_seed(cfg.train.seed)
    loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, generator=gen,
                        num_workers=cfg.data.num_workers, pin_memory=device == "cuda")

    ckpt_dir = Path(cfg.output.checkpoint_dir) / cfg.exp_id
    res_dir = Path(cfg.output.results_dir) / cfg.exp_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    best = {"eer": float("inf"), "loss": float("inf"), "epoch": -1, "state": None}
    history, stale, start_epoch = [], 0, 0
    if args.resume:
        last = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=True)
        if last["resume_sha256"] != resume_hash(cfg):
            raise SystemExit("refusing to resume: config differs from the one last.pt was trained with (only epochs, device, workers, data_root, preload and output dirs may change)")
        load_weights(model, last["model_state"])
        opt.load_state_dict(last["optimizer"])
        scaler.load_state_dict(last["scaler"])
        gen.set_state(last["generator_state"])
        torch.set_rng_state(last["torch_rng_state"])
        best, history, stale, start_epoch = last["best"], last["history"], last["stale"], last["epoch"] + 1
        print(f"resumed from {ckpt_dir / 'last.pt'} at epoch {start_epoch} (best so far: epoch {best['epoch']}, dev EER {best['eer']:.3f})")
    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()
        train_ds.epoch = epoch
        model.train()
        total = 0.0
        opt.zero_grad()
        for step, (wave, label, _) in enumerate(loader, start=1):
            wave, label = wave.to(device, non_blocking=True), label.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext():
                loss = loss_fn(model(wave), label)
            scaler.scale(loss / cfg.train.grad_accum).backward()
            if step % cfg.train.grad_accum == 0 or step == len(loader):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            total += loss.item() * len(label)
        dev = score_dataset(model, dev_ds, device)
        dev_eer, _ = eer(dev["spoof_logit"], dev["label"])
        logits2 = torch.tensor(np.stack([np.zeros(len(dev)), dev["spoof_logit"]], axis=1), dtype=torch.float32)
        dev_loss = nn.functional.cross_entropy(logits2, torch.tensor(dev["label"].to_numpy())).item()
        rec = {"epoch": epoch, "train_loss": total / len(train_ds), "dev_loss": dev_loss, "dev_eer": dev_eer, "sec": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"epoch {epoch:2d}  train_loss {rec['train_loss']:.4f}  dev_loss {dev_loss:.4f}  dev_EER {dev_eer:.3f}  ({rec['sec']}s)")
        if (dev_eer, dev_loss) < (best["eer"], best["loss"]):
            best = {"eer": dev_eer, "loss": dev_loss, "epoch": epoch, "state": checkpoint_state(model)}
            stale = 0
        else:
            stale += 1
        if hasattr(model, "layer_weights"):
            rec["layer_weights"] = [round(float(w), 5) for w in model.layer_weights()]
        torch.save({"model_state": checkpoint_state(model), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "generator_state": gen.get_state(), "torch_rng_state": torch.get_rng_state(), "epoch": epoch,
                    "best": best, "history": history, "stale": stale, "resume_sha256": resume_hash(cfg)}, ckpt_dir / "last.pt")
        if stale >= cfg.train.early_stop_patience:
            print(f"early stop at epoch {epoch} (no dev improvement for {stale} epochs)")
            break

    load_weights(model, best["state"])
    ckpt = {
        "model_kind": cfg.model.kind,
        "model_config": model.config,
        "model_state": best["state"],  # CPU tensors -> portable (frozen XLS-R: trainable weights only)
        "sample_rate": 16000,
        "crop_samples": cfg.data.crop_samples,
        "best_epoch": best["epoch"],
        "config": cfg.model_dump(mode="json"),
        "config_sha256": config_hash(cfg),
        "manifest": manifest,
        "manifest_sha256": sha256_manifest(manifest),
        "n_train": len(train_rows),
        "n_dev": len(dev_rows),
        "trained_on_device": f"{device}:{gpu}" if gpu else device,
        "git_commit": git_state(".").get("commit"),
        "torch_version": str(torch.__version__),
        "created_at": datetime.now(UTC).isoformat(),
    }
    # Final dev scoring uses the exact inference path (windowed mean), so the threshold matches deployment.
    from ml.inference.predict import score_rows

    dev = score_rows(model, ckpt, dev_rows, data_root)
    dev_eer, thr_p = eer(dev["p_spoof"], dev["label"])  # threshold on p_spoof at the dev EER point
    ckpt.update({"threshold_p_spoof": float(thr_p), "dev_eer": float(dev_eer)})
    dev.to_csv(res_dir / "dev_scores.csv", index=False)
    torch.save(ckpt, ckpt_dir / "best.pt")
    (res_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    run_info = {"config_file": str(args.config), "config_sha256": ckpt["config_sha256"], "device": ckpt["trained_on_device"],
                "amp": use_amp, "num_workers": cfg.data.num_workers, "n_train": len(train_rows), "n_dev": len(dev_rows),
                "checkpoint": str(ckpt_dir / "best.pt"), "best_epoch": best["epoch"], "dev_eer": float(dev_eer),
                "threshold_p_spoof": float(thr_p), "torch": str(torch.__version__), "cuda": torch.version.cuda, "git_commit": ckpt["git_commit"]}
    (res_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(f"best epoch {best['epoch']}: dev EER {dev_eer:.3f} (full-clip windowed scoring; per-epoch lines use one centre crop), threshold p_spoof >= {thr_p:.4f}")
    print(f"saved {ckpt_dir / 'best.pt'}  |  results in {res_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
