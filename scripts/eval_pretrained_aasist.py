"""Score the OFFICIAL pretrained AASIST (clovaai, trained on ASVspoof 2019 LA train only) on our LA dev + eval splits.

No leakage: the full manifest's dev = official LA dev and test = official LA eval; the checkpoint saw only LA train.
The dev split picks the EER threshold; test (LA eval, 71k trials) is scored once with that fixed threshold.

Preprocessing:
  --preproc upstream  (default) first 64,600 samples, tiled if shorter, no silence trim: exactly as clovaai evaluates.
  --preproc vaani     VaaniShield inference path (silence trim + mean over overlapping 4 s windows).

Usage:
  python scripts/eval_pretrained_aasist.py --data-root data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data.audio import CROP_SAMPLES, crop_or_pad, load_audio, trim_silence, windows  # noqa: E402
from ml.evaluation.metrics import eer, rates_at_threshold, roc_auc  # noqa: E402
from ml.models.aasist import AASISTDetector  # noqa: E402
from ml.models.lfcc_cnn import spoof_logit  # noqa: E402


def upstream_input(wave: np.ndarray) -> np.ndarray:
    """clovaai eval: first 64,600 samples; shorter clips are tiled."""
    return crop_or_pad(wave[:CROP_SAMPLES], CROP_SAMPLES) if wave.size >= CROP_SAMPLES else crop_or_pad(wave, CROP_SAMPLES)


class Clips(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: Path, preproc: str):
        self.df, self.root, self.preproc = df.reset_index(drop=True), data_root, preproc

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        wave = load_audio(self.root / r.path)
        if self.preproc == "upstream":
            x = upstream_input(wave)[None]
        else:
            x = np.stack(windows(trim_silence(wave)))
        return torch.from_numpy(x.astype(np.float32)), i


def collate(batch):
    xs, idx = zip(*batch)
    counts = [x.shape[0] for x in xs]
    return torch.cat(xs), counts, list(idx)


@torch.no_grad()
def score(model, df, data_root, preproc, device, batch, workers) -> np.ndarray:
    dl = DataLoader(Clips(df, data_root, preproc), batch_size=batch, num_workers=workers, collate_fn=collate,
                    persistent_workers=workers > 0)
    out = np.zeros(len(df), dtype=np.float64)
    t0, done = time.time(), 0
    for x, counts, idx in dl:
        s = spoof_logit(model(x.to(device))).float().cpu().numpy()
        pos = 0
        for c, i in zip(counts, idx):
            out[i] = s[pos : pos + c].mean()
            pos += c
        done += len(idx)
        if done % (batch * 50) < batch:
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(df)}  {rate:.0f} clips/s  eta {(len(df) - done) / rate / 60:.1f} min", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/vaanishield_v1_full.csv"))
    ap.add_argument("--variant", default="AASIST", choices=["AASIST", "AASIST-L"])
    ap.add_argument("--preproc", default="upstream", choices=["upstream", "vaani"])
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sources", default="asvspoof2019_la", help="comma list of source_dataset values, or 'all'")
    ap.add_argument("--device", default=None, help="cuda/cpu (default: cuda if available)")
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AASISTDetector(args.variant)
    weights = model.load_upstream_pretrained()
    model.to(device).eval()
    out_dir = Path(args.out or ROOT / f"runs/full/results/PRETRAINED_{args.variant}_official_{args.preproc}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model {args.variant} official weights {weights}  device {device}  preproc {args.preproc}")

    df = pd.read_csv(args.manifest, low_memory=False)
    if args.sources != "all":
        df = df[df.source_dataset.isin(args.sources.split(","))]
    res = {"model": f"{args.variant} (official clovaai weights, LA train only)", "weights": weights,
           "preproc": args.preproc, "device": device}
    thr = None
    for split in args.splits.split(","):
        part = df[df.split == split]
        print(f"scoring {split}: {len(part)} clips")
        t0 = time.time()
        s = score(model, part, Path(args.data_root), args.preproc, device, args.batch, args.workers)
        y = (part.label.values == "spoof").astype(int)
        pd.DataFrame({"clip_id": part.clip_id.values, "label": y, "spoof_logit": s,
                      "generator": part.generator.values}).to_csv(out_dir / f"{split}_scores.csv", index=False)
        e, t = eer(s, y)
        if thr is None:
            thr = t  # first split (dev) fixes the threshold
        r = rates_at_threshold(s, y, thr)
        r["balanced_accuracy"] = 1 - r["hter"]
        r["accuracy"] = (r["tp"] + r["tn"]) / len(y)
        res[split] = {"n": int(len(y)), "n_spoof": int(y.sum()), "eer": e, "auc": roc_auc(s, y),
                      "at_dev_threshold": r, "seconds": round(time.time() - t0, 1)}
        print(f"  {split}: EER {e * 100:.2f}%  AUC {res[split]['auc']:.4f}  balanced acc {r['balanced_accuracy'] * 100:.2f}%")
    res["threshold_spoof_logit"] = thr
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"saved {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
