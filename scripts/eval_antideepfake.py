"""Zero-shot evaluation of the NII AntiDeepfake MMS-300M detector (no VaaniShield training).

Sets: Hinglish pilot dev + test (all 140 clips) and a seeded sample of ASVspoof 2019 LA eval.
Decision threshold: upstream rule P(fake) >= 0.5, i.e. spoof-logit >= 0 (no tuning on our data).
Caveat: ASVspoof 2019 LA is part of AntiDeepfake's post-training data, so the English number is optimistic.

Usage: python scripts/eval_antideepfake.py [--en-n 2000]
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data.audio import load_audio  # noqa: E402
from ml.evaluation.metrics import eer, rates_at_threshold, roc_auc  # noqa: E402
from ml.models.antideepfake import AntiDeepfakeDetector  # noqa: E402
from ml.models.lfcc_cnn import spoof_logit  # noqa: E402

OUT = ROOT / "runs/multilingual/antideepfake_zeroshot"
MAX_S = 10  # score at most the first 10 s of each clip


@torch.no_grad()
def score(model, paths, device):
    out = []
    for p in paths:
        w = load_audio(ROOT / "data" / p)[: MAX_S * 16000]
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out.append(float(spoof_logit(model(torch.from_numpy(w)[None].to(device))).float()))
    return np.array(out)


def summary(s, y, thr=0.0):
    r = rates_at_threshold(s, y, thr)
    return {"n": int(len(y)), "eer": eer(s, y)[0], "auc": roc_auc(s, y), "balanced_accuracy": 1 - r["hter"],
            "fake_caught": r["recall"], "real_flagged": r["false_alarm"], "threshold": thr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--en-n", type=int, default=2000)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    model = AntiDeepfakeDetector("mms-300m").to(device).eval()

    h = pd.read_csv(ROOT / "data/manifests/hinglish_pilot_v1.csv", low_memory=False)
    la = pd.read_csv(ROOT / "data/manifests/vaanishield_v1_full.csv", low_memory=False)
    la = la[(la.source_dataset == "asvspoof2019_la") & (la.split == "test")]
    en = pd.concat([g.sample(min(args.en_n // 2, len(g)), random_state=2026) for _, g in la.groupby("label")])
    sets = {"hing_dev": h[h.split == "dev"], "hing_test": h[h.split == "test"], "en_test": en}
    hi_path = ROOT / "data/manifests/hindi_fleurs_mms.csv"
    if hi_path.exists():
        hi = pd.read_csv(hi_path)
        sets["hindi_test"] = hi[hi.split == "test"]
    res = {"model": "NII AntiDeepfake MMS-300M, zero-shot (no VaaniShield training)", "threshold_spoof_logit": 0.0,
           "caveat": "ASVspoof 2019 LA is in the model's post-training data; English result is optimistic."}
    for k, df in sets.items():
        t0 = time.time()
        s = score(model, df.path, device)
        y = (df.label.values == "spoof").astype(int)
        res[k] = summary(s, y)
        pd.DataFrame({"clip_id": df.clip_id.values, "label": y, "spoof_logit": s, "generator": df.generator.values,
                      "source": df.source_dataset.values}).to_csv(OUT / f"{k}_scores.csv", index=False)
        print(f"{k}: n={len(y)}  EER {res[k]['eer'] * 100:.2f}%  AUC {res[k]['auc']:.4f}  balanced acc "
              f"{res[k]['balanced_accuracy'] * 100:.1f}%  fake caught {res[k]['fake_caught'] * 100:.0f}%  "
              f"real flagged {res[k]['real_flagged'] * 100:.0f}%  ({time.time() - t0:.0f}s)", flush=True)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
