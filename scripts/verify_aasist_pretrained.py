#!/usr/bin/env python
"""Integration check of the OFFICIAL AASIST checkpoint on ASVspoof 2019 LA *eval* clips (unseen by that checkpoint).

Purpose: confirm that the vendored model + wrapper + label convention + weights work as upstream intends
(score direction, sensible separation). This is NOT a VaaniShield result and is NOT comparable to the README's
EER (that uses all 71,237 eval trials; we use the small pilot downloaded with
`python scripts/acquire_asvspoof.py --pilot 20 --partition eval --accept-terms`).

Two input variants are reported:
  upstream : raw file, first 64,600 samples, no silence trim   (exactly upstream eval preprocessing)
  ours     : VaaniShield preprocessing (trim + windows, ml.inference.predict) used for every model
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from ml.data.audio import preprocess
from ml.data.provenance import DataPaths
from ml.evaluation.metrics import eer, roc_auc
from ml.inference.predict import clip_logit
from ml.models.aasist import AASISTDetector


def upstream_pad(x: np.ndarray, max_len: int = 64600) -> np.ndarray:
    """Copy of upstream data_utils.pad: first max_len samples, or tile."""
    if x.shape[0] >= max_len:
        return x[:max_len]
    return np.tile(x, int(max_len / x.shape[0]) + 1)[:max_len]


def main() -> int:
    root = DataPaths.resolve().data_root / "raw/asvspoof2019_la/LA"
    proto = root / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    flac = root / "ASVspoof2019_LA_eval/flac"
    on_disk = {p.stem: p for p in flac.glob("*.flac")}
    rows = [ln.split() for ln in proto.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r[1] in on_disk]
    if len(rows) < 4:
        raise SystemExit("too few LA eval clips on disk; run the eval pilot download first")
    model = AASISTDetector("AASIST")
    weights = model.load_upstream_pretrained()
    model.eval()
    y = np.array([1 if r[4] == "spoof" else 0 for r in rows])
    s_up, s_ours = [], []
    with torch.no_grad():
        for r in rows:
            x, sr = sf.read(str(on_disk[r[1]]), dtype="float32")
            assert sr == 16000
            logits = model(torch.from_numpy(upstream_pad(x))[None])
            s_up.append(float(logits[0, 1] - logits[0, 0]))
            s_ours.append(clip_logit(model, preprocess(on_disk[r[1]]), 64600)[0])
    out = {"weights": weights, "n_clips": len(rows), "n_spoof": int(y.sum()),
           "attacks": sorted({r[3] for r in rows if r[4] == "spoof"}),
           "upstream_preprocessing": {"eer": eer(s_up, y)[0], "auc": roc_auc(s_up, y),
                                      "mean_spoof_logit": {"spoof": float(np.mean(np.array(s_up)[y == 1])), "bonafide": float(np.mean(np.array(s_up)[y == 0]))}},
           "vaanishield_preprocessing": {"eer": eer(s_ours, y)[0], "auc": roc_auc(s_ours, y)}}
    out["score_direction_ok"] = out["upstream_preprocessing"]["mean_spoof_logit"]["spoof"] > out["upstream_preprocessing"]["mean_spoof_logit"]["bonafide"]
    dest = Path("runs/upstream/aasist/verify_pretrained_la_eval_pilot.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if out["score_direction_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
