"""Copy demo clips from the ASVspoof 2019 LA EVAL set (never seen in training) into demo_clips/real and demo_clips/fake.

Picks clips that the pretrained AASIST classifies correctly with a clear margin and that are at least 3 s long, spread
over different attacks (A07–A19) and speakers. Also exports them as MP3 (for WhatsApp/YouTube uploads) and writes
demo_clips/README.md listing each clip with its ground truth and score.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "runs/full/results/PRETRAINED_AASIST_official_upstream"
OUT = ROOT / "demo_clips"
N = 10


def main() -> int:
    import json

    thr = json.loads((RES / "metrics.json").read_text())["threshold_spoof_logit"]
    s = pd.read_csv(RES / "test_scores.csv")
    man = pd.read_csv(ROOT / "data/manifests/vaanishield_v1_full.csv", low_memory=False).set_index("clip_id")
    s = s.join(man[["path", "duration_s", "speaker_id"]], on="clip_id")
    s = s[s.duration_s >= 3.0]
    real = s[(s.label == 0) & (s.spoof_logit < thr - 3)].sort_values("spoof_logit")
    fake = s[(s.label == 1) & (s.spoof_logit > thr + 3)].sort_values("spoof_logit", ascending=False)
    real = real.drop_duplicates("speaker_id").head(N)
    fake = fake.groupby("generator").head(1).head(N)  # one per attack type
    rows = []
    for kind, part in (("real", real), ("fake", fake)):
        d = OUT / kind
        d.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(part.itertuples(), 1):
            name = f"{kind}_{i:02d}_{r.clip_id}" + (f"_{r.generator}" if kind == "fake" else "")
            src = ROOT / "data" / r.path
            shutil.copy(src, d / f"{name}.flac")
            mp3 = d / f"{name}.mp3"
            if shutil.which("ffmpeg"):
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-b:a", "128k", str(mp3)], check=False)
            rows.append({"file": f"{kind}/{name}.flac", "truth": kind.upper(), "attack": r.generator if kind == "fake" else "-",
                         "duration_s": r.duration_s, "aasist_spoof_logit": round(r.spoof_logit, 2)})
    t = pd.DataFrame(rows)
    (OUT / "README.md").write_text(
        "# Demo clips\n\nFrom the official ASVspoof 2019 LA **evaluation** set (unseen speakers and attacks; never used "
        "for training or threshold selection). Licence: ODC-By 1.0 (ASVspoof 2019, University of Edinburgh).\n"
        "`.mp3` copies are for uploading to WhatsApp / YouTube to demo the extension.\n\n"
        + t.to_markdown(index=False) + "\n", encoding="utf-8")
    print(t.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
