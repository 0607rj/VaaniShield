"""Language-gap chart + table: EER of each detector on English, Hindi and Hinglish test sets.

Reads saved score files / metrics; writes runs/demo/language_gap.png, runs/demo/language_gap.md and copies the PNG to
server/static/img/. Usage: python scripts/make_language_chart.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.evaluation.metrics import eer  # noqa: E402


def from_scores(p):
    p = ROOT / p
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return round(eer(d.spoof_logit, d.label)[0] * 100, 2)


def from_metrics(p, key):
    p = ROOT / p
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return round(m[key]["eer"] * 100, 2) if key in m else None


rows = [
    ("AASIST\n(English-only, pretrained)", from_scores("runs/full/results/PRETRAINED_AASIST_official_upstream/test_scores.csv"),
     from_scores("runs/hindi/AASIST_zeroshot/test_scores.csv"), from_scores("runs/hinglish/AASIST_zeroshot/test_scores.csv")),
    ("LFCC-CNN\n(ours, English-trained)", from_scores("runs/full/results/FULL_lfcc_cnn_asvspoof_gpu/test_scores.csv"),
     from_scores("runs/hindi/LFCC_zeroshot/test_scores.csv"), from_scores("runs/hinglish/LFCC_zeroshot/test_scores.csv")),
    ("Multilingual\n(MMS-300M AntiDeepfake)", from_metrics("runs/multilingual/antideepfake_zeroshot/metrics.json", "en_test"),
     from_metrics("runs/multilingual/antideepfake_zeroshot/metrics.json", "hindi_test"),
     from_metrics("runs/multilingual/antideepfake_zeroshot/metrics.json", "hing_test")),
]
ft = ROOT / "runs/multilingual/antideepfake_finetuned/metrics.json"
if ft.exists():
    rows.append(("Multilingual\n+ VaaniShield fine-tune", from_metrics(ft.relative_to(ROOT), "en_test"),
                 from_metrics(ft.relative_to(ROOT), "hindi_test"), from_metrics(ft.relative_to(ROOT), "hing_test")))

langs = ["English", "Hindi", "Hinglish"]
colors = ["#3b3bd6", "#ff8a1f", "#e5484d"]
plt.rcParams.update({"figure.dpi": 160, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8.2, 3.8))
x = np.arange(len(rows))
w = 0.26
for k, (lang, c) in enumerate(zip(langs, colors)):
    vals = [r[k + 1] if r[k + 1] is not None else np.nan for r in rows]
    bars = ax.bar(x + (k - 1) * w, vals, w, color=c, label=lang)
    for b, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8, color="#16133a")
ax.axhline(50, color="#999", ls="--", lw=1)
ax.text(len(rows) - 0.55, 51.5, "coin toss", color="#777", fontsize=8, ha="right")
ax.set_xticks(x, [r[0] for r in rows], fontsize=9)
ax.set_ylabel("Error rate, EER %  (lower is better)")
ax.set_ylim(0, 65)
ax.set_title("Same task, three languages: English-only models fail on Hindi & Hinglish", fontsize=11)
ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
fig.tight_layout()
out = ROOT / "runs/demo"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "language_gap.png")
shutil.copy(out / "language_gap.png", ROOT / "server/static/img/language_gap.png")

table = pd.DataFrame([{"Model": r[0].replace("\n", " "), "English EER %": r[1], "Hindi EER %": r[2], "Hinglish EER %": r[3]}
                      for r in rows])
(out / "language_gap.md").write_text(
    "# Error rate by language (EER %, lower is better)\n\n"
    "English = ASVspoof 2019 LA eval (AASIST/LFCC: all 71,237 clips; multilingual: seeded sample — note ASVspoof 2019 "
    "is in its post-training data). Hindi = FLEURS Hindi real vs MMS-TTS Hindi fake, same sentences, test split "
    "(196 clips). Hinglish = pilot test set (120 clips; AASIST/LFCC: original test; fine-tuned: held-out part).\n\n"
    + table.to_markdown(index=False) + "\n", encoding="utf-8")
print(table.to_string(index=False))
