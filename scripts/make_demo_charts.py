"""Charts and a results table for the demo / slides, from the saved score CSVs.

Usage:  python scripts/make_demo_charts.py
Output: runs/demo/  (roc.png, eer_by_attack.png, confusion_<model>.png, results.md)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.evaluation.metrics import det_curve, eer, rates_at_threshold, roc_auc  # noqa: E402

OUT = ROOT / "runs/demo"
MANIFEST = ROOT / "data/manifests/vaanishield_v1_full.csv"
MODELS = {
    "AASIST (pretrained)": ROOT / "runs/full/results/PRETRAINED_AASIST_official_upstream",
    "LFCC-CNN (ours)": ROOT / "runs/full/results/FULL_lfcc_cnn_asvspoof_gpu",
}
COLORS = {"AASIST (pretrained)": "#2f6fdf", "LFCC-CNN (ours)": "#e07b20"}
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load(res_dir: Path, split: str) -> pd.DataFrame | None:
    p = res_dir / f"{split}_scores.csv"
    return pd.read_csv(p) if p.exists() else None


def dev_threshold(res_dir: Path) -> float:
    dev = load(res_dir, "dev")
    return eer(dev.spoof_logit, dev.label)[1]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gen = pd.read_csv(MANIFEST, usecols=["clip_id", "generator"]).set_index("clip_id").generator
    rows, have = [], {}
    for name, d in MODELS.items():
        test = load(d, "test")
        if test is None or load(d, "dev") is None:
            print(f"skip {name}: scores missing in {d}")
            continue
        thr = dev_threshold(d)
        e, _ = eer(test.spoof_logit, test.label)
        r = rates_at_threshold(test.spoof_logit, test.label, thr)
        rows.append({"Model": name, "Test clips": len(test), "EER %": round(e * 100, 2),
                     "AUC": round(roc_auc(test.spoof_logit, test.label), 4),
                     "Balanced accuracy %": round((1 - r["hter"]) * 100, 2),
                     "Accuracy %": round((r["tp"] + r["tn"]) / len(test) * 100, 2),
                     "Fake caught %": round(r["recall"] * 100, 2),
                     "Real wrongly flagged %": round(r["false_alarm"] * 100, 2)})
        have[name] = (test, thr)

        # confusion matrix
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum() * 100:.1f}%)", ha="center", va="center",
                        color="white" if cm[i, j] / cm[i].sum() > .5 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["Real", "Fake"]); ax.set_yticks([0, 1], ["Real", "Fake"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name, fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / f"confusion_{name.split()[0].lower()}.png"); plt.close(fig)

    if not have:
        print("no scores yet")
        return 1

    # ROC
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for name, (t, _) in have.items():
        fa, miss, _ = det_curve(t.spoof_logit, t.label)
        ax.plot(fa * 100, (1 - miss) * 100, color=COLORS[name], lw=2,
                label=f"{name}  (EER {eer(t.spoof_logit, t.label)[0] * 100:.2f}%)")
    ax.plot([0, 100], [0, 100], color="#999", lw=1, ls="--")
    ax.set_xlabel("Real voices wrongly flagged (%)"); ax.set_ylabel("Fake voices caught (%)")
    ax.set_title("ROC – ASVspoof 2019 LA evaluation set"); ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "roc.png"); plt.close(fig)

    # EER per attack (each spoof generator vs all bona fide)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    attacks = sorted({g for t, _ in have.values() for g in gen.reindex(t.clip_id).dropna() if g not in ("none", "-")})
    w = 0.8 / len(have)
    for k, (name, (t, _)) in enumerate(have.items()):
        g = gen.reindex(t.clip_id).values
        vals = []
        for a in attacks:
            m = (t.label.values == 0) | (g == a)
            vals.append(eer(t.spoof_logit[m], t.label[m])[0] * 100)
        ax.bar(np.arange(len(attacks)) + k * w, vals, w, color=COLORS[name], label=name)
    ax.set_xticks(np.arange(len(attacks)) + w * (len(have) - 1) / 2, attacks)
    ax.set_ylabel("EER (%)  ↓ better"); ax.set_title("Error per voice-cloning attack (A07–A19, unseen in training)")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(OUT / "eer_by_attack.png"); plt.close(fig)

    table = pd.DataFrame(rows)
    md = ["# VaaniShield results — ASVspoof 2019 LA evaluation set", "",
          "Threshold chosen on the dev set only; test set scored once.", "", table.to_markdown(index=False)]

    # Hinglish zero-shot (English-trained models on the Hinglish pilot test set, 120 clips)
    hrows = []
    for name, d, en in (("AASIST (pretrained)", ROOT / "runs/hinglish/AASIST_zeroshot", "AASIST (pretrained)"),
                        ("LFCC-CNN (ours)", ROOT / "runs/hinglish/LFCC_zeroshot", "LFCC-CNN (ours)")):
        t = load(d, "test")
        if t is None:
            continue
        en_eer = next((r["EER %"] for r in rows if r["Model"] == en), None)
        hrows.append({"Model": name, "English EER %": en_eer,
                      "Hinglish EER %": round(eer(t.spoof_logit, t.label)[0] * 100, 1),
                      "Hinglish AUC": round(roc_auc(t.spoof_logit, t.label), 3)})
    if hrows:
        ht = pd.DataFrame(hrows)
        md += ["", "## English-trained models on Hinglish (zero-shot, pilot test set, 120 clips)", "",
               "Small pilot set: wide uncertainty. Shows the language gap, not a final Hinglish benchmark.", "",
               ht.to_markdown(index=False)]
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        x = np.arange(len(ht))
        ax.bar(x - 0.2, ht["English EER %"].fillna(0), 0.4, color="#2f6fdf", label="English (ASVspoof eval)")
        ax.bar(x + 0.2, ht["Hinglish EER %"], 0.4, color="#e5484d", label="Hinglish (pilot test)")
        ax.axhline(50, color="#999", ls="--", lw=1)
        ax.text(len(ht) - 0.5, 51, "random guess", color="#777", fontsize=8, ha="right")
        ax.set_xticks(x, ht.Model); ax.set_ylabel("EER (%)  ↓ better"); ax.set_title("The language gap: English → Hinglish")
        ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(OUT / "hinglish_gap.png"); plt.close(fig)
        print(ht.to_string(index=False))
    (OUT / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"saved charts to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
