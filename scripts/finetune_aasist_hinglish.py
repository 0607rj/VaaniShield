"""PILOT: fine-tune the official pretrained AASIST on the Hinglish pilot + an English replay set.

Why: zero-shot, the English-trained AASIST scores 58 % EER on the Hinglish pilot test set.
Data: Hinglish pilot train (60 clips) + a seeded English replay sample from LA train (to limit forgetting).
Selection: Hinglish pilot dev (20 clips) EER, tie-break English dev-subset EER. Test (120 Hinglish clips +
an LA eval subset) is scored once, at the end, with the dev-chosen threshold.

CAVEAT (dataset shortcut): in the pilot TRAIN split every real clip is OpenSLR-104 and every fake is CS-FLEURS,
so a model can learn the corpus instead of the fake. The pilot TEST split also contains real CS-FLEURS clips,
which partly exposes that shortcut. Report this as a pilot result only.

Usage: python scripts/finetune_aasist_hinglish.py [--device cpu] [--epochs 8]
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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data.audio import CROP_SAMPLES, crop_or_pad, load_audio  # noqa: E402
from ml.evaluation.metrics import eer, rates_at_threshold, roc_auc  # noqa: E402
from ml.models.aasist import AASISTDetector  # noqa: E402
from ml.models.lfcc_cnn import spoof_logit  # noqa: E402

DATA = ROOT / "data"
HING = ROOT / "data/manifests/hinglish_pilot_v1.csv"
LA = ROOT / "data/manifests/vaanishield_v1_full.csv"
OUT = ROOT / "runs/hinglish/AASIST_finetuned"


def sample(df, n_per_class, seed):
    return pd.concat([g.sample(min(n_per_class, len(g)), random_state=seed) for _, g in df.groupby("label")])


def load_all(df):
    return [load_audio(DATA / p) for p in df.path]


def fixed_input(w):  # evaluation: first 4 s (upstream protocol), tiled if shorter
    return crop_or_pad(w[:CROP_SAMPLES], CROP_SAMPLES)


@torch.no_grad()
def score(model, waves, device, batch=16):
    model.eval()
    out = []
    for i in range(0, len(waves), batch):
        x = torch.from_numpy(np.stack([fixed_input(w) for w in waves[i : i + batch]]).astype(np.float32)).to(device)
        out.append(spoof_logit(model(x)).float().cpu().numpy())
    return np.concatenate(out)


def metrics(s, y, thr):
    e, t = eer(s, y)
    r = rates_at_threshold(s, y, thr if thr is not None else t)
    return {"n": int(len(y)), "eer": e, "auc": roc_auc(s, y), "balanced_accuracy": 1 - r["hter"],
            "fake_caught": r["recall"], "real_flagged": r["false_alarm"], "eer_threshold": t}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--replay", type=int, default=300, help="English LA-train clips per class mixed into training")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--resplit", action="store_true",
                    help="move half of the test split's real CS-FLEURS and MMS-concat fakes into train (breaks the "
                         "real=OpenSLR / fake=CS-FLEURS shortcut; CS-FLEURS real is ONE speaker, so it is shared)")
    ap.add_argument("--extra", default=None, help="folder with real/ and fake/ sub-folders of own recordings (80%% train, 20%% dev)")
    ap.add_argument("--tag", default="AASIST_finetuned")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    global OUT
    OUT = ROOT / "runs/hinglish" / args.tag
    OUT.mkdir(parents=True, exist_ok=True)

    hing = pd.read_csv(HING, low_memory=False)
    la = pd.read_csv(LA, low_memory=False)
    la = la[la.source_dataset == "asvspoof2019_la"]
    h_train, h_dev, h_test = hing[hing.split == "train"], hing[hing.split == "dev"], hing[hing.split == "test"]
    if args.resplit:
        movable = h_test[((h_test.source_dataset == "cs_fleurs") & (h_test.label == "bonafide"))
                         | (h_test.generator == "mms_concat_csfleurs")]
        moved = pd.concat([g.sample(frac=0.5, random_state=args.seed) for _, g in movable.groupby(["label", "generator"])])
        h_train = pd.concat([h_train, moved])
        h_test = h_test.drop(moved.index)
    if args.extra:
        rows = []
        for lab, sub in (("bonafide", "real"), ("spoof", "fake")):
            for f in sorted((Path(args.extra) / sub).glob("*")):
                if f.suffix.lower() in {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".webm", ".flac", ".mp4", ".aac"}:
                    rows.append({"clip_id": f"own_{sub}_{f.stem}", "path": str(f.resolve()), "label": lab,
                                 "source_dataset": "own_recordings", "generator": "own" if lab == "spoof" else "none"})
        own = pd.DataFrame(rows)
        if len(own):
            dev_part = pd.concat([g.sample(frac=0.2, random_state=args.seed) for _, g in own.groupby("label")])
            h_train = pd.concat([h_train, own.drop(dev_part.index)])
            h_dev = pd.concat([h_dev, dev_part])
            print(f"own recordings: {len(own)} ({(own.label == 'bonafide').sum()} real), {len(dev_part)} kept for dev", flush=True)
    sets = {
        "train": pd.concat([h_train, sample(la[la.split == "train"], args.replay, args.seed)]),
        "hing_dev": h_dev,
        "en_dev": sample(la[la.split == "dev"], 250, args.seed),
        "hing_test": h_test,
        "en_test": sample(la[la.split == "test"], 1000, args.seed),
    }
    print({k: len(v) for k, v in sets.items()}, flush=True)
    waves = {k: load_all(v) for k, v in sets.items()}
    labels = {k: (v.label.values == "spoof").astype(int) for k, v in sets.items()}

    device = torch.device(args.device)
    model = AASISTDetector("AASIST")
    model.load_upstream_pretrained()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    y_tr = labels["train"]
    w = torch.tensor([len(y_tr) / (2 * (y_tr == 0).sum()), len(y_tr) / (2 * (y_tr == 1).sum())], dtype=torch.float32)
    # Hinglish clips are few: weight them up so they are not drowned by the English replay
    sw = np.where(sets["train"].source_dataset.values != "asvspoof2019_la", 3.0, 1.0)
    lossf = nn.CrossEntropyLoss(weight=w.to(device), reduction="none")

    def evaluate_dev():
        hs, es = score(model, waves["hing_dev"], device), score(model, waves["en_dev"], device)
        return metrics(hs, labels["hing_dev"], None), metrics(es, labels["en_dev"], None), hs

    h0, e0, _ = evaluate_dev()
    print(f"epoch -1 (pretrained)  hing_dev EER {h0['eer'] * 100:.1f}%  en_dev EER {e0['eer'] * 100:.2f}%", flush=True)
    history = [{"epoch": -1, "hing_dev_eer": h0["eer"], "en_dev_eer": e0["eer"]}]
    best = (h0["eer"], e0["eer"], -1)
    torch.save(model.net.state_dict(), OUT / "best_net.pth")

    n = len(y_tr)
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        order = rng.permutation(n)
        for i in range(0, n, args.batch):
            idx = order[i : i + args.batch]
            x = torch.from_numpy(np.stack([crop_or_pad(waves["train"][j], CROP_SAMPLES, rng) for j in idx]).astype(np.float32)).to(device)
            y = torch.from_numpy(y_tr[idx]).to(device)
            loss = (lossf(model(x), y) * torch.from_numpy(sw[idx]).float().to(device)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        h, e, _ = evaluate_dev()
        history.append({"epoch": ep, "train_loss": tot / n, "hing_dev_eer": h["eer"], "en_dev_eer": e["eer"]})
        print(f"epoch {ep}  loss {tot / n:.4f}  hing_dev EER {h['eer'] * 100:.1f}%  en_dev EER {e['eer'] * 100:.2f}%  ({time.time() - t0:.0f}s)", flush=True)
        if e["eer"] <= 0.05 and (h["eer"], e["eer"]) < best[:2]:  # keep English working
            best = (h["eer"], e["eer"], ep)
            torch.save(model.net.state_dict(), OUT / "best_net.pth")

    print(f"best epoch {best[2]}  hing_dev EER {best[0] * 100:.1f}%  en_dev EER {best[1] * 100:.2f}%", flush=True)
    model.net.load_state_dict(torch.load(OUT / "best_net.pth", map_location=device, weights_only=True))
    # thresholds: combined dev (Hinglish + English) EER threshold, used for the server and for test
    dev_s = np.r_[score(model, waves["hing_dev"], device), score(model, waves["en_dev"], device)]
    dev_y = np.r_[labels["hing_dev"], labels["en_dev"]]
    thr = eer(dev_s, dev_y)[1]
    res = {"model": "AASIST fine-tuned on Hinglish pilot + English replay (PILOT)", "best_epoch": best[2],
           "threshold_spoof_logit": thr, "history": history, "caveat": __doc__.split("CAVEAT")[1].split("Usage")[0].strip()}
    for k in ("hing_test", "en_test"):
        s = score(model, waves[k], device)
        res[k] = metrics(s, labels[k], thr)
        pd.DataFrame({"clip_id": sets[k].clip_id.values, "label": labels[k], "spoof_logit": s,
                      "generator": sets[k].generator.values, "source": sets[k].source_dataset.values}).to_csv(OUT / f"{k}_scores.csv", index=False)
        print(f"{k}: EER {res[k]['eer'] * 100:.2f}%  AUC {res[k]['auc']:.3f}  balanced acc {res[k]['balanced_accuracy'] * 100:.1f}%", flush=True)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
