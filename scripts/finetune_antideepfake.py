"""Light fine-tune of the multilingual AntiDeepfake MMS-300M detector for Hindi / Hinglish (+ English replay).

Trainable: the top K transformer layers + final encoder layer norm + classifier head (rest frozen), so the model keeps
what it learned from ~74k hours of post-training. Mixed precision, batch 4, fits a 6 GB GPU.

Data (train / dev / test kept apart):
  - Hindi  : data/manifests/hindi_fleurs_mms.csv (FLEURS real vs MMS-TTS fake, split by sentence)
  - Hinglish: hinglish_pilot_v1 (train; plus half of the test split's real CS-FLEURS and MMS-concat fakes moved into
              train to break the "real = OpenSLR only" shortcut — the CS-FLEURS real speaker is then shared)
  - English: seeded ASVspoof 2019 LA sample (replay, to keep English working)
  - Own    : own_recordings/{real,fake}/ (80 % train, 20 % dev), if present
Augmentation: random 4 s crop, random gain, light Gaussian noise (both classes alike).

Selection: mean EER over the dev groups (Hindi, Hinglish, English). Epoch -1 = the zero-shot model.
ACCEPTANCE: the fine-tuned weights are only written to runs/multilingual/antideepfake_finetuned/ (which the server
loads) if they beat the zero-shot model on dev. Otherwise they go to ..._rejected/ and the server keeps zero-shot.

Usage: python scripts/finetune_antideepfake.py [--epochs 4] [--top-k 4]
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

from ml.data.audio import load_audio  # noqa: E402
from ml.evaluation.metrics import eer, rates_at_threshold, roc_auc  # noqa: E402
from ml.models.antideepfake import AntiDeepfakeDetector  # noqa: E402
from ml.models.lfcc_cnn import spoof_logit  # noqa: E402

DATA = ROOT / "data"
SR = 16000
CROP = 4 * SR
EVAL_MAX = 8 * SR
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".webm", ".flac", ".mp4", ".aac"}


def sample(df, n, seed):
    return pd.concat([g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby("label")])


def build_sets(seed: int, own_dir: Path, en_n: int):
    sets = {k: [] for k in ("train", "dev_hi", "dev_hing", "dev_en", "test_hi", "test_hing", "test_en")}
    hi = pd.read_csv(ROOT / "data/manifests/hindi_fleurs_mms.csv")
    sets["train"].append(hi[hi.split == "train"])
    sets["dev_hi"].append(hi[hi.split == "dev"])
    sets["test_hi"].append(hi[hi.split == "test"])

    hg = pd.read_csv(ROOT / "data/manifests/hinglish_pilot_v1.csv", low_memory=False)
    tr, dv, te = hg[hg.split == "train"], hg[hg.split == "dev"], hg[hg.split == "test"]
    movable = te[((te.source_dataset == "cs_fleurs") & (te.label == "bonafide")) | (te.generator == "mms_concat_csfleurs")]
    moved = pd.concat([g.sample(frac=0.5, random_state=seed) for _, g in movable.groupby(["label", "generator"])])
    sets["train"].append(pd.concat([tr, moved]))
    sets["dev_hing"].append(dv)
    sets["test_hing"].append(te.drop(moved.index))

    la = pd.read_csv(ROOT / "data/manifests/vaanishield_v1_full.csv", low_memory=False)
    la = la[la.source_dataset == "asvspoof2019_la"]
    sets["train"].append(sample(la[la.split == "train"], en_n, seed))
    sets["dev_en"].append(sample(la[la.split == "dev"], 150, seed))
    sets["test_en"].append(sample(la[la.split == "test"], 500, seed))

    rows = []
    for lab, sub in (("bonafide", "real"), ("spoof", "fake")):
        for f in sorted((own_dir / sub).glob("*")):
            if f.suffix.lower() in AUDIO_EXT:
                rows.append({"clip_id": f"own_{sub}_{f.stem}", "path": str(f.resolve()), "label": lab,
                             "source_dataset": "own_recordings", "generator": "own"})
    own = pd.DataFrame(rows)
    if len(own) >= 4:
        dev_part = pd.concat([g.sample(frac=0.2, random_state=seed) for _, g in own.groupby("label")])
        sets["train"].append(own.drop(dev_part.index))
        sets["dev_own"] = [dev_part]
        print(f"own recordings: {len(own)} ({(own.label == 'bonafide').sum()} real); {len(dev_part)} kept for dev")
    return {k: pd.concat(v, ignore_index=True) for k, v in sets.items() if v}


def load_waves(df):
    return [load_audio(DATA / p) for p in df.path]


@torch.no_grad()
def score(model, waves, device):
    model.eval()
    out = []
    for w in waves:
        x = torch.from_numpy(w[:EVAL_MAX])[None].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out.append(float(spoof_logit(model(x)).float()))
    return np.array(out)


def summary(s, y, thr):
    r = rates_at_threshold(s, y, thr)
    return {"n": int(len(y)), "eer": eer(s, y)[0], "auc": roc_auc(s, y), "balanced_accuracy": 1 - r["hter"],
            "fake_caught": r["recall"], "real_flagged": r["false_alarm"]}


def augment(w, rng):
    if w.size > CROP:
        s = int(rng.integers(0, w.size - CROP + 1))
        w = w[s : s + CROP]
    w = w * float(10 ** (rng.uniform(-6, 6) / 20))
    if rng.random() < 0.5:
        w = w + rng.normal(0, rng.uniform(0.001, 0.01), w.size).astype(np.float32)
    return w.astype(np.float32)


def pad_batch(ws):
    n = max(w.size for w in ws)
    return torch.from_numpy(np.stack([np.pad(w, (0, n - w.size), mode="wrap") for w in ws]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--en-n", type=int, default=250, help="English replay clips per class")
    ap.add_argument("--own", default=str(ROOT / "own_recordings"))
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sets = build_sets(args.seed, Path(args.own), args.en_n)
    print({k: f"{len(v)} ({(v.label == 'spoof').sum()} fake)" for k, v in sets.items()}, flush=True)
    t0 = time.time()
    waves = {k: load_waves(v) for k, v in sets.items()}
    labels = {k: (v.label.values == "spoof").astype(int) for k, v in sets.items()}
    print(f"audio loaded in {time.time() - t0:.0f}s", flush=True)
    dev_keys = [k for k in sets if k.startswith("dev_")]

    model = AntiDeepfakeDetector("mms-300m").to(device)
    n_layers = len(model.encoder.encoder.layers)
    train_prefixes = tuple([f"encoder.encoder.layers.{i}." for i in range(n_layers - args.top_k, n_layers)]
                           + ["encoder.encoder.layer_norm.", "head."])
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(train_prefixes)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M "
          f"(top {args.top_k} layers + head)", flush=True)
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")], "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.lr_head}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    y_tr = labels["train"]
    cw = torch.tensor([len(y_tr) / (2 * (y_tr == 0).sum()), len(y_tr) / (2 * (y_tr == 1).sum())], dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=cw)

    def dev_eval():
        res = {k: eer(score(model, waves[k], device), labels[k])[0] for k in dev_keys}
        return float(np.mean(list(res.values()))), res

    def snapshot():
        return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n in trainable}

    base, base_res = dev_eval()
    print(f"epoch -1 (zero-shot)  mean dev EER {base * 100:.2f}%  " + "  ".join(f"{k} {v * 100:.1f}%" for k, v in base_res.items()), flush=True)
    history = [{"epoch": -1, "mean_dev_eer": base, **base_res}]
    best = (base, -1, snapshot())

    for ep in range(args.epochs):
        model.train()
        model.encoder.eval()  # keep dropout/LN behaviour of the frozen part deterministic
        t0, tot = time.time(), 0.0
        order = rng.permutation(len(y_tr))
        for i in range(0, len(order), args.batch):
            idx = order[i : i + args.batch]
            x = pad_batch([augment(waves["train"][j], rng) for j in idx]).to(device)
            y = torch.from_numpy(y_tr[idx]).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = lossf(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(idx)
        m, res = dev_eval()
        history.append({"epoch": ep, "train_loss": tot / len(y_tr), "mean_dev_eer": m, **res})
        print(f"epoch {ep}  loss {tot / len(y_tr):.4f}  mean dev EER {m * 100:.2f}%  "
              + "  ".join(f"{k} {v * 100:.1f}%" for k, v in res.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
        if m < best[0]:
            best = (m, ep, snapshot())

    accepted = best[1] >= 0
    model.load_state_dict(best[2], strict=False)
    dev_s = np.concatenate([score(model, waves[k], device) for k in dev_keys])
    dev_y = np.concatenate([labels[k] for k in dev_keys])
    thr = eer(dev_s, dev_y)[1] if accepted else 0.0
    out = {"model": "AntiDeepfake MMS-300M " + ("fine-tuned (VaaniShield)" if accepted else "zero-shot (fine-tune rejected)"),
           "accepted": accepted, "best_epoch": best[1], "threshold_spoof_logit": thr, "history": history,
           "trainable": f"top {args.top_k} layers + head", "train_size": int(len(y_tr))}
    name_map = {"test_hi": "hindi_test", "test_hing": "hing_test", "test_en": "en_test"}
    for k, nm in name_map.items():
        s = score(model, waves[k], device)
        out[nm] = summary(s, labels[k], thr)
        print(f"{nm}: EER {out[nm]['eer'] * 100:.2f}%  AUC {out[nm]['auc']:.4f}  balanced acc "
              f"{out[nm]['balanced_accuracy'] * 100:.1f}%  fake caught {out[nm]['fake_caught'] * 100:.0f}%  "
              f"real flagged {out[nm]['real_flagged'] * 100:.0f}%", flush=True)
    d = ROOT / "runs/multilingual" / ("antideepfake_finetuned" if accepted else "antideepfake_finetuned_rejected")
    d.mkdir(parents=True, exist_ok=True)
    if accepted:
        torch.save(best[2], d / "best_head_and_top.pth")
    (d / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(("ACCEPTED" if accepted else "REJECTED (zero-shot was better on dev)") + f" -> {d}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
