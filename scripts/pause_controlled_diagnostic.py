#!/usr/bin/env python
"""Pause-controlled diagnostic (team decision D-H4): how much does a detector depend on pause structure?

    python scripts/pause_controlled_diagnostic.py [--seeds 2026 2027 2028]

DIAGNOSTIC — NOT FINAL RESULTS. Tiny LFCC-CNN (14k parameters) trained on CPU on the 200-clip v2.1 train split.
Design (2 x 2, per seed):
  train on  pauses kept (primary v2.1)   | pauses removed (diag copy, ml.data.audio.remove_pauses)
  test on   pauses kept                  | pauses removed          (same test clips, same speakers)
Dependence on pauses = EER(train kept, test removed) - EER(train kept, test kept).
Thresholds come from each model's own dev split; nothing is fitted on test.
Also reports the B-probes (ml.data.hindi_gate.diagnostics) on both versions.
Writes runs/diag/pause_controlled/summary.json and summary.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml.data.hindi_gate import _file_level, diagnostics
from ml.data.manifest import read_manifest
from ml.data.provenance import DataPaths
from ml.evaluation.evaluate import evaluate_checkpoint
from ml.evaluation.metrics import eer

BANNER = "DIAGNOSTIC — NOT FINAL RESULTS"
OUT = Path("runs/diag/pause_controlled")
TRAIN = {"pauses_kept": "ml/configs/diag_hindi_v2_1_lfcc_pauses_kept.yaml",
         "pauses_removed": "ml/configs/diag_hindi_v2_1_lfcc_pauses_removed.yaml"}
TEST = {"pauses_kept": "data/manifests/hindi_v2_1_pilot.csv", "pauses_removed": "data/manifests/hindi_v2_1_pilot_pauses_removed.csv"}


def eer_by_family(scores: pd.DataFrame, manifest: str) -> dict:
    fam = {r.clip_id: r.generator_family for r in read_manifest(Path(manifest))}
    s = scores.assign(family=scores.clip_id.map(fam))
    out = {"all": eer(s.spoof_logit, s.label)[0]}
    for f in ("xtts", "freevc"):
        sub = s[(s.label == 0) | (s.family == f)]
        if sub.label.nunique() == 2:
            out[f] = eer(sub.spoof_logit, sub.label)[0]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    args = ap.parse_args()
    (OUT / "configs").mkdir(parents=True, exist_ok=True)
    rows = []
    for tr, template in TRAIN.items():
        base = yaml.safe_load(Path(template).read_text(encoding="utf-8"))
        for seed in args.seeds:
            cfg = {**base, "exp_id": f"{base['exp_id']}_s{seed}", "train": {**base["train"], "seed": seed}}
            cpath = OUT / "configs" / f"{cfg['exp_id']}.yaml"
            cpath.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            ckpt = Path(cfg["output"]["checkpoint_dir"]) / cfg["exp_id"] / "best.pt"
            if not ckpt.exists():  # re-runs reuse finished models (training is deterministic on CPU)
                subprocess.run([sys.executable, "-m", "ml.training.train", "--config", str(cpath)], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for te, manifest in TEST.items():
                out_dir = OUT / "eval" / cfg["exp_id"] / f"test_{te}"
                if not (out_dir / "test_scores.csv").exists():
                    evaluate_checkpoint(ckpt, "test", manifest=manifest, out_dir=out_dir)
                fam = eer_by_family(pd.read_csv(out_dir / "test_scores.csv"), manifest)
                rows.append({"train": tr, "test": te, "seed": seed, **{f"eer_{k}": round(v, 4) for k, v in fam.items()}})
                print(f"  train {tr:14s} seed {seed} test {te:14s} EER all {fam['all']:.3f} xtts {fam.get('xtts', float('nan')):.3f} "
                      f"freevc {fam.get('freevc', float('nan')):.3f}", flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby(["train", "test"])[["eer_all", "eer_xtts", "eer_freevc"]].agg(["mean", "std"]).round(3)
    dep = {tr: {k: round(float(df[(df.train == tr) & (df.test == "pauses_removed")][k].mean()
                                 - df[(df.train == tr) & (df.test == "pauses_kept")][k].mean()), 3)
                for k in ("eer_all", "eer_xtts", "eer_freevc")} for tr in TRAIN}
    root = DataPaths.resolve().data_root
    probes = {}
    for te, manifest in TEST.items():
        rws = [r for r in read_manifest(Path(manifest)) if r.language == "hi"]
        d = diagnostics(rws, root, _file_level(rws, root))
        probes[te] = {k: {"worst_eer": v["worst_eer"], "reading": v["reading"], "cue_type": v["cue_type"],
                          "top": [(t["feature"], t["sep_auc"]) for t in v["top_single"]]} for k, v in d.items()}
    flat = {f"{a}|{b}": {f"{m}_{s}": float(x) for (m, s), x in v.items()} for (a, b), v in agg.to_dict("index").items()}
    summary = {"label": BANNER, "seeds": args.seeds, "runs": rows, "mean_std": flat,
               "pause_dependence_delta_eer": dep, "probes": probes}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    md = [f"# Pause-controlled diagnostic\n\n> **{BANNER}.** LFCC-CNN, 200 training clips, {len(args.seeds)} seeds, CPU.\n",
          "| train | test | EER all (mean ± sd) | EER seen XTTS | EER held-out FreeVC |", "|---|---|---|---|---|"]
    for (a, b), v in agg.to_dict("index").items():
        md.append(f"| {a} | {b} | {v[('eer_all', 'mean')]:.3f} ± {v[('eer_all', 'std')]:.3f} | {v[('eer_xtts', 'mean')]:.3f} ± {v[('eer_xtts', 'std')]:.3f} "
                  f"| {v[('eer_freevc', 'mean')]:.3f} ± {v[('eer_freevc', 'std')]:.3f} |")
    md += ["", f"Pause dependence (EER test-pauses-removed minus test-pauses-kept): {dep}", ""]
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(json.dumps(dep))
    print(json.dumps(probes, indent=1))
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    raise SystemExit(main())
