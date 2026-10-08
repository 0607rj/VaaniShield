"""Robustness evaluation: models x {clean, simulated codecs} x {en, hi, hi-en} on a fixed test subset.

    python -m ml.evaluation.robustness --config ml/configs/robustness_smoke.yaml [--device cuda] [--rescore]

Steps (all deterministic):
  1. subset   n_per_class real + spoof TEST clips per language pool -> subset manifest
  2. codecs   ml.data.build_codec_variants on that subset (clean parents + sim:<condition> variants)
  3. leakage  no evaluated clip (sha256) or speaker group may occur in any model's train/dev manifest rows
  4. scoring  every model scores every row via the inference path (windowed mean); cached per checkpoint sha256
  5. report   metrics per model x slice (EER, AUC, dev-threshold false alarm / miss / HTER / F1, ECE, speaker-bootstrap
              CI for EER), CSV + JSON + Markdown + PNG plots in output_dir
Thresholds are each model's own DEV threshold stored in its checkpoint; nothing is fitted on these test clips.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

from ml.data.manifest import read_manifest, write_manifest  # noqa: E402
from ml.data.provenance import DataPaths  # noqa: E402
from ml.evaluation.evaluate import evaluate_frame  # noqa: E402
from ml.evaluation.metrics import det_curve, eer, group_bootstrap  # noqa: E402
from ml.training.repro import sha256_file  # noqa: E402

BANNER = "SMOKE / DEVELOPMENT RESULTS — NOT FINAL RESULTS"
LANGS = ("en", "hi", "hi-en")


class Pool(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: str
    manifest: str


class ModelRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    checkpoint: str


class RobustCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    seed: int = 2026
    n_per_class: int
    pools: list[Pool]
    subset_manifest: str
    codec_config: str
    models: list[ModelRef]
    device: str = "cpu"
    bootstrap_n: int = 200
    output_dir: str


def build_subset(cfg: RobustCfg) -> list:
    rng = np.random.default_rng(cfg.seed)
    out = []
    for pool in cfg.pools:
        rows = sorted((r for r in read_manifest(Path(pool.manifest)) if r.split == "test" and r.language == pool.language
                       and r.transform == "clean"), key=lambda r: r.clip_id)
        for lab in ("bonafide", "spoof"):
            cand = [r for r in rows if r.label == lab]
            if len(cand) < cfg.n_per_class:
                raise SystemExit(f"{pool.language}: only {len(cand)} {lab} test clips in {pool.manifest}")
            out += [cand[i] for i in sorted(rng.choice(len(cand), cfg.n_per_class, replace=False))]
    if len({r.clip_id for r in out}) != len(out):
        raise SystemExit("clip_id collision across pools")
    return out


def leakage(eval_rows: list, cfg: RobustCfg) -> dict:
    """Conservative: compare with ALL train+dev rows of each model's training manifest (a superset of what it saw)."""
    import torch

    ev_sha, ev_spk = {r.sha256 for r in eval_rows}, {r.speaker_group for r in eval_rows}
    res = {}
    for m in cfg.models:
        man = torch.load(m.checkpoint, map_location="cpu", weights_only=True)["manifest"]
        seen = [r for r in read_manifest(Path(man)) if r.split in ("train", "dev")]
        res[m.name] = {"training_manifest": man, "clip_overlap": len(ev_sha & {r.sha256 for r in seen}),
                       "speaker_overlap": sorted(ev_spk & {r.speaker_group for r in seen})}
    return res


def score_all(cfg: RobustCfg, rows: list, root: Path, out: Path, device: str, rescore: bool) -> pd.DataFrame:
    from ml.inference.predict import load_checkpoint, score_rows

    frames = []
    for m in cfg.models:
        f, info = out / "scores" / f"{m.name}.csv", out / "scores" / f"{m.name}.json"
        sha = sha256_file(Path(m.checkpoint))
        ids = sorted(r.clip_id for r in rows)
        cached = (not rescore and f.exists() and info.exists()
                  and json.loads(info.read_text())["checkpoint_sha256"] == sha and json.loads(info.read_text())["clip_ids_n"] == len(ids))
        if cached:
            df = pd.read_csv(f)
        else:
            model, ckpt = load_checkpoint(m.checkpoint, device)
            print(f"  scoring {m.name} ({ckpt['model_kind']}) on {len(rows)} clips ...", flush=True)
            df = score_rows(model, ckpt, rows, root)
            df["threshold_p_spoof"] = ckpt["threshold_p_spoof"]
            f.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(f, index=False)
            info.write_text(json.dumps({"checkpoint": m.checkpoint, "checkpoint_sha256": sha, "clip_ids_n": len(ids),
                                        "model_kind": ckpt["model_kind"], "n_train": ckpt.get("n_train"),
                                        "trained_on": ckpt["manifest"], "threshold_p_spoof": ckpt["threshold_p_spoof"]}, indent=2))
        frames.append(df.assign(model=m.name))
    return pd.concat(frames, ignore_index=True)


def slices(meta: pd.DataFrame) -> dict[str, pd.Series]:
    t = meta["transform"]
    sl = {"clean (all languages)": t == "clean", "codec (all sim, pooled)": t != "clean"}
    for lang in LANGS:
        sl[f"{lang} clean"] = (meta.language == lang) & (t == "clean")
        sl[f"{lang} codec"] = (meta.language == lang) & (t != "clean")
    for cond in sorted(t[t != "clean"].unique()):
        sl[cond] = t == cond
    return sl


def report(cfg: RobustCfg, scores: pd.DataFrame, meta: pd.DataFrame, out: Path) -> pd.DataFrame:
    df = scores.merge(meta[["clip_id", "language", "transform", "speaker_group", "source_dataset", "parent_clip_id"]], on="clip_id")
    rows = []
    for name, g in df.groupby("model", sort=False):
        thr_p = float(g.threshold_p_spoof.iloc[0])
        thr_logit = math.log(thr_p / (1 - thr_p)) if 0 < thr_p < 1 else (math.inf if thr_p >= 1 else -math.inf)
        for sname, mask in slices(meta.set_index("clip_id").loc[g.clip_id].reset_index()).items():
            s = g[mask.to_numpy()]
            if s.label.nunique() < 2:
                continue
            m = evaluate_frame(s[["clip_id", "label", "spoof_logit", "p_spoof"]], thr_logit)
            try:
                ci = group_bootstrap(s, lambda d: eer(d.spoof_logit, d.label)[0], "speaker_group", n=cfg.bootstrap_n, seed=cfg.seed)
                lo, hi = ci.get("lo"), ci.get("hi")
            except ValueError:
                lo = hi = None
            a = m["at_threshold"]
            rows.append({"model": name, "slice": sname, "n": m["n"], "n_spoof": m["n_spoof"], "eer": round(m["eer"], 3),
                         "eer_ci95_lo": lo, "eer_ci95_hi": hi, "auc": round(m["auc"], 3), "false_alarm": round(a["false_alarm"], 3),
                         "miss": round(a["miss"], 3), "hter": round(a["hter"], 3), "f1": round(a["f1"], 3),
                         "ece": round(m["calibration"]["ece_equal_mass_15"], 3), "threshold_p_spoof": round(thr_p, 4)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "metrics.csv", index=False)
    (out / "metrics.json").write_text(json.dumps({"label": BANNER, "rows": res.to_dict("records")}, indent=2, default=str))
    plots(res, df, out)
    return res


def _stamp(fig):
    fig.text(0.5, 0.005, BANNER, ha="center", va="bottom", fontsize=9, color="crimson", weight="bold")


def plots(res: pd.DataFrame, df: pd.DataFrame, out: Path) -> None:
    models = list(dict.fromkeys(res.model))
    # 1. heatmap: EER per model x headline slice
    head = ["clean (all languages)", "codec (all sim, pooled)"] + [f"{l} {k}" for l in LANGS for k in ("clean", "codec")]
    piv = res.pivot(index="model", columns="slice", values="eer").reindex(index=models, columns=head)
    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.6 * len(models)))
    im = ax.imshow(piv.to_numpy(dtype=float), cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(piv.to_numpy(dtype=float)):
        ax.text(j, i, "" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(head)), head, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)), models, fontsize=8)
    ax.set_title("EER by model and condition (0.5 = chance)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); _stamp(fig)
    fig.savefig(out / "eer_heatmap.png", dpi=120); plt.close(fig)
    # 2. EER per codec condition, one line per model
    conds = [s for s in res.slice.unique() if not any(s.startswith(p) for p in ("clean", "codec", "en ", "hi ", "hi-en "))]
    fig, ax = plt.subplots(figsize=(10, 4))
    for mname in models:
        r = res[res.model == mname].set_index("slice")
        ax.plot(["clean (all languages)"] + conds, [r.eer.get("clean (all languages)")] + [r.eer.get(c) for c in conds], "o-", label=mname)
    ax.axhline(0.5, ls="--", color="grey", lw=1)
    ax.set_ylabel("EER"); ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=30, labelsize=8); ax.legend(fontsize=8)
    ax.set_title("EER clean vs each simulated codec (all languages pooled)", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); _stamp(fig)
    fig.savefig(out / "eer_by_codec.png", dpi=120); plt.close(fig)
    # 3. ROC per language (clean), one panel per language
    fig, axes = plt.subplots(1, len(LANGS), figsize=(12, 4))
    for ax, lang in zip(axes, LANGS, strict=True):
        for mname in models:
            s = df[(df.model == mname) & (df.language == lang) & (df["transform"] == "clean")]
            if s.label.nunique() == 2:
                fa, miss, _ = det_curve(s.spoof_logit, s.label)
                ax.step(fa, 1 - miss, where="post", label=mname)
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.set_title(f"{lang} clean (n={len(s)})", fontsize=9); ax.set_xlabel("false alarm"); ax.set_ylabel("spoof detected")
    axes[0].legend(fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); _stamp(fig)
    fig.savefig(out / "roc_by_language.png", dpi=120); plt.close(fig)
    # 4. score shift: codec variant logit minus its clean parent logit, per condition and class
    par = df[df["transform"] == "clean"][["model", "clip_id", "spoof_logit"]].rename(columns={"clip_id": "parent_clip_id", "spoof_logit": "parent_logit"})
    d = df[df["transform"] != "clean"].merge(par, on=["model", "parent_clip_id"])
    d["shift"] = d.spoof_logit - d.parent_logit
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 2.2 * len(models)), sharex=True)
    for ax, mname in zip(np.atleast_1d(axes), models, strict=True):
        s = d[d.model == mname]
        cs = sorted(s["transform"].unique())
        for k, (lab, col) in enumerate(((0, "tab:green"), (1, "tab:red"))):
            data = [s[(s["transform"] == c) & (s.label == lab)]["shift"].to_numpy() for c in cs]
            ax.boxplot(data, positions=np.arange(len(cs)) + (k - 0.5) * 0.35, widths=0.3, patch_artist=True,
                       boxprops={"facecolor": col, "alpha": 0.5}, showfliers=False)
        ax.axhline(0, color="grey", lw=1); ax.set_ylabel(mname, fontsize=7)
        ax.set_xticks(range(len(cs)), [c.removeprefix("sim:") for c in cs], rotation=30, fontsize=7)
    np.atleast_1d(axes)[0].set_title("Spoof-logit shift caused by each codec (variant − clean parent); green = real, red = spoof", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); _stamp(fig)
    fig.savefig(out / "codec_score_shift.png", dpi=120); plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--device")
    ap.add_argument("--data-root")
    ap.add_argument("--rescore", action="store_true", help="ignore cached score files")
    args = ap.parse_args(argv)
    cfg = RobustCfg(**yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = DataPaths.resolve(args.data_root).data_root

    sub = build_subset(cfg)
    write_manifest(sub, Path(cfg.subset_manifest))
    print(f"[1] subset: {len(sub)} clean test clips -> {cfg.subset_manifest}")
    from ml.data.build_codec_variants import load_cfg
    from ml.data.build_codec_variants import main as build_codecs

    build_codecs(["--config", cfg.codec_config] + (["--data-root", args.data_root] if args.data_root else []))
    codec_manifest = load_cfg(cfg.codec_config)[0].output.manifest
    rows = read_manifest(Path(codec_manifest))
    print(f"[2] codec manifest: {len(rows)} rows ({sum(r.transform == 'clean' for r in rows)} clean)")
    leak = leakage(rows, cfg)
    print(f"[3] leakage vs training manifests: {leak}")
    if any(v["clip_overlap"] or v["speaker_overlap"] for v in leak.values()):
        raise SystemExit("evaluation clips/speakers overlap a model's train/dev data; fix the subset")
    scores = score_all(cfg, rows, root, out, args.device or cfg.device, args.rescore)
    meta = pd.DataFrame([r.model_dump() for r in rows])
    res = report(cfg, scores, meta, out)
    head = res[res.slice.isin(["clean (all languages)", "codec (all sim, pooled)"] + [f"{l} {k}" for l in LANGS for k in ("clean", "codec")])]
    summary = head.pivot(index="slice", columns="model", values="eer").reindex(columns=[m.name for m in cfg.models])
    md = [f"# {cfg.run_id}", "", f"> **{BANNER}.** Tiny subsets ({cfg.n_per_class} real + {cfg.n_per_class} spoof clean test clips per "
          "language) and models trained on 8–60 clips. These numbers verify that evaluation, metrics, plots and outputs work. "
          "They say nothing about which model or condition is better.", "",
          f"Models: {', '.join(f'{m.name} (`{m.checkpoint}`)' for m in cfg.models)}", "",
          f"Data: `{codec_manifest}`; leakage check: {json.dumps(leak)}", "", "## EER (headline slices)", "",
          "| slice | " + " | ".join(summary.columns) + " |", "|---|" + "---|" * len(summary.columns)]
    md += [f"| {s} | " + " | ".join("" if pd.isna(v) else f"{v:.2f}" for v in summary.loc[s]) + " |" for s in summary.index]
    md += ["", "Full table (per codec, CIs, false alarm / miss at dev threshold, ECE): `metrics.csv`.",
           "Plots: `eer_heatmap.png`, `eer_by_codec.png`, `roc_by_language.png`, `codec_score_shift.png`.", ""]
    (out / "README.md").write_text("\n".join(md), encoding="utf-8")
    (out / "run_info.json").write_text(json.dumps({"label": BANNER, "config": cfg.model_dump(), "config_file": args.config,
                                                   "codec_manifest": codec_manifest, "n_rows": len(rows), "leakage": leak}, indent=2))
    print(f"[5] {BANNER}\n{summary.round(2).to_string()}\noutputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
