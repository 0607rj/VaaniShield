"""Validate the ML vertical slice: data integrity, leakage, checkpoint, score direction, reproducibility, plots.

Usage:
    python -m ml.evaluation.validate --checkpoint runs/B0_lfcc_cnn_pilot/best.pt \
        --repro runs/repro_1/B0_lfcc_cnn_pilot runs/repro_2/B0_lfcc_cnn_pilot

Writes <checkpoint dir>/validation/{validation.json, confusion_matrix_test.png, roc_test.png}.
Test metrics are computed from the existing test_scores.csv (the test split is not re-scored here).
Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from ml.data.audio import CROP_SAMPLES, crop_or_pad, preprocess  # noqa: E402
from ml.data.manifest import ManifestRow, read_manifest  # noqa: E402
from ml.data.provenance import DataPaths  # noqa: E402
from ml.evaluation.metrics import det_curve, eer, rates_at_threshold, roc_auc  # noqa: E402
from ml.inference.predict import load_checkpoint, predict_file, score_rows  # noqa: E402
from ml.training.repro import item_rng, sha256_file, sha256_manifest  # noqa: E402

SPLITS = ("train", "dev", "test")


def _result(ok: bool, **details) -> dict:
    return {"ok": bool(ok), **details}


# 1 ------------------------------------------------------------------------------------------------------------
def check_preprocessing(rows: list[ManifestRow], root: Path) -> dict:
    mismatches = []
    for r in rows:
        a, b = preprocess(root / r.path), preprocess(root / r.path)
        same = a.dtype == b.dtype and np.array_equal(a, b)
        same &= np.array_equal(crop_or_pad(a), crop_or_pad(b))  # eval crop
        same &= np.array_equal(crop_or_pad(a, rng=item_rng(2026, 3, r.clip_id)), crop_or_pad(b, rng=item_rng(2026, 3, r.clip_id)))
        if not same:
            mismatches.append(r.clip_id)
    return _result(not mismatches, n_files=len(rows), mismatches=mismatches,
                   rule="preprocess twice -> bit-identical; eval crop and seeded train crop identical")


# 2 ------------------------------------------------------------------------------------------------------------
def check_labels(rows: list[ManifestRow], root: Path) -> dict:
    """Re-derive label/speaker/system from the official ASVspoof protocol and re-hash every audio file."""
    proto = root / "raw/asvspoof2019_la/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    official = {}
    for line in proto.read_text(encoding="utf-8").splitlines():
        spk, utt, _, system, key = line.split()
        official[utt] = (key, f"asvspoof19:{spk}", system)
    problems = []
    for r in rows:
        if r.clip_id not in official:
            problems.append(f"{r.clip_id}: not in official protocol")
            continue
        key, spk, system = official[r.clip_id]
        expected_gen = system if key == "spoof" else "none"
        if (r.label, r.speaker_id, r.generator) != (key, spk, expected_gen):
            problems.append(f"{r.clip_id}: manifest {(r.label, r.speaker_id, r.generator)} != protocol {(key, spk, expected_gen)}")
        if sha256_file(root / r.path) != r.sha256:
            problems.append(f"{r.clip_id}: file sha256 differs from manifest")
    return _result(not problems, n_checked=len(rows), problems=problems, label_counts=dict(Counter(r.label for r in rows)))


# 3 ------------------------------------------------------------------------------------------------------------
def check_split(rows: list[ManifestRow]) -> dict:
    counts = {sp: dict(Counter(r.label for r in rows if r.split == sp)) for sp in SPLITS}
    missing = [r.clip_id for r in rows if r.split not in SPLITS]
    both = all(set(counts[sp]) == {"bonafide", "spoof"} for sp in SPLITS)
    return _result(not missing and both and sum(sum(c.values()) for c in counts.values()) == len(rows),
                   counts=counts, rows_without_split=missing)


# 4 ------------------------------------------------------------------------------------------------------------
def check_speaker_leakage(rows: list[ManifestRow]) -> dict:
    groups = {sp: {r.speaker_group for r in rows if r.split == sp} for sp in SPLITS}
    overlaps = {f"{a}-{b}": sorted(groups[a] & groups[b]) for a, b in combinations(SPLITS, 2) if groups[a] & groups[b]}
    return _result(not overlaps, speakers_per_split={k: len(v) for k, v in groups.items()}, overlaps=overlaps)


# 5 ------------------------------------------------------------------------------------------------------------
def check_source_leakage(rows: list[ManifestRow]) -> dict:
    """A source that supplies only one class (overall or within a split) lets a model learn 'which corpus' instead of
    'real vs fake'. Also report the spoof generator families per split (seen/unseen information)."""
    table = Counter((r.split, r.source_dataset, r.label) for r in rows)
    one_class = []
    for src in {r.source_dataset for r in rows}:
        if len({r.label for r in rows if r.source_dataset == src}) < 2:
            one_class.append(f"{src}: single class overall")
        for sp in SPLITS:
            labs = {r.label for r in rows if r.source_dataset == src and r.split == sp}
            if len(labs) == 1:
                one_class.append(f"{src} in {sp}: only {labs.pop()}")
    families = {sp: dict(Counter(r.generator_family for r in rows if r.split == sp and r.label == "spoof")) for sp in SPLITS}
    unseen = {sp: sorted(set(families[sp]) - set(families["train"])) for sp in ("dev", "test")}
    return _result(not one_class, source_label_counts={"|".join(k): v for k, v in sorted(table.items())},
                   single_class_sources=one_class, spoof_families_per_split=families, families_unseen_in_train=unseen)


# 6 ------------------------------------------------------------------------------------------------------------
def check_duplicates(rows: list[ManifestRow], root: Path) -> dict:
    """Exact duplicates by file sha256 and by decoded+preprocessed audio content (catches re-encoded copies of the same samples)."""
    def dup_groups(key_of):
        seen: dict[str, list[ManifestRow]] = {}
        for r in rows:
            seen.setdefault(key_of(r), []).append(r)
        return [g for g in seen.values() if len(g) > 1]

    file_dups = dup_groups(lambda r: r.sha256)
    audio_dups = dup_groups(lambda r: hashlib.sha256(preprocess(root / r.path).tobytes()).hexdigest())
    cross = [[(r.clip_id, r.split) for r in g] for g in file_dups + audio_dups if len({r.split for r in g}) > 1]
    return _result(not cross, file_duplicate_groups=len(file_dups), audio_duplicate_groups=len(audio_dups), cross_split=cross,
                   note="exact duplicates only; near-duplicates (same text/speaker, different recording) not checked")


# 7 ------------------------------------------------------------------------------------------------------------
def check_descendants(rows: list[ManifestRow]) -> dict:
    by_id = {r.clip_id: r for r in rows}
    derived = [r for r in rows if r.parent_clip_id or r.transform != "clean"]
    bad = []
    external: dict[str, set[str]] = {}  # parent not a manifest row (e.g. a long recording cut into segments)
    for r in derived:
        if not r.parent_clip_id:
            bad.append(f"{r.clip_id}: transform {r.transform!r} without parent_clip_id")
            continue
        parent = by_id.get(r.parent_clip_id)
        if parent is None:
            external.setdefault(r.parent_clip_id, set()).add(r.split)
        elif parent.split != r.split:
            bad.append(f"{r.clip_id} ({r.split}) derived from {parent.clip_id} ({parent.split})")
    for pid, splits in external.items():  # all children of one external parent must share a split
        if len(splits) > 1:
            bad.append(f"children of external parent {pid} are in several splits: {sorted(splits)}")
    return _result(not bad, n_derived_rows=len(derived), n_external_parents=len(external), problems=bad,
                   note="no augmented/transformed rows exist yet; check is in place for when they are added" if not derived else "")


# 8 ------------------------------------------------------------------------------------------------------------
def check_checkpoint(ckpt_path: Path, manifest: Path) -> dict:
    m1, c1 = load_checkpoint(ckpt_path)
    m2, _ = load_checkpoint(ckpt_path)
    x = torch.randn(4, CROP_SAMPLES, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        same = torch.equal(m1(x), m2(x))
    required = {"model_kind", "model_config", "model_state", "threshold_p_spoof", "manifest_sha256", "config_sha256"}
    missing = sorted(required - set(c1))
    manifest_same = c1.get("manifest_sha256") == sha256_manifest(manifest)
    thr_ok = 0.0 < c1.get("threshold_p_spoof", -1) < 1.0
    return _result(same and not missing and manifest_same and thr_ok, reload_outputs_identical=same, missing_keys=missing,
                   manifest_unchanged_since_training=manifest_same, threshold_p_spoof=c1.get("threshold_p_spoof"),
                   best_epoch=c1.get("best_epoch"), safe_load="torch.load(weights_only=True)")


# 9 ------------------------------------------------------------------------------------------------------------
def check_score_direction(ckpt_path: Path, rows: list[ManifestRow], root: Path) -> dict:
    """Higher score must mean 'more spoof': on TRAIN data the fitted model must rank spoof above bona fide."""
    model, ckpt = load_checkpoint(ckpt_path)
    train = score_rows(model, ckpt, [r for r in rows if r.split == "train"], root)
    auc = roc_auc(train["spoof_logit"], train["label"])
    mean_sp = float(train.loc[train.label == 1, "p_spoof"].mean())
    mean_bf = float(train.loc[train.label == 0, "p_spoof"].mean())
    one = next(r for r in rows if r.split == "train")
    pred = predict_file(model, ckpt, root / one.path)
    consistent = (pred["label"] == "SPOOF") == (pred["p_spoof"] >= ckpt["threshold_p_spoof"])
    probs_ok = abs(pred["p_spoof"] + pred["p_real"] - 1) < 1e-3
    spread = float(train["p_spoof"].std())
    warning = "near-constant outputs (std p_spoof < 0.01): the model has not learned; metrics are not meaningful" if spread < 0.01 else ""
    return _result(auc > 0.5 and mean_sp > mean_bf and consistent and probs_ok, train_auc=auc,
                   train_mean_p_spoof={"spoof": mean_sp, "bonafide": mean_bf}, train_p_spoof_std=spread, warning=warning,
                   label_matches_threshold_rule=consistent, p_spoof_plus_p_real_is_1=probs_ok)


# 10 -----------------------------------------------------------------------------------------------------------
def check_reproducibility(run_dirs: list[Path], atol: float = 1e-6) -> dict:
    ref = run_dirs[0]
    ref_ckpt = torch.load(ref / "best.pt", map_location="cpu", weights_only=True)
    ref_hist = json.loads((ref / "history.json").read_text())
    ref_dev = pd.read_csv(ref / "dev_scores.csv")
    comparisons = []
    for d in run_dirs[1:]:
        ck = torch.load(d / "best.pt", map_location="cpu", weights_only=True)
        hist = json.loads((d / "history.json").read_text())
        dev = pd.read_csv(d / "dev_scores.csv")
        max_w = max(float((ref_ckpt["model_state"][k].float() - ck["model_state"][k].float()).abs().max()) for k in ref_ckpt["model_state"])
        max_loss = max(abs(a["train_loss"] - b["train_loss"]) for a, b in zip(ref_hist, hist, strict=True)) if len(ref_hist) == len(hist) else float("inf")
        max_dev = float((ref_dev["spoof_logit"] - dev["spoof_logit"]).abs().max()) if list(ref_dev.clip_id) == list(dev.clip_id) else float("inf")
        comparisons.append({"run": str(d), "same_n_epochs": len(ref_hist) == len(hist), "same_best_epoch": ref_ckpt["best_epoch"] == ck["best_epoch"],
                            "max_abs_weight_diff": max_w, "max_abs_train_loss_diff": max_loss, "max_abs_dev_logit_diff": max_dev,
                            "threshold_diff": abs(ref_ckpt["threshold_p_spoof"] - ck["threshold_p_spoof"])})
    ok = all(c["same_n_epochs"] and c["same_best_epoch"] and c["max_abs_weight_diff"] <= atol and c["max_abs_dev_logit_diff"] <= atol
             and c["max_abs_train_loss_diff"] <= atol and c["threshold_diff"] <= atol for c in comparisons)
    return _result(ok, reference=str(ref), tolerance=atol, comparisons=comparisons)


# report -------------------------------------------------------------------------------------------------------
def test_report(ckpt_path: Path, out: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    df = pd.read_csv(ckpt_path.parent / "test_scores.csv")
    y, p = df["label"].to_numpy(), df["p_spoof"].to_numpy()
    thr = ckpt["threshold_p_spoof"]
    e, _ = eer(p, y)
    auc = roc_auc(p, y)
    r = rates_at_threshold(p, y, thr)
    cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])

    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred REAL", "pred SPOOF"])
    ax.set_yticks([0, 1], ["true REAL", "true SPOOF"])
    ax.set_title(f"Test confusion matrix (n={len(df)})\nthreshold p_spoof >= {thr:.3f} (chosen on dev)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix_test.png", dpi=120)
    plt.close(fig)

    fa, miss, _ = det_curve(p, y)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.step(fa, 1 - miss, where="post", label=f"LFCC-CNN (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="chance")
    i = int(np.argmin(np.abs(fa - miss)))
    ax.plot(fa[i], 1 - miss[i], "o", color="red", label=f"EER {e:.3f}")
    ax.set_xlabel("False alarm rate (real flagged as spoof)")
    ax.set_ylabel("True spoof detection rate")
    ax.set_title(f"Test ROC (n={len(df)}; pipeline check only)", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "roc_test.png", dpi=120)
    plt.close(fig)
    return {"n": int(len(df)), "n_spoof": int(y.sum()), "eer": e, "roc_auc": auc, "f1_spoof_at_dev_threshold": r["f1"],
            "threshold_p_spoof": thr, "confusion_matrix": {"rows=true [REAL, SPOOF]": cm.tolist()},
            "false_alarm": r["false_alarm"], "miss": r["miss"], "plots": ["confusion_matrix_test.png", "roc_test.png"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate the VaaniShield vertical slice.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repro", nargs="*", default=[], help="extra run dirs trained with the same config")
    args = ap.parse_args(argv)
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    manifest = Path(ckpt["manifest"])
    rows = read_manifest(manifest)
    root = DataPaths.resolve().data_root
    out = ckpt_path.parent / "validation"
    out.mkdir(exist_ok=True)

    checks = {
        "1_deterministic_preprocessing": check_preprocessing(rows, root),
        "2_labels_match_official_protocol": check_labels(rows, root),
        "3_split_complete_and_both_classes": check_split(rows),
        "4_no_speaker_leakage": check_speaker_leakage(rows),
        "5_no_source_shortcut": check_source_leakage(rows),
        "6_no_duplicate_leakage": check_duplicates(rows, root),
        "7_no_derived_rows_crossing_splits": check_descendants(rows),
        "8_checkpoint_save_load": check_checkpoint(ckpt_path, manifest),
        "9_score_direction": check_score_direction(ckpt_path, rows, root),
    }
    if args.repro:
        checks["10_reproducibility_same_seed"] = check_reproducibility([ckpt_path.parent] + [Path(d) for d in args.repro])
    report = {"checks": checks, "test_report": test_report(ckpt_path, out)}
    (out / "validation.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    for name, c in checks.items():
        print(f"{'PASS' if c['ok'] else 'FAIL'}  {name}" + (f"   WARNING: {c['warning']}" if c.get("warning") else ""))
    t = report["test_report"]
    print(f"test: n={t['n']}  EER={t['eer']:.3f}  ROC-AUC={t['roc_auc']:.3f}  F1(spoof)={t['f1_spoof_at_dev_threshold']:.3f}  "
          f"confusion [[TN FP][FN TP]]={t['confusion_matrix']['rows=true [REAL, SPOOF]']}")
    print(f"written: {out}")
    return 0 if all(c["ok"] for c in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
