"""Leakage, shortcut and distribution checks for a canonical manifest.

Usage:
    python -m ml.data.check_manifest --manifest data/manifests/vaanishield_v1.csv

FAIL (exit 1) = hard leakage that invalidates evaluation: duplicates across splits, speaker overlap, derived rows in a
different split than their parent, generator inputs (cloned/VC source speakers) or TTS sentences crossing splits.
WARNING = shortcut risks a model could exploit (single-class sources, codec / sample-rate perfectly tied to the label);
these are reported with evidence and must be handled (or acknowledged) before training on the affected pool.
Writes <manifest>_report.json and <manifest>_report.md next to the manifest.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ml.data.manifest import ManifestRow, read_manifest
from ml.data.provenance import DataPaths
from ml.evaluation.validate import check_descendants, check_duplicates, check_speaker_leakage, check_split

SPLITS = ("train", "dev", "test")
EVAL_ONLY_SOURCES = {"openslr104_hien"}  # declared single-class, test-only sources (never trained on)


def generator_leakage(rows: list[ManifestRow]) -> dict:
    bona_splits: dict[str, set[str]] = defaultdict(set)  # every split holding REAL audio of each speaker
    for r in rows:
        if r.label == "bonafide":
            bona_splits[r.speaker_id].add(r.split)
    problems = []
    for r in rows:
        if r.label != "spoof":
            continue
        inputs = [r.speaker_id] + ([r.notes.split("vc_source=")[1]] if r.notes and "vc_source=" in r.notes else [])
        for spk in inputs:  # cloned voice / VC source speaker must not have bona fide audio in another split
            other = bona_splits.get(spk, set()) - {r.split}
            if other:
                problems.append(f"{r.clip_id} ({r.split}) uses speaker {spk} whose real audio is in {sorted(other)}")
    tx = defaultdict(set)
    for r in rows:
        if r.transcript_id:
            tx[r.transcript_id].add(r.split)
    cross_tx = [t for t, s in tx.items() if len(s) > 1]
    fam = {sp: dict(Counter(r.generator_family for r in rows if r.split == sp and r.label == "spoof")) for sp in SPLITS}
    unseen = {sp: sorted(set(fam[sp]) - set(fam["train"])) for sp in ("dev", "test")}
    return {"ok": not problems and not cross_tx, "generator_input_problems": problems[:20], "n_generator_input_problems": len(problems),
            "transcripts_crossing_splits": len(cross_tx), "spoof_families_per_split": fam, "families_unseen_in_train": unseen,
            "note": "unseen families in dev/test are by design for ASVspoof (official eval attacks); they are an evaluation condition, not leakage"}


def shortcut_risks(rows: list[ManifestRow]) -> dict:
    warnings = []
    labels_by_source, splits_by_source = defaultdict(set), defaultdict(set)
    for r in rows:
        labels_by_source[r.source_dataset].add(r.label)
        splits_by_source[r.source_dataset].add(r.split)
    eval_only = {s for s in EVAL_ONLY_SOURCES if splits_by_source.get(s) == {"test"}}  # only if really test-only here
    for src, labs in sorted(labels_by_source.items()):
        if len(labs) == 1:
            role = "declared eval-only" if src in eval_only else "USED FOR TRAINING"
            warnings.append(f"source {src} is single-class ({next(iter(labs))}); {role}")
    pools = defaultdict(list)
    for r in rows:
        if r.source_dataset not in eval_only:
            pools[r.language].append(r)
    confounds = {}
    for lang, pool in sorted(pools.items()):
        for scope, sub in (("", pool), (" train split", [r for r in pool if r.split == "train"])):  # train = what the model learns from
            for attr in ("codec", "sample_rate_orig", "source_dataset"):
                by_label = {lab: Counter(str(getattr(r, attr)) for r in sub if r.label == lab) for lab in ("bonafide", "spoof")}
                confounds[f"{lang}{scope.replace(' ', ':')}:{attr}"] = {k: dict(v) for k, v in by_label.items()}
                if by_label["bonafide"] and by_label["spoof"] and not (set(by_label["bonafide"]) & set(by_label["spoof"])):
                    warnings.append(f"language '{lang}'{scope}: {attr} perfectly separates the classes "
                                    f"(bonafide={dict(by_label['bonafide'])}, spoof={dict(by_label['spoof'])})")
    dur = {f"{lang}:{lab}": round(sum(r.duration_s or 0 for r in pool if r.label == lab) / max(1, sum(r.label == lab for r in pool)), 2)
           for lang, pool in pools.items() for lab in ("bonafide", "spoof")}
    return {"warnings": warnings, "attribute_by_label": confounds, "mean_duration_s": dur}


def distributions(rows: list[ManifestRow]) -> dict:
    return {
        "class_by_split": {sp: dict(Counter(r.label for r in rows if r.split == sp)) for sp in SPLITS},
        "class_by_source": {s: dict(Counter(r.label for r in rows if r.source_dataset == s)) for s in sorted({r.source_dataset for r in rows})},
        "language_by_split": {sp: dict(Counter(r.language for r in rows if r.split == sp)) for sp in SPLITS},
        "language_by_class": {lab: dict(Counter(r.language for r in rows if r.label == lab)) for lab in ("bonafide", "spoof")},
        "source_by_split": {sp: dict(Counter(r.source_dataset for r in rows if r.split == sp)) for sp in SPLITS},
        "codec_by_source": {s: dict(Counter(r.codec for r in rows if r.source_dataset == s)) for s in sorted({r.source_dataset for r in rows})},
        "speakers_by_split": {sp: len({r.speaker_group for r in rows if r.split == sp}) for sp in SPLITS},
        "hours_by_split": {sp: round(sum(r.duration_s or 0 for r in rows if r.split == sp) / 3600, 2) for sp in SPLITS},
        "total_clips": len(rows),
    }


def to_markdown(rep: dict) -> str:
    lines = [f"# Manifest report: `{rep['manifest']}`", "", "| Check | Result |", "|---|---|"]
    for k, v in rep["checks"].items():
        lines.append(f"| {k} | {'PASS' if v['ok'] else 'FAIL'} |")
    lines += ["", "## Shortcut warnings"] + ([f"- {w}" for w in rep["shortcuts"]["warnings"]] or ["- none"])
    d = rep["distributions"]
    lines += ["", "## Class distribution by split", "| split | bonafide | spoof | hours | speaker groups |", "|---|---|---|---|---|"]
    for sp in SPLITS:
        c = d["class_by_split"][sp]
        lines.append(f"| {sp} | {c.get('bonafide', 0)} | {c.get('spoof', 0)} | {d['hours_by_split'][sp]} | {d['speakers_by_split'][sp]} |")
    lines += ["", "## Language distribution by split", "| split | en | hi | hi-en |", "|---|---|---|---|"]
    for sp in SPLITS:
        c = d["language_by_split"][sp]
        lines.append(f"| {sp} | {c.get('en', 0)} | {c.get('hi', 0)} | {c.get('hi-en', 0)} |")
    lines += ["", "## Source × class", "| source | bonafide | spoof | codec |", "|---|---|---|---|"]
    for s, c in d["class_by_source"].items():
        lines.append(f"| {s} | {c.get('bonafide', 0)} | {c.get('spoof', 0)} | {d['codec_by_source'][s]} |")
    g = rep["checks"]["generator_leakage"]
    lines += ["", "## Spoof generator families by split"] + [f"- {sp}: {g['spoof_families_per_split'][sp]}" for sp in SPLITS]
    lines += [f"- unseen in train: {g['families_unseen_in_train']}"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check a canonical manifest for leakage and shortcuts.")
    ap.add_argument("--manifest", default="data/manifests/vaanishield_v1.csv")
    args = ap.parse_args(argv)
    path = Path(args.manifest)
    rows = read_manifest(path)
    root = DataPaths.resolve().data_root
    missing_fields = [r.clip_id for r in rows if not (r.path and r.label and r.language and r.speaker_id and r.generator
                                                      and r.source_dataset and r.codec and r.split)]
    checks = {
        "required_fields_present": {"ok": not missing_fields, "rows_missing_fields": missing_fields[:20]},
        "split_complete_and_both_classes": check_split(rows),
        "no_duplicate_leakage": check_duplicates(rows, root),
        "no_speaker_leakage": check_speaker_leakage(rows),
        "no_derived_rows_crossing_splits": check_descendants(rows),
        "generator_leakage": generator_leakage(rows),
    }
    rep = {"manifest": str(path), "checks": checks, "shortcuts": shortcut_risks(rows), "distributions": distributions(rows)}
    out_json = path.with_name(path.stem + "_report.json")
    out_json.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    path.with_name(path.stem + "_report.md").write_text(to_markdown(rep), encoding="utf-8")
    print(to_markdown(rep))
    print(f"written: {out_json} (+ .md)")
    return 0 if all(c["ok"] for c in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
