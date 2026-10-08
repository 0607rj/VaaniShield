"""Hindi dataset gate (team decision D-H6): A. ACCEPTANCE GATE + B. DIAGNOSTIC PROBES.

    python -m ml.data.hindi_gate --manifest data/manifests/hindi_v2_1_pilot.csv
    python -m ml.data.hindi_gate --manifest data/manifests/vaanishield_v1.csv --language hi      # v1 has no metadata

A. ACCEPTANCE (pass/fail): nuisance and leakage properties a dataset must not have.
   file metadata, format / sample rate / codec / bitrate, duration, edge-silence and digital-silence statistics,
   overall level, peak / clipping processing, speaker / source / duplicate / sentence leakage, generator protocol,
   identical preprocessing, class balance.
   Numeric limits are PROVISIONAL ENGINEERING ACCEPTANCE CRITERIA (D-H3), not scientifically established thresholds.
B. DIAGNOSTICS (reported, never pass/fail): metadata-only, nuisance-only, pause-only, waveform-statistics and spectral
   probes, each with EER, a strength reading and whether the cue is a nuisance or possibly a legitimate synthesis cue.
   A spectrum probe that separates real from fake does NOT fail the dataset (VaaniShield is meant to learn synthesis
   artefacts); internal pause structure is kept in the primary data (D-H4) and is quantified here instead.

Verdicts: ACCEPTANCE_FAIL | ACCEPTANCE_PASS_PROVISIONAL. The gate never says READY; readiness for training is a
separate, evidence-based judgement (research/hindi_v2_1_validation.md).
Writes <out>/gate_report.json and gate_report.md (default runs/gate/<manifest stem>_<language>/).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from ml.data.audio import load_audio
from ml.data.check_manifest import generator_leakage
from ml.data.manifest import read_manifest
from ml.data.provenance import DataPaths
from ml.evaluation.shortcut_probe import (
    LTAS,
    PAUSE_COLS,
    VIEW_COLS,
    centre_crop,
    pause_features,
    probe,
    single_feature_tests,
    trivial_classifiers,
    view_features,
)
from ml.evaluation.validate import check_descendants, check_duplicates, check_speaker_leakage, check_split

SPLITS = ("train", "dev", "test")
NUISANCE_COLS = ["dur_s", "dur_trim_s", "needs_pad", "lead_sil_s", "trail_sil_s", "rms_active_db", "dc_offset", "clip_frac"]
SPECTRAL_COLS = ["band_0_1k", "band_4_8k", "band_6_8k", "rolloff95_hz"] + LTAS


def worst_eer(res: dict) -> float:
    """Most pessimistic EER of a trivial_classifiers() result (both models; speaker-grouped CV and train->test)."""
    return min(v for m in ("logreg", "forest") for k in ("cv_eer", "split_test_eer") if not np.isnan(v := float(res[m][k])))


def strength(e: float) -> str:
    return ("strong separation" if e < 0.20 else "moderate separation" if e < 0.35 else "weak separation" if e < 0.45
            else "no usable separation (near chance)")


def _file_level(rows, root: Path) -> pd.DataFrame:
    out = []
    for r in rows:
        p = root / r.path
        x = load_audio(p)
        fv = view_features(x)  # stored file as decoded, NOT trimmed/cropped: edge silence and level as stored
        info = sf.info(str(p)) if p.suffix.lower() == ".wav" else None
        out.append({**probe(p), "label": r.label, "split": r.split, "speaker_group": r.speaker_group,
                    "duration": r.duration_s if r.duration_s is not None else len(x) / 16000,
                    "lead_sil_s": fv["lead_sil_s"], "trail_sil_s": fv["trail_sil_s"], "rms_active_db": fv["rms_active_db"],
                    "zero_frac": float((x == 0).mean()), "subtype": info.subtype if info else None,
                    "fullscale_frac": float((np.abs(x) >= 32767 / 32768).mean()) if info and info.subtype.startswith("PCM") else 0.0})
    return pd.DataFrame(out)


def acceptance(rows, root: Path, cfg: dict, metadata: dict | None, header: dict | None, fl: pd.DataFrame) -> dict:
    th = cfg["acceptance"]
    df = pd.DataFrame([r.model_dump() for r in rows])
    a: dict[str, dict] = {}
    fl = fl.assign(is_aac=(fl.codec == "aac").astype(float), sr=fl.sr_native.astype(float), is_wav=(fl.container == "wav").astype(float))

    meta_res = trivial_classifiers(fl, ["sr", "is_aac", "is_wav", "channels", "bitrate_kbps", "file_kb", "duration"])
    a["A1_file_metadata_classifier"] = {"ok": worst_eer(meta_res) >= th["metadata_min_eer"], "worst_eer": worst_eer(meta_res),
                                        "criterion": f">= {th['metadata_min_eer']} (provisional)", "detail": meta_res}
    fmt = {lab: sorted({(m.container, m.codec, int(m.sr_native), int(m.channels), str(m.sample_fmt), str(m.subtype), float(m.bitrate_kbps))
                        for m in fl[fl.label == lab].itertuples()}) for lab in ("bonafide", "spoof")}
    a["A2_format_samplerate_codec_bitrate_identical"] = {"ok": fmt["bonafide"] == fmt["spoof"] and len(fmt["bonafide"]) == 1,
                                                         "formats_by_class": fmt, "criterion": "exactly one identical format tuple"}
    dur = trivial_classifiers(fl, ["duration"])
    a["A3_duration"] = {"ok": worst_eer(dur) >= th["duration_min_eer"], "worst_eer": worst_eer(dur),
                        "quantiles_s": {lab: [round(float(fl[fl.label == lab].duration.quantile(q)), 2) for q in (0.1, 0.5, 0.9)]
                                        for lab in ("bonafide", "spoof")}, "criterion": f">= {th['duration_min_eer']} (provisional)"}
    sil = trivial_classifiers(fl, ["lead_sil_s", "trail_sil_s", "zero_frac"])
    a["A4_edge_and_digital_silence"] = {"ok": worst_eer(sil) >= th["silence_min_eer"], "worst_eer": worst_eer(sil),
                                        "medians": fl.groupby("label")[["lead_sil_s", "trail_sil_s", "zero_frac"]].median().round(4).to_dict("index"),
                                        "criterion": f">= {th['silence_min_eer']} (provisional); internal pauses -> diagnostics (D-H4)"}
    lvl = trivial_classifiers(fl, ["rms_active_db"])
    params = (header or {}).get("preproc_params") or (header or {}).get("preproc_v2") or {}  # v2 headers used "preproc_v2"
    target = params.get("loudness_target_dbfs")
    metas = list((metadata or {}).values()) if target is not None else []
    off = [abs(m.get("post_active_rms_dbfs", 99) - target) for m in metas]
    lim_ok = target is not None and all(o <= th["loudness_tolerance_db"] or m.get("peak_limited") for o, m in zip(off, metas, strict=True))
    a["A5_overall_level"] = {"ok": worst_eer(lvl) >= th["level_min_eer"] and lim_ok and target is not None, "worst_eer": worst_eer(lvl),
                             "target_dbfs": target, "rows_off_target": sum(o > th["loudness_tolerance_db"] for o in off),
                             "criterion": f"level-only classifier >= {th['level_min_eer']} and all rows within {th['loudness_tolerance_db']} dB (provisional)"}
    if metadata:
        frac = {lab: float(np.mean([metadata[r.clip_id].get("peak_limited", False) for r in rows if r.label == lab])) for lab in ("bonafide", "spoof")}
        introduced = int((fl.fullscale_frac > 0).sum())
        a["A6_peak_clipping_processing_class_neutral"] = {
            "ok": abs(frac["bonafide"] - frac["spoof"]) <= th["max_limiter_fraction_gap"] and introduced == 0,
            "peak_limited_fraction": {k: round(v, 3) for k, v in frac.items()}, "rows_with_full_scale_samples_in_integer_files": introduced,
            "criterion": f"limiter fraction gap <= {th['max_limiter_fraction_gap']} (provisional) and no full-scale samples in integer files"}
    else:
        a["A6_peak_clipping_processing_class_neutral"] = {"ok": False, "reason": "no preprocessing metadata"}
    sl, cs = check_speaker_leakage(rows), check_split(rows)
    a["A7_speaker_leakage_zero"] = {"ok": sl["ok"] and cs["ok"], "overlaps": sl.get("overlaps"), "split_classes": cs.get("counts")}
    desc, gl = check_descendants(rows), generator_leakage(rows)
    ref_problems = []
    if metadata:
        split_of_real = {r.clip_id.removeprefix("hv2_r_"): r.split for r in rows if r.label == "bonafide"}
        for r in rows:
            m = metadata.get(r.clip_id, {})
            for ref in (m.get("target_ref"), m.get("source_ref")):
                if ref and ref in split_of_real and split_of_real[ref] != r.split:
                    ref_problems.append(f"{r.clip_id} ({r.split}) uses reference {ref} ({split_of_real[ref]})")
    a["A8_source_leakage_zero"] = {"ok": desc["ok"] and gl["n_generator_input_problems"] == 0 and not ref_problems,
                                   "descendant_problems": desc.get("problems"), "generator_input_problems": gl["generator_input_problems"],
                                   "reference_split_problems": ref_problems[:20]}
    du = check_duplicates(rows, root)
    a["A9_duplicate_leakage_zero"] = {"ok": du["ok"], "cross_split": du.get("cross_split"), "file_dup_groups": du.get("file_duplicate_groups"),
                                      "audio_dup_groups": du.get("audio_duplicate_groups")}
    t_test = set(df[(df.split == "test") & df.transcript_id.notna()].transcript_id)
    cross = t_test & set(df[(df.split != "test") & df.transcript_id.notna()].transcript_id)
    a["A10_sentence_leakage_zero"] = {"ok": not cross and int(df.transcript_id.isna().sum()) == 0, "test_sentences_in_train_or_dev": len(cross),
                                      "rows_without_transcript_id": int(df.transcript_id.isna().sum())}
    fam = {sp: set(df[(df.split == sp) & (df.label == "spoof")].generator_family) for sp in SPLITS}
    hold = cfg["protocol"].get("holdout_family")
    a["A11_generator_protocol"] = {"ok": bool(fam["train"]) and fam["dev"] <= fam["train"] and (not hold or (hold not in fam["train"] | fam["dev"] and hold in fam["test"])),
                                   "families_per_split": {k: sorted(v) for k, v in fam.items()}, "holdout_family": hold}
    if metadata:
        hashes = {lab: sorted({metadata[r.clip_id].get("preproc_hash") for r in rows if r.label == lab}) for lab in ("bonafide", "spoof")}
        dc_bad = sum(abs(m.get("post_dc", 1)) > th["max_abs_dc"] for m in metadata.values())
        a["A12_identical_preprocessing"] = {"ok": hashes["bonafide"] == hashes["spoof"] and len(hashes["bonafide"]) == 1 and None not in hashes["bonafide"] and dc_bad == 0,
                                            "preproc_hash_by_class": hashes, "rows_failing_dc": dc_bad, "params": (header or {}).get("preproc_params") or (header or {}).get("preproc_v2")}
    else:
        a["A12_identical_preprocessing"] = {"ok": False, "reason": "no preprocessing metadata: cannot be shown identical"}
    ratio = {}
    for sp in SPLITS:
        c = Counter(df[df.split == sp].label)
        ratio[sp] = round(c["spoof"] / c["bonafide"], 3) if c["bonafide"] else float("inf")
    lo, hi = th["class_ratio_range"]
    a["A13_class_balance"] = {"ok": all(lo <= v <= hi for v in ratio.values()), "spoof_per_real": ratio, "criterion": f"{lo}-{hi} per split (provisional)"}
    return a


def diagnostics(rows, root: Path, fl: pd.DataFrame) -> dict:
    recs = []
    for r in rows:
        x = load_audio(root / r.path)
        recs.append({**view_features(centre_crop(x)), **pause_features(x), "label": r.label, "split": r.split, "speaker_group": r.speaker_group})
    mv = pd.DataFrame(recs)
    fl = fl.assign(is_aac=(fl.codec == "aac").astype(float), sr=fl.sr_native.astype(float), is_wav=(fl.container == "wav").astype(float))
    defs = {
        "B1_metadata_only": (fl, ["sr", "is_aac", "is_wav", "channels", "bitrate_kbps", "file_kb", "duration"], "nuisance",
                             "container/codec/rate/bitrate/size/duration describe the file, not the speech"),
        "B2_nuisance_only": (mv, NUISANCE_COLS, "nuisance",
                             "duration, edge silence, level, DC and clipping are recording/processing properties"),
        "B3_pause_only": (mv, PAUSE_COLS, "mixed / unresolved",
                          "pause frequency/length and the noise floor in pauses reflect recording conditions (nuisance) and how the generator renders pauses (possibly legitimate)"),
        "B4_waveform_statistics": (mv, VIEW_COLS, "mixed",
                                   "combines level, silence, pause and coarse spectral statistics"),
        "B5_spectral": (mv, SPECTRAL_COLS, "potentially legitimate synthesis cue (not proven)",
                        "long-term spectrum of speech frames; after speaker pairing, matched codec and identical preprocessing, differences are plausibly vocoder/cloning artefacts, but a residual channel component cannot be excluded"),
    }
    out = {}
    for name, (d, cols, kind, why) in defs.items():
        res = trivial_classifiers(d, cols)
        e = worst_eer(res)
        out[name] = {"worst_eer": round(e, 3), "logreg_cv_eer": res["logreg"]["cv_eer"], "forest_cv_eer": res["forest"]["cv_eer"],
                     "logreg_test_eer": res["logreg"]["split_test_eer"], "forest_test_eer": res["forest"]["split_test_eer"],
                     "reading": strength(e), "cue_type": kind, "why": why, "top_features": res.get("top_features", [])[:4],
                     "top_single": single_feature_tests(d, cols).head(3)[["feature", "sep_auc"]].to_dict("records")}
    return out


def to_md(rep: dict, manifest: str) -> str:
    lines = [f"# Hindi gate: `{manifest}`", "", f"**Acceptance verdict: {rep['verdict']}**  ({rep['criteria_status']})", "",
             f"Rows {rep['n_rows']}; per split {rep['counts']}; speaker groups {rep['groups_per_split']}", "",
             "## A. Acceptance gate", "| Check | Result | Key numbers |", "|---|---|---|"]
    show = ("worst_eer", "formats_by_class", "quantiles_s", "medians", "target_dbfs", "rows_off_target", "peak_limited_fraction",
            "rows_with_full_scale_samples_in_integer_files", "overlaps", "descendant_problems", "reference_split_problems",
            "cross_split", "test_sentences_in_train_or_dev", "rows_without_transcript_id", "families_per_split",
            "preproc_hash_by_class", "rows_failing_dc", "spoof_per_real", "reason")
    for k, c in rep["acceptance"].items():
        nums = "; ".join(f"{x}={c[x]}" for x in show if x in c).replace("|", "/")
        lines.append(f"| {k} | {'PASS' if c['ok'] else 'FAIL'} | {nums} |")
    lines += ["", "## B. Diagnostic probes (not pass/fail)", "| Probe | worst EER | Reading | Cue type | Strongest single features (sep AUC) |",
              "|---|---|---|---|---|"]
    for k, d in rep["diagnostics"].items():
        feats = ", ".join(f"{t['feature']} ({t['sep_auc']:.2f})" for t in d["top_single"])
        lines.append(f"| {k} | {d['worst_eer']:.3f} | {d['reading']} | {d['cue_type']} | {feats} |")
    lines += ["", "Numeric acceptance limits are PROVISIONAL ENGINEERING ACCEPTANCE CRITERIA (team decision D-H3), not scientifically "
              "established thresholds. EER: 0.5 = chance; 'worst' = lowest over logistic regression / random forest, speaker-grouped CV "
              "and train->test."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Hindi acceptance gate + diagnostic probes.")
    ap.add_argument("--config", default="ml/configs/hindi_gate.yaml")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--metadata", help="per-row metadata JSONL (default: <manifest stem>_metadata.jsonl if it exists)")
    ap.add_argument("--language", default="hi")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = [r for r in read_manifest(Path(args.manifest)) if r.language == args.language]
    mpath = Path(args.metadata) if args.metadata else Path(args.manifest).with_name(Path(args.manifest).stem + "_metadata.jsonl")
    metadata, header = None, None
    if mpath.exists():
        metadata = {}
        for line in mpath.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if "_header" in rec:
                header = rec["_header"]
            elif "clip_id" in rec:
                metadata[rec["clip_id"]] = rec
    root = DataPaths.resolve().data_root
    fl = _file_level(rows, root)
    acc = acceptance(rows, root, cfg, metadata, header, fl)
    diag = diagnostics(rows, root, fl)
    df = pd.DataFrame([r.model_dump() for r in rows])
    rep = {"verdict": "ACCEPTANCE_PASS_PROVISIONAL" if all(c["ok"] for c in acc.values()) else "ACCEPTANCE_FAIL",
           "criteria_status": "numeric limits are PROVISIONAL ENGINEERING ACCEPTANCE CRITERIA (D-H3)",
           "n_rows": len(rows), "counts": {sp: dict(Counter(df[df.split == sp].label)) for sp in SPLITS},
           "groups_per_split": {sp: int(df[df.split == sp].speaker_group.nunique()) for sp in SPLITS},
           "acceptance": acc, "diagnostics": diag}
    out = Path(args.out or f"runs/gate/{Path(args.manifest).stem}_{args.language}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    md = to_md(rep, args.manifest)
    (out / "gate_report.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"written {out / 'gate_report.json'} (+ .md)")
    return 0 if rep["verdict"] != "ACCEPTANCE_FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
