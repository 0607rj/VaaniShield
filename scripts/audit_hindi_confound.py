#!/usr/bin/env python
"""Audit the Hindi REAL/SPOOF confound (research/hindi_data_leakage_audit.md) and test candidate fixes on copies.

    python scripts/audit_hindi_confound.py [--out runs/audit/hindi]

Reads the Hindi rows of data/manifests/vaanishield_v1.csv (Kathbath = real, IndicSynth = spoof), plus ASVspoof and
the Hinglish pilot as controls. Never modifies data/: every transformed signal lives in memory only.
Outputs: features.csv (one row per clip x variant), results.json, and printed tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.audio import load_audio
from ml.data.manifest import read_manifest
from ml.data.provenance import DataPaths
from ml.evaluation.shortcut_probe import (
    LEVEL_FREE,
    LTAS,
    SEED,
    VIEW_COLS,
    aac_roundtrip,
    bandwidth_hz,
    centre_crop,
    dc_highpass,
    loudnorm,
    native,
    probe,
    single_feature_tests,
    trivial_classifiers,
    view_features,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="runs/audit/hindi")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = DataPaths.resolve().data_root
    v1 = read_manifest(Path("data/manifests/vaanishield_v1.csv"))
    hindi = [r for r in v1 if r.language == "hi"]
    kb_kbps = None
    recs = []
    # pass 1: metadata + native stats for Hindi (needed to pick the matched AAC bitrate)
    meta = {}
    for r in hindi:
        p = root / r.path
        xn, sr = native(p)
        meta[r.clip_id] = {**probe(p), "bandwidth_native_hz": bandwidth_hz(xn, sr), "native_dur_s": len(xn) / sr}
    kb_kbps = float(np.median([meta[r.clip_id]["bitrate_kbps"] for r in hindi if r.source_dataset == "kathbath_hi"]))
    rng = np.random.default_rng(SEED)
    for r in hindi:
        p = root / r.path
        base = {"clip_id": r.clip_id, "label": r.label, "split": r.split, "source": r.source_dataset, "generator": r.generator,
                "speaker_id": r.speaker_id, "speaker_group": r.speaker_group, "pool": "hindi", **meta[r.clip_id]}
        x16 = load_audio(p)  # decoded + resampled to 16 kHz, untrimmed
        xn, sr = native(p)
        recs.append({**base, "variant": "F0_file_untrimmed", **view_features(x16)})  # silence diagnostics only
        aac = aac_roundtrip(xn, sr, kb_kbps)  # spoof: first AAC pass; real: second AAC pass
        spoof = r.label == "spoof"
        variants = {  # model view = trim_silence + centre crop/repeat-pad to 4.04 s (exactly what the detector sees)
            "M0_current_pipeline": x16,
            "M1_loudnorm": loudnorm(x16),
            "M2_spoof_aac_matched+loudnorm": loudnorm(aac) if spoof else loudnorm(x16),
            "M3_both_aac+loudnorm": loudnorm(aac),
            "M4_spoof_aac+loudnorm+dc_highpass": dc_highpass(loudnorm(aac) if spoof else loudnorm(x16)),
        }
        for v, x in variants.items():
            recs.append({**base, "variant": v, **view_features(centre_crop(x))})
    # controls: ASVspoof (en) and Hinglish pilot (hi-en), current pipeline only
    controls = [r for r in v1 if r.language == "en"]
    controls = [controls[i] for i in rng.choice(len(controls), min(560, len(controls)), replace=False)]
    controls += read_manifest(Path("data/manifests/hinglish_pilot_v1.csv"))
    for r in controls:
        p = root / r.path
        recs.append({"clip_id": r.clip_id, "label": r.label, "split": r.split, "source": r.source_dataset, "generator": r.generator,
                     "speaker_id": r.speaker_id, "speaker_group": r.speaker_group, "pool": "english" if r.language == "en" else "hinglish",
                     **probe(p), "variant": "M0_current_pipeline", **view_features(centre_crop(load_audio(p)))})
    df = pd.DataFrame(recs)
    df.to_csv(out / "features.csv", index=False)

    res = {"kathbath_median_kbps": kb_kbps, "metadata": {}, "single_feature": {}, "classifiers": {}}
    h0 = df[(df.pool == "hindi") & (df.variant == "M0_current_pipeline")]
    for col in ("container", "codec", "sr_native", "channels", "sample_fmt", "wav_subtype", "encoder_tag", "n_tags"):
        res["metadata"][col] = {lab: h0[h0.label == lab][col].astype(str).value_counts().to_dict() for lab in ("bonafide", "spoof")}
    for col in ("bitrate_kbps", "file_kb", "bandwidth_native_hz", "native_dur_s"):
        res["metadata"][col] = {f"{lab}_{s}": round(float(getattr(h0[h0.label == lab][col], s)()), 2)
                                for lab in ("bonafide", "spoof") for s in ("min", "median", "max")}
    res["metadata"]["by_generator_bandwidth_native_hz"] = h0.groupby("generator").bandwidth_native_hz.median().to_dict()
    meta_X = h0.assign(is_aac=(h0.codec == "aac").astype(float), sr=h0.sr_native.astype(float))
    res["classifiers"]["hindi:E1_metadata_only(sr,codec,bitrate,file_kb,duration)"] = trivial_classifiers(
        meta_X, ["sr", "is_aac", "bitrate_kbps", "file_kb", "native_dur_s"])
    variants = ["F0_file_untrimmed", "M0_current_pipeline", "M1_loudnorm", "M2_spoof_aac_matched+loudnorm",
                "M3_both_aac+loudnorm", "M4_spoof_aac+loudnorm+dc_highpass"]
    hd = df[df.pool == "hindi"]
    both = hd[hd.variant == "M0_current_pipeline"].groupby("speaker_group").label.nunique()
    matched_groups = set(both[both == 2].index)
    res["speaker_balance"] = {"groups_total": int(len(both)), "groups_with_both_classes": len(matched_groups),
                              "clips_in_matched_groups": {lab: int(((hd.variant == "M0_current_pipeline") & hd.speaker_group.isin(matched_groups)
                                                                    & (hd.label == lab)).sum()) for lab in ("bonafide", "spoof")}}
    res["class_medians"] = {}
    for v in variants:
        d = hd[hd.variant == v]
        res["class_medians"][v] = d.groupby("label")[VIEW_COLS].median().round(4).to_dict("index")
        res["single_feature"][f"hindi:{v}"] = single_feature_tests(d, VIEW_COLS).head(8).to_dict("records")
        for fs_name, cols in (("all_nonsemantic", VIEW_COLS), ("level_free", LEVEL_FREE), ("ltas_spectrum", LTAS)):
            res["classifiers"][f"hindi:{v}:{fs_name}"] = trivial_classifiers(d, cols)
    d = hd[(hd.variant == "M4_spoof_aac+loudnorm+dc_highpass") & hd.speaker_group.isin(matched_groups)]
    if d.label.nunique() == 2 and d.speaker_group.nunique() >= 2 :
        res["classifiers"]["hindi:M5_M4+speaker_matched_groups:all_nonsemantic"] = trivial_classifiers(d, VIEW_COLS)
        res["classifiers"]["hindi:M5_M4+speaker_matched_groups:ltas_spectrum"] = trivial_classifiers(d, LTAS)
    res["by_generator_M0"] = hd[hd.variant == "M0_current_pipeline"].groupby("generator")[
        ["noise_floor_db", "sil_frac", "peak", "rms_active_db", "dc_offset", "band_6_8k"]].median().round(4).to_dict("index")
    res["single_feature"]["hindi:M4+matched_groups"] = single_feature_tests(d, VIEW_COLS).head(8).to_dict("records") if len(d) else []
    for pool in ("english", "hinglish"):
        d = df[(df.pool == pool)]
        res["single_feature"][f"{pool}:M0_current_pipeline"] = single_feature_tests(d, VIEW_COLS).head(5).to_dict("records")
        res["classifiers"][f"{pool}:M0_current_pipeline:all_nonsemantic"] = trivial_classifiers(d, VIEW_COLS)
        res["classifiers"][f"{pool}:M0_current_pipeline:level_free"] = trivial_classifiers(d, LEVEL_FREE)
        res["classifiers"][f"{pool}:M0_current_pipeline:ltas_spectrum"] = trivial_classifiers(d, LTAS)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")

    pd.set_option("display.width", 220)
    print("== metadata (Hindi, as stored)")
    for k, v in res["metadata"].items():
        print(f"  {k}: {v}")
    print("\n== trivial classifiers (EER; 0.5 = chance, 0 = perfect shortcut)")
    for k, v in res["classifiers"].items():
        print(f"  {k:62s} logreg cv {v['logreg']['cv_eer']:.3f} test {v['logreg']['split_test_eer']:.3f} | forest cv {v['forest']['cv_eer']:.3f} "
              f"test {v['forest']['split_test_eer']:.3f} | top {v.get('top_features', [])[:4]}")
    print("\n== strongest single features (separation AUC; 0.5 = none, 1 = perfect)")
    for k, v in res["single_feature"].items():
        print(f"  {k:30s} " + ", ".join(f"{r['feature']} {r['sep_auc']:.2f}" for r in v[:5]))
    print(f"\nwritten {out / 'features.csv'}, {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
