#!/usr/bin/env python
"""Before/after diagnostics for the Hindi repair (research/hindi_repair_report.md). Reads only; writes runs/audit/hindi_v2/.

    python scripts/hindi_v2_diagnostics.py

1. distributions  stored format, duration, class balance, speakers/groups: v1 Hindi vs v2 pilot
2. ablation       which feature families still separate REAL/SPOOF in v2 (all / without silence / without level / ...)
3. candidates     two possible further fixes, tested IN MEMORY ONLY (nothing written to data/):
                  T30  v2 chain with a -30 dBFS loudness target (less peak limiting of transient-rich real clips)
                  NF   identical low-level noise floor added to BOTH classes after v2 (masks digital-silence pauses)
                  VAD  internal pauses removed from BOTH classes (keep 25 ms frames within 40 dB of the loudest frame,
                       10 ms cross-fades) so pause structure cannot be a cue
                  VAD+LP6k  VAD and a 6 kHz low-pass on BOTH classes: attribution test for the remaining 6-8 kHz cue
                       (diagnostic only; discarding 6-8 kHz would also discard genuine synthesis artefacts)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.audio import highpass_dc, load_audio, loudness_normalise, trim_silence
from ml.data.manifest import read_manifest
from ml.data.provenance import DataPaths
from ml.evaluation.shortcut_probe import (
    LTAS,
    SEED,
    VIEW_COLS,
    aac_roundtrip,
    centre_crop,
    native,
    probe,
    single_feature_tests,
    trivial_classifiers,
    view_features,
)

OUT = Path("runs/audit/hindi_v2")
SILENCE = ["lead_sil_s", "trail_sil_s", "sil_frac", "noise_floor_db", "sil_flatness"]
LEVEL = ["rms_active_db", "peak", "crest_db", "clip_frac", "dc_offset"]
DURATION = ["dur_s", "dur_trim_s", "needs_pad"]
SPECTRAL = ["band_0_1k", "band_4_8k", "band_6_8k", "rolloff95_hz"]


def worst(res: dict) -> float:
    return min(v for m in ("logreg", "forest") for k in ("cv_eer", "split_test_eer") if not np.isnan(v := float(res[m][k])))


def summarise(df: pd.DataFrame) -> dict:
    sets = {"all_nonsemantic": VIEW_COLS, "without_silence": [c for c in VIEW_COLS if c not in SILENCE],
            "without_level": [c for c in VIEW_COLS if c not in LEVEL], "without_silence_and_level": [c for c in VIEW_COLS if c not in SILENCE + LEVEL],
            "silence_only": SILENCE, "level_only": LEVEL, "spectral_bands_only": SPECTRAL, "ltas_40_bands": LTAS}
    out = {}
    for name, cols in sets.items():
        r = trivial_classifiers(df, cols)
        out[name] = {"worst_eer": round(worst(r), 3), "forest_cv_eer": r["forest"]["cv_eer"], "logreg_cv_eer": r["logreg"]["cv_eer"]}
    out["top_features"] = single_feature_tests(df, VIEW_COLS).head(5)[["feature", "sep_auc"]].to_dict("records")
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    root = DataPaths.resolve().data_root
    v1 = [r for r in read_manifest(Path("data/manifests/vaanishield_v1.csv")) if r.language == "hi"]
    v2 = read_manifest(Path("data/manifests/hindi_v2_pilot.csv"))
    meta = {json.loads(ln).get("clip_id"): json.loads(ln) for ln in Path("data/manifests/hindi_v2_pilot_metadata.jsonl").read_text(encoding="utf-8").splitlines()}
    res: dict = {"distributions": {}, "ablation": {}, "candidates": {}}

    # 1. distributions
    for name, rows in (("v1_hindi", v1), ("v2_pilot", v2)):
        d = pd.DataFrame([{**probe(root / r.path), "label": r.label, "split": r.split, "dur": r.duration_s, "spk": r.speaker_id,
                           "grp": r.speaker_group, "gen": r.generator} for r in rows])
        spk = {lab: set(d[d.label == lab].spk) for lab in ("bonafide", "spoof")}
        res["distributions"][name] = {
            "format_by_class": {lab: d[d.label == lab].groupby(["container", "codec", "sr_native"]).size().rename("n").reset_index().to_dict("records")
                                for lab in ("bonafide", "spoof")},
            "duration_s_by_class": {lab: {q: round(float(d[d.label == lab].dur.quantile(q)), 2) for q in (0.1, 0.5, 0.9)} for lab in ("bonafide", "spoof")},
            "class_by_split": {sp: d[d.split == sp].label.value_counts().to_dict() for sp in ("train", "dev", "test")},
            "generators": d[d.label == "spoof"].gen.value_counts().to_dict(),
            "speakers": {"real": len(spk["bonafide"]), "spoof_target": len(spk["spoof"]), "shared": len(spk["bonafide"] & spk["spoof"])},
            "groups": int(d.grp.nunique()), "groups_with_both_classes": int((d.groupby("grp").label.nunique() == 2).sum()),
        }

    # 2. ablation on v2 as stored (model view)
    base = pd.DataFrame([{**view_features(centre_crop(load_audio(root / r.path))), "label": r.label, "split": r.split,
                          "speaker_group": r.speaker_group, "clip_id": r.clip_id} for r in v2])
    res["ablation"]["v2_as_built"] = summarise(base)
    res["ablation"]["v2_peak_limited_fraction"] = {lab: round(float(np.mean([meta[r.clip_id]["peak_limited"] for r in v2 if r.label == lab])), 3)
                                                   for lab in ("bonafide", "spoof")}

    # 3a. T30: same v2 chain, -30 dBFS target (recomputed from the original files; in memory)
    rows_t30 = []
    for r in v2:
        m = meta[r.clip_id]
        p = root / m["raw_path"]
        if r.label == "spoof":
            xn, sr = native(p)
            x16 = aac_roundtrip(xn, sr, m["codec_matching"]["bitrate_kbps"])
        else:
            x16 = load_audio(p)
        y, info = loudness_normalise(trim_silence(highpass_dc(x16)), target_dbfs=-30.0)
        rows_t30.append({**view_features(centre_crop(y)), "label": r.label, "split": r.split, "speaker_group": r.speaker_group,
                         "peak_limited": info["peak_limited"]})
    t30 = pd.DataFrame(rows_t30)
    res["candidates"]["T30_target_-30dBFS"] = {**summarise(t30), "peak_limited_fraction": t30.groupby("label").peak_limited.mean().round(3).to_dict()}

    # 3b. NF: identical noise floor for both classes (white noise, fixed seed per clip, -60 dBFS RMS) on the stored v2 audio
    rows_nf = []
    for i, r in enumerate(v2):
        x = load_audio(root / r.path)
        noise = np.random.default_rng(SEED + i).standard_normal(x.size).astype(np.float32) * 10 ** (-60 / 20)
        rows_nf.append({**view_features(centre_crop(x + noise)), "label": r.label, "split": r.split, "speaker_group": r.speaker_group})
    res["candidates"]["NF_noise_floor_-60dBFS_both"] = summarise(pd.DataFrame(rows_nf))

    # 3c. VAD: keep voiced frames only (both classes), 10 ms linear cross-fade between kept runs
    def voiced_only(x, hop=400, fade=160):
        n = x.size // hop
        fr = x[: n * hop].reshape(n, hop)
        e = 10 * np.log10((fr**2).mean(1) + 1e-12)
        keep = e > e.max() - 40
        runs, out = [], []
        i = 0
        while i < n:
            if keep[i]:
                j = i
                while j < n and keep[j]:
                    j += 1
                runs.append(x[i * hop : j * hop])
                i = j
            else:
                i += 1
        for seg in runs:
            if out and seg.size > fade and out[-1].size > fade:
                w = np.linspace(0, 1, fade, dtype=np.float32)
                out[-1][-fade:] = out[-1][-fade:] * (1 - w) + seg[:fade] * w
                seg = seg[fade:]
            out.append(seg.copy())
        return np.concatenate(out) if out else x
    rows_vad = [{**view_features(centre_crop(voiced_only(load_audio(root / r.path)))), "label": r.label, "split": r.split,
                 "speaker_group": r.speaker_group} for r in v2]
    res["candidates"]["VAD_pauses_removed_both"] = summarise(pd.DataFrame(rows_vad))
    from scipy.signal import butter, sosfiltfilt

    sos = butter(8, 6000, "lowpass", fs=16000, output="sos")
    rows_lp = [{**view_features(centre_crop(sosfiltfilt(sos, voiced_only(load_audio(root / r.path))).astype(np.float32))),
                "label": r.label, "split": r.split, "speaker_group": r.speaker_group} for r in v2]
    res["candidates"]["VAD+LP6k_both"] = summarise(pd.DataFrame(rows_lp))

    (OUT / "diagnostics.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
