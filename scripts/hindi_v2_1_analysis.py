#!/usr/bin/env python
"""Peak/clipping, level, noise-floor and spectrum analysis of Hindi v2 vs v2.1 (research/hindi_v2_1_validation.md).

    python scripts/hindi_v2_1_analysis.py        # reads only; writes runs/audit/hindi_v2_1/analysis.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.audio import load_audio, trim_silence
from ml.data.manifest import read_manifest
from ml.data.provenance import DataPaths
from ml.evaluation.metrics import roc_auc
from ml.evaluation.shortcut_probe import FRAME, LTAS, centre_crop, native, view_features

OUT = Path("runs/audit/hindi_v2_1")


def meta(path: str) -> dict:
    return {(r := json.loads(ln)).get("clip_id"): r for ln in Path(path).read_text(encoding="utf-8").splitlines()}


def sep(df: pd.DataFrame, col: str) -> float:
    a = roc_auc(df[col].to_numpy(float), (df.label == "spoof").to_numpy(int))
    return round(max(a, 1 - a), 3)


def frame_db(x: np.ndarray) -> np.ndarray:
    n = x.size // FRAME
    return 10 * np.log10((x[: n * FRAME].reshape(n, FRAME) ** 2).mean(1) + 1e-12)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    root = DataPaths.resolve().data_root
    res: dict = {}
    m2, m21 = meta("data/manifests/hindi_v2_pilot_metadata.jsonl"), meta("data/manifests/hindi_v2_1_pilot_metadata.jsonl")
    rows21 = read_manifest(Path("data/manifests/hindi_v2_1_pilot.csv"))

    # --- peaks / clipping
    recs = []
    for r in rows21:
        a, b = m2[r.clip_id], m21[r.clip_id]
        x = load_audio(root / r.path)
        xn, _ = native(root / b["raw_path"])
        recs.append({"label": r.label, "gen": r.generator, "v2_peak_limited": a["peak_limited"], "v2_gain_db": a["gain_db"],
                     "v21_gain_db": b["gain_db"], "v21_peak": float(np.abs(x).max()), "v21_over_1": bool(np.abs(x).max() >= 1.0),
                     "raw_peak": float(np.abs(xn).max()), "raw_fullscale_frac": float((np.abs(xn) >= 0.999).mean()),
                     "crest_db": 20 * np.log10(np.abs(x).max() + 1e-9) - b["post_active_rms_dbfs"]})
    pk = pd.DataFrame(recs)
    res["peaks"] = {
        "v2_peak_limited_by_class": pk.groupby("label").v2_peak_limited.mean().round(3).to_dict(),
        "v21_peak_quantiles_by_class": {lab: [round(float(pk[pk.label == lab].v21_peak.quantile(q)), 3) for q in (0.5, 0.9, 0.99, 1.0)] for lab in ("bonafide", "spoof")},
        "v21_clips_with_peak_ge_1_by_class": pk.groupby("label").v21_over_1.sum().astype(int).to_dict(),
        "raw_clips_with_fullscale_samples_by_class": pk.groupby("label").apply(lambda d: int((d.raw_fullscale_frac > 0).sum()), include_groups=False).to_dict(),
        "crest_db_median_by_class": pk.groupby("label").crest_db.median().round(2).to_dict(), "crest_sep_auc": sep(pk, "crest_db"),
        "gain_change_v2_to_v21_db_for_limited_clips": round(float((pk[pk.v2_peak_limited].v21_gain_db - pk[pk.v2_peak_limited].v2_gain_db).median()), 2)
        if pk.v2_peak_limited.any() else 0.0,
    }

    # --- level and noise floor: whole file vs the 4.04 s model crop
    lv = []
    for r in rows21:
        x = load_audio(root / r.path)
        c = centre_crop(x)
        ef, ec = frame_db(trim_silence(x)), frame_db(c)
        act = ec > ec.max() - 40
        lv.append({"label": r.label, "gen": r.generator, "file_active_rms": m21[r.clip_id]["post_active_rms_dbfs"],
                   "crop_active_rms": float(10 * np.log10(np.mean(10 ** (ec[act] / 10)) + 1e-12)),
                   "crop_active_frames": float(act.mean()), "file_noise_floor": float(np.percentile(ef, 10) - ef.max()),
                   "speech_noise_floor": float(np.percentile(ef[ef > ef.max() - 40], 10) - ef.max()),
                   **{k: v for k, v in view_features(c).items() if k in LTAS or k in ("band_6_8k", "band_4_8k")}})
    lvd = pd.DataFrame(lv)
    res["level_and_floor"] = {
        "median_by_class": lvd.groupby("label")[["file_active_rms", "crop_active_rms", "crop_active_frames", "file_noise_floor", "speech_noise_floor"]].median().round(3).to_dict("index"),
        "median_by_generator": lvd.groupby("gen")[["crop_active_rms", "crop_active_frames", "file_noise_floor", "speech_noise_floor"]].median().round(3).to_dict("index"),
        "sep_auc": {k: sep(lvd, k) for k in ("file_active_rms", "crop_active_rms", "crop_active_frames", "file_noise_floor", "speech_noise_floor")},
    }

    # --- spectrum: class means, per generator, and paired by target speaker
    edges = np.linspace(0, 8000, 41)
    centres = ((edges[:-1] + edges[1:]) / 2).round().astype(int).tolist()
    mean_db = lvd.groupby("label")[LTAS].mean()
    diff = (mean_db.loc["spoof"] - mean_db.loc["bonafide"]).round(2)
    res["spectrum"] = {
        "band_centres_hz": centres, "spoof_minus_real_db": diff.tolist(),
        "spoof_minus_real_db_by_generator": {g: (lvd[lvd.gen == g][LTAS].mean() - mean_db.loc["bonafide"]).round(2).tolist()
                                             for g in ("xtts_v2", "freevc24")},
        "band_sep_auc": {f"{centres[i]}Hz": sep(lvd, c) for i, c in enumerate(LTAS)},
    }
    spk = {r.clip_id: r.speaker_id for r in rows21}
    lvd["spk"] = [spk[r.clip_id] for r in rows21]
    paired = []
    for s, d in lvd.groupby("spk"):
        if d.label.nunique() == 2:
            paired.append((d[d.label == "spoof"][LTAS].mean() - d[d.label == "bonafide"][LTAS].mean()).to_numpy())
    paired = np.array(paired)
    res["spectrum"]["paired_by_speaker"] = {
        "n_speakers": int(len(paired)), "median_spoof_minus_real_db": np.round(np.median(paired, 0), 2).tolist(),
        "fraction_of_speakers_spoof_lower": np.round((paired < 0).mean(0), 2).tolist(),
        "high_band_4_8k_spoof_lower_in_speakers": f"{int((paired[:, 20:].mean(1) < 0).sum())}/{len(paired)}",
    }
    (OUT / "analysis.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "spectrum"}, indent=1, default=str))
    s = res["spectrum"]
    print("spoof-real dB by band:", list(zip(s["band_centres_hz"][::4], s["spoof_minus_real_db"][::4], strict=True)))
    print("paired:", s["paired_by_speaker"]["n_speakers"], "speakers; median dB every 4th band", s["paired_by_speaker"]["median_spoof_minus_real_db"][::4],
          "| 4-8 kHz lower in", s["paired_by_speaker"]["high_band_4_8k_spoof_lower_in_speakers"])
    print("xtts:", s["spoof_minus_real_db_by_generator"]["xtts_v2"][::4]); print("freevc:", s["spoof_minus_real_db_by_generator"]["freevc24"][::4])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
