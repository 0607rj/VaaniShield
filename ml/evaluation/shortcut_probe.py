"""Shortcut probes: non-semantic features of what a detector sees, and trivial classifiers over them.

Used by scripts/audit_hindi_confound.py (diagnosis) and ml/data/hindi_gate.py (acceptance gate), so both measure
exactly the same thing. See research/hindi_data_leakage_audit.md for definitions and interpretation.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pandas as pd
import soundfile as sf
import soxr
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.data.audio import CROP_SAMPLES, SAMPLE_RATE, trim_silence
from ml.data.codec import Condition, _pyav_roundtrip
from ml.evaluation.metrics import eer, roc_auc

SEED = 2026
FRAME = 400  # 25 ms at 16 kHz


# ---------------------------------------------------------------- file-level metadata (what the container says)
def probe(path: Path) -> dict:
    with av.open(str(path)) as c:
        s = c.streams.audio[0]
        meta = {**{f"tag_{k}": v for k, v in (c.metadata or {}).items()}, **{f"stag_{k}": v for k, v in (s.metadata or {}).items()}}
        out = {"container": c.format.name.split(",")[0], "codec": s.codec_context.name, "sr_native": s.rate,
               "channels": s.codec_context.channels, "sample_fmt": s.codec_context.format.name if s.codec_context.format else None,
               "bitrate_kbps": round((s.bit_rate or c.bit_rate or 0) / 1000, 1), "file_kb": round(path.stat().st_size / 1024, 1),
               "encoder_tag": meta.get("tag_encoder") or meta.get("stag_encoder") or meta.get("stag_handler_name") or "",
               "n_tags": len(meta)}
    if path.suffix.lower() == ".wav":
        out["wav_subtype"] = sf.info(str(path)).subtype
    return out


def native(path: Path) -> tuple[np.ndarray, int]:
    if path.suffix.lower() in (".wav", ".flac"):
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return x.mean(1), sr
    with av.open(str(path)) as c:
        s = c.streams.audio[0]
        parts = [f.to_ndarray() for f in c.decode(s)]
        sr = s.rate
    x = np.concatenate([p.reshape(p.shape[0], -1) if p.ndim == 2 else p[None] for p in parts], axis=1)
    return x.astype(np.float32).mean(0), sr


def bandwidth_hz(x: np.ndarray, sr: int, floor_db: float = 50.0) -> float:
    win = 1024 if sr > 16000 else 512
    fr = np.lib.stride_tricks.sliding_window_view(np.r_[x, np.zeros(win, np.float32)], win)[:: win // 2] * np.hanning(win)
    p = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    e = p.sum(1)
    act = 10 * np.log10(e + 1e-12) > 10 * np.log10(e.max() + 1e-12) - 30
    lt = 10 * np.log10(p[act].mean(0) + 1e-12)
    return float(np.nonzero(lt > lt.max() - floor_db)[0].max() * sr / win)


# ---------------------------------------------------------------- model-view features (16 kHz mono, as the detector sees it)
def view_features(x: np.ndarray) -> dict:
    sr = SAMPLE_RATE
    n = len(x) // FRAME * FRAME
    fr = x[:n].reshape(-1, FRAME)
    e_db = 10 * np.log10((fr**2).mean(1) + 1e-12)
    act = e_db > e_db.max() - 40
    idx = np.nonzero(act)[0]
    lead = idx[0] * FRAME / sr if idx.size else 0.0
    trail = (len(act) - 1 - idx[-1]) * FRAME / sr if idx.size else 0.0
    t = trim_silence(x)
    spec = np.abs(np.fft.rfft(fr * np.hanning(FRAME), axis=1)) ** 2
    f = np.fft.rfftfreq(FRAME, 1 / sr)
    pa = spec[act].sum(0) + 1e-12
    band = lambda lo, hi: float(pa[(f >= lo) & (f < hi)].sum() / pa.sum())  # noqa: E731
    sil = spec[~act] if (~act).sum() >= 3 else spec[np.argsort(e_db)[: max(3, len(e_db) // 10)]]
    flat = np.exp(np.log(sil + 1e-12).mean(1)) / (sil.mean(1) + 1e-12)
    rms_act = 10 * np.log10((fr[act] ** 2).mean() + 1e-12)
    peak = float(np.abs(x).max())
    cum = np.cumsum(pa)
    return {
        "dur_s": len(x) / sr, "dur_trim_s": len(t) / sr, "lead_sil_s": lead, "trail_sil_s": trail,
        "sil_frac": float(1 - act.mean()), "rms_active_db": rms_act, "peak": peak,
        "crest_db": 20 * np.log10(peak + 1e-9) - rms_act, "dc_offset": float(x.mean()),
        "noise_floor_db": float(np.percentile(e_db, 10) - e_db.max()), "clip_frac": float((np.abs(x) >= 0.999).mean()),
        "band_0_1k": band(0, 1000), "band_4_8k": band(4000, 8001), "band_6_8k": band(6000, 8001),
        "rolloff95_hz": float(f[np.searchsorted(cum, 0.95 * cum[-1])]), "sil_flatness": float(flat.mean()),
        "needs_pad": float(len(t) < CROP_SAMPLES),
        **ltas(pa, f),
    }


def ltas(pa: np.ndarray, f: np.ndarray, n_bands: int = 40) -> dict:
    """Long-term average spectrum of active frames in n_bands linear bands (0-8 kHz), in dB relative to the mean band."""
    edges = np.linspace(0, 8000, n_bands + 1)
    b = np.array([pa[(f >= lo) & (f < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:], strict=True)]) + 1e-12
    db = 10 * np.log10(b)
    return {f"ltas_{i:02d}": float(v) for i, v in enumerate(db - db.mean())}


VIEW_COLS = ["dur_s", "dur_trim_s", "lead_sil_s", "trail_sil_s", "sil_frac", "rms_active_db", "peak", "crest_db", "dc_offset",
             "noise_floor_db", "clip_frac", "band_0_1k", "band_4_8k", "band_6_8k", "rolloff95_hz", "sil_flatness", "needs_pad"]
LEVEL_FREE = [c for c in VIEW_COLS if c not in ("rms_active_db", "peak", "crest_db", "clip_frac")]
SHAPE_ONLY = [c for c in LEVEL_FREE if c not in ("dur_s", "dur_trim_s", "needs_pad")]
LTAS = [f"ltas_{i:02d}" for i in range(40)]


# ---------------------------------------------------------------- candidate fixes (in memory only)
def loudnorm(x: np.ndarray, target_db: float = -26.0) -> np.ndarray:
    n = len(x) // FRAME * FRAME
    fr = x[:n].reshape(-1, FRAME)
    e = 10 * np.log10((fr**2).mean(1) + 1e-12)
    act = fr[e > e.max() - 40]
    g = 10 ** ((target_db - 10 * np.log10((act**2).mean() + 1e-12)) / 20)
    y = x * g
    pk = np.abs(y).max()
    return (y * (0.95 / pk) if pk > 0.95 else y).astype(np.float32)


def aac_roundtrip(xn: np.ndarray, sr: int, kbps: float) -> np.ndarray:
    """Native signal -> 44.1 kHz -> AAC-LC (FFmpeg native) at kbps -> decode -> 16 kHz (mirrors Kathbath's path)."""
    x44 = soxr.resample(xn, sr, 44100, quality="HQ").astype(np.float32) if sr != 44100 else xn
    c = Condition(id="aac_match", kind="codec", role="eval_heldout", encoder="aac", container="adts",
                  codec_sample_rate=44100, bitrate=int(kbps * 1000))
    y, dec_sr, _ = _pyav_roundtrip(x44, c)
    return soxr.resample(y, dec_sr, SAMPLE_RATE, quality="HQ").astype(np.float32)


def dc_highpass(x: np.ndarray, fc: float = 40.0) -> np.ndarray:
    """Remove DC and sub-audio rumble (2nd-order Butterworth high-pass, zero-phase)."""
    from scipy.signal import butter, sosfiltfilt

    return sosfiltfilt(butter(2, fc, "highpass", fs=SAMPLE_RATE, output="sos"), x).astype(np.float32)


def centre_crop(x: np.ndarray) -> np.ndarray:
    t = trim_silence(x)
    if len(t) >= CROP_SAMPLES:
        s = (len(t) - CROP_SAMPLES) // 2
        return t[s : s + CROP_SAMPLES]
    return np.resize(t, CROP_SAMPLES)  # repeat-pad, as in training


# ---------------------------------------------------------------- tests
def single_feature_tests(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    y = (df.label == "spoof").to_numpy()
    rows = []
    for c in cols:
        a, b = df.loc[~y, c].to_numpy(float), df.loc[y, c].to_numpy(float)
        if np.nanstd(np.r_[a, b]) == 0:
            rows.append({"feature": c, "auc_spoof_high": 0.5, "sep_auc": 0.5, "p_mwu": 1.0, "real_median": np.median(a), "spoof_median": np.median(b)})
            continue
        auc = roc_auc(df[c].to_numpy(float), y.astype(int))
        rows.append({"feature": c, "auc_spoof_high": round(auc, 3), "sep_auc": round(max(auc, 1 - auc), 3),
                     "p_mwu": float(mannwhitneyu(a, b).pvalue), "real_median": round(float(np.median(a)), 4),
                     "spoof_median": round(float(np.median(b)), 4)})
    return pd.DataFrame(rows).sort_values("sep_auc", ascending=False)


def trivial_classifiers(df: pd.DataFrame, cols: list[str]) -> dict:
    """Speaker-group 5-fold CV (no group in both train and test fold) + the manifest train->test split."""
    X, y, g = df[cols].to_numpy(float), (df.label == "spoof").to_numpy(int), df.speaker_group.to_numpy()
    out = {}
    models = {"logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
              "forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)}
    for name, mk in models.items():
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=min(5, len(set(g)))).split(X, y, g):
            m = mk().fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        tr, te = (df.split == "train").to_numpy(), (df.split == "test").to_numpy()
        out[name] = {"cv_eer": round(eer(oof, y)[0], 3), "cv_auc": round(roc_auc(oof, y), 3),
                     "split_test_eer": float("nan"), "split_test_auc": float("nan")}
        if len(set(y[tr])) == 2 and len(set(y[te])) == 2:  # manifest train -> test (speaker-disjoint by construction)
            m = mk().fit(X[tr], y[tr])
            pt = m.predict_proba(X[te])[:, 1]
            out[name].update(split_test_eer=round(eer(pt, y[te])[0], 3), split_test_auc=round(roc_auc(pt, y[te]), 3))
        if name == "forest":
            out["top_features"] = [c for _, c in sorted(zip(m.feature_importances_, cols, strict=True), reverse=True)[:6]]
    return out


# ---- pause structure (pause-only probe; research/hindi_v2_1_validation.md) ----------------------------------------
PAUSE_COLS = ["pause_frac", "pauses_per_s", "mean_pause_s", "max_pause_s", "pause_noise_floor_db", "pause_flatness"]


def pause_features(x: np.ndarray, top_db: float = 40.0, min_pause_s: float = 0.1) -> dict:
    """Internal pauses of a trimmed clip: runs of >= min_pause_s of frames below (loudest frame - top_db)."""
    t = trim_silence(x)
    n = t.size // FRAME
    if n < 4:
        return dict.fromkeys(PAUSE_COLS, 0.0)
    fr = t[: n * FRAME].reshape(n, FRAME)
    e = 10 * np.log10((fr**2).mean(1) + 1e-12)
    sil = e <= e.max() - top_db
    runs, i = [], 0
    while i < n:
        if sil[i]:
            j = i
            while j < n and sil[j]:
                j += 1
            if (j - i) * FRAME / SAMPLE_RATE >= min_pause_s:
                runs.append((i, j))
            i = j
        else:
            i += 1
    lens = [(j - i) * FRAME / SAMPLE_RATE for i, j in runs]
    idx = np.concatenate([np.arange(i, j) for i, j in runs]) if runs else np.argsort(e)[: max(1, n // 10)]
    spec = np.abs(np.fft.rfft(fr[idx] * np.hanning(FRAME), axis=1)) ** 2
    flat = float((np.exp(np.log(spec + 1e-12).mean(1)) / (spec.mean(1) + 1e-12)).mean())
    return {"pause_frac": float(sum(lens) / (n * FRAME / SAMPLE_RATE)), "pauses_per_s": len(runs) / (n * FRAME / SAMPLE_RATE),
            "mean_pause_s": float(np.mean(lens)) if lens else 0.0, "max_pause_s": float(max(lens)) if lens else 0.0,
            "pause_noise_floor_db": float(np.median(e[idx]) - e.max()), "pause_flatness": flat}
