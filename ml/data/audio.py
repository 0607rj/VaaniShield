"""Audio preprocessing: load -> mono -> 16 kHz -> trim silence -> fixed-length crop/pad.

The same functions are used for training, evaluation and inference, so all three see identical input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SAMPLE_RATE = 16000
CROP_SAMPLES = 64600  # ~4.04 s, the length used by AASIST-family code (DEC-10)


AV_SUFFIXES = {".m4a", ".mp4", ".aac", ".webm", ".opus", ".mp3"}  # decoded with PyAV (bundled FFmpeg)


def _decode_av(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode compressed audio to float32 [T, C] at the native rate (no resampling/downmix inside FFmpeg)."""
    import av

    chunks, sr = [], None
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp")  # sample-format conversion only
        for frame in container.decode(stream):
            sr = frame.sample_rate
            for f in resampler.resample(frame):
                chunks.append(f.to_ndarray())  # [C, n]
        for f in resampler.resample(None):  # flush
            chunks.append(f.to_ndarray())
    if not chunks or sr is None:
        raise ValueError(f"{path}: no audio decoded")
    return np.concatenate(chunks, axis=1).T.astype(np.float32), int(sr)


def load_audio(path: str | Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Read WAV/FLAC/OGG via libsndfile or m4a/AAC/MP3/Opus via PyAV, average channels, resample to ``target_sr``."""
    if Path(path).suffix.lower() in AV_SUFFIXES:
        wave, sr = _decode_av(path)
    else:
        wave, sr = sf.read(str(path), dtype="float32", always_2d=True)  # [T, C]
    if wave.size == 0:
        raise ValueError(f"{path}: empty audio")
    wave = wave.mean(axis=1)
    if sr != target_sr:
        wave = soxr.resample(wave, sr, target_sr, quality="HQ").astype(np.float32)
    return wave


def trim_silence(wave: np.ndarray, sr: int = SAMPLE_RATE, top_db: float = 40.0, frame_ms: float = 25.0) -> np.ndarray:
    """Remove leading/trailing frames quieter than (loudest frame - top_db). Same rule for both classes.

    Applied to every clip because ASVspoof bona fide clips have longer silences than spoofed ones, which models
    can exploit as a shortcut (Müller et al. 2021, research/literature_matrix.md #13).
    """
    hop = int(sr * frame_ms / 1000)
    if wave.size < hop:
        return wave
    n_frames = wave.size // hop
    frames = wave[: n_frames * hop].reshape(n_frames, hop)
    energy_db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    voiced = np.flatnonzero(energy_db > energy_db.max() - top_db)
    if voiced.size == 0:
        return wave
    return wave[voiced[0] * hop : (voiced[-1] + 1) * hop]


def crop_or_pad(wave: np.ndarray, length: int = CROP_SAMPLES, rng: np.random.Generator | None = None) -> np.ndarray:
    """Fixed length. Short clips are repeated (tiled), never zero-padded; long clips are cropped.

    ``rng`` given -> random crop (training); ``rng`` None -> centre crop (evaluation/inference, deterministic).
    """
    if wave.size == 0:
        raise ValueError("cannot crop/pad empty audio")
    if wave.size < length:
        reps = int(np.ceil(length / wave.size))
        return np.tile(wave, reps)[:length]
    start = int(rng.integers(0, wave.size - length + 1)) if rng is not None else (wave.size - length) // 2
    return wave[start : start + length]


def windows(wave: np.ndarray, length: int = CROP_SAMPLES, hop: int = CROP_SAMPLES // 2) -> list[np.ndarray]:
    """Overlapping fixed-length windows covering the whole clip (inference on long files). Last window is right-aligned."""
    if wave.size <= length:
        return [crop_or_pad(wave, length)]
    starts = list(range(0, wave.size - length + 1, hop))
    if starts[-1] != wave.size - length:
        starts.append(wave.size - length)
    return [wave[s : s + length] for s in starts]


def preprocess(path: str | Path) -> np.ndarray:
    """Full preprocessing of one file, before cropping."""
    return trim_silence(load_audio(path))


# ---- Preprocessing v2 (Hindi repair, research/hindi_repair_report.md). v1 above is unchanged. ----------------------
# v2 = decode -> mono -> 16 kHz -> DC/rumble high-pass -> trim silence -> loudness normalisation (+ peak limit).
# Applied identically to both classes; the parameters below are recorded with every processed file.
PREPROC_V2 = {"id": "v2", "sample_rate": SAMPLE_RATE, "resampler": "soxr HQ", "highpass_hz": 40.0, "highpass_order": 2,
              "highpass_filter": "Butterworth, zero-phase (sosfiltfilt)", "trim_top_db": 40.0, "trim_frame_ms": 25.0,
              "loudness_target_dbfs": -26.0, "loudness_measure": "RMS of active 25 ms frames (> loudest - 40 dB)",
              "peak_limit": 0.95, "output": "PCM_16 WAV, mono"}


def highpass_dc(wave: np.ndarray, sr: int = SAMPLE_RATE, fc: float = PREPROC_V2["highpass_hz"],
                order: int = PREPROC_V2["highpass_order"]) -> np.ndarray:
    """Remove DC offset and sub-audio rumble (zero-phase, so no delay)."""
    from scipy.signal import butter, sosfiltfilt

    if wave.size < 64:
        return wave - wave.mean()
    return sosfiltfilt(butter(order, fc, "highpass", fs=sr, output="sos"), wave).astype(np.float32)


def active_rms_dbfs(wave: np.ndarray, sr: int = SAMPLE_RATE, top_db: float = 40.0, frame_ms: float = 25.0) -> float:
    hop = int(sr * frame_ms / 1000)
    n = wave.size // hop
    if n == 0:
        return float(10 * np.log10(np.mean(wave**2) + 1e-12))
    frames = wave[: n * hop].reshape(n, hop)
    e = np.mean(frames**2, axis=1)
    e_db = 10 * np.log10(e + 1e-12)
    return float(10 * np.log10(e[e_db > e_db.max() - top_db].mean() + 1e-12))


def loudness_normalise(wave: np.ndarray, sr: int = SAMPLE_RATE, target_dbfs: float = PREPROC_V2["loudness_target_dbfs"],
                       peak_limit: float | None = PREPROC_V2["peak_limit"]) -> tuple[np.ndarray, dict]:
    """Scale to the target active RMS; if that would exceed the peak limit, scale to the peak limit instead.
    peak_limit=None: pure linear gain, no limiting (v2.1)."""
    gain = 10 ** ((target_dbfs - active_rms_dbfs(wave, sr)) / 20)
    peak = float(np.abs(wave).max()) * gain
    limited = peak_limit is not None and peak > peak_limit
    if limited:
        gain *= peak_limit / peak
    return (wave * gain).astype(np.float32), {"gain_db": round(float(20 * np.log10(gain + 1e-12)), 3), "peak_limited": bool(limited)}


def preprocess_v2_wave(wave16k: np.ndarray) -> tuple[np.ndarray, dict]:
    """v2 chain on an already decoded 16 kHz mono waveform (so both classes enter at the same point)."""
    y = trim_silence(highpass_dc(wave16k), SAMPLE_RATE, PREPROC_V2["trim_top_db"], PREPROC_V2["trim_frame_ms"])
    return loudness_normalise(y)


# ---- Preprocessing v2.1 (research/hindi_v2_1_validation.md; team decision D-H5) ------------------------------------
# Same as v2 but WITHOUT the 0.95 peak limiter (it touched 9.4 % of real clips and 0 % of fakes, i.e. class-dependent),
# and stored as 32-bit float WAV so no sample can clip after the linear loudness gain. One linear gain per clip, the
# same rule for both classes; nothing non-linear is applied to either class.
PREPROC_V2_1 = {**PREPROC_V2, "id": "v2.1", "peak_limit": None, "output": "FLOAT (32-bit) WAV, mono; no limiter, no clipping"}


def preprocess_wave(wave16k: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    """v2 / v2.1 chain on an already decoded 16 kHz mono waveform, driven by a recorded parameter set."""
    y = trim_silence(highpass_dc(wave16k, SAMPLE_RATE, params["highpass_hz"], params["highpass_order"]),
                     SAMPLE_RATE, params["trim_top_db"], params["trim_frame_ms"])
    return loudness_normalise(y, SAMPLE_RATE, params["loudness_target_dbfs"], params["peak_limit"])


def remove_pauses(wave: np.ndarray, sr: int = SAMPLE_RATE, top_db: float = 40.0, frame_ms: float = 25.0,
                  fade_ms: float = 10.0) -> np.ndarray:
    """DIAGNOSTIC ONLY (pause-controlled experiment, D-H4): keep frames within top_db of the loudest frame and join
    the kept runs with short linear cross-fades. Never part of the primary preprocessing."""
    hop, fade = int(sr * frame_ms / 1000), int(sr * fade_ms / 1000)
    n = wave.size // hop
    if n == 0:
        return wave
    fr = wave[: n * hop].reshape(n, hop)
    e = 10 * np.log10((fr**2).mean(1) + 1e-12)
    keep = e > e.max() - top_db
    out: list[np.ndarray] = []
    i = 0
    while i < n:
        if not keep[i]:
            i += 1
            continue
        j = i
        while j < n and keep[j]:
            j += 1
        seg = wave[i * hop : j * hop].copy()
        if out and seg.size > fade and out[-1].size > fade:
            w = np.linspace(0, 1, fade, dtype=np.float32)
            out[-1][-fade:] = out[-1][-fade:] * (1 - w) + seg[:fade] * w
            seg = seg[fade:]
        out.append(seg)
        i = j
    return np.concatenate(out).astype(np.float32) if out else wave
