"""Deterministic codec / channel round-trips for 16 kHz mono speech (architecture.md §4.7 M07, data_flow.md S9).

Every condition is encode -> decode -> back to 16 kHz -> delay-compensated -> same length as the input.
Two backends:
  pyav        bundled FFmpeg in PyAV (opus, mp3, aac, G.711 mu-law/A-law, G.722, plain resampling); always available
  ffmpeg_cli  a system `ffmpeg` binary, for encoders PyAV lacks (AMR-NB / AMR-WB); used only if the encoder is listed
An unavailable encoder raises CodecUnavailable; it is never silently replaced by a different codec.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tempfile
from functools import cache
from pathlib import Path
from typing import Literal

import av
import numpy as np
import soundfile as sf
import soxr
from pydantic import BaseModel, ConfigDict

SR = 16000
MAX_LAG_S = 0.25  # codec delays are a few ms to ~50 ms; search a generous window


class CodecUnavailable(RuntimeError):
    pass


class Condition(BaseModel):
    """One channel condition. Everything that changes the output is a field, so `params_hash` identifies the output."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str                                    # e.g. opus_voip_16k ; used in transform="sim:<id>" and folder names
    kind: Literal["codec", "resample"]
    role: Literal["train_aug", "eval_heldout"]  # eval_heldout = codec/bitrate never used for training augmentation
    backend: Literal["pyav", "ffmpeg_cli"] = "pyav"
    encoder: str | None = None                 # FFmpeg encoder name (libopus, libmp3lame, aac, pcm_mulaw, g722, libopencore_amrnb...)
    container: str | None = None               # FFmpeg muxer (ogg, mp3, adts, wav, amr...)
    codec_sample_rate: int = SR                # rate the encoder runs at (8000 for narrowband codecs)
    bitrate: int | None = None                 # requested bits/s (None for fixed-rate codecs such as G.711)
    options: dict[str, str] = {}               # encoder options, e.g. {"application": "voip"}
    description: str = ""

    def params_hash(self) -> str:
        keys = self.model_dump(exclude={"description", "role"})
        return hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()[:8]


def tool_versions() -> dict:
    v = {"pyav": av.__version__, "libavcodec": ".".join(map(str, av.library_versions["libavcodec"])),
         "soxr": soxr.__version__, "numpy": np.__version__}
    exe = shutil.which("ffmpeg")
    if exe:
        v["ffmpeg_cli"] = subprocess.run([exe, "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    return v


@cache
def _cli_encoders() -> frozenset[str]:
    exe = shutil.which("ffmpeg")
    if not exe:
        return frozenset()
    out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return frozenset(line.split()[1] for line in out.splitlines() if line.startswith(" A"))


def available(c: Condition) -> tuple[bool, str]:
    if c.kind == "resample":
        return True, "soxr"
    if c.backend == "pyav":
        try:
            av.codec.Codec(c.encoder, "w")
            return True, f"pyav/{c.encoder}"
        except Exception:  # noqa: BLE001 (UnknownCodecError and friends)
            return False, f"encoder {c.encoder} not in PyAV's bundled FFmpeg"
    if not shutil.which("ffmpeg"):
        return False, "no system ffmpeg binary on PATH"
    if c.encoder not in _cli_encoders():
        return False, f"system ffmpeg lacks encoder {c.encoder}"
    return True, f"ffmpeg_cli/{c.encoder}"


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    return x if sr_in == sr_out else soxr.resample(x, sr_in, sr_out, quality="HQ").astype(np.float32)


def _to_s16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 32767 / 32768) * 32768).round().astype(np.int16)


def _pyav_roundtrip(x: np.ndarray, c: Condition) -> tuple[np.ndarray, int, int]:
    buf = io.BytesIO()
    with av.open(buf, "w", format=c.container) as out:
        st = out.add_stream(c.encoder, rate=c.codec_sample_rate, layout="mono")
        if c.bitrate:
            st.bit_rate = c.bitrate
        st.options = dict(c.options)
        frame = av.AudioFrame.from_ndarray(_to_s16(x)[None, :], format="s16", layout="mono")
        frame.sample_rate = c.codec_sample_rate
        for pkt in st.encode(frame):
            out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)
    data = buf.getvalue()
    with av.open(io.BytesIO(data)) as inp:
        s = inp.streams.audio[0]
        parts, dec_sr = [], s.rate
        for f in inp.decode(s):
            a = f.to_ndarray()
            a = a.astype(np.float32) / 32768 if a.dtype == np.int16 else a.astype(np.float32)
            parts.append(a.reshape(-1) if f.format.is_planar or a.shape[0] == 1 else a.reshape(-1, len(f.layout.channels)).mean(1))
    return np.concatenate(parts), dec_sr, len(data)


def _cli_roundtrip(x: np.ndarray, c: Condition) -> tuple[np.ndarray, int, int]:
    exe = shutil.which("ffmpeg")
    with tempfile.TemporaryDirectory() as td:
        src, enc, dec = Path(td) / "in.wav", Path(td) / f"enc.{c.container}", Path(td) / "dec.wav"
        sf.write(src, _to_s16(x), c.codec_sample_rate, subtype="PCM_16")
        cmd = [exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-c:a", c.encoder]
        if c.bitrate:
            cmd += ["-b:a", str(c.bitrate)]
        for k, v in c.options.items():
            cmd += [f"-{k}", v]
        subprocess.run(cmd + [str(enc)], check=True)
        subprocess.run([exe, "-nostdin", "-loglevel", "error", "-y", "-i", str(enc), "-c:a", "pcm_s16le", str(dec)], check=True)
        y, dec_sr = sf.read(str(dec), dtype="float32", always_2d=True)
        return y.mean(1), dec_sr, enc.stat().st_size


def align(ref: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[np.ndarray, int]:
    """Shift y by the cross-correlation lag (codec delay) and crop/zero-pad to len(ref). Returns (aligned, lag)."""
    n = len(ref) + len(y)
    nfft = 1 << (n - 1).bit_length()
    xc = np.fft.irfft(np.fft.rfft(y, nfft) * np.conj(np.fft.rfft(ref, nfft)), nfft)
    lags = np.r_[np.arange(0, max_lag + 1), np.arange(-max_lag, 0)]
    lag = int(lags[np.argmax(xc[lags])])  # y[t + lag] ~ ref[t]
    y = y[lag:] if lag >= 0 else np.r_[np.zeros(-lag, np.float32), y]
    return (np.r_[y, np.zeros(max(0, len(ref) - len(y)), np.float32)][: len(ref)]).astype(np.float32), lag


def apply(x16k: np.ndarray, c: Condition) -> tuple[np.ndarray, dict]:
    """Round-trip a 16 kHz mono float waveform through condition `c`. Output: 16 kHz, len(x16k) samples."""
    ok, why = available(c)
    if not ok:
        raise CodecUnavailable(f"{c.id}: {why}")
    x = _resample(x16k.astype(np.float32), SR, c.codec_sample_rate)
    if c.kind == "resample":
        y, dec_sr, nbytes = x, c.codec_sample_rate, None
    else:
        y, dec_sr, nbytes = (_pyav_roundtrip if c.backend == "pyav" else _cli_roundtrip)(x, c)
    y = _resample(y, dec_sr, SR)
    y, lag = align(x16k, y, int(MAX_LAG_S * SR)) if c.kind == "codec" else (y[: len(x16k)], 0)
    dur = len(x16k) / SR
    return y, {"implementation": why, "decoded_sample_rate": dec_sr, "delay_samples_16k": lag,
               "encoded_bytes": nbytes, "bitrate_actual_bps": round(nbytes * 8 / dur) if nbytes else None}


def quality(ref: np.ndarray, y: np.ndarray, sr: int = SR) -> dict:
    """Basic, reference-based quality stats (no PESQ/ViSQOL: not installed / licence-restricted)."""
    err = y - ref
    snr = 10 * np.log10((ref**2).sum() / max((err**2).sum(), 1e-12))
    win, hop = 512, 256
    def spec(s):
        fr = np.lib.stride_tricks.sliding_window_view(np.r_[s, np.zeros(win, np.float32)], win)[::hop] * np.hanning(win)
        return np.abs(np.fft.rfft(fr, axis=1)) ** 2
    pr, py = spec(ref), spec(y)
    active = 10 * np.log10(pr.sum(1) + 1e-12) > 10 * np.log10(pr.sum(1).max() + 1e-12) - 40  # ignore near-silent frames
    lsd = np.sqrt(((10 * np.log10(pr[active] + 1e-10) - 10 * np.log10(py[active] + 1e-10)) ** 2).mean(1)).mean() if active.any() else float("nan")
    def bw(p):  # effective bandwidth: highest frequency whose long-term power is within 50 dB of the spectral peak
        lt = 10 * np.log10(p[active].mean(0) + 1e-12) if active.any() else 10 * np.log10(p.mean(0) + 1e-12)
        return float(np.nonzero(lt > lt.max() - 50)[0].max() * sr / win)
    rms = lambda s: 20 * np.log10(np.sqrt((s**2).mean()) + 1e-9)  # noqa: E731
    return {"snr_db": round(float(snr), 2), "lsd_db": round(float(lsd), 2), "bandwidth_hz_ref": round(bw(pr)),
            "bandwidth_hz_out": round(bw(py)), "rms_change_db": round(float(rms(y) - rms(ref)), 2),
            "peak_out": round(float(np.abs(y).max()), 4), "clipped_frac_out": float((np.abs(y) >= 0.999).mean())}
