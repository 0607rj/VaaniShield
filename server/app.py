"""VaaniShield demo backend: FastAPI server that scores audio as REAL or FAKE (synthetic speech).

Run from the repository root:
    .venv\\Scripts\\python -m uvicorn server.app:app --host 127.0.0.1 --port 8000

Endpoints:
    GET  /               web demo page (upload / record from microphone)
    GET  /api/health     server + model status
    GET  /api/models     available models and their measured accuracy
    POST /api/analyze    multipart form: file=<audio/video>, model=aasist|lfcc  ->  verdict JSON

Used by the web page (server/static) and the browser extension (client/extension).
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ml.data.audio import CROP_SAMPLES, SAMPLE_RATE, _decode_av, load_audio, trim_silence, windows
from ml.models.lfcc_cnn import spoof_logit

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
MAX_SECONDS = 600  # longer uploads are cut to the first 10 minutes
SILENCE_DBFS = -55.0  # windows quieter than this are ignored (no speech to judge)
DISCLAIMER = ("Model estimate of synthetic-speech artefacts, not forensic proof. Trained on English ASVspoof 2019 data; "
              "accuracy can drop on other languages, new voice generators and heavily compressed audio.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


class Detector:
    """One model + its dev-chosen decision threshold (on the spoof-logit scale)."""

    def __init__(self, key: str, name: str, model: torch.nn.Module, threshold: float, trim: bool, metrics: dict,
                 crop: int = CROP_SAMPLES, batch: int = 16, tile_short: bool = True, fp16: bool = False):
        self.key, self.name, self.model, self.threshold, self.trim, self.metrics = key, name, model, threshold, trim, metrics
        self.crop, self.batch, self.tile_short, self.fp16 = crop, batch, tile_short, fp16

    @torch.no_grad()
    def score(self, wave: np.ndarray) -> dict:
        if self.trim:
            wave = trim_silence(wave)
        crop = self.crop
        hop = crop // 2
        if wave.size <= crop and not self.tile_short:
            wins = [wave]  # SSL models accept any length: no tiling of short clips
        else:
            wins = windows(wave, crop, hop)
        starts = [0] if wave.size <= crop else [min(i * hop, wave.size - crop) for i in range(len(wins))]
        rms_db = [20 * math.log10(float(np.sqrt(np.mean(w**2))) + 1e-9) for w in wins]
        keep = [i for i, d in enumerate(rms_db) if d > SILENCE_DBFS]
        if not keep:
            return {"error": "no_speech"}
        x = torch.from_numpy(np.stack([wins[i] for i in keep]).astype(np.float32)).to(DEVICE)
        t_model = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16 and DEVICE == "cuda"):
            logits = torch.cat([spoof_logit(self.model(x[j : j + self.batch])).float().cpu()
                                for j in range(0, len(x), self.batch)]).numpy()
        # Calibrated so that 50 % = the model's decision threshold.
        probs = 1 / (1 + np.exp(-(logits - self.threshold)))
        clip_logit = float(logits.mean())
        p_fake = float(1 / (1 + math.exp(-(clip_logit - self.threshold))))
        timeline = [{"start_s": round(starts[i] / SAMPLE_RATE, 2),
                     "end_s": round(min(starts[i] + crop, wave.size if not self.tile_short else max(wave.size, crop)) / SAMPLE_RATE, 2),
                     "fake_probability": round(float(p) * 100, 1)} for i, p in zip(keep, probs)]
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        return {"p_fake": p_fake, "timeline": timeline, "n_windows": len(keep), "spoof_logit": round(clip_logit, 3),
                "model_ms": round((time.perf_counter() - t_model) * 1000), "n_total_windows": len(wins),
                "window_s": round(self.crop / SAMPLE_RATE, 1)}


def _load_aasist() -> Detector | None:
    from ml.models.aasist import AASISTDetector

    model = AASISTDetector("AASIST")
    model.load_upstream_pretrained()
    model.to(DEVICE).eval()
    m = _read_json(ROOT / "runs/full/results/PRETRAINED_AASIST_official_upstream/metrics.json")
    thr = float(m.get("threshold_spoof_logit", 0.0))
    return Detector("aasist", "AASIST (graph attention network, pretrained)", model, thr, trim=False,
                    metrics=_summary(m))


ANTIDEEPFAKE_CROP = 6 * SAMPLE_RATE  # 6 s windows, 3 s hop


def _load_antideepfake() -> Detector | None:
    from ml.models.antideepfake import AntiDeepfakeDetector

    model = AntiDeepfakeDetector("mms-300m")
    ft = ROOT / "runs/multilingual/antideepfake_finetuned/best_head_and_top.pth"
    m = _read_json(ROOT / "runs/multilingual/antideepfake_zeroshot/metrics.json")
    thr = 0.0  # upstream decision rule: softmax P(fake) >= 0.5
    if ft.exists():  # optional fine-tuned weights (VaaniShield Hinglish adaptation)
        model.load_state_dict(torch.load(ft, map_location="cpu", weights_only=True), strict=False)
        m = _read_json(ft.parent / "metrics.json") or m
        thr = float(m.get("threshold_spoof_logit", 0.0))
    model.to(DEVICE).eval()
    metrics = {}
    for k, split in (("hing_test", "test_hinglish"), ("en_test", "test")):
        if k in m:
            metrics[split] = {"eer_percent": round(m[k]["eer"] * 100, 2), "auc": round(m[k]["auc"], 4),
                              "balanced_accuracy_percent": round(m[k]["balanced_accuracy"] * 100, 2), "n_clips": m[k]["n"]}
    return Detector("antideepfake", "VaaniShield multilingual (MMS-300M AntiDeepfake)", model, thr, trim=False,
                    metrics=metrics, crop=ANTIDEEPFAKE_CROP, batch=4, tile_short=False, fp16=True)


HINGLISH_RUNS = ["AASIST_hinglish_own", "AASIST_hinglish_resplit", "AASIST_finetuned"]  # first one found is used


def _load_hinglish() -> Detector | None:
    from ml.models.aasist import AASISTDetector

    for tag in HINGLISH_RUNS:
        d = ROOT / "runs/hinglish" / tag
        if (d / "metrics.json").exists() and (d / "best_net.pth").exists():
            break
    else:
        return None
    m = _read_json(d / "metrics.json")
    model = AASISTDetector("AASIST")
    model.net.load_state_dict(torch.load(d / "best_net.pth", map_location="cpu", weights_only=True))
    model.to(DEVICE).eval()
    metrics = {}
    for k, split in (("hing_test", "test"), ("en_test", "english")):
        if k in m:
            metrics[split] = {"eer_percent": round(m[k]["eer"] * 100, 2), "auc": round(m[k]["auc"], 4),
                              "balanced_accuracy_percent": round(m[k]["balanced_accuracy"] * 100, 2), "n_clips": m[k]["n"]}
    return Detector("hinglish", "AASIST fine-tuned for Hinglish (pilot)", model, float(m["threshold_spoof_logit"]),
                    trim=False, metrics=metrics)


def _load_lfcc() -> Detector | None:
    from ml.inference.predict import load_checkpoint

    ckpt_path = ROOT / "runs/full/checkpoints/FULL_lfcc_cnn_asvspoof_gpu/best.pt"
    if not ckpt_path.exists():
        return None
    model, ckpt = load_checkpoint(ckpt_path, DEVICE)
    p = float(ckpt["threshold_p_spoof"])
    thr = math.log(p / (1 - p))
    res = ROOT / "runs/full/results/FULL_lfcc_cnn_asvspoof_gpu"
    m = _read_json(res / "test_metrics.json") or {}
    summary = _summary_lfcc(m, _read_json(res / "run_info.json"))
    return Detector("lfcc", "LFCC-CNN (VaaniShield baseline, trained by us)", model, thr, trim=True, metrics=summary)


def _summary(m: dict) -> dict:
    out = {}
    for split in ("dev", "test"):
        if split in m:
            s = m[split]
            out[split] = {"eer_percent": round(s["eer"] * 100, 2), "auc": round(s["auc"], 4),
                          "balanced_accuracy_percent": round(s["at_dev_threshold"]["balanced_accuracy"] * 100, 2),
                          "n_clips": s["n"]}
    return out


def _summary_lfcc(test: dict, run_info: dict) -> dict:
    out = {}
    if run_info.get("dev_eer") is not None:
        out["dev"] = {"eer_percent": round(run_info["dev_eer"] * 100, 2), "n_clips": run_info.get("n_dev")}
    if test.get("eer") is not None:
        out["test"] = {"eer_percent": round(test["eer"] * 100, 2), "auc": round(test.get("auc", float("nan")), 4),
                       "n_clips": test.get("n")}
        if "at_threshold" in test:
            out["test"]["balanced_accuracy_percent"] = round((1 - test["at_threshold"]["hter"]) * 100, 2)
    return out


app = FastAPI(title="VaaniShield API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
DETECTORS: dict[str, Detector] = {}


@app.on_event("startup")
def _startup() -> None:
    for loader in (_load_antideepfake, _load_aasist, _load_hinglish, _load_lfcc):
        try:
            d = loader()
            if d:
                DETECTORS[d.key] = d
                print(f"loaded {d.name}  threshold {d.threshold:.3f}  device {DEVICE}")
        except Exception as e:  # keep the server up with whatever loaded
            print(f"WARNING: could not load {loader.__name__}: {e}")
    # warm-up so the first real request is fast
    for d in DETECTORS.values():
        d.score(np.random.default_rng(0).normal(0, 0.05, CROP_SAMPLES).astype(np.float32))


def _decode(data: bytes, filename: str) -> np.ndarray:
    suffix = Path(filename or "").suffix.lower() or ".webm"
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"upload{suffix}"
        p.write_bytes(data)
        try:
            wave = load_audio(p)
        except Exception:
            import soxr

            w, sr = _decode_av(p)  # PyAV handles almost any container (webm, ogg/opus, mp4 video, ...)
            wave = w.mean(axis=1)
            if sr != SAMPLE_RATE:
                wave = soxr.resample(wave, sr, SAMPLE_RATE, quality="HQ").astype(np.float32)
    return wave[: MAX_SECONDS * SAMPLE_RATE]


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "device": DEVICE, "models": list(DETECTORS)}


SHORT = {"auto": "Auto", "antideepfake": "Multilingual", "aasist": "English · AASIST", "hinglish": "Hinglish", "lfcc": "Baseline · LFCC"}
DEFAULT_MODEL = "auto"


@app.get("/api/models")
def models() -> list[dict]:
    keys = sorted(DETECTORS, key=lambda k: list(SHORT).index(k) if k in SHORT else 99)
    return [{"key": k, "name": DETECTORS[k].name, "short": SHORT.get(k, k), "metrics": DETECTORS[k].metrics,
             "default": k == DEFAULT_MODEL} for k in keys]


@app.post("/api/contact")
async def contact(msg: dict) -> dict:
    rec = {k: str(msg.get(k, ""))[:4000] for k in ("name", "email", "topic", "message")}
    rec["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = ROOT / "runs/contact_messages.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True}


def pick_detector(requested: str) -> tuple[Detector, str]:
    """'auto' (default): the Hinglish-adapted model when it is available — it was fine-tuned on Hinglish AND English
    replay, so it covers both — otherwise the English AASIST. An explicit key forces that model (used for testing)."""
    if requested in DETECTORS:
        return DETECTORS[requested], "selected manually"
    if "antideepfake" in DETECTORS:
        return DETECTORS["antideepfake"], ("auto: multilingual model (trained on ~74,000 hours of real and fake speech "
                                           "in 100+ languages: Hindi, English, Hinglish and more)")
    if "hinglish" in DETECTORS:
        return DETECTORS["hinglish"], "auto: Hinglish-adapted model (handles Hindi, English and mixed speech)"
    if "aasist" in DETECTORS:
        return DETECTORS["aasist"], "auto: English model (Hinglish model not trained yet)"
    d = next(iter(DETECTORS.values()))
    return d, "auto: only available model"


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), model: str = Form("auto")) -> dict:
    if not DETECTORS:
        raise HTTPException(503, "no model loaded")
    det, reason = pick_detector(model)
    t0 = time.perf_counter()
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    t_dec = time.perf_counter()
    try:
        wave = _decode(data, file.filename)
    except Exception as e:
        raise HTTPException(415, f"could not decode audio: {e}") from e
    if wave.size < SAMPLE_RATE // 2:
        raise HTTPException(422, "audio too short (need at least 0.5 s)")
    decode_ms = round((time.perf_counter() - t_dec) * 1000)
    r = det.score(wave)
    if r.get("error") == "no_speech":
        raise HTTPException(422, "no speech detected (audio is silent)")
    p = r["p_fake"]
    label = "FAKE" if p >= 0.5 else "REAL"
    return {
        "label": label,
        "verdict": "AI-generated / synthetic voice suspected" if label == "FAKE" else "Likely a genuine human voice",
        "fake_probability": round(p * 100, 1),
        "confidence": round(max(p, 1 - p) * 100, 1),
        "model": det.key,
        "model_name": det.name,
        "model_reason": reason,
        "duration_s": round(wave.size / SAMPLE_RATE, 2),
        "n_windows": r["n_windows"],
        "timeline": r["timeline"],
        "latency_ms": round((time.perf_counter() - t0) * 1000),
        "stages": {
            "upload": {"bytes": len(data), "format": Path(file.filename or "").suffix.lstrip(".").upper() or "audio"},
            "decode": {"ms": decode_ms, "seconds": round(wave.size / SAMPLE_RATE, 2), "sample_rate": SAMPLE_RATE},
            "slice": {"windows": r["n_windows"], "skipped_silent": r["n_total_windows"] - r["n_windows"],
                      "window_s": r["window_s"]},
            "model": {"ms": r["model_ms"], "device": DEVICE.upper(), "name": det.name,
                      "params_millions": round(sum(p.numel() for p in det.model.parameters()) / 1e6, 1)},
        },
        "note": DISCLAIMER,
    }


PAGES = {"": "index.html", "features": "features.html", "how-it-works": "how-it-works.html", "try": "try.html",
         "extension": "extension.html", "faq": "faq.html", "contact": "contact.html"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/{page}")
def page(page: str) -> FileResponse:
    name = PAGES.get(page.removesuffix(".html"))
    if not name:
        raise HTTPException(404, "page not found")
    return FileResponse(STATIC / name)


app.mount("/static", StaticFiles(directory=STATIC), name="static")
