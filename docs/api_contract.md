# VaaniShield — API Contract (v1)

Design only. Two layers:
- **(A)** the Python inference contract (`Scorer.score`), shared by CLI, evaluation and service;
- **(B)** the HTTP API (FastAPI) plus the Gradio UI.

The HTTP response extends the project-guide contract [DOC:pg §12].

## A. Python contract

```python
scorer = Scorer(bundle_dir="bundles/xlsr_aasist_lite-v1", device="cpu")
result: ScoreResult = scorer.score(waveform: np.ndarray, sample_rate: int)
```

**Pre-conditions:**
- `waveform` is 1-D float32/float64 in [-1, 1] (2-D input is accepted as [channels, T] and averaged);
- `sample_rate` ≥ 8000;
- duration within limits (§B.2).

**Post-conditions:**
- `0 ≤ p_spoof ≤ 1`;
- `label == "spoof"` iff `p_spoof ≥ threshold`;
- `segments` cover [0, duration] with 50 % overlap;
- no I/O side effects.

**Errors:** `AudioValidationError(code)` with codes from §B.4.

A bundle directory contains:
- `weights.pt`;
- `config.yaml` (ModelCfg);
- `calibrator.json`;
- `thresholds.json` (per condition plus global);
- `channel_estimator.joblib`;
- `bundle.json` (model_version, run_id, checkpoint sha256, encoder HF id + revision, created_at, training manifests' sha256);
- `MODEL_CARD.md`.

## B. HTTP API

Base path `/v1`. JSON responses use UTF-8. No authentication in the local demo; the optional gateway is out of MVP scope (DEC-5).

### B.1 Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/health` | liveness: `{"status":"ok"}` |
| GET | `/v1/model` | `{"model_version","bundle_sha256","encoder","trained_on":[manifest names],"thresholds":{...},"limits":{...},"disclaimer"}` |
| POST | `/v1/analyze` | multipart form field `file` (audio). Optional form field `return_segments` (bool, default true) |

### B.2 Limits (enforced before decoding; also shown in the UI)
| Limit | Value | Source |
|---|---|---|
| Formats | wav, flac, mp3, ogg/opus, m4a/aac, webm | DOC:pg §02 + browser recording |
| Max upload size | 10 MB | [REC] |
| Duration | 2.0 s ≤ d ≤ 20.0 s (after decode, before VAD) | DOC:pg §02 |
| Sample rate | ≥ 8 kHz input; resampled to 16 kHz | DOC:pg |
| Concurrency | 1 worker, queue length 4 (demo) | [REC] |

### B.3 `POST /v1/analyze` success response (200)
```json
{
  "request_id": "6f1c2a9e-…",
  "label": "spoof",
  "spoof_probability": 0.82,
  "threshold": 0.57,
  "operating_point": "dev_eer",
  "channel_condition": {"estimate": "opus_lo", "confidence": 0.74, "calibration_used": "conditional"},
  "duration_seconds": 6.4,
  "input": {"format": "ogg", "codec": "opus", "sample_rate": 48000, "channels": 1, "resampled": true},
  "quality_warnings": ["heavy_compression"],
  "segments": [
    {"start_s": 0.00, "end_s": 4.04, "spoof_probability": 0.77},
    {"start_s": 2.02, "end_s": 6.06, "spoof_probability": 0.86},
    {"start_s": 2.36, "end_s": 6.40, "spoof_probability": 0.84}
  ],
  "model_version": "xlsr-aasist-lite-v1",
  "bundle_sha256": "3b9f…(12 chars)",
  "processing_ms": 318,
  "disclaimer": "This model estimates whether a clip contains synthetic-speech artifacts. It can be wrong, especially on unseen languages, generators and channels. It is not forensic proof."
}
```
Field rules:
- `label` ∈ {`bonafide`, `spoof`}. The UI may show "likely real" / "likely synthetic" but must show the probability and threshold next to it.
- `spoof_probability` is calibrated and rounded to 2 decimals. Raw logits are not exposed.
- `segments[*].spoof_probability` uses the same calibrator as the clip. The UI labels the timeline "localisation aid, not an explanation" [DOC:syn 4.2.4].
- `quality_warnings` ⊆ {`heavy_compression`, `narrowband`, `clipping`, `low_level`, `mostly_silence`, `short_speech`, `unseen_channel`, `long_input_truncated`}.
  - `unseen_channel` is emitted when channel-estimator confidence < 0.5.
- The disclaimer text is fixed and returned with every response [DOC:pg §12].

### B.4 Error responses
```json
{"request_id":"…","error":{"code":"TOO_SHORT","message":"Audio is 1.3 s; minimum is 2.0 s."}}
```
| HTTP | code |
|---|---|
| 413 | `FILE_TOO_LARGE` |
| 415 | `UNSUPPORTED_FORMAT` |
| 422 | `DECODE_FAILED`, `EMPTY`, `TOO_SHORT`, `TOO_LONG`, `SILENT_AUDIO` |
| 503 | `BUSY` (queue full) |
| 500 | `INTERNAL` (no stack trace in body; logged server-side without audio) |

### B.5 Privacy and logging rules
- The uploaded file is written only to a `tempfile.NamedTemporaryFile` and deleted in a `finally` block. Audio is never logged and never stored by default [DOC:syn 4.4.2].
- Server logs hold only: request_id, timestamp, duration_s, input format, label, p_spoof, processing_ms, error code.
- A downloadable JSON of the response is offered in the UI. There is no server-side history in the MVP (MongoDB deferred, DEC-5).

### B.6 Gradio UI (MVP)
- Upload or microphone record, with accepted formats and the duration limit visible.
- Waveform preview with duration and a resampling note.
- Result card: label, probability, threshold, channel estimate, warnings, model version, processing time, disclaimer.
- Segment timeline plot (matplotlib) aligned with the waveform.
- "Download JSON" button.

### B.7 Contract tests (M23, must pass in CI)
1. 4 s WAV at 16 kHz returns 200 with all fields and valid types.
2. 48 kHz stereo OGG/Opus returns 200, `resampled=true`.
3. 1 s clip returns 422 `TOO_SHORT`; 25 s clip returns 422 `TOO_LONG`.
4. Random bytes return 422 `DECODE_FAILED`; `.txt` returns 415.
5. A 12 MB file returns 413.
6. All-zero audio returns 422 `SILENT_AUDIO`.
7. After a request, the temp directory holds no new files.
8. `label`/`threshold` consistency: `label=="spoof"` ⇔ `spoof_probability ≥ threshold`.
9. Latency is recorded in `processing_ms` and is > 0.
