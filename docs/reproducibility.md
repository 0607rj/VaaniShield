# VaaniShield — Reproducibility Specification (v1)

Design only. Goal: **another teammate, on a fresh clone, can regenerate any reported number from committed files plus lawfully re-downloaded data** [DOC:pg §20].

## 1. What is versioned where

| Artefact | Location | In Git? |
|---|---|---|
| Code, configs, manifests (CSV), leakage reports, run directories (metrics, scores CSV, env, config, hashes) | repo | yes |
| `requirements-lock.txt` (pip, hashes) and `environment.md` (ffmpeg version, CUDA, OS) | repo | yes |
| Dataset download instructions + `DOWNLOAD.json` templates + `LICENSES_REGISTER.md` | repo | yes |
| Raw/processed audio, simulated variants, real-app round-trip files | `VAANI_DATA_ROOT` | **no** |
| Checkpoints, inference bundles | `VAANI_CKPT_ROOT` (+ team Drive / Kaggle private dataset) | **no** (sha256 only) |
| Consent forms, speaker identity map | `VAANI_DATA_ROOT/private` (restricted) | **never** |
| Secrets (W&B key, HF token) | `.env` (gitignored) / Colab secrets | **never** |

`.gitignore` must include at least the list in DOC:pg §13 plus `*.ogg *.opus *.m4a *.webm *.pt *.joblib private/ .env`. A CI job fails if any file with an audio or weight extension is tracked.

## 2. Environment

- **Python 3.12** (DEC-6). Hashed locks were produced by `uv pip compile` in Phase 1 (see `environment/README.md`):
  - `environment/lock/py312-windows.txt`: laptop, CPU;
  - `py312-linux.txt`: Linux with CUDA torch;
  - `py312-linux-cpu.txt`: CI and Docker.
  - Locked on 2026-10-07: torch 2.14.1, transformers 5.19.0, numpy 2.5.3. **No torchaudio** (DEC-14).
- **ffmpeg:**
  - Local: a pinned static build, with its version string recorded.
  - Colab/Kaggle: the image ffmpeg, version recorded per run.
  - `scripts/check_ffmpeg.py` asserts that encoders `libopus`, `libmp3lame` and `libopencore_amrnb` exist. If AMR-NB is missing, `amr_nb` is disabled and `narrowband_sim` is used, labelled "AMR-like" (DOC:pg §09).
- **HF checkpoints:** referenced by id **and revision hash**, e.g. `facebook/wav2vec2-xls-r-300m@<sha>`. Download once to a cache dir; the cache path is recorded.
- **Vendored code:** AASIST is copied at a pinned upstream commit, with LICENSE and an `UPSTREAM.md` (URL, commit, local modifications). The same applies to the RawBoost reference.

## 3. Determinism

`seed_everything(seed)` sets `random`, `numpy`, `torch` (CPU and CUDA), `PYTHONHASHSEED`, `torch.backends.cudnn.deterministic=True` and `benchmark=False`. It also calls `torch.use_deterministic_algorithms(True, warn_only=True)`, and warnings are logged to the run dir.

- **DataLoader:** a `worker_init_fn` derives worker seeds from (seed, epoch, worker_id). Per-item RNG is `default_rng(hash((seed, epoch, clip_id)))` (architecture §4.6), so results don't depend on `num_workers`.
- **Augmentation:** codec round-trips are deterministic given (codec, bitrate, ffmpeg version). The ffmpeg version is part of `env.json`.
- **Accepted non-determinism:** GPU atomics can make the same seed differ slightly. Tolerance: dev EER within 0.1 pp across two identical 1-epoch runs (experiment_protocol §8). If larger, investigate before running the matrix.
- **Bootstrap** uses a fixed seed (0) recorded in `metrics.json`.

## 4. Run manifest (`ml/experiments/<run_id>/run.json`)

```json
{
  "run_id": "E3_Tenhics_xlsr_aasist_lite__s2026__20261210T101500__a1b2c3d",
  "exp_id": "E3", "hypotheses": ["H1","H2","H8"],
  "status": "completed",
  "config_path": "config.yaml", "config_sha256": "…",
  "git": {"commit": "a1b2c3d…", "dirty": false, "branch": "main"},
  "env": {"python": "3.12.x", "torch": "…", "cuda": "…", "gpu": "Tesla T4", "ffmpeg": "…", "requirements_sha256": "…"},
  "data": {"manifest_version": "v3", "manifests": {"splits/v3/train.csv": "sha256…", "splits/v3/dev.csv": "sha256…"}, "leakage_report_sha256": "…"},
  "encoder": {"hf_id": "facebook/wav2vec2-xls-r-300m", "revision": "…"},
  "seed": 2026,
  "checkpoint": {"path": "VAANI_CKPT_ROOT/<run_id>/best.pt", "sha256": "…", "epoch": 7, "dev_eer": 0.0},
  "calibrator": {"method": "conditional_temp", "json_sha256": "…"},
  "test_scoring": [{"test_set": "test_hinglish_clean", "manifest_sha256": "…", "scored_at": "…", "scores_sha256": "…"}],
  "compute": {"gpu_hours": 0.0, "peak_mem_gb": 0.0, "platform": "kaggle"},
  "notes": "NOTES.md"
}
```
Numeric values above are placeholders, not results.

## 5. Handoff package per experiment [DOC:pg §15]
- [ ] config file and exact command
- [ ] git commit + requirements lock hash
- [ ] manifest hashes + split summary (stats.csv)
- [ ] checkpoint sha256 + best dev epoch
- [ ] per-file score CSV + metrics.json
- [ ] NOTES.md: what failed, changed or was excluded

## 6. Colab / Kaggle procedure
1. Clone the repo at a **tagged commit**. Install from `requirements-lock.txt`. Run `scripts/check_env.py`, which prints and saves `env.json`.
2. Mount data, either as a Kaggle private dataset (processed audio + sim variants) or as a Drive folder. Set `VAANI_DATA_ROOT` and `VAANI_CKPT_ROOT`.
3. Run `manifest validate` and `manifest leakage` on the configured manifests. **Abort on failure.**
4. Run `train --config ...`. Checkpoints go to `VAANI_CKPT_ROOT` on persistent storage (Drive / Kaggle output), so a session timeout doesn't lose them. Resume from `last.pt` with `--resume` (restores optimizer, scheduler, scaler, RNG and epoch).
5. Copy the run directory, without weights, back into the repo and commit it with the message `run: <run_id>`.

## 7. Independent reproduction check (Phase 9 gate)

A teammate who did not run the original experiment does the following from a fresh clone:
1. Re-runs `score` + `evaluate` on the archived bundle → `metrics.json` must be identical (deterministic scoring).
2. Re-trains one key P2 configuration with seed 2026 → dev EER within the documented tolerance and test EER within the 3-seed range.

The result is recorded in PROJECT_STATE.
