# VaaniShield — Data Flow (v1)

Design only. This document describes where every byte lives, what transforms it, and what is versioned. Module IDs refer to `docs/architecture.md`.

## 1. Storage layout

```
VAANI_DATA_ROOT/                         (outside Git; local disk / Google Drive / Kaggle dataset)
├── raw/<source>/...                     original archives + extracted files, untouched; + DOWNLOAD.json (url, date, sha256, licence)
├── processed/<source>/<clip_id>.wav     mono, 16 kHz, PCM16, VAD-trimmed (policy v1), peak-normalised
├── processed/<source>/qc.csv            QCReport per clip;  qc_rejected.csv (reason)
├── sim/<source>/<clip_id>__<codec>_<kbps>.wav   offline codec variants (optional cache, architecture.md §4.7)
├── realapp/sent/<batch_id>/...          files sent through apps (copies of processed test clips)
├── realapp/received/<batch_id>/...      files downloaded back from apps (original container kept)
├── realapp/logs/<batch_id>.csv          app, version, OS, device, account, date, send_mode, sent file, received file
├── private/consent/                     signed forms, NEVER synced to shared drives or Git
└── private/speaker_map.csv              real identity → pseudonymous speaker_id (restricted access, M1 only)

repo/data/manifests/                      (in Git — text only)
├── sources/<source>.csv                 one row per processed clip, split=None
├── links/speaker_links.csv              (speaker_a, speaker_b, reason) e.g. VC source↔target, clone reference↔voice
├── splits/<manifest_version>/{train,dev,test_*}.csv
├── splits/<manifest_version>/leakage_report.json
└── splits/<manifest_version>/stats.csv  per split × label × language × source × family × transform counts

repo/ml/configs/test_lock.yaml           {test manifest path: sha256, locked_on, locked_by}
VAANI_CKPT_ROOT/<run_id>/{best.pt,last.pt}    (outside Git; sha256 recorded in run dir)
```

## 2. Stage-by-stage flow

| # | Stage | Module | Input | Output | Deterministic? | Logged |
|---|---|---|---|---|---|---|
| S1 | Acquire | data_scripts | official URL / HF id | `raw/<source>/` + `DOWNLOAD.json` | yes (pinned revision where possible) | url, date, sha256 of archives, licence text copy |
| S2 | Validate + decode | M03 | raw file | `DecodedAudio` | yes | orig sr, channels, codec |
| S3 | Mono + 16 kHz | M03 | DecodedAudio | float32 `[T]` @16k | yes (fixed resampler params) | resampler name/params |
| S4 | QC + VAD trim + peak-normalise | M04 | `[T]` | processed WAV + `qc.csv` row; or rejected | yes | QC metrics, trim amounts, reject reason |
| S5 | Source manifest | M05 | processed dir + source metadata (IndicSynth metadata.csv, OpenSLR utt IDs, consent sheet) | `manifests/sources/<source>.csv` | yes | row count, sha256 per clip |
| S6 | Speaker graph + split | M05 | all source manifests + `speaker_links.csv` + split config | `splits/<ver>/{train,dev,test_*}.csv` | yes (seeded) | config, seed, holdout families |
| S7 | Leakage audit | M05 | split CSVs | `leakage_report.json`; **training is blocked if not ok** | yes | all violations |
| S8 | Test lock | M18 TestLock | test CSVs | `test_lock.yaml` entry | n/a | who/when |
| S9 | Derived test conditions | M07 codec (offline) / M21 | **locked** test clips | `sim/` variants and `realapp/` round-trips as new rows with `parent_clip_id` and `transform`; same split as parent | yes | ffmpeg flags; app/version/date/mode |
| S10 | Load + crop/pad | M06 | split CSV row | `[L]` tensor | train: seeded random crop; eval: centre crop | crop offset in `aug_log` |
| S11 | Augment (train only) | M07 | `[L]` | `[L]` + `aug_log` | seeded per (seed, epoch, clip_id) | ops + params (sampled 1 % of items written to run dir) |
| S12 | Encode → aggregate → head | M09–M12 | `[B,L]` | `DetectorOutput` | yes given seed and deterministic flags | layer weights per epoch |
| S13 | Score dev | M18 | dev CSV | `dev_scores.csv` | yes | per-clip logits |
| S14 | Calibrate | M16 | dev logits (+ dev condition labels) | `calibrator.json`, threshold(s) | yes | method, T_c, thresholds, n per condition |
| S15 | Bundle | M14 | best.pt + calibrator + threshold | `bundle/` | yes | bundle sha256 |
| S16 | Score test (final only) | M17/M18 | **locked** test CSVs + bundle | `test_scores.csv` | yes | entry in `test_access_log.csv` |
| S17 | Metrics + slices + bootstrap | M15/M18 | scores CSV | `metrics.json` | yes (seeded bootstrap) | all slices |
| S18 | Paper tables | M22 | `metrics.json` files | `paper/tables/*.tex/.md` | yes | source run_ids per cell |

## 3. Split policy (normative)

1. **Unit of splitting:** the speaker-graph connected component (`speaker_group`). Edges come from:
   - IndicSynth VC metadata (source speaker ↔ target speaker);
   - TTS metadata (generated voice ↔ target/reference speaker);
   - our clone references (consented participant ↔ the synthetic voice made from them);
   - cross-corpus identity (the same participant in multiple recordings).
2. **Order:** split groups first. Crops, augmentations, simulated variants and real-app variants are created **afterwards, inside the parent's split** (DOC:pg §04 golden rule).
3. **Holdouts:** a `generator_family` held out for an unseen-family fold appears **only** in the test split of that fold. Dev uses only seen families, so model selection never sees the held-out family.
4. **Transcripts (Hindi):** test `transcript_id`s are excluded from train. IndicSynth reuses IndicSUPERB transcripts.
5. **OpenSLR 104:** if speaker IDs are absent, `speaker_group` = recording/tutorial ID parsed from the utterance ID. Documented as a weaker guarantee (R3).
6. **ASVspoof 2019 LA:** keep the official train/dev/eval protocol unchanged for comparability. Its speakers do not overlap other sources.
7. **Ratios (non-ASVspoof pools):** 70/15/15 by group. Balanced by class within ±5 % per split where the data allows. Achieved ratios are reported, not forced.

## 4. Test sets (named, versioned, locked)

| Name | Content | Purpose |
|---|---|---|
| `test_la19` | ASVspoof 2019 LA eval (official) | in-domain parity (E0/E1 sanity) |
| `test_hi_seen` / `test_hi_unseen_<family>` | Kathbath-Hi bona fide + IndicSynth-Hi spoof, held-out groups; one fold per held-out family | RQ1, RQ3 |
| `test_hinglish_clean` | HinglishFake-Test, clean | RQ1, RQ3 |
| `test_hinglish_sim_<codec>` | same clips, simulated codecs incl. ≥1 codec/bitrate unseen in training | RQ2 |
| `test_hinglish_app_<app>_<mode>` | same clips after verified real-app round-trip | RQ2, RQ4 |
| `test_itw` | In-the-Wild, untouched until the pipeline is frozen | external realism |

Real-app and simulated sets are **paired** with the clean set by `parent_clip_id`, which enables paired bootstrap (H5–H7, H11–H12).

## 5. What can and cannot cross boundaries

| From → To | Allowed? |
|---|---|
| Test audio → training (any form, incl. augmented/app copies) | **Never** |
| Test labels → threshold, temperature, checkpoint or hyperparameter choice | **Never**. All selection uses dev only |
| Real-app audio → training | Only experiment A4, only from speakers in a dedicated `train_realapp` pool disjoint from all test groups |
| Simulated codec conditions → dev (for calibration, C4) | Allowed. This is the intended design (TD-19) |
| In-the-Wild → anything but final evaluation | **Never** |
| Consent forms / speaker_map → Git, shared drive, paper | **Never** |
| Raw dataset audio → Git | **Never**. `.gitignore` plus a CI check that fails on audio extensions |

## 6. Inference-time data flow (demo / API)

```
upload (bytes) ─► size/format check (M20 limits) ─► temp file (tmpfs / NamedTemporaryFile)
 ─► M03 validate+decode+mono16k ─► M04 compute_qc (warnings; NO VAD trim beyond policy v1)
 ─► M16 channel_features + ChannelEstimator → condition c
 ─► M17 make_windows (64600 / 32300) ─► batched Detector forward ─► window logits
 ─► per-window p_spoof via calibrator(c) ─► clip logit = mean(window logits) ─► p_spoof(c), label via threshold_c
 ─► JSON response ─► temp file deleted in `finally` (no retention by default)
```
