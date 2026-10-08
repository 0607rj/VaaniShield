# VaaniShield — Phased Implementation Checklist (v1)

Design only. This plan **supersedes the phase table in `docs/PLAN_v0.md` §G**.

- Owners: **M1** data & governance · **M2** models · **M3** audio & augmentation · **M4** infra, evaluation & demo (architecture §6).
- Dates are targets for the 2026-10-08 → 2027-03-31 window. Phases overlap where the dependency graph allows.
- Task IDs are `P<phase>.<n>`. A task is done only when its listed test or artefact exists.

## Calendar overview

```
            Oct        Nov        Dec        Jan        Feb        Mar
P1 Found.   ███
P2 Data      ████████████████████████
P3 Base.         ██████████
P4 XLS-R               ████████
P5 Robust.               ███████████
P6 Contrib.                         ████████
P7 Eval.                                ██████████
P8 Demo              ███(MVP)                   ██████(final)
P9 Paper                                   ████████████████████
P10 Patent                                 ██████████████
```

## Integration points (all four lanes meet)
| ID | Date | What must work end-to-end |
|---|---|---|
| I1 | 2026-10-24 | synthetic audio → M03 → M04 → M05 manifest → M06 loader → DummyDetector → M15 metrics (CPU, CI) |
| I2 | 2026-11-14 | ASVspoof LA manifest → B0 train 2 epochs → checkpoint → dev scores → EER (parity test passes) |
| I3 | 2026-11-28 | B1 bundle → Scorer → FastAPI/Gradio MVP |
| I4 | 2026-12-19 | P2 frozen XLS-R trains on T_en+hi with codec+RawBoost; HinglishFake-Test locked |
| I5 | 2027-01-24 | full recipe + calibration + channel estimator + compact model; pre-registration committed |
| I6 | 2027-02-14 | all matrix cells measured or marked unavailable; final bundle in demo |

---

## PHASE 1 — Foundation (2026-10-08 → 10-21)

**Inputs:**
- `PROJECT_STATE.md`, `docs/*`, `research/*`
- laptop (Python 3.12 to install; ffmpeg missing)
- Colab/Kaggle accounts

**Outputs:**
- Git repo with skeleton package, CI and lockfile
- licence register, consent protocol draft
- systematic search log
- M01, M02, M15 and M19 implemented with tests

**Tasks:**
- [x] P1.1 (M4) `git init`; create the directory tree from architecture §3. `pyproject.toml` (package `vaanishield`, src layout). `.gitignore` per reproducibility §1. MIT or Apache-2.0 licence for our code (team decision).
  - **Status:** DONE 2026-10-07: tree created, `git init -b main` (no commit yet). Code licence still a team decision (Q-L).
- [x] P1.2 (M4) Create a Python 3.12 venv. Choose torch/torchaudio/transformers versions with wheels for CPU-Windows and CUDA-Colab. Write `requirements.in` and compile `requirements-lock.txt` with hashes. Write `scripts/check_env.py`.
  - **Status:** DONE: uv venv Python 3.12.13; hashed locks environment/lock/py312-{windows,linux,linux-cpu}.txt; scripts/environment_check.py. torchaudio dropped (DEC-14).
- [ ] P1.3 (M3) Install a pinned ffmpeg static build. Write `scripts/check_ffmpeg.py` (libopus, libmp3lame, libopencore_amrnb). Record versions in `environment.md`.
  - **Status:** PARTIAL: encoder check built into environment_check.py; ffmpeg NOT installed on laptop yet.
- [ ] P1.4 (M4) CI (GitHub Actions or a local pre-commit fallback): ruff, pytest on CPU, and a "no audio/weights tracked" check.
  - **Status:** PARTIAL: .github/workflows/ci.yml + .pre-commit-config.yaml written; hygiene check verified locally (fails on force-added .wav). CI not yet run (no remote).
- [x] P1.5 (M4) M01 config schemas + `load_config` + `config_hash`. Test: round-trip YAML → model → hash is stable across key order.
  - **Status:** DONE: ml/src/utils/config.py + tests.
- [x] P1.6 (M4) M02 `seed_everything`, `capture_env`, `git_state`, `sha256_file`, `sha256_manifest`. Test: manifest hash is invariant to row order.
  - **Status:** DONE: ml/src/utils/repro.py + tests.
- [x] P1.7 (M4) M15 `eer`, `roc_auc`, `rates_at_threshold`, `ece`, `nll`, `brier`, `speaker_bootstrap`. Tests:
  - **Status:** DONE: ml/src/metrics/core.py + tests (official-script parity deferred to P3.6).
  - perfect separation → EER 0;
  - identical distributions → EER ≈ 0.5;
  - known hand-computed toy cases;
  - score-sign test.
- [x] P1.8 (M4) M19 `RunContext` writing the run directory layout. Test: a dummy run produces all files.
  - **Status:** DONE: ml/src/utils/registry.py + tests.
- [x] P1.9 (M1) `LICENSES_REGISTER.md`: one row per resource (licence, URL, access date, redistribution allowed?, NC?, citation). Pre-fill from `research/references.md`.
  - **Status:** DONE: LICENSES_REGISTER.md (several rows still 'to verify').
- [ ] P1.10 (M1) `CONSENT_PROTOCOL.md` + consent form draft, covering: recording, research retention, optional voice cloning for the spoof class, real-app transmission, withdrawal, data deletion. Submit to the supervisor / college ethics process (Q2).
  - **Status:** PENDING: needs supervisor/ethics process (Q2).
- [ ] P1.11 (M2) Run the systematic search from `research_gap.md` §7. Write `research/search_log.md` (query, database, date, hits, relevant papers). Update the literature matrix.
  - **Status:** PENDING.
- [ ] P1.12 (M3) Write the M03/M04 test fixtures generator (synthetic sine/noise/silence/clipped/stereo/48k files created at test time).
  - **Status:** PENDING (moves with M03 in Phase 2).

**Acceptance criteria:**
- A fresh clone + `pip install -r requirements-lock.txt` + `pytest` passes on the laptop (CPU) and in a Colab notebook.
- `check_ffmpeg.py` passes, or AMR fallback is documented.
- The CI no-audio check fails on a deliberately added `.wav`, verified once.
- The search log exists with ≥ 4 queries × ≥ 3 databases.

**Failure condition:** no torch build supports Python 3.12 on both platforms, **or** ffmpeg codecs are unavailable, **or** the CI cannot run.

**Fallback:**
- Python 3.11.
- ffmpeg from conda-forge.
- AMR-NB replaced by `narrowband_sim` (8 kHz low-pass + resample), labelled honestly.
- CI replaced by a mandatory `pre-commit run --all-files && pytest` before each merge, checked in review.

---

## PHASE 2 — Data (2026-10-13 → 12-20)

**Inputs:**
- licence register; consent approval (for P2.9+)
- M03/M04 implementations
- official download pages

**Outputs:**
- processed audio for all sources
- source manifests, speaker links, split manifests v1+, leakage reports, stats
- locked HinglishFake-Test incl. simulated + real-app variants
- `DATASET_CARD.md` draft

**Tasks:**
- [ ] P2.1 (M3) M03 `validate_file`, `decode`, `to_mono_16k` + tests (architecture §4.3).
- [ ] P2.2 (M3) M04 `compute_qc`, `vad_trim`, `safe_peak_normalize` + tests. Write the VAD policy v1 (energy-based or webrtcvad; thresholds in config) to `docs/decisions/vad_policy_v1.md`.
- [ ] P2.3 (M1) `preprocess` CLI: idempotent, writes `qc.csv` / `qc_rejected.csv`, skips existing outputs by sha.
- [ ] P2.4 (M1) **ASVspoof 2019 LA:** download (7.12 GB). Record DOWNLOAD.json and the licence text. Build a source manifest that keeps the official protocol splits. Map attack IDs A01–A19 to `generator` and `generator_family`. Check that LA eval has 13 attacks.
- [ ] P2.5 (M1) **OpenSLR 104 test archive (443 MB) first:**
  - listen to 20 clips;
  - inspect transcripts and utterance-ID structure;
  - decide the `speaker_group` derivation rule and document it;
  - then the train archive (7.3 GB).
- [ ] P2.6 (M1) **Kathbath-Hindi + IndicSynth-Hindi:**
  - download Hindi subsets only;
  - parse IndicSynth `metadata.csv` (TTS: target speaker, target clip; VC: source and target speaker/clip) into `speaker_links.csv`;
  - verify that IndicSynth speaker IDs match Kathbath IDs on a 50-row sample;
  - choose a speaker-balanced subset (target: ≥ 60 speaker groups, ~20–40 h total; size set by Phase 3 throughput).
- [ ] P2.7 (M1) M05 `ManifestRow`, `read_manifest`, `validate_manifest`, `build_speaker_graph`, `assign_splits`, `check_leakage` + **negative tests** (synthetic leaky manifests must fail each of the 6 rules).
- [ ] P2.8 (M1) Create split manifests v1 (`T_en`, `T_hi`, dev, `test_hi_*` folds), plus `leakage_report.json` and `stats.csv`. Lock `test_la19` and `test_hi_*` in `test_lock.yaml`.
- [ ] P2.9 (M1) **HinglishFake-Test design:**
  - ≥ 150 natural code-switched prompts (daily-life, banking/OTP-scam style without real names, family requests);
  - Devanagari + Latin script per the XTTS guidance [DOC:dg p5];
  - per-prompt switch-point annotation.
- [ ] P2.10 (M1) **Consented recordings:**
  - target ≥ 20 speakers, 15–30 prompts each, on their own phones;
  - pseudonymous IDs; consent records stored privately;
  - speakers who consent to cloning are flagged.
- [ ] P2.11 (M3) **Spoof generation** (only from consented voices or the model's own built-in voices):
  - Hinglish-XTTS (4 voices + consented clones where allowed);
  - ≥ 1 non-XTTS family, chosen in P1.11/Q6 with a licence check;
  - generation parameters logged per clip.
- [ ] P2.12 (M1) Hinglish pool split by speaker group into `train_cs` / `dev_cs` / `test_hinglish`. At least one generator family appears in **test only**. Lock `test_hinglish_clean`.
- [ ] P2.13 (M3) Simulated test conditions: for locked test clips, offline Opus {12, 16, 24 kbps}, MP3 {32, 64 kbps}, AMR-NB 12.2 kbps (or narrowband_sim), and resample 8 kHz. At least one setting is excluded from the training codec list. Rows carry `parent_clip_id`.
- [ ] P2.14 (M1+M3) **Real-app round-trips** for a balanced test subset (target ≥ 200 clips per app/mode) via M21:
  - ≥ 2 apps × ≥ 2 send modes (audio file as media; voice note recorded by playing the file into the phone mic = a "re-recorded" condition, labelled separately);
  - log app version, OS, device and date;
  - `verify_reencoded` on every file; passthrough files are excluded and counted.
- [ ] P2.15 (M1) `DATASET_CARD.md` v1: composition, licences, consent, transformations, known biases (lecture-domain OpenSLR, read-prompt team speech, limited speakers).

**Acceptance criteria:**
- `manifest leakage` returns `ok=true` for every split version used.
- Every test manifest is in `test_lock.yaml`.
- Every row has licence + sha256.
- Random listening audit: 20 clips per source, labels confirmed by provenance (not by ear) [DOC:dg p8]. Recorded in `docs/audits/listening_audit_v1.md`.
- `stats.csv` produced.
- Real-app passthrough rate is reported.

**Failure condition:**
- consent approval not granted by 2026-11-15; or
- fewer than 10 consenting speakers; or
- IndicSynth/Kathbath IDs cannot be linked; or
- the apps do not re-encode in any tested mode.

**Fallback:**
- **No consent:** HinglishFake-Test uses OpenSLR 104 bona fide (test groups) + Hinglish-XTTS/CS-FLEURS XTTS-Test2 spoof only. The paper states the reduced scope (no natural phone speech, no cloned participant voices).
- **Fewer speakers:** report per-speaker results and widen CIs. Do not pad with same-speaker clips.
- **Unlinkable IDs:** treat each IndicSynth target speaker as its own group and exclude VC-source-ambiguous clips.
- **No re-encoding:** report that finding (it is itself a result) and rely on simulated conditions for RQ2. Re-scope H5–H7 to "simulated only".

---

## PHASE 3 — Baselines (2026-10-27 → 11-28)

**Inputs:**
- `T_en` manifests (P2.4/P2.8)
- M01/M02/M15/M19
- vendored AASIST at a pinned commit

**Outputs:**
- M06 dataset, M08 LFCC, LCNN head, B1 adapter, M12 Detector, M13 train loop, M14 checkpoint, M18 runner
- E0/E1 runs on `T_en`; throughput measurements; parity report

**Tasks:**
- [ ] P3.1 (M4) M06 `ClipDataset`, `crop_or_pad` (repeat-pad), sampler, collate + tests:
  - same (seed, epoch, clip) gives an identical tensor;
  - the augmenter guard raises on non-train splits.
- [ ] P3.2 (M2) M08 `lfcc` + test (shape; matches a reference implementation on one file within tolerance). M11 `LCNNHead`.
- [ ] P3.3 (M2) Vendor official AASIST (+ LICENSE, UPSTREAM.md). Write a B1 adapter to the `Detector` contract. Load the official pretrained checkpoint and score LA eval; record score direction and EER as an **implementation check** (not our result).
- [ ] P3.4 (M2) M12 `Detector`, `DetectorOutput`, `build_detector` + shape tests for every `ModelCfg.kind` with random input.
- [ ] P3.5 (M4) M13 `train` with AMP, grad-accum, early stop, `--overfit-32`, `--swap-labels`, `--resume`. M14 save/load/export.
- [ ] P3.6 (M4) M18 `score_manifest`, `evaluate`, `TestLock`. Parity test of `eer` against the official ASVspoof 2019 eval script on B1 scores (≤ 1e-6). min t-DCF for LA.
- [ ] P3.7 (M2) Train B0 on `T_en`, 2 seeds; train B1 from scratch on `T_en`, 2 seeds. Evaluate on LA dev, then LA eval (`test_la19`, `--confirm-final`).
- [ ] P3.8 (M4) Measure throughput on a T4: clips/s for B1 and for a frozen XLS-R forward pass at batch 8, fp16 (using random input). Compute the GPU-hour budget of the experiment matrix. Record in PROJECT_STATE.
- [ ] P3.9 (M4) Silence probe on LA (experiment_protocol §8). Record the outcome with and without VAD policy v1.

**Acceptance criteria:**
- The sanity checks (overfit, label swap, score direction, determinism) pass.
- The EER parity test passes.
- B1 trained by us on LA lands in a range consistent with the published behaviour; any large deviation is investigated and explained before moving on (DOC:syn review checkpoint).
- Two seeds are reported with their spread.
- The budget table exists.

**Failure condition:**
- B1 reproduction is far from published behaviour (e.g. LA eval EER more than ~3× the paper's average) with no explanation; or
- the training loop is non-deterministic beyond tolerance.

**Fallback:**
- Use the official pretrained AASIST checkpoint for B1 and declare "not retrained".
- Debug data first: protocol parsing, label mapping, crop length 64,600, silence policy. Then the model. **Stop rule:** do not start Phase 4 training until the baseline pipeline reproduces (DOC:pg §14).

---

## PHASE 4 — XLS-R (2026-11-17 → 12-19)

**Inputs:**
- Phase 3 pipeline
- `T_en`, `T_hi` manifests
- HF checkpoint `facebook/wav2vec2-xls-r-300m` (revision pinned)

**Outputs:**
- M09 encoder, M10 aggregations, adapter, M11 `AttnStatsHead` and `AASISTLiteHead`
- E2 and E3 on `T_en` and `T_en+hi` (seed 2026), then 3 seeds
- layer-weight logs; memory/latency profile

**Tasks:**
- [ ] P4.1 (M2) M09 `SSLEncoder` with `freeze`, `unfreeze_top_k`, `truncate_after_layer`, and output `[B,25,T,1024]`. Tests:
  - shape for L = 64600 (T ≈ 201 frames);
  - no grad when frozen;
  - truncation reduces the parameter count;
  - per-utterance normalisation matches the HF feature extractor.
- [ ] P4.2 (M2) M10 `WeightedLayerSum` (LayerNorm per layer, softmax α, init uniform), `LastLayer`, `FixedLayer` + tests (α sums to 1; gradient flows to w).
- [ ] P4.3 (M2) M11 `Adapter`, `AttnStatsHead`, `AASISTLiteHead`. The AASIST-lite config (widths, graph nodes, pooling ratios) is fixed in `configs/model/aasist_lite.yaml`. Parameter count is printed and asserted ≤ a configured maximum.
- [ ] P4.4 (M4) Memory profile on a T4: batch {4, 8, 16} × fp16 × frozen. Choose batch and grad_accum so the effective batch is 32.
- [ ] P4.5 (M2) Run E2 and E3 on `T_en` (seed 2026). Evaluate LA dev. Run the sanity checks again for SSL models.
- [ ] P4.6 (M2) Run E3 on `T_en+hi` (seed 2026). Log α per epoch. Plot α (`paper/figures/layer_weights_*.png`).
- [ ] P4.7 (M4) Implement `--resume` robustness for Colab/Kaggle timeouts. Test by killing a run mid-epoch and resuming to the same final dev EER (within tolerance).
- [ ] P4.8 (M2) Choose and pin the English-only wav2vec2 checkpoint for A3a (comparable size; document the reason).

**Acceptance criteria:**
- E3 trains end-to-end on a T4 within session limits (with resume).
- Dev EER on LA dev and Hindi dev is reported.
- α logged.
- No leakage report failures.
- Peak memory and step time are recorded.

**Failure condition:**
- out-of-memory at batch 4 fp16 with frozen encoder; or
- the T4 budget cannot cover ≥ 3 seeds of E3; or
- AASIST-lite does not converge (overfit-32 fails).

**Fallback:**
- Cache only the CNN-feature-extractor output, or use `truncate_after_layer=12`.
- Reduce the crop to 3 s (documented as a deviation).
- Replace AASIST-lite with `AttnStatsHead` as the main P model (H8 is still testable).
- Reduce the matrix via the budget gate (experiment_protocol §3).

---

## PHASE 5 — Robustness (2026-11-24 → 2027-01-04)

**Inputs:**
- M03 and the ffmpeg codec check
- Phase 4 model
- Hinglish train/dev pools (P2.12)

**Outputs:**
- M07 (CodecAug, RawBoostAug, LangMixAug, ResampleChain, AugPipeline)
- optional offline sim cache
- splice probe results (A2p)
- `T_en+hi+cs` full-recipe E3 (seed 2026)

**Tasks:**
- [ ] P5.1 (M3) `CodecAug`: ffmpeg round-trip for opus/mp3/amr_nb/amr_wb/speex_wb. Lag compensation by cross-correlation; output length preserved. Tests:
  - output length = input length;
  - the alignment lag after compensation is ≤ 1 ms on a chirp;
  - the logged flags reproduce identical output.
- [ ] P5.2 (M3) `RawBoostAug` ported from the reference implementation (algorithms 1–8). Test: reproduces reference output for a fixed seed on one file.
- [ ] P5.3 (M3) `LangMixAug`, symmetric design (architecture §4.7): pool index by (split, class, language, speaker_group); pause-boundary detection via VAD; 10–30 ms cross-fade; RMS matching.
- [ ] P5.4 (M3) Splice probe (A2p): boundary features (energy/spectral discontinuity in ±50 ms around the splice vs random positions) → logistic regression. Run for the symmetric and asymmetric designs (H3 control, H4).
- [ ] P5.5 (M3) `AugPipeline` with per-op probabilities from config and an `aug_log`. Offline sim cache CLI (K = 4 variants per train clip), if online codec cost exceeds 30 % of step time.
- [ ] P5.6 (M2) Small dev grid (≤ 6 configurations, seed 2026): RawBoost algorithm × p_codec × p_langmix. Freeze the chosen recipe into `configs/recipes/full_v1.yaml`.
- [ ] P5.7 (M2) E3 full recipe on `T_en+hi+cs` (seed 2026). Dev EER on clean and simulated dev.

**Acceptance criteria:**
- All augmentations are deterministic under a fixed seed and are logged.
- The symmetric splice probe gives AUC within 0.50 ± 0.05, **or** the language-mix design is revised until it does.
- The full-recipe E3 trains and its dev EER is reported per dev condition.
- Augmentation is never applied to dev/test (guard test passes).

**Failure condition:** symmetric splices remain detectable (probe AUC > 0.55) after two design revisions; or codec augmentation makes clean dev EER worse by more than 1 pp absolute.

**Fallback:**
- Drop language-mix augmentation from the main recipe. Report A2 as a negative/inconclusive result, and keep the probe finding as a contribution to the shortcut discussion.
- Lower p_codec, or use codec-only on a 50 % sub-batch.
- Report the trade-off honestly (H7 non-inferiority fails).

---

## PHASE 6 — Proposed contribution (2027-01-01 → 01-24)

The contribution is: (a) channel-conditioned calibration (C4); (b) the paired sim-vs-real channel analysis tooling (C2); (c) the compact detector (K1). The C1 benchmark is produced in Phase 2.

**Inputs:**
- Phase 5 model
- dev pool with simulated condition labels
- locked paired test sets

**Outputs:**
- M16 (GlobalTemperature, ConditionalTemperature, ChannelEstimator)
- M17 Scorer with conditional calibration; compact-model bundle
- `docs/preregistration.md` (committed before Phase 7)

**Tasks:**
- [ ] P6.1 (M4) `GlobalTemperature`: fit log T (and optional bias) by LBFGS on dev NLL. Test: ECE on synthetic miscalibrated logits decreases; ranking is unchanged (AUC equal).
- [ ] P6.2 (M3) `channel_features` + `ChannelEstimator` (logistic regression or a small MLP; scikit-learn). Trained on simulated conditions from train/dev only. Report a confusion matrix on **simulated dev**. Test-set behaviour is checked only in Phase 7.
- [ ] P6.3 (M4) `ConditionalTemperature`: per-condition (T_c, b_c, thr_c), with fallback to global when n_c < 200, and serialisation. Tests: JSON round-trip; fallback path.
- [ ] P6.4 (M4) M17 `Scorer`: `make_windows`, batched window scoring, mean-logit aggregation, calibration by estimated condition, warnings, `ScoreResult`. Test: window coverage for d < L, d = L, d = 2.5 L; deterministic output.
- [ ] P6.5 (M2) K1 compact model: choose k* = the highest layer index with cumulative α ≥ 0.9 (rule fixed now). Train the truncated P2 (seed 2026) and measure CPU latency (experiment_protocol §5).
- [ ] P6.6 (M1) Write `docs/preregistration.md`: H1–H13 margins, run IDs, test sets, decision rules. Get team + supervisor sign-off and commit before any Phase 7 test scoring.
- [ ] P6.7 (M1) Prepare the switch-point annotations for the exploratory analysis X1 (from P2.9 prompt annotations, aligned by forced alignment or manual marking on a subset).

**Acceptance criteria:**
- Calibrators reduce dev ECE versus uncalibrated, or the lack of reduction is reported.
- The ChannelEstimator reaches a reported accuracy on simulated dev.
- The Scorer passes its unit tests and gives identical outputs to `score_manifest` on 50 clips.
- The pre-registration is committed (git hash recorded in PROJECT_STATE).

**Failure condition:**
- the channel estimator is near chance on simulated dev (balanced accuracy < 0.5 for 6 classes); or
- conditional calibration is indistinguishable from global on dev.

**Fallback:**
- Reduce the conditions to {clean, compressed} (2 classes).
- If still uninformative, keep C4 as a negative result: the global temperature fitted on clean+sim becomes the deployed method, and H12 is reported as not supported.

---

## PHASE 7 — Evaluation (2027-01-15 → 02-14)

**Inputs:**
- pre-registration
- frozen recipes and configs for the full matrix
- locked tests
- GPU budget

**Outputs:**
- all matrix runs (3 seeds) with run dirs
- test scores for pre-registered runs
- `metrics.json` with slices and CIs
- hypothesis verdict table
- error analysis
- In-the-Wild external check

**Tasks:**
- [ ] P7.1 (M2) Launch the 3-seed runs for E0–E3, A1–A6, K1 per the experiment protocol (budget-gated). Track them in `docs/run_tracker.md` (cell → run_ids → status).
- [ ] P7.2 (M4) For each completed run: calibrate (dev), bundle, and score each pre-registered test set once with `--confirm-final`. Entries go to `test_access_log.csv`.
- [ ] P7.3 (M4) `evaluate` with all slices (experiment_protocol §6) + speaker bootstrap (1,000) + paired comparisons per hypothesis.
- [ ] P7.4 (M1) Write the hypothesis verdict table `docs/results/hypotheses.md`: H1–H13 → supported / not supported / inconclusive, with effect size and CI. Every cell links to run IDs.
- [ ] P7.5 (M3) Channel analysis (C2): paired clean vs sim vs app per clip. Compute ΔEER, Δfalse-alarm, ΔECE, and Spearman ρ across configurations (H6). Add the passthrough/re-recorded condition notes.
- [ ] P7.6 (M2) Error analysis: 50 worst false alarms and 50 worst misses on Hinglish tests. Listen and tag (generator family, channel, QC flags, switch density). No re-training based on this.
- [ ] P7.7 (M4) In-the-Wild: score the final P2 and B1 bundles once.
- [ ] P7.8 (M4) Latency table (CPU + T4) for B1, P2 and K1 bundles.

**Acceptance criteria:**
- Every pre-registered cell is "measured" or "not measured — reason".
- No test set is scored outside the logged procedure.
- Hypothesis table complete.
- Tables regenerate from `metrics.json` via M22 without manual edits.

**Failure condition:**
- the GPU budget runs out before all pre-registered E/A cells have ≥ 1 seed; or
- a leakage bug is discovered after test scoring.

**Fallback:**
- **Budget:** report single-seed cells clearly marked, and prioritise H1, H2, H5, H7 and H8 (core RQs).
- **Leakage bug:** invalidate the affected test scores, fix and rebuild the splits, construct replacement test data if the old test was exposed, and document it in PROJECT_STATE and the paper.

---

## PHASE 8 — Demo (MVP 2026-11-24 → 12-05; final 2027-02-10 → 02-28)

**Inputs:**
- M17 Scorer and a bundle (B1 for MVP; best P2 or K1 for final)
- API contract

**Outputs:**
- M20 FastAPI + Gradio app; contract tests; Dockerfile (optional); demo video; MODEL_CARD.md

**Tasks:**
- [ ] P8.1 (M4) FastAPI `/v1/health`, `/v1/model`, `/v1/analyze` per `api_contract.md`; limits module; error mapping; temp-file deletion in `finally`.
- [ ] P8.2 (M4) Gradio UI per api_contract §B.6, including the segment timeline plot and JSON download.
- [ ] P8.3 (M4) Contract tests B.7 (1–9) in CI using a `DummyDetector` bundle; then the real bundle locally.
- [ ] P8.4 (M3) Demo clips: ≥ 6 consented or public-domain clips, each in clean + ≥ 2 channel conditions [DOC:pg §06 item 7]. No real-person impersonation.
- [ ] P8.5 (M1) `MODEL_CARD.md`: intended use, out-of-scope uses, training data, metrics by slice (copied from `metrics.json` by M22), limitations, ethical warnings, contact.
- [ ] P8.6 (M4) Fresh-clone demo test: README instructions → running demo in ≤ 15 min on the laptop (CPU).
- [ ] P8.7 (M4) Record the demo video (3–5 min).

**Acceptance criteria:**
- All contract tests pass.
- Invalid audio fails safely.
- Model version and disclaimer are visible.
- No files are retained after requests (test 7).
- CPU p95 latency for a 10 s clip is reported.
- The fresh-clone test passes.

**Failure condition:** CPU latency of the final model > 10 s per 10 s clip on the laptop; or the bundle cannot load on CPU.

**Fallback:**
- Serve the K1 compact bundle, or B1, for the live demo.
- Run the XLS-R bundle on Colab GPU with a public-share Gradio link for the viva only, with privacy text unchanged.
- ONNX export of the head only (the encoder stays in PyTorch) if that helps latency.

---

## PHASE 9 — Paper (2027-01-20 → 03-15)

**Inputs:**
- `metrics.json`, hypothesis table, `research/related_work.md`, figures, dataset/model cards

**Outputs:**
- IEEE-style manuscript, auto-generated tables/figures, reproducibility appendix
- independent-reproduction record; final synopsis/report text; viva slides

**Tasks:**
- [ ] P9.1 (M4) M22 `make_tables.py` / `make_figures.py`:
  - Table 1: data stats.
  - Table 2: main EER/AUC across clean / sim / app.
  - Table 3: ablations with params and latency.
  - Figure 1: architecture.
  - Figure 2: cross-lingual train-test matrix.
  - Figure 3: EER vs codec/bitrate.
  - Figure 4: DET/ROC and reliability diagrams [DOC:pg §11].
- [ ] P9.2 (M2) Methods and experimental setup sections, generated from the configs (hyperparameters table from YAML).
- [ ] P9.3 (M1) Introduction, related work (from `research/related_work.md` after re-verifying every SEC-level claim), contributions worded per `research_gap.md` §3/§6, ethics and limitations, threats to validity [DOC:pg §16].
- [ ] P9.4 (M3) Channel study section (C2) with the real-app protocol description and passthrough statistics.
- [ ] P9.5 (all) Independent reproduction (reproducibility §7). Record the outcome.
- [ ] P9.6 (M1) Turnitin self-check [DOC:syn declaration]. Supervisor review. Venue choice.
- [ ] P9.7 (M1) **Do not post a preprint or a public repo until P10.5 clears it.**

**Acceptance criteria:**
- Every number in the paper traces to a run ID.
- The novelty wording passes a checklist against `research_gap.md` §3.
- The independent reproduction succeeds within tolerance.
- The supervisor has signed off.

**Failure condition:** core results are inconclusive or negative.

**Fallback:** write it as a benchmark-and-protocol paper. Negative and inconclusive findings, the channel-proxy analysis and the shortcut probes are publishable as evaluation findings (DOC:syn §5.4 success criterion). Do not re-tune on test to "rescue" results.

---

## PHASE 10 — Patent preparation (2027-01-20 → 03-01)

No patentability is claimed. This phase prepares material for the college IPR cell, which decides.

**Inputs:**
- C4 / compact-model / Scorer design and Phase 6–7 results
- `PROJECT_STATE.md` §12 (P-1, P-2)

**Outputs:**
- invention disclosure draft; prior-art search log
- recommendation memo to the IPR cell; publication-timing decision

**Tasks:**
- [ ] P10.1 (M1) Prior-art search for P-1 (channel estimation → condition-specific calibration/threshold → segment timeline with quality warnings) and P-2 (code-switch-aware segment analysis):
  - Google Patents, InPASS (Indian Patent Office), WIPO Patentscope, and the literature (incl. biometrics quality-aware calibration);
  - log queries and hits in `docs/ip/prior_art_log.md` (kept private until filing decision).
- [ ] P10.2 (M4) Technical description: system block diagram, data flow, the claims-relevant technical effect (e.g. measured reduction in false-alarm rate on compressed bona fide speech at a fixed operating point, **only if Phase 7 shows it**), alternative embodiments, and the exact parameters used.
- [ ] P10.3 (M1) Inventorship and contribution record (who conceived what, with dates and commit references). Check NC-licensed data dependencies (IndicSynth, CPML models) and state that the claimed method does not depend on any specific dataset.
- [ ] P10.4 (M1) Meet the IPR cell with the disclosure + prior-art log + results. Note: Indian Patents Act s.3(k) ("computer programme per se") requires professional framing.
- [ ] P10.5 (M1) Publication-timing decision recorded in PROJECT_STATE: provisional filing before the preprint/public repo/viva demo, or a decision not to file.

**Acceptance criteria:**
- The disclosure draft is complete.
- The prior-art log has ≥ 3 databases.
- The IPR cell's written response or meeting note is recorded.
- The publication-timing decision is recorded before any public disclosure.

**Failure condition:**
- prior art substantially anticipates P-1/P-2; or
- the C4 results are negative (H12 not supported); or
- the IPR cell is unavailable before the planned publication.

**Fallback:**
- Do not file; publish openly as research.
- If timing conflicts, delay public posting (not the internal report or viva) until the IPR decision, or proceed with publication once the team accepts losing patent options. Record the decision.

---

## Cross-phase definition of done
- Code merged to `main` with tests and review.
- Run dirs committed.
- PROJECT_STATE updated with the 8-item phase report.
- No new failures in the leakage report.
- No numbers anywhere that are not traceable to `metrics.json`.
