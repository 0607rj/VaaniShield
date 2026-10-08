# VaaniShield — PROJECT_STATE

> The persistent source of truth for the project. Update it at the end of every phase.
> Tags: [DOC] verified from the supplied PDFs · [EXT] checked externally (dated) · [REC] engineering recommendation · [HYP] hypothesis or assumption.
> Full plan: `docs/PLAN_v0.md`.

**Last updated:** 2026-10-08  **Current phase:** SCOPE RESET. Simplified end-to-end build (see §0).
**Next step:** pre-handoff audit done (research/PRE_HANDOFF_AUDIT.md, §0.23): register + Windows notes fixed; create a PRIVATE repo, commit, push; then the handoff.

---

## 0. SIMPLIFIED SCOPE (2026-10-07, team decision): this section overrides everything below where they conflict

### 0.1 Goal
One working, debuggable pipeline that 4 B.Tech students can understand:

**dataset → audio preprocessing → ML model → training → evaluation → inference → simple server → simple client → end-to-end demo**

The research goals (Hinglish, codec robustness, calibration; RQ1–RQ4) stay. They are reached by extending this pipeline step by step, not by building infrastructure first.

### 0.2 Target repository
```
VaaniShield/
├── client/            simple web page (upload → result)
├── server/            simple HTTP server wrapping ml/inference
├── ml/
│   ├── data/          download helpers, manifests, audio preprocessing, PyTorch Dataset
│   ├── models/        detector models (baseline first)
│   ├── training/      training loop
│   ├── evaluation/    metrics + evaluate script
│   ├── inference/     score one audio file
│   └── configs/       a few plain YAML configs
├── data/{raw,processed,manifests}/
├── research/  tests/  scripts/  docs/
├── README.md  PROJECT_STATE.md  DATASET_CARD.md  MODEL_CARD.md  pyproject.toml  .gitignore
└── project source PDFs (kept locally; ignored by Git)
```

### 0.3 Out of scope until further notice
- Dockerfiles, docker-compose, Kubernetes, cloud deployment.
- CI/CD, GitHub Actions, pre-commit.
- MLflow, Weights & Biases, experiment registry.
- MongoDB, Redis, Celery, queues; authentication; monitoring.
- Paper automation; patent folder.
- Placeholder files; abstractions without an immediate use.

### 0.4 Implementation order (each step must run before the next starts)
1. **Pilot data:** a small ASVspoof 2019 LA subset (+ IndicSynth / OpenSLR pilots already on disk) → manifests (CSV).
2. **Preprocessing:** load → mono 16 kHz → trim → fixed-length crop/pad; PyTorch Dataset.
3. **Baseline model:** one small, well-known model (LFCC + LCNN, or AASIST) trained on the ASVspoof subset.
4. **Evaluation:** EER/AUC on held-out data (metrics already implemented and tested).
5. **Inference:** `score(file) → {label, probability}`.
6. **Simple server:** one endpoint.
7. **Simple client:** one upload page.
8. **Demo**, then scale data → XLS-R → Hindi/Hinglish → codec augmentation → calibration.

### 0.5 Status of pilot data (2026-10-07)
| Source | On disk (gitignored) | Provenance recorded | Note |
|---|---|---|---|
| ASVspoof 2019 LA | 3 CM protocols + README.LA + 20 bona fide + 20 spoof train FLACs (44 files, 6.4 MB) | yes | ODC-By 1.0; full LA.zip (7.6 GB) only ~124 KB/s here, so download it on Colab/Kaggle |
| IndicSynth Hindi | 10 xtts_v2 + 10 freevc24 clips + metadata | yes | CC BY-NC 4.0 |
| OpenSLR 104 | 2 whole lecture WAVs (~41 MB), no transcripts | yes | segment transcripts need the full 443 MB test archive |
| In-the-Wild | `meta.csv` + attribution note only | yes | external test only |
| Kathbath Hindi | nothing | inspect only | licence conflict; m4a needs ffmpeg; HF route needs your HF_TOKEN |

### 0.6 Repository clean-up (done 2026-10-07)
**Removed (unused scaffolding):**
- CI/tooling: `.github/` (CI), `.pre-commit-config.yaml`, ruff config in `pyproject.toml`.
- Docker/service: `docker-compose.yml`, `ml-service/`, `environment/requirements-service.in`, `environment/lock/py312-linux-cpu.txt`.
- Scripts: `scripts/check_repo_hygiene.py`, `ml/data_scripts/verify_environment.py`.
- Experiment infrastructure: `ml/src/utils/registry.py` + its test, `ml/src/train.py` (stub that refused to train), `ml/configs/test_lock.yaml`, `ml/configs/examples/`.
- Placeholders: `ml/src/augment/` (empty), `ml/experiments/`, `ml/notebooks/`, `paper/`, `tests/integration/`, `data/interim/`, placeholder READMEs in `client/`, `server/`, `ml/data_scripts/`, and `.env.example`.

**Edited only to drop references:**
- `tests/unit/test_repo_hygiene_and_cli.py` → `tests/unit/test_evaluate.py` (only the evaluate tests kept).
- `tests/unit/test_config.py` no longer uses the deleted template.
- `README.md`, `environment/README.md`, `pyproject.toml`, `provenance.py` (no `interim` folder).

**Result:** 59 unit tests pass; all downloaders, `environment_check.py` and `vaanishield.evaluate` still run.

### 0.7 Restructure into the §0.2 layout (done 2026-10-07)
- **Code moves:**

| From | To |
|---|---|
| `ml/src/datasets/{manifest,provenance}.py` | `ml/data/` |
| `ml/src/metrics/core.py` | `ml/evaluation/metrics.py` |
| `ml/src/evaluate.py` | `ml/evaluation/evaluate.py` |
| `ml/src/utils/{config,repro}.py` | `ml/training/` |
| `ml/src/utils/env_info.py` | `scripts/` |
| `ml/data_scripts/acquire_*.py` | `scripts/` |

- **Data moves:**

| From | To |
|---|---|
| `data/metadata/{licence_registry.yaml,provenance/,licence_texts/}` | `data/manifests/` |
| `data/external/in_the_wild/` | `data/raw/in_the_wild/` (provenance paths updated + a `relocate` event appended) |
| download logs | `data/raw/_logs/` |

- **Packaging:** `ml` is now the Python package (`from ml.evaluation.metrics import eer`); DEC-15 (the `vaanishield` → `ml/src` mapping) is withdrawn.
- **Checks:** 59 tests pass; all scripts load; 69/69 provenance-recorded files exist with matching sha256.
- `ml/models/`, `ml/inference/` and `ml/configs/` exist but are empty until their code is written.

Docs in `docs/` are kept for reference but are **not** the plan.

### 0.8 First end-to-end ML vertical slice (done 2026-10-07)
**Works:** manifest → speaker-disjoint split → preprocessing → train → checkpoint → reload → evaluate → inference on a new file.

**Data:** ASVspoof 2019 LA pilot only.
- 40 clips (20 bona fide / 20 spoof) from 18 speakers; manifest `data/manifests/asvspoof_pilot.csv`.
- Split by speaker (seed 2026): train 24 clips (13 speakers), dev 8 (2), test 8 (3). Both classes in every split; no speaker overlap (checked).
- IndicSynth / OpenSLR were *not* mixed into training: single-class sources would teach a corpus shortcut. They are used only as out-of-domain inference examples.

**Preprocessing** (`ml/data/audio.py`, identical for train, eval and inference):
- load → mono → 16 kHz (soxr) → trim leading/trailing silence (−40 dB, both classes);
- 64,600-sample crop: random for train, centre for eval;
- repeat-padding for short clips; 50 %-overlap windows for long files.

**Model** (`ml/models/lfcc_cnn.py`): LFCC (20 + Δ + ΔΔ, per-utterance normalised) → 3 conv blocks → global average pool → linear; 14,274 parameters.
- A simplified stand-in for LFCC-LCNN (no max-feature-map layers). Our own implementation.

**Training** (`ml/training/train.py`):
- AdamW, lr 1e-3, batch 8, 30 epochs, class-weighted cross-entropy, CPU, ~0.3 s per epoch.
- Best epoch chosen on dev EER (tie-break dev loss).
- Threshold chosen on dev p_spoof at the dev EER point, using the inference scoring path; stored in the checkpoint.
- Two identical runs gave identical logs (seeded).

**Measured results** (pipeline check only; **not research evidence**):

| Set | Size | Result |
|---|---|---|
| Dev | 8 clips / 2 speakers | EER 0.000 |
| Test | 8 clips / 3 speakers | EER 0.367, AUC 0.80; at the dev threshold (p_spoof ≥ 0.2736): FA 0.667, miss 0.200 |
| IndicSynth Hindi, out of domain | 20 clips | FreeVC24 10/10 labelled SPOOF; XTTS-v2 5/10 SPOOF, scores near the threshold |
| OpenSLR Hinglish lecture (real) | 8 min | REAL, p_spoof 0.03 |

The test set is far too small for conclusions: a single clip moves EER by 12–33 points.

**Commands:**
```
python -m ml.data.build_manifest
python -m ml.training.train --config ml/configs/baseline_lfcc_cnn.yaml
python -m ml.evaluation.evaluate --checkpoint runs/B0_lfcc_cnn_pilot/best.pt --split test
python -m ml.inference.predict --checkpoint runs/B0_lfcc_cnn_pilot/best.pt --audio <file>
```

**Failures fixed during the slice:**
- `torch.load(weights_only=True)` rejected `torch.__version__` (a `TorchVersion` object); it is now stored as `str`. Safe loading is kept.

**Known limitations:**
- 40-clip dataset; dev and test are tiny.
- No augmentation and no calibration; probabilities are uncalibrated (test ECE 0.34).
- The dev-EER threshold is unstable with 8 dev clips.

**Next:** scale ASVspoof to a meaningful subset (e.g. several thousand clips, ideally the full LA train/dev on Colab) with the same code, then evaluate on the official LA dev/eval protocol. After that: real Hindi bona fide data (Kathbath) plus IndicSynth for a Hindi train/test.

### 0.9 Validation of the vertical slice (done 2026-10-07): all 10 checks PASS
**How:**
```
python -m ml.evaluation.validate --checkpoint runs/B0_lfcc_cnn_pilot/best.pt \
  --repro runs/repro_1/B0_lfcc_cnn_pilot runs/repro_2/B0_lfcc_cnn_pilot
```
Output: `runs/B0_lfcc_cnn_pilot/validation/{validation.json, confusion_matrix_test.png, roc_test.png}`.

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Deterministic preprocessing | PASS | all 40 files: preprocess ×2 bit-identical; centre crop and seeded train crop identical |
| 2 | Correct labels | PASS | label / speaker / attack re-derived from the official ASVspoof CM protocol for 40/40 clips; every file's sha256 re-hashed = manifest |
| 3 | Split correct | PASS | train 12 bona fide / 12 spoof, dev 5 / 3, test 3 / 5; no row without split |
| 4 | Speaker leakage | PASS | 13 / 2 / 3 speakers, no overlap |
| 5 | Source leakage | PASS | single source (ASVspoof), both classes in every split. Note: all spoof attacks in dev/test (A01–A06) are also seen in train, so this measures *seen-attack* detection only |
| 6 | Duplicate leakage | PASS | 0 duplicate groups by file sha256 and by decoded-audio hash (exact duplicates only; near-duplicates not checked) |
| 7 | Derived rows crossing splits | PASS (vacuous) | no augmented/transformed rows exist yet. The check works: a unit test proves it fails on a cross-split derived row |
| 8 | Checkpoint save/load | PASS | reloaded twice → identical outputs; safe load (`weights_only=True`); manifest unchanged since training (sha256 match) |
| 9 | Score direction | PASS | train ROC-AUC 1.00; mean p_spoof 0.746 (spoof) vs 0.063 (bona fide); label = SPOOF ⇔ p_spoof ≥ dev threshold; p_spoof + p_real = 1 |
| 10 | Reproducibility | PASS | 3 runs with the same config (seed 2026, CPU): identical epochs, best epoch 29, max weight / train-loss / dev-logit / threshold differences = **0.0** (tolerance 1e-6) |

**Test-set report** (8 clips, 3 speakers; from the single existing test scoring, not re-scored; **pipeline check only**):

| Metric | Value |
|---|---|
| EER | 0.367 |
| ROC-AUC | 0.800 |
| F1 (spoof class, dev threshold 0.274) | 0.727 |
| Confusion [[TN FP] [FN TP]] | [[1 2] [1 4]], i.e. false alarm 0.667, miss 0.200 |

Train AUC 1.00 vs test 0.80 shows the expected overfitting on 24 training clips.

**Code added (minimal):**
- `ml/evaluation/validate.py`: the 10 checks plus confusion matrix and ROC plots.
- `tests/unit/test_validate.py`: proves the leakage checks fail on broken data.

**Dependency added:** `matplotlib` (+7 transitive pins, nothing else changed) in `environment/requirements-core.in` and both locks.

**Tests:** 72 passing.

**Caveats:**
- Reproducibility was verified on CPU only. GPU (cuDNN) runs may differ slightly and need their own check.
- Check 2 is specific to ASVspoof. Other sources need their own protocol cross-check when they are added.

### 0.10 AASIST baseline (B1) integrated and compared (2026-10-07)

**Upstream verification:**

| Item | Verified value |
|---|---|
| Source | `clovaai/aasist` @ `a04c9863f63d44471dde8a6abcb3b082b07cd1d1`, vendored **unchanged** in `ml/models/vendored/aasist/` (`AASIST.py` sha256 `9e0d3e80…`, configs, LICENSE, NOTICE, UPSTREAM.md) |
| Licence | MIT (NAVER Corp.); NOTICE subcomponents RawNet2 baseline (MIT) and Jungjee/RawNet (MIT) |
| Input format | raw 16 kHz waveform, 64,600 samples, no normalisation (upstream eval: first 64,600 samples; ours: centre/windows, same for every model) |
| Label convention | upstream bona fide = 1 (score = column 1); the wrapper `ml/models/aasist.py` swaps the columns (unit-tested) |
| Checkpoint | official `AASIST.pth` sha256 `51d2d9cf…`, fetched on demand into `runs/upstream/aasist/` (gitignored); 297,866 params; loads with `weights_only=True` |

**Integration check of the official checkpoint** (`scripts/verify_aasist_pretrained.py`):
- Data: 40 ASVspoof 2019 LA **eval** clips (20 bona fide / 20 spoof, attacks A07–A19), never seen by that checkpoint. Downloaded with the new `--partition eval` option.
- Upstream preprocessing: EER 0.000, AUC 1.000; mean spoof-logit +9.32 (spoof) vs −13.27 (bona fide), so wiring, weights and score direction are correct.
- Same weights through VaaniShield preprocessing (silence trim + centre windows): EER 0.150, AUC 0.960.
  - Hypothesis to test at scale: the official model partly relies on leading/trailing silence (ASVspoof silence shortcut, R50).
- 40 clips only. This is an integration check, not comparable to the README's 0.83 % on 71,237 trials.

**Why the official weights are NOT scored on our split:** they were trained on the ASVspoof 2019 LA **train** partition, which contains all 40 pilot clips, so it would be train-on-test leakage.

**B1 = AASIST trained from scratch on the pilot split:**
- Same split, training loop, model-selection rule and evaluation protocol as B0.
- Config `ml/configs/baseline_aasist.yaml`: lr 1e-4 / wd 1e-4 as upstream; batch 8; CPU, ~30 s/epoch.
- Result: **did not learn**. Outputs are near-constant (dev/test p_spoof 0.5142–0.5144; train p_spoof std < 0.01). Early stopping at epoch 11.
- The selected "best" epoch 1 (dev EER 0.000) is an artefact of 4th-decimal differences on 8 dev clips.
- Expected: a 297k-parameter raw-waveform model trained on 24 clips (upstream: ~25k clips × 100 epochs).
- No tuning was done on test.

**Comparison** (same 8-clip test split, scored once each; dev-chosen thresholds; **pipeline-level evidence only**):

| | B0 LFCC-CNN | B1 AASIST (from scratch) |
|---|---|---|
| Parameters | 14,274 | 297,866 |
| Test EER | 0.367 | 0.367 (degenerate: near-constant scores) |
| Test ROC-AUC | 0.800 | 0.533 |
| F1 (spoof, dev threshold) | 0.727 | 0.000 |
| Confusion [[TN FP] [FN TP]] | [[1 2] [1 4]] | [[3 0] [5 0]] (predicts REAL for everything) |
| Dev-chosen threshold (p_spoof) | 0.274 | 0.514 |
| CPU latency, 6.2 s file (2 windows), median / p95 | 8.6 / 9.5 ms | 826 / 1035 ms |
| CPU latency, one 4.04 s window forward, median | 4.5 ms | 472 ms |

Latency: laptop CPU (i5-1235U, 10 torch threads), 20 runs after 3 warm-ups, `predict.py --benchmark`.

**Interpretation:**
- At 24 training clips, the small LFCC model learns something and AASIST does not.
- This says nothing about AASIST in general. It is the expected data-hunger of a raw-waveform graph model.
- A meaningful B0 vs B1 comparison needs the scaled ASVspoof subset (next step).

**Changes:**

| File | Change |
|---|---|
| `ml/models/aasist.py` | new wrapper + pinned weight fetch |
| `ml/models/vendored/aasist/*` | vendored upstream (unchanged) |
| `ml/models/__init__.py` | `aasist` kind registered |
| `ml/configs/baseline_aasist.yaml` | new |
| `scripts/verify_aasist_pretrained.py` | new |
| `scripts/acquire_asvspoof.py` | `--partition` option |
| `ml/inference/predict.py` | `--benchmark` |
| `ml/evaluation/validate.py` | near-constant-output warning in check 9 |
| `tests/unit/test_pipeline.py` | wrapper column-swap test |
| `LICENSES_REGISTER.md` | AASIST row verified |

Tests: 73 passing.

**Data on disk now:** + 40 LA eval clips (`data/raw/asvspoof2019_la/LA/ASVspoof2019_LA_eval/`, provenance recorded). Used **only** for the checkpoint integration check, never for training or selection.

**Next:** scale ASVspoof (several thousand train clips + official dev; ideally full LA on Colab GPU, since AASIST at 0.5 s/window on CPU is too slow to train locally). Retrain B0 and B1 with the same code and protocol. Optionally fine-tune from the official AASIST weights, but only when evaluating on data the checkpoint never saw (LA dev/eval, Hindi). Sections 2–13 below describe the earlier, larger plan and remain as research context only.

---

## 1. Project identity [DOC]
- Title: *VaaniShield: Towards Code-Switch-Robust Audio Deepfake Detection for Hindi-English Speech under Messaging-App Compression*
- B.Tech CSE 2023–27, AKGEC Ghaziabad / AKTU, Group-20; supervisor Ms. Swati Tomar; 4 students.
- Primary RQ [DOC:syn 1.3.1]: Can a multilingual SSL detector, trained with codec and language-mix augmentation, improve detection of Hindi-English code-switched deepfake speech under messaging-app compression?

## 2. Research questions (revised in Phase 0b; full text and hypotheses H1–H13 in `research/research_gap.md` §4–5)
| ID | Question (short) | Merges |
|---|---|---|
| RQ1 | Language / code-switch shift: English → Hindi → Hinglish degradation; does *symmetric* splice augmentation help without a shortcut? | syn SRQ1, pg RQ1 |
| RQ2 | Messaging channel: is simulated codec processing a valid proxy for real store-and-forward app re-encoding (EER, bona fide false alarms)? | syn SRQ3, pg RQ2 |
| RQ3 | Representation and generator-family generalisation: XLS-R vs LFCC / AASIST / monolingual SSL | syn SRQ2+SRQ4, pg RQ3+RQ4 |
| RQ4 | Deployment reliability: calibration and fixed-threshold errors under shift; channel-conditioned calibration; CPU latency | syn SRQ5 |

## 3. Completed work
- [x] Read all 5 source PDFs in full: synopsis 26 pp, project guide 21 pp, dataset guide 8 pp, HAV-DF 22 pp, IndicSynth 24 pp.
- [x] Targeted prior-work and licence check (web, 2026-10-07). Sources are in `docs/PLAN_v0.md` §H.
- [x] Plan written: architecture, gap, dataset, experiment, repo and phase plan (`docs/PLAN_v0.md`).
- [x] Phase 0b research audit:
  - 31 sources in the literature matrix (9 read in full text);
  - novelty classification and do-not-claim list;
  - RQ1–4 with H1–H13;
  - Novelty Boundary;
  - related-work draft;
  - TD-1…TD-20.
- [x] Phase 0c implementation blueprint (`docs/` × 6).
- [x] **PHASE 1 (repository + environment), 2026-10-07:**
  - repo tree per the project guide; `git init -b main` (no commits yet);
  - `pyproject.toml` maps import name `vaanishield` → `ml/src/`;
  - hashed dependency locks: Windows-CPU, Linux-CUDA, Linux-CPU;
  - `.venv` on Python 3.12.13;
  - `scripts/environment_check.py` and `scripts/check_repo_hygiene.py`;
  - typed config (M01), repro utils (M02), metrics (M15), run registry (M19);
  - `train.py` (config validation only) and `evaluate.py` (metrics from a score CSV);
  - 83 unit tests passing; ruff clean;
  - CI workflow and pre-commit config;
  - README / MODEL_CARD / DATASET_CARD / LICENSES_REGISTER;
  - `.gitignore`, `.env.example`, Docker skeleton.

## 4. Pending work
The authoritative task list is `docs/implementation_checklist.md`: PHASE 1 Foundation … PHASE 10 Patent preparation, each with inputs, outputs, acceptance criteria, failure conditions and fallbacks.

Integration milestones:
| ID | Date | Milestone |
|---|---|---|
| I1 | 2026-10-24 | CPU skeleton end-to-end |
| I2 | 2026-11-14 | B0 on LA + EER parity |
| I3 | 2026-11-28 | demo MVP (B1) |
| I4 | 2026-12-19 | P2 on T_en+hi; HinglishFake-Test locked |
| I5 | 2027-01-24 | full recipe + calibration; pre-registration committed |
| I6 | 2027-02-14 | full matrix measured |

Immediate next tasks: P1.1–P1.12 (repo, Python 3.12 lock, ffmpeg check, CI, M01/M02/M15/M19, licence register, consent draft, systematic search log).

## 5. Decisions log
| # | Date | Decision | Status |
|---|---|---|---|
| DEC-1 | 2026-10-07 | Canonical labels `bonafide`/`spoof`; internal score = spoof logit (higher = more spoof); converted once in metrics | proposed |
| DEC-2 | 2026-10-07 | Primary Hindi real class = Kathbath-Hindi (IndicSUPERB, CC0), because it is the source corpus of IndicSynth fakes (reduces corpus shortcut) | proposed |
| DEC-3 | 2026-10-07 | Splits by connected component of a speaker graph (VC source + target + clone reference) | proposed |
| DEC-4 | 2026-10-07 | "Unseen generator" claims are made at **generator-family** level (XTTS appears in 3 sources) | proposed |
| DEC-5 | 2026-10-07 | Demo MVP = FastAPI + Gradio; React/Express/MongoDB deferred | proposed (follows DOC:pg) |
| DEC-6 | 2026-10-07 | Python 3.12 (not the system 3.14) | proposed |
| DEC-7 | 2026-10-07 | File-based run registry is the source of truth; W&B/MLflow optional mirror | proposed |
| DEC-8 | 2026-10-07 | Architecture (XLS-R weighted layers + AASIST-lite) is treated as a **baseline re-implementation**, not the novelty (SLS and Tak et al. 2022 exist) | proposed |

## 6. Open questions for the team (blocking marked ★)
- ★Q1 GPU: which accounts are available (Colab Pro? Kaggle 30 h/week)? Is there a college GPU?
- ★Q2 Is there an institutional ethics/consent process for recording participants and cloning their voices?
- Q3 Keep or drop the "Indian English" claim (synopsis abstract)? No corpus is identified for it.
- Q4 Which apps are in scope for real re-encoding (WhatsApp, Telegram, others)? Which devices and account owners?
- Q5 Will the college IPR cell be consulted before any arXiv/GitHub/demo publication?
- Q6 Second non-XTTS Hinglish generator: which one, and is its licence acceptable?
- Q7 Should source PDFs be stored in the repo (docs/sources/) or kept out?
- Q8 Experiment tracker preference (none / W&B / MLflow).
- Q-L Licence for our own code (MIT or Apache-2.0)? Needed before any public repository.
- Q-G Who owns the Git remote (GitHub org/user), and is it private until the IPR decision (P10.5)?

## 7. Dataset status
| Dataset | Role | Licence | Access | Status |
|---|---|---|---|---|
| ASVspoof 2019 LA | English baseline | **verify** (not shown on DataShare page) [EXT] | Edinburgh DataShare, LA.zip 7.12 GB [EXT] | not downloaded |
| Kathbath-Hindi (IndicSUPERB) | Hindi bona fide | CC0 [EXT][DOC:p2] | HF `ai4bharat/Kathbath` | not downloaded |
| IndicSynth-Hindi | Hindi spoof (XTTS-v2, FreeVC24; Diversity only) | CC BY-NC 4.0 [EXT][DOC:p2] | HF `vdivyasharma/IndicSynth`, ungated [EXT] | not downloaded |
| OpenSLR 104 Hi-En | Hinglish bona fide | CC BY-SA 4.0 [EXT] | 7.3 GB train / 443 MB test [EXT] | not downloaded; **speaker IDs unverified** |
| SwitchLingua audio | Hinglish bona fide (opt.) | CC BY-NC 4.0 [DOC:dg] | gated HF | not requested |
| CS-FLEURS | Read-Test = real*; XTTS/MMS = spoof | verify | HF `byan/cs-fleurs` | not downloaded |
| Hinglish XTTS-v2 | Hinglish spoof generator | CPML-derived, non-commercial [EXT] | HF | not downloaded |
| In-the-Wild | external held-out | Apache-2.0 [EXT] | HF `mueller91/In-The-Wild` | **do not touch until pipeline frozen** |
| HinglishFake-Test | project benchmark | own consent terms | – | not started |
| HAV-DF | related work only | on request [DOC:p1] | – | not used |

## 8. Experiment status
None run. All result cells are "not measured".

| ID | Description | Status |
|---|---|---|
| E0 | LFCC-LCNN | not started |
| E1 | AASIST | not started |
| E2 | XLS-R + light head | not started |
| E3 | XLS-R + AASIST-lite | not started |
| A1–A5 | ablations | not started |
| C2/C4 | channel gap / channel-conditioned calibration | not started |

## 9. Known issues and risks
| ID | Risk | Mitigation |
|---|---|---|
| R1 | No local NVIDIA GPU; no ffmpeg; Python 3.14 default (observed on team laptop) | Colab/Kaggle T4; install ffmpeg; Python 3.12 venv |
| R2 | Corpus shortcut: real = lectures (OpenSLR), fake = clean TTS | Kathbath↔IndicSynth matched pair; team real + consented fakes from same speakers; per-source metrics; corpus-probe |
| R3 | OpenSLR 104 may lack speaker IDs → speaker-disjointness unverifiable | Group by recording/tutorial ID; document weaker guarantee |
| R4 | XTTS family in IndicSynth, Hinglish-XTTS and CS-FLEURS → fake "unseen" claims | `generator_family` column; family-level holdout; need ≥1 non-XTTS/non-FreeVC Hinglish generator |
| R5 | Hindi has only 2 generators → 2-fold LOGO only | State as a limitation; add a third family via HinglishFake-Test |
| R6 | Non-commercial licences (IndicSynth, CPML, SwitchLingua) vs patent/commercial goals | Keep research use only; patent claims must not depend on NC data; flag to IPR cell |
| R7 | Hindi clips ~2.65 s avg [DOC:p2] vs 4 s crops → padding could become a class cue | Same pad policy for both classes; duration-bucket metrics; consider 3 s crops |
| R8 | Silence/duration shortcuts (known on ASVspoof) | Consistent VAD trim for both classes; report silence-ablated eval |
| R9 | Language-mix splice: real splices are cross-speaker (Kathbath Hi + English corpus) but fake splices could be same-voice (XTTS) → speaker-change becomes a "real" cue | Symmetric splice design (cross-voice for both classes); splice-only probe |
| R10 | App "send as document" may not re-encode; in-app voice note = acoustic re-recording | Verify codec/hash change per file; separate conditions |
| R11 | App-version drift | Record app/version/OS/date; treat new version as new condition |
| R12 | Calibration on clean dev breaks under compression | Include simulated-codec conditions in dev; never real-app |
| R13 | Overclaiming novelty (Indic-CodecFake/SATYAM, SEA-Spoof, RTCFake, ADD-C, SLS exist) | Say "underexplored"; systematic review in Phase 1 |
| R14 | Schedule ~1 month behind the synopsis Gantt | Compressed plan; parallel Phases 6/7 |
| R15 | Consent/privacy for team recordings | Consent form, pseudonymous IDs, private storage, withdrawal procedure |
| R16 | Publication before patent filing destroys novelty | IPR-cell disclosure before arXiv/GitHub/demo/viva publication |
| R17 | min t-DCF needs ASV scores | Report only on ASVspoof19 LA with official ASV scores |
| R18 | **CS-FLEURS has no real Hindi-English**: per its card, Hindi appears only in the synthetic XTTS-Test2 subset | Real Hinglish only from OpenSLR 104 + consented recordings (TD-9) |
| R19 | Strong prior work on channel robustness (RTCFake paired real platforms; ADD-C; ASVspoof 2021 real telephony) shrinks the C2 novelty | Frame C2 narrowly: store-and-forward messaging, Hinglish, simulation-validity test (research_gap §6) |
| R20 | Calibration under shift already audited (Zhou & Wang 2026) | Evaluate C4 by HTER/false alarm/ECE, never EER; search biometrics prior art |
| R21 | Neural-codec-as-attack (ICF/CodecFake) vs codec-as-channel confusion could make detectors flag compressed bona fide speech | H5 tests it; never label codec-processed bona fide as spoof (TD-16) |
| R22 | Power: HinglishFake-Test is small, so CIs may be wide | Pre-register margins; report "inconclusive" honestly |
| R23 | The laptop's `py` launcher lists `C:\Python312\python.exe`, but that path does not exist (stale registration) | Use uv-managed Python 3.12 (`uv venv --python 3.12`); optionally repair or uninstall the stale entry |
| R24 | Low free RAM on the laptop (2.89 / 15.68 GiB free at check time) | Close other apps for local tests; all model training on Colab/Kaggle |
| R25 | Fast-moving major versions (transformers 5.x, pandas 3.x, numpy 2.5, librosa 1.0) may break reference code (AASIST, RawBoost, HF examples) written for older APIs | Locked versions; smoke-tested (soundfile+soxr+librosa+tiny Wav2Vec2 forward OK); vendor reference code with adapters; record any shims in UPSTREAM.md |
| R26 | CI workflow never executed (no Git remote yet) | Push to the team remote and confirm a green run (P1.4) |

## 10. Reproducibility information
- **Environment (created 2026-10-07):** `.venv` with uv-managed CPython 3.12.13. Locks in `environment/lock/`:
  - `py312-windows.txt`: 69 pins, hashed;
  - `py312-linux.txt`: 86 pins, CUDA torch;
  - `py312-linux-cpu.txt`: torch 2.14.1+cpu.
- **Key pins:** torch 2.14.1, transformers 5.19.0, numpy 2.5.3, scipy 1.18.1, pandas 3.0.6, librosa 1.0.0, soundfile 0.14.0, soxr 1.1.0, scikit-learn 1.9.1, pydantic 2.13.5, pytest 9.1.1, ruff 0.16.10.
- **Laptop environment check (2026-10-07):**

| Item | Value |
|---|---|
| OS | Windows 11 (10.0.26300) |
| CPU | Intel Core i5-1235U (10C/12T) |
| RAM | 15.68 GiB total (only 2.89 GiB free at check time) |
| PyTorch | 2.14.1+cpu |
| CUDA | **not available**; only GPU is Intel Iris Xe (integrated) |
| ffmpeg | **not found** |

  The report JSON is saved locally in `environment/reports/` (gitignored, machine-specific).
- Seeds: default 2026 [DOC:pg §07]; ≥ 3 seeds per reported cell.
- Every run stores: config snapshot, git commit, manifest SHA-256, checkpoint SHA-256 (weights outside Git), scores.csv, metrics.json, environment dump.
- Data root: outside Git via `VAANI_DATA_ROOT`. Never commit raw audio, consent forms, `.env` or weights.

## 11. Candidate novelty — classification after Phase 0b audit (provisional; systematic search pending)
| ID | Contribution | Class |
|---|---|---|
| C1 | HinglishFake-Test (Hindi-English code-switched, audio-only, consented, speaker-disjoint, generator-family holdout) | underexplored |
| C2 | Paired clean / simulated / real **store-and-forward messaging** re-encoding; simulation-as-proxy test; bona fide false alarms | underexplored in this form; real-channel evaluation generally **partially solved** (ASVspoof 2021 LA, RTCFake) |
| C3 | Symmetric code-switch splice augmentation | unclear (splice-shortcut risk, R52) |
| C4 | Channel-conditioned calibration | unclear (threshold-transfer audit R53; biometrics prior art unsearched) |
| C5 | Switch-point segment analysis | underexplored (exploratory only) |
| Combo | Intersection: Hinglish × real messaging channel × unseen family × calibration, compact detector | **potentially novel as an evaluation setting** |

**Already solved, do not claim:** Indic/Hindi ADD; English-trained detectors failing on Hindi; XLS-R+AASIST(+RawBoost); learned layer weighting; codec augmentation; temperature scaling; segment timelines; real-channel evaluation as such; the first code-switched deepfake dataset (ArEnAV). Full list in `research/research_gap.md` §3.

## 12. Patent-sensitive ideas — keep internal until the IPR cell is consulted
- P-1 (from C4): system that estimates the transmission/compression condition of an incoming voice clip and selects condition-specific calibration parameters and decision thresholds for a spoof detector, outputting a calibrated probability plus per-segment timeline and channel-quality warnings.
- P-2 (from C3 + C5): code-switch-aware segmentation, where segment scores are aligned with detected language-switch points.
- Phase 0b update: P-1 has nearby prior work. Zhou & Wang 2026 (R53) audit global and unlabeled calibration under domain shift, and RTCFake (R46) covers platform-robust training. The *channel-estimation → conditional calibration → timeline* system is still unsearched in patent databases.
- Notes: no patentability is claimed. Prior art is not yet searched; quality-dependent calibration exists in biometrics generally [HYP]. Indian law excludes "computer programme per se" (Patents Act s.3(k)), so any claim must be framed as a technical system and needs professional advice. Do not publish details before a filing decision.

## 13. Phase log
| Phase | Date | Summary | Files | Commands | Experiments |
|---|---|---|---|---|---|
| 0 | 2026-10-07 | Read 5 PDFs; prior-work and licence check; plan | `PROJECT_STATE.md`, `docs/PLAN_v0.md` (created) | `pdfinfo`/`pdftotext` (to session scratchpad), tool-availability checks | none |
| 0b | 2026-10-07 | Research audit: 31 sources, novelty classification, RQ1–4, H1–H13, Novelty Boundary, TD-1…20 | `research/*.md` (5 created); `PROJECT_STATE.md` (updated) | web search/fetch; `pdftotext` + `grep` on downloaded arXiv PDFs (scratchpad only) | none |
| 0c | 2026-10-07 | Implementation blueprint: 23 modules with typed interfaces, dependency graph, 4-lane parallel plan, data flow, experiment protocol, API contract, reproducibility spec, 10-phase checklist | `docs/` (6 created); `PROJECT_STATE.md` (updated) | none (documentation only) | none |
| 1 | 2026-10-07 | Repository + environment foundation (details in §3; leftovers in `docs/implementation_checklist.md` P1 status lines) | 66 addable files incl. `pyproject.toml`, `ml/src/**`, `tests/unit/*` (6 files), `scripts/*` (2), `environment/**`, `.github/workflows/ci.yml`, README/MODEL_CARD/DATASET_CARD/LICENSES_REGISTER; docs updated (architecture §3/§4.3, experiment_protocol §5 EER, reproducibility §2) | `uv pip compile` ×3 locks; `uv venv --python 3.12`; `uv pip sync`; `uv pip install -e .`; `pytest` (83 passed); `ruff check/format`; `scripts/environment_check.py`; `git init -b main`; hygiene check incl. force-added `.wav` probe | none (no data, no training) |

### Decisions added in Phase 1
- DEC-14: **no torchaudio**. Its latest release (2.11.0) lags torch (2.14.1) and its wheels are ABI-tied to torch. Audio I/O = soundfile + ffmpeg subprocess; resampling = soxr HQ; LFCC = torch.stft.
- DEC-15: code stays in the guide's `ml/src/{datasets,models,augment,metrics,utils}`, installed as package **`vaanishield`** via a setuptools `package-dir` mapping. This avoids `ml/src/datasets` shadowing Hugging Face `datasets`.
- DEC-16: EER uses the official ASVspoof det-curve rule (mean of FA/miss at the minimum |FA−miss|, no interpolation). `experiment_protocol.md` §5 is corrected.
- DEC-17: Colab/Kaggle keep their image's CUDA torch. All other packages are constrained to the Linux lock; actual versions are recorded per run.

### Decisions added in Phase 0c (proposed)
- DEC-9: one `Detector` contract (`[B,L] → logits [B,2]`, `spoof_logit = l1 − l0`) for all models; evaluation, inference and API never branch on model type.
- DEC-10: crop 64,600 samples, repeat-padding; inference windows hop 32,300; clip score = mean window logit.
- DEC-11: test scoring only via `TestLock` (clean git tree + `--confirm-final` + access log).
- DEC-12: pre-registration (`docs/preregistration.md`) committed before Phase 7.
- DEC-13: frozen-encoder checkpoints store only head weights + HF revision.

### Errata to `docs/PLAN_v0.md`
- §A.3 D10 / §D.1: CS-FLEURS Read-Test does **not** contain Hindi-English per the dataset card; Hindi appears only in XTTS-Test2 (synthetic). See TD-9.
- §C.1: add ASVspoof 2021 LA (real VoIP/PSTN transmission) and RADAR 2026 / ML-ITW as channel and realism prior work; add Zhou & Wang 2026 for calibration.
- §E.3 / RQs: superseded by RQ1–4 and H1–H13 in `research/research_gap.md`.

### 0.11 Scaled data pipeline: canonical manifest v1 (completed 2026-10-08)
- **Manifest:** `data/manifests/vaanishield_v1.csv`, 1,934 clips (2.47 h), from `python -m ml.data.build_canonical_manifest`. Composition and split rules are in DATASET_CARD.md.

| Source | Clips | Notes |
|---|---|---|
| ASVspoof LA | 1,074 | official partitions |
| IndicSynth-Hi | 320 | spoof |
| Kathbath-Hi | 240 | bona fide; m4a/AAC decoded via PyAV |
| OpenSLR104 Hinglish | 300 | bona fide segments, test only; 30 speakers from `utt2spk`, so **R3 resolved** |

- **Checks** (`python -m ml.data.check_manifest` → `vaanishield_v1_report.{md,json}`): all six hard checks **PASS** (required fields, split, duplicates, speaker leakage, derived/segment rows, generator-input + transcript leakage).
- **Distribution reports:**
  - Class by split: train 421 bona fide / 543 spoof; dev 196/125; test 459/190.
  - Language by split: train en 634 / hi 330; dev en 200 / hi 121; test en 240 / hi 109 / hi-en 300.
- **BLOCKER for Hindi training (warning):** in the Hindi pool, codec (AAC vs PCM), sample rate (44.1 vs 24 kHz) and source each perfectly separate real from fake. Options are listed in DATASET_CARD.md; this is a team decision before any Hindi training.
- **Code:**

| File | Change |
|---|---|
| `ml/data/build_canonical_manifest.py` | new |
| `ml/data/check_manifest.py` | new; reuses `ml/evaluation/validate.py` checks |
| `ml/data/manifest.py` | `codec` column; legacy manifests still load |
| `ml/data/audio.py` | m4a/AAC via PyAV |
| `ml/evaluation/validate.py` | external-parent rule: segments of one recording must share a split |
| `scripts/acquire_asvspoof.py` | `--partition`; parallel CRC-checked extraction (600 clips in 3 min vs ~85 min) |
| `scripts/acquire_kathbath.py` | `--subset`: tar-header probing; 60 MB fetched instead of a 3.6 GB tar |
| `scripts/acquire_indicsynth.py` | `--subset` |

- **Dependency:** `av` 19.0.1 added (locks updated; only new pin).
- **Tests:** 76 passing. The new checks have tests proving they fail on broken data; one such test caught a real bug in `generator_leakage` during development.
- **Downloaded (total ≈ 0.5 GB on disk):**
  - ASVspoof subset ≈ 70 MB;
  - IndicSynth ≈ 333 MB transferred;
  - Kathbath 24 MB;
  - OpenSLR test archive 444 MB (sha256 `93e358b3…`, kept in `data/raw/openslr104_hien/archives/`).
- **Not done (by instruction):** no training.

**Next:**
1. Decide the Hindi codec fix.
2. Retrain B0 (and B1 on GPU) on the ASVspoof part of v1 with the existing code. B0/B1 still point to the frozen pilot manifest `asvspoof_pilot.csv`.

### 0.12 Portability: laptop = development/smoke, GPU machine = full training (2026-10-08)
Workflow doc: `docs/LOCAL_VS_GPU_WORKFLOW.md` (what runs where, what to transfer, how to reproduce).

**Configurable, with no hardcoded paths** (audit: no absolute paths in `ml/` or `scripts/`; only download URLs):

| Setting | Config key | CLI override |
|---|---|---|
| Data root | `data.data_root` (null → `$VAANI_DATA_ROOT` / `<repo>/data`) | `--data-root` |
| Manifest | `data.train_manifests` | — |
| Sources | `data.include_sources` | — |
| Subset size | `data.max_clips_per_split` (deterministic, label-stratified) | — |
| Audio loading | `data.preload_audio` (off = lazy loading for large data) | — |
| Workers | `data.num_workers` | `--num-workers` |
| Model | `model.kind` + `model.params` | — |
| Batch size | `train.batch_size` | — |
| Epochs | `train.epochs` | `--epochs` |
| Seed | `train.seed` | — |
| Device | `train.device` (auto / cpu / cuda; asking for cuda without a GPU is an error) | `--device` |
| AMP | `train.amp` (CUDA only) | — |
| Gradient accumulation | `train.grad_accum` (now implemented) | — |
| Checkpoint dir | `output.checkpoint_dir` | `--checkpoint-dir` |
| Results dir | `output.results_dir` | `--results-dir` |

**Commands (existing entry points, no new ones):**
- `python -m ml.training.train --config <yaml>`
- `python -m ml.evaluation.evaluate --config <yaml>` (new mode; the old `--checkpoint` mode remains)
- `python -m ml.inference.predict --checkpoint <pt> --audio <wav> [--device]`

Checkpoints store CPU tensors (portable) plus the device trained on. Each run also writes `run_info.json`.

**Configs:**
- `ml/configs/smoke_cpu.yaml`: ASVspoof-only, 16 / 8 / 8 clips, batch 4, 1 worker, 1 epoch, CPU, seed 2026.
- `ml/configs/full_gpu_aasist.yaml`: AASIST, all ASVspoof LA, batch 24, 8 workers, 100 epochs, cuda, lazy loading, upstream lr/wd; manifest `vaanishield_v1_full.csv`, built on the GPU machine. Not run here.

**Smoke test on this laptop: all 8 verifications passed.**
- Dataset loads (16 train / 8 dev from `vaanishield_v1.csv`, source-filtered) → preprocessing → model init → 1 epoch (3.9 s).
- Checkpoint saved (`runs/smoke/checkpoints/…/best.pt`) and reloaded.
- `evaluate --config` scored 8 test clips.
- `predict` scored an 8-min OpenSLR file (245 windows).
- Total ≈ 35 s.
- Metrics are meaningless at this size (test EER 0.75).

**Regression checks:**
- The refactored trainer reproduces the original B0 pilot run bit-for-bit (weight / loss / threshold diff 0.0).
- `--device cuda` and the full GPU config refuse to start on this CPU-only laptop.
- 78 tests passing (+2: subset selection, config overrides).

**Files:**

| File | Change |
|---|---|
| `ml/training/config.py` | new fields + dotted overrides |
| `ml/training/repro.py` | `resolve_device`, `resolve_path` |
| `ml/data/dataset.py` | `select_rows`, lazy loading |
| `ml/training/train.py` | device / AMP / grad-accum / output dirs / CLI overrides; legacy `--out-dir` kept |
| `ml/evaluation/evaluate.py` | `--config` mode, config-driven selection |
| `ml/inference/predict.py` | `--device` |
| `ml/configs/smoke_cpu.yaml`, `ml/configs/full_gpu_aasist.yaml` | new |
| `docs/LOCAL_VS_GPU_WORKFLOW.md` | new |
| `tests/unit/test_pipeline.py` | +2 tests |

**Next (on the GPU machine):** full ASVspoof download → build `vaanishield_v1_full.csv` → `check_manifest` → `full_gpu_aasist.yaml`. Locally, still pending: the Hindi codec-shortcut decision.

### 0.13 XLS-R baseline (P1) implemented and smoke-tested on CPU (2026-10-08). Full training NOT run.
**Weights:** `facebook/wav2vec2-xls-r-300m` @ `1a640f32ac3e…`
- Apache-2.0, ungated.
- 1.27 GB `pytorch_model.bin` in the HF cache (not in Git); downloaded in 77 s.
- `preprocessor_config` do_normalize = true, so the model normalises each utterance (zero mean, unit variance).

**Model:** `ml/models/xlsr.py` (`model.kind: xlsr_mlp`)
- Pipeline: XLS-R hidden states (K = 25) → softmax layer weights (logged per epoch in `history.json`) → Linear 1024→128 + SELU → attentive-stats pooling → Linear → 2 logits (index 1 = spoof).
- Encoder options come from the existing ModelCfg fields: `freeze_encoder`, `unfreeze_top_k`, `truncate_after_layer`, `adapter_dim`.
- The CNN feature extractor is always frozen.
- The encoder always runs in eval mode, with SpecAugment / layerdrop / dropout off (documented choice; deterministic frozen runs).

**Checkpoints are compact:**
- Trainable weights only, plus the HF id and revision: `best.pt` is 565 KB, not 1.27 GB.
- Unfrozen layers are included in the checkpoint when fine-tuning.

**Trainer** (`ml/training/train.py`):
- Uses the model's parameter groups when it has them (`lr_head` vs `lr_encoder`).
- Writes `last.pt` every epoch (weights, optimizer, AMP scaler, shuffle-generator state, torch RNG state, history, best-so-far).
- New `--resume`:
  - Refuses if the config differs, apart from epochs, device, workers, data_root, preload and output dirs; those may change between interruptions or machines.
  - Verified: 1 epoch + resume gives weights bit-identical to an uninterrupted run (LFCC smoke).
  - XLS-R resume 1 → 2 epochs works.
  - A changed learning rate is refused (exit 1).

**Configs:**
- `ml/configs/smoke_cpu_xlsr.yaml`: frozen XLS-R, ASVspoof 8 / 4 / 4 clips, batch 2, 1 epoch, CPU.
- `ml/configs/full_gpu_xlsr.yaml`: frozen XLS-R, all ASVspoof LA (`vaanishield_v1_full.csv`), batch 32 × grad_accum 2, AMP on CUDA, 8 workers, 30 epochs, lazy loading. Partial fine-tuning (`unfreeze_top_k`) is a separate config / exp_id.

**Smoke verification on this laptop (CPU):** all passed.
1. Model loads: 315,577,756 params, 140,060 trainable (frozen). The load report's UNEXPECTED keys are pretraining-only quantizer/projection heads, which is expected.
2. Forward pass: [2, 2] in 2.6 s (batch 2).
3. Loss is finite.
4. Gradients:
   - frozen: every trainable param has a non-zero gradient and the encoder has none;
   - `unfreeze_top_k=2`: only transformer layers 22–23 receive gradients, the CNN extractor none.
5. One training epoch on 8 clips: 25.6 s.
6. Checkpoint saved, then reloaded by `evaluate --config` (4 test clips scored).
7. `predict --device auto` on a Hindi file: 3 windows.

Smoke metrics are meaningless (8 training clips).

**Tests and regression:**
- 80 tests passing (+2 using a tiny random wav2vec2 config: no download needed).
- The B0 pilot still reproduces bit-for-bit after the trainer changes.
- `LICENSES_REGISTER.md`: XLS-R row verified.

**Next (GPU machine):** full manifest → `full_gpu_aasist.yaml` and `full_gpu_xlsr.yaml` (both ASVspoof-only until the Hindi codec fix).

### 0.14 Portability test + GPU handoff checklist (2026-10-08)
- **Handoff doc:** `docs/GPU_HANDOFF.md` (exact bash + PowerShell commands)
  - environment: route A = exact Linux lock (driver CUDA ≥ 13); route B = any OS/driver: `torch==2.14.1` from the PyTorch `cu126`/`cu130`/`cu132` index, then all other packages constrained to the lock;
  - CUDA/GPU checks; data paths via `VAANI_DATA_ROOT` / `HF_HOME`;
  - manifest build + validation; smoke; model loading; full training; `--resume`; evaluation;
  - what to send back; troubleshooting.
- **Portability test (simulated teammate machine):**
  - Setup: only clone-able files (91 files, 2.3 MB, no audio) copied to a different folder whose path contains spaces; a new venv built with route B (CPU torch build standing in for CUDA); audio read from an external `VAANI_DATA_ROOT`.
  - Passed: 80 tests; environment and CUDA checks; data-root resolution; `check_manifest` (6/6 PASS); LFCC smoke train / evaluate / predict; XLS-R smoke + resume; full GPU config refused on a CPU-only machine; outputs stayed inside the copy.
  - Losses identical to the original environment (LFCC 0.7792; XLS-R 1.9439 → 0.9548).
- **Not testable here:** an actual CUDA run (no NVIDIA GPU). The GPU-only checklist items are for the teammate.
- **Known handoff prerequisite:** the repository has **no git commit yet**. Commit and push (or zip the clone-able files) before handing over.

### 0.15 200-clip Hinglish pilot (2026-10-08): dataset + protocol verified on the laptop
- **Built:**
  - `ml/data/build_hinglish_pilot.py` (`plan` / `build`);
  - `scripts/acquire_csfleurs.py` (pinned HF revision, licence-gated, provenance recorded);
  - registry entry `cs_fleurs` (CC BY-NC 4.0, verified from the HF licence tag);
  - `ml/configs/pilot_hinglish_lfcc.yaml`;
  - manifest `data/manifests/hinglish_pilot_v1.csv` + report.
  - Downloaded: 150 CS-FLEURS clips (84 MB) + metadata (22 MB). OpenSLR segments are cut from the local archive.
    **No local synthesis.**
- **Design:** 100 real / 100 spoof, all `hi-en`, 16 kHz. Details in DATASET_CARD.md § Hinglish pilot.
  - train: 30 OpenSLR + 30 XTTS hin-spk0;
  - dev: 10 + 10 (hin-spk1);
  - test T1: 10 OpenSLR + 10 XTTS hin-spk2 (seen family);
  - test T2: 50 CS-FLEURS read + 50 MMS concatenative (**held-out family**).
  - Speaker-, voice- and sentence-disjoint.
- **Checks:** all 6 hard checks PASS.
- **`check_manifest` fixes (bugs found by the pilot):**
  - a declared eval-only source (OpenSLR) used for training was excluded from the confound analysis; it is now
    excluded only if it really is test-only;
  - confounds are now also checked within the **train split** (the pilot's train source confound had been hidden by
    pooling).
  - Unit test added; 81 tests pass; v1's hard checks are unchanged (only new train-split warnings).
- **Protocol run** (LFCC-CNN, CPU, 10 epochs × 4 s): best epoch 9, dev EER 0.00 (windowed; 0.10 on single crops).
  - Test (dev-fixed threshold): T1 EER 0.00; **T2 (held-out family) EER 0.72, AUC 0.24.**
  - Breakdown: `runs/pilot/results/PILOT_hinglish_lfcc_cnn_cpu/test_conditions.csv`.
  - **Not a result** (n tiny).
- **Shortcut floor:** trivial-feature logistic regression gets T1 EER 0.00 and dev 0.20 → T1 measures nothing.
  Recorded in research_gap.md §8.
- **Train log fix:** the final train log line now says it uses full-clip windowed scoring (it had looked
  inconsistent with the per-epoch crop EER).
- **Open team decisions:**
  - **D-P1:** real Hinglish speech from the same conditions as the fakes (team recordings with consent + cloning those
    speakers) to remove the train-source confound.
  - **D-P2:** loudness/peak normalisation of both classes (decide on dev/probe evidence, before Phase 9).
  - **D-P3:** whether non-commercial CS-FLEURS is acceptable for the final dataset (it is fine for research use).

### 0.16 Codec robustness pipeline (2026-10-08): built, pilot run on the laptop only
- **Code:**
  - `ml/data/codec.py`: `Condition`, `apply`, `align`, `quality`, `available`;
  - `ml/data/build_codec_variants.py` (driver);
  - `ml/configs/codec_conditions.yaml`: shared condition definitions;
  - `ml/configs/codec_pilot.yaml` (laptop) and `ml/configs/codec_full_gpu.yaml` (GPU/many-core machine, dev + test,
    8 workers, AMR required);
  - `tests/unit/test_codec.py` (8 tests; 89 total pass);
  - GPU_HANDOFF.md step 4b.
- **How a variant is made:**
  - parent decoded with `load_audio` (mono, soxr HQ 16 kHz, no trimming) → encode → decode → back to 16 kHz;
  - cross-correlation delay compensation (±250 ms), then same length as the parent;
  - stored as PCM-16 WAV in `<data_root>/processed/codec_sim/<condition>-<params_hash>/<source>/<parent>.wav`.
  - Changing any condition setting changes `params_hash`, so stale files are never reused. Re-runs skip existing
    files (resume).
- **Conditions (TD-13):**
  - `train_aug`: Opus VoIP 8/16/32 kb/s; MP3 16/32 kb/s; G.711 μ-law (8 kHz); resample 16→8→16 and 16→22.05→16;
    AMR-NB 12.2.
  - `eval_heldout` (never applied to train-split clips; enforced in code and tested): AAC-LC 24/48 kb/s; G.722;
    AMR-WB 12.65.
  - **AMR-NB/WB are unavailable on the laptop.** PyAV's bundled FFmpeg has no AMR encoder and there is no system
    ffmpeg, so they are recorded as `unavailable`. They run via the `ffmpeg_cli` backend where a system ffmpeg has the
    encoders. Nothing is substituted or relabelled (G.711 is not called "AMR").
- **Per-variant metadata** (`*_metadata.jsonl`; first line holds the config, conditions and tool versions):
  - provenance: parent id + parent sha256, source, split, label, language;
  - condition: condition id, role, backend, encoder, container, codec sample rate, requested **and** measured bitrate,
    encoder options, params_hash, delay;
  - output: sha256;
  - quality: SNR, log-spectral distance, effective bandwidth (−50 dB), RMS change, clipping.
- **Manifest:** variants inherit split, label, source, speaker, language and generator from the parent;
  `parent_clip_id` is set; `transform=sim:<condition>`; the notes repeat the codec/bitrate/params.
- **Pilot** (`codec_pilot_v1*`):
  - inputs: 12 parents from the Hinglish pilot (2 real + 2 spoof per split) → 120 variants (+20 recorded unavailable);
  - cost: 40 MB, ~15 s CPU.
  - Results:
    - `check_manifest`: all 6 hard checks PASS;
    - inheritance verified for all 6 fields;
    - **bit-identical** re-encode (120/120 sha256 equal with `--overwrite`).
- **Pilot quality** (medians; sanity only):
  - bandwidth: resample_8k / G.711 3.8 kHz; Opus 8k 4.5 kHz; MP3 16k 5.5 kHz; AAC / G.722 / Opus 16–32k ≈ 7.5–8 kHz;
  - measured bitrate within ~10 % of requested (Opus 32k uses 28.6 kb/s; VBR);
  - delays: AAC 64 ms (priming, removed by alignment); others ≈ 0.
  - Waveform SNR is low for perceptual codecs (Opus ≈ 8–10 dB) by design; LSD and bandwidth are the useful
    indicators. PESQ/ViSQOL are not used (not installed).
- **Fixes found during the pilot:**
  - the 99 %-energy bandwidth metric could not show band-limiting → replaced with −50 dB bandwidth;
  - held-out codecs had been applied to train clips → now blocked.
- **Not done (as instructed):** no training, no full-dataset processing.
  - The training loader does not yet choose among `sim:` rows. Before codec-augmented training, decide whether
    `train_aug` is applied online (architecture.md §4.7) or sampled from pre-encoded train variants.
- **Open:**
  - **D-C1:** AMR-capable ffmpeg on the GPU machine (or record that AMR is omitted).
  - **D-C2:** confirm bitrates/conditions before the full run, then never change them after seeing test results.
  - Real messaging-app round-trips (`app:` transforms) are a separate, manual, consented step and are not part of
    this pipeline.

### 0.17 Robustness smoke experiment (2026-10-08): SMOKE / DEVELOPMENT RESULTS — NOT FINAL RESULTS
- **Code:**
  - `ml/evaluation/robustness.py`, one command: subset → codec variants → leakage check → scoring (cached per
    checkpoint sha256) → metrics → plots;
  - `ml/configs/robustness_smoke.yaml` + `ml/configs/codec_robust_smoke.yaml`;
  - `tests/unit/test_robustness.py` (2 tests; 91 total pass).
- **Data:**
  - 6 real + 6 spoof clean TEST clips per language:
    - en = ASVspoof LA eval;
    - hi = Kathbath vs IndicSynth (codec-confounded; DATASET_CARD);
    - hi-en = Hinglish pilot test.
  - × 11 available codecs (AMR unavailable on the laptop) = 36 clean + 396 codec rows: `robust_smoke_codec.csv`
    (+ metadata, quality).
- **Models** (existing checkpoints, each with its own dev threshold; nothing fitted on test):
  - B0 LFCC smoke (16 ASVspoof clips);
  - B1 AASIST pilot (24 ASVspoof clips; known not to learn);
  - P1 XLS-R frozen smoke (8 ASVspoof clips);
  - B0 LFCC trained on the 60 Hinglish-pilot clips.
- **Leakage check:** 0 clip and 0 speaker overlap between the evaluated rows and every model's train+dev rows.
- **Outputs** (`runs/robustness/ROBUST_smoke_v1/`):
  - `README.md` (banner), `run_info.json`, `scores/<model>.csv/json`;
  - `metrics.csv/json`: 76 rows = 4 models × 19 slices; EER + speaker-bootstrap 95 % CI, AUC, false alarm / miss /
    HTER / F1 at the dev threshold, ECE;
  - plots: `eer_heatmap.png`, `eer_by_codec.png`, `roc_by_language.png`, `codec_score_shift.png`, all stamped with
    the banner.
- **Pipeline verification:**
  - evaluation, metrics, plots and outputs all work;
  - `evaluate.py --scores` and `--checkpoint` both work on the new files;
  - re-running the report from cached scores reproduces the outputs.
  - CPU cost: ≈ 25 min for 432 clips × 4 models (XLS-R dominates).
- **Numbers:** for illustration only; no interpretation.
  - Clean-pooled EER 0.22–0.56 with 95 % CIs about 0.4 wide (e.g. LFCC smoke 0.22 [0.12, 0.50]).
  - These numbers cannot rank models or conditions.
- **Bugs and gotchas found by this run:**
  - `df.transform` in pandas is a method, not the column; the unit test caught it in the plotting code (fixed).
  - `evaluate --checkpoint --manifest X` reuses the checkpoint's *training* selection rules (sources, max clips).
    With the smoke checkpoints it scored only 8 English rows of the codec manifest. This is intended (same selection
    as training), so robustness runs use `ml.evaluation.robustness`, which scores every row.
  - The B1 AASIST pilot's scores move only ~1e-4 under any codec (near-constant output); its EERs are noise.
- **For the GPU machine:** copy `robustness_smoke.yaml` to a new config with the full-run checkpoints, a larger
  `n_per_class`, a new `subset_manifest`, and a copy of `codec_robust_smoke.yaml` whose `input_manifests` is that
  subset (`codec_full_gpu.yaml` reads the full manifests, so it is not the right file here). Same command,
  `--device cuda`.

### 0.18 GPU-machine handoff package (2026-10-08)
- **`docs/GPU_HANDOFF.md` rewritten** as one guide in run order:
  - overview table with time/disk estimates (labelled as guesses) and the integrity rules;
  - steps: environment (two install routes), CUDA check, paths, data verification + GPU smoke test, full ASVspoof +
    manifest, full training, evaluation, full codec run, robustness;
  - what to send back; troubleshooting (the quoted error messages were checked against the code).
- **New files:**
  - `ml/configs/full_gpu_lfcc.yaml`: B0 full run; lr/wd from the pilot recipe, batch 32, 50 epochs, not tuned.
  - `ml/configs/robustness_full_gpu.yaml` + `ml/configs/codec_robust_full.yaml`: 35 + 35 test clips per language;
    the 3 full models; AMR required.
  - `scripts/verify_data.py`: every manifest path exists and its sha256 matches. It passes on all 2,698 local rows and
    fails clearly when the data root is wrong.
- **Changed:**
  - `codec_full_gpu.yaml` capped at 3,000 dev + 6,000 test parents per manifest. Uncapped it was ~1.2 M files
    (~100 GB); now ~119k files (10–15 GB).
  - `.gitignore` now ignores the `constraints.txt` created by install route B.
- **Data transfer:** the teammate cannot re-create the exact pilot manifests by re-downloading, so the laptop's
  `data/raw` + `data/processed` are copied.
  - Command: `tar -cf vaanishield-data.tar --exclude "raw/openslr104_hien/archives" --exclude "raw/_logs" -C data raw processed`
    (tested: 1.04 GB, 2,802 entries; the already-extracted 444 MB OpenSLR tarball is excluded).
- **Blocker before handoff:** the Git repository still has **no commit and no remote**.
- Tests: 91 pass.

### 0.19 Hindi confound audit + handoff licence audit (2026-10-08): handoff READY_WITH_CONDITIONS
- **New documents:**
  - `research/hindi_data_leakage_audit.md` (sections A–I);
  - `research/handoff_licence_audit.md`;
  - `research/HANDOFF_DECISION.md`;
  - `scripts/audit_hindi_confound.py` (reproduces every number; writes `runs/audit/hindi/`; never touches `data/`).
- **Hindi cause:**
  - two upstream chains: Kathbath = phone recordings released only as AAC m4a, 44.1 kHz, raw level; IndicSynth =
    generator output, PCM 24 kHz, peak-normalised;
  - **speaker mismatch:** 4/19 target speakers overlap, and 0/323 of the fakes' reference recordings are in the real
    pool (our Kathbath came from test-known).
  - Our scripts did not re-encode anything.
- **Probes** (speaker-grouped CV / train→test):
  - metadata EER 0.000;
  - waveform stats through the current pipeline: 0.17–0.26 (ASVspoof control 0.33–0.41);
  - loudness normalisation → about 0.27–0.31;
  - AAC matching: almost no effect (AAC-trace probe EER 0.43–0.46);
  - within speaker-matched groups the spectral probe is at chance (0.53–0.59);
  - residual "synthetic silence" cue: noise floor / silent-frame fraction, sep AUC 0.78–0.86.
- **Plan (proposed, nothing applied):** pair fakes with their own Kathbath references (D-H1); preprocessing v2 = 40 Hz
  high-pass + loudness normalisation for all classes (D-H2); acceptance gate (D-H3); repair, not removal; build a new
  v2 manifest, keep v1.
- **Licences:**
  - verified for transfer: ASVspoof (ODC-By), IndicSynth (CC BY-NC), OpenSLR (CC BY-SA);
  - "Needs verification from current dataset terms.": Kathbath (CC0 vs CC-BY-4.0), CS-FLEURS (tag vs card text),
    In-the-Wild.
  - The earlier claim in GPU_HANDOFF.md that "the licences allow this sharing" was unsupported; corrected.
- **Re-download test:** Kathbath and CS-FLEURS re-downloaded from the original sources into an isolated root are
  **bit-identical** (240/240, 150/150), so they are not copied.
  - Restricted archive (ASVspoof + IndicSynth + OpenSLR + segments) tested: 794 MB.
- **Updated:**
  - `docs/GPU_HANDOFF.md` step 0 (restricted archive + self-download + Hindi evaluation-only);
  - DATASET_CARD Hindi warning now points to the audit.
- **Conditions C1–C4** are listed in HANDOFF_DECISION.md. Open: Hindi repair, licence verification, codec decisions,
  Hinglish D-P items.

### 0.20 Hindi repair: v2 pilot + acceptance gate (2026-10-08): gate verdict FAIL (1 of 10 checks)
- **Report:** `research/hindi_repair_report.md`. DATASET_CARD has a new "Hindi v2 pilot" section. v1 data and
  manifests are untouched (verified).
- **Acquisition** (`scripts/acquire_kathbath.py`):
  - `--transcripts`: Hindi transcripts of all Kathbath splits, 18 MB;
  - `--fetch-ids`: parallel header walk of a tar's Hindi section, exact files, full listing.
  - Findings: all 323 IndicSynth reference recordings are in Kathbath **train**; 323/323 fetched + 230 same-speaker
    (55 MB; 96k range requests, 16 min). The walker was verified complete on test-known (3,130/3,130).
- **Code:**
  - `ml/data/audio.py` gains `PREPROC_V2` + `preprocess_v2_wave` (v1 functions unchanged);
  - `ml/data/build_hindi_v2.py` + `ml/configs/hindi_v2_pilot.yaml`;
  - `ml/data/hindi_gate.py` + `ml/configs/hindi_gate.yaml`: 10 checks; thresholds PROPOSED with `approved: false`,
    so the best possible verdict is PASS_PENDING_APPROVAL;
  - `ml/evaluation/shortcut_probe.py`: probe code moved out of the audit script; the audit reproduces identically;
  - `scripts/hindi_v2_diagnostics.py`;
  - `tests/unit/test_hindi_v2.py` (4 tests; 95 total pass).
- **v2 pilot:**
  - 640 clips (320 real references/same-speaker + 320 fakes), balanced 1:1 in every split;
  - 11 speaker groups (train 4 / dev 1 / test 6);
  - FreeVC held out (test only); sentence-disjoint; identical stored format; one preprocessing hash for both classes.
- **Gate, before → after:**
  - metadata EER 0.000 → 0.467 (PASS);
  - format, generator protocol, sentence IDs, preprocessing record and balance: FAIL → PASS;
  - G10 waveform probe 0.166 → 0.241 and spectrum probe 0.210 → 0.256: both still **FAIL**.
- **Remaining cues** (in-memory ablation, nothing adopted):
  - pause structure: removing internal pauses from both classes takes silence-only EER from 0.28 to 0.44;
  - peak limiting: 9.4 % of real clips vs 0 % of fakes;
  - spectral envelope: plausibly a genuine synthesis artefact; the spectrum probe stays at about 0.28 even after
    pause removal + 6 kHz low-pass.
- **Open team decisions:**
  - D-H3: approve the thresholds;
  - D-H4: pause removal (v2.1) for all data, or report pause probe floors;
  - D-H5: peak limiting;
  - D-H6: split G10 into a gated non-speech probe and a reported spectrum probe.
- **Not done (as instructed):** no training, no XLS-R, no full Hindi build, no deletion.
- **Handoff impact:** Kathbath train recordings and `processed/hindi_v2/` fall under the Kathbath licence condition
  (do not copy); the restricted archive in HANDOFF_DECISION.md is unchanged.

### 0.21 Hindi v2.1 (2026-10-08): team decisions D-H3..D-H6 applied; recommendation READY_WITH_LIMITATIONS
- **Report:** `research/hindi_v2_1_validation.md`. New dataset version (v1 and v2 untouched, verified):
  - `data/processed/hindi_v2_1/pilot/` (220 MB);
  - diagnostic copy `.../diag_pauses_removed/` (187 MB);
  - manifests `hindi_v2_1_pilot.csv`, `_metadata.jsonl`, `_pauses_removed.csv`;
  - config `ml/configs/hindi_v2_1_pilot.yaml`.
- **Decisions applied:**
  - **D-H3:** all numeric limits are labelled provisional engineering acceptance criteria.
  - **D-H4:** pauses kept; pause-controlled 2×2 diagnostic (`scripts/pause_controlled_diagnostic.py`, configs
    `diag_hindi_v2_1_lfcc_*`).
  - **D-H5:** limiter removed; float32 storage is the class-neutral alternative (`PREPROC_V2_1`).
  - **D-H6:** `ml/data/hindi_gate.py` rewritten as A (13 acceptance checks) + B (5 diagnostic probes with
    interpretation); it never outputs READY.
- **Code:**
  - `ml/data/audio.py`: `PREPROC_V2_1`, `preprocess_wave`, `remove_pauses` (diagnostic only); v1/v2 unchanged, and
    the v2 hash is reproduced;
  - `ml/data/build_hindi_v2.py`: `preprocessing` + `pause_diagnostic` options;
  - `ml/data/manifest.py`: transform `diag:<...>` allowed;
  - `ml/evaluation/shortcut_probe.py`: `pause_features`;
  - `scripts/hindi_v2_1_analysis.py`;
  - tests updated (96 pass).
- **Gate (same definition on all versions):**
  - v1: ACCEPTANCE_FAIL (10 of 13 fail);
  - v2: ACCEPTANCE_FAIL (only A6 peak limiter);
  - **v2.1: ACCEPTANCE_PASS_PROVISIONAL (13/13).**
- **Diagnostics v2.1** (EER): metadata 0.463, nuisance 0.341, pause 0.419, waveform 0.221, spectral 0.234
  (spectral is unchanged with pauses removed).
- **Pause dependence** (LFCC, 3 seeds, DIAGNOSTIC — NOT FINAL RESULTS): +0.014 EER overall, −0.020 XTTS,
  +0.026 FreeVC.
- **Peaks:** 29 real clips > ±1.0 (float, not clipped). This is a **downstream risk**: any PCM/codec step would clip
  real clips only, so add a fixed attenuation for both classes first (not implemented).
- **Spectrum:** fakes have less 5–7.5 kHz energy (15/19 paired speakers, both generators); plausibly a legitimate
  synthesis cue, but the generator training pipeline is unverified.
- **Allowed:** pilot/development training on v2.1, with probe floors + pause 2×2 + per-generator reporting.
- **Not allowed:** Hindi research claims (dev has 1 group; needs a full build with ≥ 2 dev groups).
- **Not done (as instructed):** no final detector, no XLS-R, no large training; v1/v2 not deleted.

### 0.22 Handoff refresh after Hindi v2.1 (2026-10-08): still READY_WITH_CONDITIONS
- **Hindi removed** from `ml/configs/robustness_full_gpu.yaml` (pools now en + hi-en) and `ml/configs/codec_full_gpu.yaml`
  (`include_sources` = ASVspoof, OpenSLR, CS-FLEURS), before any results existed.
  - Reasons: v1 Hindi is confounded; v2.1 needs the headroom guard before codec steps.
- **`docs/GPU_HANDOFF.md`:**
  - Hindi rules rewritten;
  - new optional **step H**: rebuild Hindi v2.1 from original sources, then `verify_data`, gate and pause diagnostic,
    as a cross-machine reproducibility check (no Hindi training);
  - test count 96; codec/robustness text updated; send-back list includes the step H reports.
- **`research/HANDOFF_DECISION.md`:** update section added; C3 resolved.
- **Remaining conditions:** C1 commit + push (still blocking); C2 restricted archive only; C4 follow the guide.

### 0.23 Final pre-handoff audit (2026-10-08)
- **Report:** `research/PRE_HANDOFF_AUDIT.md`. Read-only audit of the 153 commit candidates.
- **PASS:** hygiene (no data, audio, weights, venv, caches, personal paths), secrets (pattern + entropy scan,
  0 hits), configurability, handoff commands (32 commands, all configs and scripts exist, flags valid, checkpoint
  paths consistent), research integrity, tests (96).
  - Integrity detail: train/dev only in training; dev thresholds; 0 configs edited after results, only CLI overrides;
    Hindi excluded from the English benchmark.
- **FAIL found and then fixed on request:** `LICENSES_REGISTER.md` contradicted the registry (Kathbath and
  In-the-Wild "redistribute: yes", etc.); the five rows were corrected.
- **Windows notes fixed in GPU_HANDOFF.md:** `&&`, execution policy, `Select-String`.
- **Verified:** IndicSynth re-downloads bit-identical (320/320), so every dataset can be obtained directly.
- **Remaining:**
  - no commit or remote yet (create a PRIVATE repo);
  - keep it private (licences, IPR R16, supervisor name in docs);
  - optional overwrite guard for existing `exp_id`s;
  - run a dedicated secret scanner before any public release.
