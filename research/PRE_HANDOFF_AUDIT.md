# Final pre-handoff audit

**Date:** 2026-10-08. **Scope:** the whole working tree, read-only. Nothing was committed or pushed, and no project
file was changed except this report.
**Method:** local scripts. No dedicated secret scanner is installed (gitleaks, trufflehog, detect-secrets and
git-secrets are all absent), so the secret scan used regex patterns plus an entropy check over every commit
candidate. Evidence and commands are given per check.

## Overall result: **NOT YET READY TO COMMIT. 1 FAIL (documentation) + the known "no commit/remote yet" blocker**
- The code, configs, manifests and Git hygiene are in good shape.
- One committed document, `LICENSES_REGISTER.md`, misstates dataset licences and contradicts the project's own
  evidence. The GPU guide tells the teammate to read it before accepting terms, so it must be corrected first.

| Category | Result |
|---|---|
| 1. Repository hygiene | **PASS** (1 WARNING) |
| 2. Secrets / security | **PASS** (1 WARNING) |
| 3. Reproducibility / configurability | **PASS** (1 WARNING) |
| 4. GPU handoff commands | **PASS** with 3 WARNINGS (Windows shell details; Linux/GPU not executable here) |
| 5. Research integrity | **PASS** (2 WARNINGS) |
| 6. Licence / provenance documentation | **FAIL** (`LICENSES_REGISTER.md`) |
| 7. Data transfer licensing | **PASS** for the restricted set (after the register is corrected) |
| 8. Tests | **PASS** (96 passed) |
| 9. Git state | **BLOCKER**: no commit, no remote (expected; you asked not to push) |

---

## 1. Repository hygiene: PASS
| Check | Result | Evidence |
|---|---|---|
| No datasets tracked | PASS | 0 files tracked (no commit yet). The 153 commit candidates (`git ls-files -co --exclude-standard`) contain no file from `data/raw/` or `data/processed/` (both ignored) |
| No audio files | PASS | extensions of the candidates: py 57, md 33, yaml 21, csv 12, json 11, txt 5, jsonl 4, in 3, conf 2, toml 1, plus LICENSE, NOTICE, .gitignore, .gitattributes. No binary file (NUL-byte check). `.gitignore` covers wav/flac/mp3/ogg/opus/m4a/aac/amr/webm/sph |
| No model checkpoints | PASS | no .pt/.pth/.ckpt/.bin/.safetensors/.onnx candidates; `runs/` ignored |
| No API keys / tokens / passwords | PASS | §2 |
| No .env secrets | PASS | no `.env*`, `*.pem`, `*.key`, `credentials*`, `kaggle.json`, `.netrc` anywhere outside `.venv`; all are in `.gitignore` |
| No virtual environments | PASS | `.venv/` and `venv/` ignored; none among the candidates |
| No caches | PASS | `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `hf_cache/`, `.cache/` ignored; none among the candidates |
| No machine-specific absolute paths | PASS | 0 matches for Windows user-profile paths (`<drive>:\Users\<name>\`, MSYS-style `/c/Users/<name>`), `AppData`, `/home/<user>/`, `/Users/<user>/`. The only drive-letter strings are two documented *examples* (`D:\vaanishield-data` in GPU_HANDOFF.md; `D:/vaani-data` in a train.py docstring) and a historical note about the laptop's stale `C:\Python312` (PROJECT_STATE R23) |
| No personal Windows usernames | PASS | 0 matches for the username or user profile path |
| Size | PASS | 153 files, about 6 MB. Largest: `hindi_v2_1_pilot_metadata.jsonl` 695 KB, `hindi_v2_pilot_metadata.jsonl` 659 KB, `vaanishield_v1.csv` 638 KB |
| WARNING | — | The supervisor's name ("Ms. Swati Tomar") appears in PROJECT_STATE.md:276 and docs/PLAN_v0.md:26, quoted from the synopsis. Fine for a private team repository; confirm with the supervisor before making anything public. Machine-specific environment reports (`environment/reports/*.json`) are correctly ignored |

## 2. Secrets / security: PASS
- **Patterns scanned in all 153 candidates:** Hugging Face (`hf_…`), GitHub (`ghp_/gho_/…`), OpenAI/Anthropic
  (`sk-`, `sk-ant-`), AWS access keys, Google API keys, Slack tokens, PEM private-key blocks, `password/secret/
  api_key/access_token = "…"` assignments, bearer headers, Kaggle keys, and high-entropy quoted strings (> 4.3 bits per
  character, ≥ 32 characters, excluding hex hashes). **0 hits in every category.**
- **No emails:** no email address in any candidate.
- **No transcript text:** no Devanagari text runs in any candidate. The licensed Kathbath/CS-FLEURS/IndicSynth
  transcripts live only in ignored `data/raw/`; the manifests carry *hashes* only. The single "TTS Transcript" string
  in `provenance/indicsynth_hi.json` is a column name in a schema listing.
- **Tokens are never written to files:** the code reads `HF_TOKEN` only from the environment, and the handoff docs say
  so.
- **WARNING:** this is a pattern scan, not a dedicated scanner. Before any *public* release, run gitleaks or
  detect-secrets (not installed here).

## 3. Reproducibility / configurability: PASS
| Item | How it is set | Result |
|---|---|---|
| Dataset path | `--data-root` (train, evaluate, robustness, codec, Hindi builder, acquisition scripts) > `VAANI_DATA_ROOT` > `<repo>/data` | PASS |
| Training manifests / sources | YAML `data.train_manifests`, `data.include_sources`, `data.max_clips_per_split` | PASS |
| GPU/CPU | `--device auto\|cpu\|cuda` (train, evaluate, predict, robustness); YAML `train.device`; `cuda` without a GPU fails loudly | PASS |
| Batch size | YAML `train.batch_size` (+ `grad_accum`) | PASS (by design via a *new* config / new `exp_id`, not a CLI flag) |
| Workers | `--num-workers` (train, evaluate), `--workers` (codec builder); YAML | PASS |
| Random seed | YAML `train.seed`, `seed` in data configs; deterministic subset selection from (manifest, split, seed) | PASS (new config per seed, as the pause diagnostic does) |
| Checkpoint location | `--checkpoint-dir`; YAML `output.checkpoint_dir` | PASS |
| Output location | `--results-dir`; YAML `output.results_dir`; `output_dir` / `--out` for robustness and gate | PASS |
| HF cache | `HF_HOME` | PASS |
| WARNING | `ml.data.check_manifest` and `ml.data.hindi_gate` take the data root only from `VAANI_DATA_ROOT` (no `--data-root` flag). The guide sets the variable before they are used | WARNING |

## 4. GPU handoff (`docs/GPU_HANDOFF.md`): PASS with warnings
| Check | Result | Evidence |
|---|---|---|
| Every `python -m …` / `python scripts/…` command exists and every flag is accepted | PASS | 32 commands parsed; module or script exists and every `--flag` appears in its `--help`. Two automatic flags were false positives: `clean/train_audio.tar` is a valid `--tars` choice; `pytest` is an installed package |
| Every referenced script exists | PASS | acquire_asvspoof, acquire_csfleurs, acquire_kathbath, environment_check, pause_controlled_diagnostic, verify_data |
| Every referenced config exists | PASS | codec_full_gpu, codec_pilot, codec_robust_smoke, full_gpu_{lfcc,aasist,xlsr}, hindi_v2_1_pilot, robustness_full_gpu, smoke_cpu, smoke_cpu_xlsr |
| Dataset paths | PASS | every referenced manifest exists, except `vaanishield_v1_full.csv` and `codec_full_v1.csv`, which the guide creates in steps 5 and 8 before using them |
| Checkpoint paths | PASS | `robustness_full_gpu.yaml` expects `runs/full/checkpoints/FULL_{lfcc_cnn,aasist,xlsr_frozen}_asvspoof_gpu/best.pt`, exactly what the three full configs produce (`checkpoint_dir` + `exp_id`) |
| Model-loading one-liners | PASS | `build_model('aasist', {})` and `build_model('xlsr_mlp', {})` give output `[2, 2]` (tested on CPU without `.cuda()`) |
| Windows (PowerShell) commands | PASS with WARNING | tested in Windows PowerShell 5.1: the constraints one-liner (77 pins), the torch/CUDA one-liner and `$env:VAANI_DATA_ROOT` + DataPaths all work. **WARNINGS:** (a) step 0 `git clone … && cd VaaniShield` fails in PowerShell 5.1 (`&&` is a parser error; works in PowerShell 7 and bash); (b) the AMR check `ffmpeg … \| grep -E …` needs grep, which plain Windows lacks; (c) `.venv\Scripts\activate` in PowerShell can be blocked by the execution policy (`Set-ExecutionPolicy -Scope Process Bypass`, or call `.venv\Scripts\python.exe` directly) |
| Linux commands | WARNING (static only) | syntax checked (bash, `.venv/bin/python`, `source …/activate`, `export`); not executed, because no Linux machine is available here |
| GPU-specific paths | WARNING (untested) | no NVIDIA GPU here: `--device cuda`, AMP and the CUDA wheel routes are untested; the CPU equivalents and the "cuda without GPU fails loudly" path are tested |

## 5. Research integrity: PASS
| Check | Result | Evidence |
|---|---|---|
| Test data never used for model selection | PASS | `ml/training/train.py` loads only `rows_for(cfg, "train")` and `rows_for(cfg, "dev")`; best epoch and threshold come from dev. The docstring states that test is never used |
| Test results not used to tune hyperparameters | PASS | evaluate.py applies the dev threshold stored in the checkpoint (`threshold_from: dev`); no code path reads test scores back into a config. Every full GPU config is unrun |
| Configs not modified after results exist | PASS (1 documented exception) | Each saved run's config (stored in its checkpoint) was compared with the current file. All differences are allowed CLI overrides only (`output.*`, `train.epochs`, `data.num_workers`); **0 real edits**. `robustness_smoke.yaml` is identical to its stored copy. Exception: `ml/configs/hindi_gate.yaml` (a *gate* definition, not an experiment config) was restructured after v2 gate results, by explicit team decision D-H6; documented in PROJECT_STATE §0.21 |
| Experiment IDs immutable | **WARNING** | Immutability is a documented rule and is detectable afterwards (`config_sha256` in every checkpoint and run_info), but it is **not enforced**: re-running `train` with an existing `exp_id` and no `--resume` overwrites `best.pt`/results silently. `--resume` does refuse a changed config |
| Speaker / source / generator leakage checks exist | PASS | `check_speaker_leakage`, `check_descendants`, `check_duplicates` (validate.py); `generator_leakage`, `shortcut_risks` incl. train-split confounds (check_manifest.py); Hindi A7–A11 (hindi_gate.py); unit tests in test_validate.py and test_hindi_v2.py |
| Hindi not in the v1 final benchmark | PASS | `full_gpu_{lfcc,aasist,xlsr}.yaml`: `include_sources: [asvspoof2019_la]`; evaluate re-uses the checkpoint's selection, so Hindi test rows are excluded. Hindi was removed from `robustness_full_gpu.yaml` (pools en, hi-en) and `codec_full_gpu.yaml` before any results. The full manifest *contains* Hindi v1 rows (expected `check_manifest` warnings), but no GPU config selects them |
| Unresolved datasets clearly marked | PASS in research docs / **FAIL in LICENSES_REGISTER.md** | `licence_registry.yaml`, `research/handoff_licence_audit.md`, HANDOFF_DECISION and DATASET_CARD mark Kathbath, CS-FLEURS and In-the-Wild "Needs verification from current dataset terms." `LICENSES_REGISTER.md` does not (§6) |
| No unsupported novelty claims | PASS | Novelty statements are qualified: "potentially novel as an evaluation setting (provisional; systematic search pending)" (PROJECT_STATE §11), "[HYP — novelty unchecked]" (PLAN_v0 C.3), and explicit "do not claim … as novel" in the audits. MODEL_CARD's "courtroom", "100 % accuracy" entries are in its *prohibited uses* list |
| WARNING | — | PROJECT_STATE R16: "Publication before patent filing destroys novelty: IPR-cell disclosure before arXiv/GitHub/demo/viva publication". A **public** GitHub repository would be a publication. Keep it private until the team clears this |

## 6. Licence / provenance documentation: **FAIL**
`LICENSES_REGISTER.md` (a commit candidate, and the file the guide tells the teammate to read before
`--accept-terms`) contradicts the evidence in `licence_registry.yaml` and `research/handoff_licence_audit.md`:

| Row | Register says | Evidence says | Problem |
|---|---|---|---|
| ASVspoof 2019 LA | licence "to verify", redistribute "to verify" | ODC-By 1.0, verified from README §COPYING (copy saved in `data/manifests/licence_texts/`) | stale |
| Kathbath / IndicSUPERB | CC0, NC "no", **redistribute "yes"** | CONFLICT (CC0 packaging vs CC-BY-4.0 tag; third-party text); registry: redistribution "No (until conflict resolved)" | **dangerous**: invites redistribution |
| IndicSynth | redistribute "no (NC + cite)" | CC BY-NC 4.0 permits non-commercial sharing with attribution; the handoff plan transfers it | contradicts the handoff plan |
| In-the-Wild | Apache-2.0, **redistribute "yes"** | CONFLICT (CC BY-SA 4.0 HF tag vs Apache-2.0 project page) | **dangerous** |
| CS-FLEURS | CC BY-NC 4.0, "no (we do not redistribute)" | tag CC BY-NC 4.0, card text "CC BY 4.0 for Non-Commercial use": "Needs verification from current dataset terms." | should say so |

Everything else is consistent: `licence_registry.yaml`, the provenance JSONs (URL, pinned revision/ETag, sha256 per
file), DATASET_CARD, `research/handoff_licence_audit.md` and HANDOFF_DECISION.

**Required before commit:** correct these five rows to match the registry and licence audit.

## 7. Data transfer licensing (what may go to the teammate)
Re-download reproducibility was tested today into isolated folders (repository provenance untouched):
- Kathbath subset 240/240 bit-identical (earlier today);
- CS-FLEURS 150/150 bit-identical;
- **IndicSynth 320/320 bit-identical** (`acquire_indicsynth.py --pilot 10` + `--subset 12 --rows-per-shard 25`).

| Dataset / file group (bundle path) | Licence / access terms (evidence) | Redistribution permitted? | Private research sharing with a teammate? | Decision |
|---|---|---|---|---|
| ASVspoof 2019 LA pilot clips (`raw/asvspoof2019_la`, 1,074 FLAC + protocols) | ODC-By 1.0 (DataShare README §COPYING, saved copy) | Yes, with attribution | Yes, with attribution | **Safe to transfer.** The teammate also downloads the full LA (step 5), a superset containing the same files |
| IndicSynth Hindi subset (`raw/indicsynth_hi`, 320 WAV + 2 metadata CSV) | CC BY-NC 4.0 (HF licence tag + card: "non-commercial, academic research only"); not gated | Yes, non-commercial only, with attribution + licence notice | Yes (academic, non-commercial) | **Safe to transfer** once the register row is corrected; or the teammate downloads it (verified identical, ~10 min) |
| OpenSLR 104 Hindi-English test recordings (`raw/openslr104_hien/test`, without the 444 MB archive) + our lossless segments (`processed/openslr104_hien`) | CC BY-SA 4.0 (OpenSLR page, re-checked today) | Yes, with attribution; derivatives under CC BY-SA 4.0 | Yes | **Safe to transfer** (segments carry the share-alike condition) |
| Kathbath Hindi (test-known subset, 553 train references, transcripts, tar listings) | CONFLICT CC0 vs CC-BY-4.0; third-party text; HF mirror gated | Unclear | Unclear | **DO NOT TRANSFER — FRIEND MUST DOWNLOAD DIRECTLY.** (`acquire_kathbath.py --subset 300`, `--transcripts`, `--fetch-ids …`) |
| CS-FLEURS subset (150 WAV + 3 metadata JSONL with transcripts) | tag CC BY-NC 4.0 vs card text "CC BY 4.0 for Non-Commercial use" | Unclear | Unclear | **DO NOT TRANSFER — FRIEND MUST DOWNLOAD DIRECTLY.** (`acquire_csfleurs.py --metadata`, `--files data/manifests/hinglish_pilot_plan.csv`) |
| In-the-Wild `meta.csv` | CONFLICT CC BY-SA 4.0 vs Apache-2.0 | Unclear | Unclear | **DO NOT TRANSFER — FRIEND MUST DOWNLOAD DIRECTLY** (not needed for the handoff) |
| `processed/codec_sim/` (all codec variants) | inherits each parent; includes Kathbath and CS-FLEURS derivatives | Unclear for part | Unclear for part | **DO NOT TRANSFER**: regenerate (bit-identical, `build_codec_variants`) |
| `processed/hindi_v2/`, `processed/hindi_v2_1/` (incl. pause-removed copies) | Kathbath-derived (real class) | Unclear | Unclear | **DO NOT TRANSFER**: rebuild (step H) |
| XLS-R weights, upstream AASIST.pth, Python packages | Apache-2.0 / MIT / various | n/a | n/a | not transferred; downloaded by the teammate's own tools |

## 8. Tests: PASS
`python -m pytest -q` → **96 passed**. Covers config, metrics, evaluation, pipeline, reproducibility, leakage checks,
codec, robustness report and Hindi v2/v2.1 (preprocessing, splits, gate).

## 9. Git state: BLOCKER (expected)
Branch `main`, **no commits, no remote**. All 153 candidates are untracked. Nothing was pushed.

---

## Answers

### 1. Files safe to commit (after the one correction in §6)
All 153 current commit candidates, plus this report (`research/PRE_HANDOFF_AUDIT.md`):
- **Root:** `.gitattributes`, `.gitignore`, `DATASET_CARD.md`, `LICENSES_REGISTER.md` (**after correction**),
  `MODEL_CARD.md`, `PROJECT_STATE.md`, `README.md`, `pyproject.toml`
- **`data/manifests/`:**
  - `asvspoof_pilot.csv`;
  - `codec_pilot_v1.csv`, `codec_pilot_v1_metadata.jsonl`, `codec_pilot_v1_quality.csv`, `codec_pilot_v1_quality.md`,
    `codec_pilot_v1_report.json`, `codec_pilot_v1_report.md`;
  - `hindi_v2_1_pilot.csv`, `hindi_v2_1_pilot_metadata.jsonl`, `hindi_v2_1_pilot_pauses_removed.csv`,
    `hindi_v2_1_pilot_report.json`, `hindi_v2_1_pilot_report.md`;
  - `hindi_v2_pilot.csv`, `hindi_v2_pilot_metadata.jsonl`, `hindi_v2_pilot_report.json`, `hindi_v2_pilot_report.md`;
  - `hindi_v2_reference_ids.txt`;
  - `hinglish_pilot_plan.csv`, `hinglish_pilot_v1.csv`, `hinglish_pilot_v1_report.json`, `hinglish_pilot_v1_report.md`;
  - `licence_registry.yaml`;
  - `licence_texts/asvspoof2019_la/LICENSE_text.txt`, `licence_texts/asvspoof2019_la/README.txt`;
  - `provenance/{asvspoof2019_la,cs_fleurs,in_the_wild,indicsynth_hi,kathbath_hi,openslr104_hien}.json`;
  - `robust_smoke_codec.csv`, `robust_smoke_codec_metadata.jsonl`, `robust_smoke_codec_quality.csv`,
    `robust_smoke_codec_quality.md`, `robust_smoke_subset.csv`;
  - `vaanishield_v1.csv`, `vaanishield_v1_report.json`, `vaanishield_v1_report.md`
- **`docs/`:** `GPU_HANDOFF.md`, `LOCAL_VS_GPU_WORKFLOW.md`, `PLAN_v0.md`, `api_contract.md`, `architecture.md`,
  `data_flow.md`, `experiment_protocol.md`, `implementation_checklist.md`, `reproducibility.md`
- **`environment/`:** `README.md`, `lock/py312-linux.txt`, `lock/py312-windows.txt`, `requirements-core.in`,
  `requirements-dev.in`, `requirements-torch.in`
- **`ml/`:**
  - `__init__.py`;
  - `configs/`: all 20 YAML (baseline_aasist, baseline_lfcc_cnn, codec_conditions, codec_full_gpu, codec_pilot,
    codec_robust_full, codec_robust_smoke, diag_hindi_v2_1_lfcc_pauses_kept, diag_hindi_v2_1_lfcc_pauses_removed,
    full_gpu_aasist, full_gpu_lfcc, full_gpu_xlsr, hindi_gate, hindi_v2_1_pilot, hindi_v2_pilot,
    pilot_hinglish_lfcc, robustness_full_gpu, robustness_smoke, smoke_cpu, smoke_cpu_xlsr);
  - `data/`: `__init__`, audio, build_canonical_manifest, build_codec_variants, build_hindi_v2, build_hinglish_pilot,
    build_manifest, check_manifest, codec, dataset, hindi_gate, manifest, provenance (`.py`);
  - `evaluation/`: `__init__`, evaluate, metrics, robustness, shortcut_probe, validate (`.py`);
  - `inference/`: `__init__.py`, `predict.py`;
  - `models/`: `__init__.py`, `aasist.py`, `lfcc_cnn.py`, `xlsr.py`, `vendored/__init__.py`;
  - `models/vendored/aasist/`: `AASIST-L.conf`, `AASIST.conf`, `AASIST.py`, `LICENSE`, `NOTICE`, `UPSTREAM.md`,
    `__init__.py`;
  - `training/`: `__init__.py`, `config.py`, `repro.py`, `train.py`
- **`research/`:** `HANDOFF_DECISION.md`, `PRE_HANDOFF_AUDIT.md`, `handoff_licence_audit.md`,
  `hindi_data_leakage_audit.md`, `hindi_repair_report.md`, `hindi_v2_1_validation.md`, `literature_matrix.md`,
  `references.md`, `related_work.md`, `research_gap.md`, `technical_decisions.md`
- **`scripts/`:** acquire_asvspoof, acquire_csfleurs, acquire_indicsynth, acquire_inthewild, acquire_kathbath,
  acquire_openslr104, audit_hindi_confound, env_info, environment_check, hindi_v2_1_analysis, hindi_v2_diagnostics,
  pause_controlled_diagnostic, verify_aasist_pretrained, verify_data (`.py`)
- **`tests/unit/`:** test_codec, test_config, test_env_info, test_evaluate, test_hindi_v2, test_metrics,
  test_pipeline, test_repro, test_robustness, test_validate (`.py`)

*Condition:* the manifests, provenance and Hindi metadata include **identifiers and hashes** of Kathbath, CS-FLEURS
and In-the-Wild files (no audio, no transcript text). This is acceptable for a **private** team repository and needed
for reproducibility; it is **not** cleared for a public repository until those licences are verified.

### 2. Files that must NOT be committed (all are already ignored by `.gitignore`)
- **Data:** `data/raw/**` (all audio, Kathbath transcripts and tar listings, CS-FLEURS metadata JSONL with
  transcripts, `_logs/`) and `data/processed/**` (OpenSLR segments, codec_sim, hindi_v2, hindi_v2_1)
- **Runs:** `runs/**` (checkpoints, scores, gate reports, diagnostics, `runs/upstream/aasist/AASIST.pth`)
- **Environment:** `.venv/`; `environment/reports/*.json` (machine-specific); `constraints.txt`
- **Caches:** `__pycache__/`, `.pytest_cache/`, and any HF cache
- **Source PDFs:** the root `*.pdf` files (paper1/2 and the three internal VaaniShield PDFs)
- **Secrets:** any `.env*`, key or credential files (none exist)

### 3. Datasets safe to transfer (restricted archive, with `LICENSES_REGISTER.md` corrected and sent along)
- ASVspoof 2019 LA pilot clips (ODC-By 1.0, attribution)
- IndicSynth Hindi subset (CC BY-NC 4.0, non-commercial academic use, attribution)
- OpenSLR 104 Hindi-English test recordings + our segments (CC BY-SA 4.0, attribution, share-alike)

Command (tested; 794 MB):
`tar -cf vaanishield-data.tar --exclude "raw/openslr104_hien/archives" -C data raw/asvspoof2019_la raw/indicsynth_hi raw/openslr104_hien processed/openslr104_hien`

### 4. Datasets the teammate must download themselves
- **DO NOT TRANSFER — FRIEND MUST DOWNLOAD DIRECTLY:**
  - Kathbath (test-known subset, transcripts, train references);
  - CS-FLEURS;
  - In-the-Wild (not needed).
- Also downloaded directly by the teammate:
  - full ASVspoof 2019 LA;
  - XLS-R weights;
  - Python packages.
- Regenerated rather than transferred: codec variants, Hindi v2/v2.1.
- Optionally, the teammate may download ASVspoof, IndicSynth and OpenSLR directly instead of receiving the archive;
  IndicSynth is verified bit-identical.

### 5. Remaining blockers
1. **`LICENSES_REGISTER.md` must be corrected** (Kathbath, In-the-Wild, IndicSynth, CS-FLEURS, ASVspoof rows).
   Otherwise the teammate is told Kathbath and In-the-Wild may be redistributed.
2. **No commit and no remote yet** (create a **private** repository; add the teammate).
3. Recommended, not blocking:
   - fix the three Windows notes in GPU_HANDOFF.md (`&&`, grep, activation policy);
   - decide whether to add an overwrite guard for existing `exp_id`s;
   - keep the repository private (IPR note R16; unresolved licences);
   - run a dedicated secret scanner before any public release.

### 6. Is the repository genuinely ready for GitHub?
**Not yet. One documentation fix away from ready for a PRIVATE GitHub repository.**
- No secrets, data, audio, weights, caches, environments or personal paths would be committed.
- Every command and config in the handoff guide exists and its flags are valid.
- No config was edited after its results, and tests pass.
- After correcting `LICENSES_REGISTER.md` it is ready to commit and push to a **private** repository. It is **not**
  ready for a **public** repository: unresolved licences, IPR note R16 and a supervisor sign-off are still needed.

---

## Resolution (2026-10-08, after this audit, on request)
| Item | Before | Now |
|---|---|---|
| §6 `LICENSES_REGISTER.md` | **FAIL**: Kathbath and In-the-Wild "redistribute: yes"; IndicSynth "no"; CS-FLEURS without "needs verification"; ASVspoof "to verify" | **PASS**: the five rows now match `licence_registry.yaml` and `research/handoff_licence_audit.md`. Kathbath, In-the-Wild and CS-FLEURS say "Needs verification from current dataset terms." and "DO NOT TRANSFER — FRIEND MUST DOWNLOAD DIRECTLY."; IndicSynth says non-commercial sharing with attribution; ASVspoof says ODC-By verified (checked 2026-10-08) |
| §4 Windows notes in `docs/GPU_HANDOFF.md` | WARNING | **Fixed**: `git clone` and `cd` on two lines (no `&&`); execution-policy hint next to `.venv\Scripts\activate`; PowerShell `Select-String` alternative to `grep` for the AMR check (tested) |
| §1 this report | quoted the laptop's user-profile path pattern | reworded; the re-scan finds no user path or username in any commit candidate |

Re-checks after the fixes:
- the command checker is unchanged (32 commands; the same two known false positives);
- the secret scan has 0 hits;
- no "redistribute: yes" remains for an unresolved dataset;
- 154 commit candidates (the 153 + this report).

**Updated answer to question 6:**
- **Ready to commit and push to a PRIVATE GitHub repository**; the remaining blocker is only that no commit or remote
  exists yet.
- **Still not ready for a public repository:** unresolved licences, IPR note R16, a supervisor sign-off, and a
  dedicated secret scanner run.
- Still recommended: an overwrite guard for existing `exp_id`s.
