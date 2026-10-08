# GPU machine handoff: everything the teammate with the GPU needs

You are running the **heavy** part of VaaniShield: full downloads, full training, the full codec run, and robustness
evaluation. The laptop only did small pilots and smoke tests. You do **not** need to change any code.

Nothing below assumes the laptop's username, folders, Python or CUDA version. Run every command **from the repository
root**. Commands are given for **Linux (bash)** and **Windows (PowerShell)**; pick one and stay with it.

## Overview: order of work and rough cost

| # | Step | Time (rough guess) | Disk |
|---|---|---|---|
| 0 | Get repo + restricted data archive; download Kathbath + CS-FLEURS yourself | 10–20 min | 1.2 GB |
| 1–3 | Environment, GPU check, paths | 15–30 min | 6 GB (.venv) |
| 4 | Verify copied data + smoke tests on the GPU | 10 min | +1.3 GB (XLS-R weights) |
| 5 | Download full ASVspoof 2019 LA, build + check full manifest | 0.5–2 h (network) | +15 GB |
| 6 | Full training: LFCC → AASIST → XLS-R | several hours each (overnight) | small |
| 7 | Evaluate each model (dev, then test once) | minutes each | small |
| 8 | Full codec variants (CPU-bound, many cores help) | 1–3 h | +10–15 GB |
| 9 | Robustness evaluation of the full models | under 1 h | +2 GB |
| 10 | Send results back | | |
| H | *Optional* Hindi v2.1: rebuild from original sources, gate, pause diagnostic (CPU) | ~45 min | +0.5 GB |

Times are estimates, not measurements; they depend on the GPU, CPU and network. Have **≥ 50 GB free** in total.

## Rules (research integrity, please read)
- **Never** pick epochs, thresholds, configs or anything else by looking at **test** results. `dev` is for that. Run
  test evaluation once per model, at the end.
- **Never edit a config after results exist.** To change something (e.g. batch size for memory), copy the YAML to a new
  file with a **new `exp_id`**.
- **Hindi:** the old v1 Hindi pool is confounded (`research/hindi_data_leakage_audit.md`): never train on it and
  never report its numbers. The repaired **Hindi v2.1** is cleared only for *pilot* training under conditions
  (`research/hindi_v2_1_validation.md` §10). On this machine you only rebuild and verify it (step H). The full
  training, codec and robustness configs contain **no Hindi** on purpose.
- Do not commit audio, checkpoints, `.venv`, tokens or passwords to Git.
- If something fails, **report it**; do not work around it by changing data splits or thresholds.

---

## 0. Get the repository and the pilot data
```bash
git clone <team-repo-url> VaaniShield
cd VaaniShield                     # two lines: '&&' does not work in Windows PowerShell 5.1
```
The audio is **not** in Git. You receive a separate, **restricted** archive (Drive/USB, ~0.8 GB) containing only
the groups whose licences permit sharing: the ASVspoof and IndicSynth pilot clips, and the OpenSLR recordings and
segments (see `research/handoff_licence_audit.md` and `research/HANDOFF_DECISION.md`).
- **Conditions travelling with it:** non-commercial research use only (IndicSynth), share-alike (OpenSLR), attribution
  per `LICENSES_REGISTER.md`, no onward redistribution.
- **Unpack** so that you have `<DATA>/raw/...` and `<DATA>/processed/...`, where `<DATA>` is any folder you like
  (step 3).

Kathbath and CS-FLEURS have unresolved terms, so you download them **yourself** from the original sources. The
pinned scripts give files identical to the laptop's (verified 2026-10-08). After step 3, from the repository root:
```bash
python scripts/acquire_kathbath.py --subset 300 --accept-terms
python scripts/acquire_csfleurs.py --metadata --accept-terms
python scripts/acquire_csfleurs.py --files data/manifests/hinglish_pilot_plan.csv --accept-terms
python -m ml.data.build_codec_variants --config ml/configs/codec_pilot.yaml           # regenerates pilot codec variants
python -m ml.data.build_codec_variants --config ml/configs/codec_robust_smoke.yaml
```
Read `LICENSES_REGISTER.md` before passing `--accept-terms`. **Hindi: see the rules above and step H; do not train on
it or report Hindi numbers from these runs.**

## 1. Create the environment
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (one binary; it also downloads Python 3.12, so the
system Python version does not matter).

**First read your driver's CUDA version:** `nvidia-smi` → top-right "CUDA Version: X.Y". Choose the PyTorch build
whose CUDA version is **≤ X.Y**. Available builds of the pinned torch 2.14.1: `cu126`, `cu130`, `cu132`.

### Route A: Linux, driver CUDA ≥ 13.0 (exact hashed lock)
```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python environment/lock/py312-linux.txt
uv pip install --python .venv/bin/python --no-deps -e .
source .venv/bin/activate
```

### Route B: any OS / any driver (CUDA torch first, then everything else pinned to the lock)
Replace `cu126` with `cu130` or `cu132` if your driver allows.

Linux (bash):
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -c "import re,sys; pins=[l.split()[0] for l in open(sys.argv[1], encoding='utf-8') if re.match(r'^[A-Za-z0-9_.-]+==', l)]; open('constraints.txt','w').write('\n'.join(p for p in pins if not re.match(r'^(torch|nvidia-|triton|cuda-)', p)) + '\n')" environment/lock/py312-linux.txt
uv pip install --python .venv/bin/python -c constraints.txt -r environment/requirements-core.in -r environment/requirements-dev.in transformers
uv pip install --python .venv/bin/python --no-deps -e .
source .venv/bin/activate
```
Windows (PowerShell):
```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python.exe -c "import re,sys; pins=[l.split()[0] for l in open(sys.argv[1], encoding='utf-8') if re.match(r'^[A-Za-z0-9_.-]+==', l)]; open('constraints.txt','w').write('\n'.join(p for p in pins if not re.match(r'^(torch|nvidia-|triton|cuda-)', p)) + '\n')" environment\lock\py312-windows.txt
uv pip install --python .venv\Scripts\python.exe -c constraints.txt -r environment\requirements-core.in -r environment\requirements-dev.in transformers
uv pip install --python .venv\Scripts\python.exe --no-deps -e .
.venv\Scripts\activate            # if blocked by the execution policy: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass, then retry
```
(Colab/Kaggle: skip `uv venv` and the torch line; the image already has CUDA torch. Use `--system` instead of
`--python ...` and `py312-linux.txt`.)

- [ ] `python -m pytest -q` → **96 passed** (CPU-only tests; 0.5–2 min)

**System ffmpeg with AMR (needed only for step 8):** Ubuntu/Colab: `sudo apt install ffmpeg`, then check that
`ffmpeg -hide_banner -encoders | grep -E "amrnb|amrwb"` (Windows PowerShell: `ffmpeg -hide_banner -encoders | Select-String "amrnb|amrwb"`) shows `libopencore_amrnb` and `libvo_amrwbenc`.
Windows: use a "full" ffmpeg build that lists those two encoders, and put it on `PATH`.

## 2. Check PyTorch CUDA and the GPU
```bash
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| built for CUDA', torch.version.cuda, '| cuda available', torch.cuda.is_available(), '| GPUs', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
python scripts/environment_check.py          # saves a JSON report in environment/reports/ (send it back)
```
- [ ] `cuda available True` and your GPU is listed. If **False**: the torch build is CPU-only, or its CUDA is newer
  than the driver. Redo step 1 route B with a lower `cuXXX`.

## 3. Configure paths (every new shell, or put in your shell profile)
| Variable | Meaning | Default if unset |
|---|---|---|
| `VAANI_DATA_ROOT` | the `<DATA>` folder holding `raw/` and `processed/` | `<repo>/data` |
| `HF_HOME` | Hugging Face cache (XLS-R weights, 1.3 GB) | `~/.cache/huggingface` |

Linux: `export VAANI_DATA_ROOT=/path/to/vaanishield-data` and `export HF_HOME=/path/to/hf-cache`
Windows: `$env:VAANI_DATA_ROOT = "D:\vaanishield-data"` and `$env:HF_HOME = "D:\hf-cache"`

Command-line overrides also exist: `--data-root`, `--device auto|cpu|cuda`, `--num-workers N`, `--checkpoint-dir`,
`--results-dir`, and for training `--epochs`.

## 4. Verify the copied data, then smoke-test on the GPU (~10 min)
```bash
python scripts/verify_data.py              # every pilot audio file present and sha256-identical
python -m ml.training.train      --config ml/configs/smoke_cpu.yaml --device cuda
python -m ml.evaluation.evaluate --config ml/configs/smoke_cpu.yaml --device cuda
python -m ml.training.train      --config ml/configs/smoke_cpu_xlsr.yaml --device cuda    # first run downloads XLS-R into HF_HOME
```
- [ ] `verify_data.py` ends with `OK: all audio present and unchanged`.
- [ ] Training prints `device cuda (<your GPU>)`, one epoch, `saved .../best.pt`. Smoke numbers are meaningless.
- [ ] XLS-R prints `model xlsr_mlp: 315,577,756 params, 140,060 trainable`.

## 5. Full ASVspoof 2019 LA + full manifest
```bash
python scripts/acquire_asvspoof.py --full --confirm-large --accept-terms     # 7.1 GB, resumable, size-verified
python -m ml.data.build_canonical_manifest --out data/manifests/vaanishield_v1_full.csv
python -m ml.data.check_manifest --manifest data/manifests/vaanishield_v1_full.csv
python scripts/verify_data.py data/manifests/vaanishield_v1_full.csv
```
Read `LICENSES_REGISTER.md` before passing `--accept-terms`. If the download is interrupted, run it again; it resumes.
- [ ] `check_manifest`: all six hard checks **PASS** (exit 0). The Hindi **WARNINGS** are known and expected.
- [ ] Never overwrite `data/manifests/vaanishield_v1.csv` (the laptop subset); the full one has its own name.

## 6. Full training (GPU)
Run one at a time, smallest first (each fails fast if something is wrong):
```bash
python -m ml.training.train --config ml/configs/full_gpu_lfcc.yaml      # B0 LFCC-CNN, 50 epochs
python -m ml.training.train --config ml/configs/full_gpu_aasist.yaml    # B1 AASIST, 100 epochs, batch 24, AMP off
python -m ml.training.train --config ml/configs/full_gpu_xlsr.yaml      # P1 XLS-R frozen, 30 epochs, batch 32 x accum 2, AMP on
```
- All use ASVspoof only, 8 DataLoader workers (`--num-workers N` to change) and `device: cuda`. Early stopping is on
  the dev EER.
- Per-epoch lines show `dev_EER` on one centre crop. The final "best epoch" line uses full-clip windowed scoring (the
  deployment path), so the two can differ.
- **Out of GPU memory?** Copy the YAML to a new file with a **new `exp_id`**, lower `batch_size` and raise `grad_accum`
  so the product stays the same.
- **Windows:** if DataLoader workers hang or crash, add `--num-workers 0`.
- **Interrupted?** Run the same command with `--resume`; it continues from `last.pt` (saved every epoch). Extend with
  `--resume --epochs N`. Device, workers, data root, output dirs and epochs may change; anything else is refused.

## 7. Evaluate each model
```bash
python -m ml.evaluation.evaluate --config ml/configs/full_gpu_lfcc.yaml   --split dev
python -m ml.evaluation.evaluate --config ml/configs/full_gpu_aasist.yaml --split dev
python -m ml.evaluation.evaluate --config ml/configs/full_gpu_xlsr.yaml   --split dev
# only after all training is finished, once each:
python -m ml.evaluation.evaluate --config ml/configs/full_gpu_lfcc.yaml   --split test
python -m ml.evaluation.evaluate --config ml/configs/full_gpu_aasist.yaml --split test
python -m ml.evaluation.evaluate --config ml/configs/full_gpu_xlsr.yaml   --split test
```
Threshold = the dev threshold stored in each checkpoint. Results go to `runs/full/results/<exp_id>/`.

## 8. Full codec / channel variants (CPU-bound; needs ffmpeg with AMR from step 1)
```bash
python -m ml.data.build_codec_variants --config ml/configs/codec_full_gpu.yaml --workers 8   # use about your CPU core count
python -m ml.data.check_manifest --manifest data/manifests/codec_full_v1.csv
```
- [ ] The first lines list **every** condition as `ok`. If AMR shows `SKIP`, the run stops (by design): install the AMR
  ffmpeg, or copy the config with `skip_unavailable: true` and tell the team AMR is missing.
- Dev + test only; English (ASVspoof) and Hinglish (OpenSLR, CS-FLEURS) only, **no Hindi**; capped at 3,000 dev +
  6,000 test parents of the full manifest (plus all Hinglish pilot dev/test), giving about 120k files (10–15 GB). Held-out codecs (AAC, G.722, AMR-WB) are never applied to train clips.
- Interrupted? Run the same command again: finished files are reused.

## 9. Robustness evaluation of the full models
```bash
python -m ml.evaluation.robustness --config ml/configs/robustness_full_gpu.yaml --device cuda
```
- It evaluates 35 real + 35 fake test clips per language (en, hi-en; **no Hindi**) × clean + every codec, for the
  three full models, after checking that none of those clips or speakers were used in training.
- Outputs are in `runs/full/robustness/ROBUST_full_v1/`: `README.md`, `metrics.csv/json`, 4 plots and `scores/`.
- These are still **development** results (the Hinglish pool is the 200-clip pilot).
  The banner on every output says so; leave it.

## 10. Send back (Drive/USB; small files)
- [ ] `runs/full/checkpoints/<exp_id>/best.pt` for the 3 models (only `best.pt`; a few MB each)
- [ ] `runs/full/results/` (whole folder: `run_info.json`, `history.json`, `*_scores.csv`, `*_metrics.json`)
- [ ] `runs/full/robustness/ROBUST_full_v1/` (whole folder)
- [ ] `data/manifests/vaanishield_v1_full.csv`, `codec_full_v1*`, `robust_full_*` and their `_report.*` files
- [ ] `environment/reports/env_*.json` from step 2
- [ ] If you did step H: `runs/gate/hindi_v2_1_pilot_hi/`, `runs/diag/pause_controlled/summary.{md,json}` and the
      `verify_data.py` output (no audio)
- [ ] A short note: GPU model, total time per step, anything that failed or was changed (new `exp_id`s and why)

**Do NOT send:** `.venv/`, audio (`raw/`, `processed/`), the HF cache, `last.pt`, tokens/secrets.
Checkpoints contain CPU tensors, so they load on the laptop with `--device cpu`.

## H. Optional: rebuild and verify Hindi v2.1 (CPU; ~45 min, mostly network)
The Kathbath-derived Hindi audio may **not** be copied to you (licence "Needs verification from current dataset terms."),
so you rebuild it from the original sources. Purpose: an independent cross-machine check that the repaired dataset is
reproducible and passes the same gate. It does **not** authorise Hindi training on this machine.
```bash
python scripts/acquire_kathbath.py --transcripts --accept-terms                       # 18 MB of Hindi transcripts
python scripts/acquire_kathbath.py --fetch-ids data/manifests/hindi_v2_reference_ids.txt --tars clean/train_audio.tar --allow-train --extra-per-speaker 10 --accept-terms
#   ^ walks only the Hindi headers of the 85 GB train tar (~96k small range requests, ~15-30 min), fetches 553 files (55 MB)
python -m ml.data.build_hindi_v2 --config ml/configs/hindi_v2_1_pilot.yaml           # needs raw/indicsynth_hi from the archive
python scripts/verify_data.py data/manifests/hindi_v2_1_pilot.csv
python -m ml.data.hindi_gate --manifest data/manifests/hindi_v2_1_pilot.csv
python scripts/pause_controlled_diagnostic.py                                         # optional, ~13 min CPU
```
- [ ] The fetch log ends with `323 of 323 wanted found`.
- [ ] Gate prints `ACCEPTANCE_PASS_PROVISIONAL` (13/13) with diagnostic EERs close to the laptop's (metadata 0.463,
  nuisance 0.341, pause 0.419, waveform 0.221, spectral 0.234).
- [ ] `verify_data.py`: identical hashes are expected with the same environment lock. If hashes differ but the gate
  numbers match, report it (floating-point differences between CPUs are possible); do not "fix" anything.
- Never send the rebuilt audio back; send only the reports (step 10).

## Troubleshooting
| Symptom | Fix |
|---|---|
| `device 'cuda' requested but torch.cuda.is_available() is False` | CPU-only torch or driver too old → step 1 route B with a lower `cuXXX` |
| `verify_data.py` reports missing files | `VAANI_DATA_ROOT` not set in this shell, or the archive was unpacked one level too deep/shallow (it must contain `raw/` directly) |
| `FileNotFoundError` under `raw/asvspoof2019_la` during full training | step 5 not finished, or the manifest was built before the download completed: rebuild it |
| `refusing to resume: config differs` | the YAML was edited; resume with the original file, or start a new `exp_id` |
| codec run: `some conditions are unavailable` | ffmpeg without AMR encoders on `PATH` (step 1, last paragraph) |
| CUDA out of memory | new config + new `exp_id` with smaller `batch_size`, larger `grad_accum` |
| `Warning: You are sending unauthenticated requests to the HF Hub` | harmless; optional `HF_TOKEN` env var for faster downloads |
| XLS-R load report lists `project_q`, `quantizer.*` as UNEXPECTED | expected: pretraining-only heads are not used |
