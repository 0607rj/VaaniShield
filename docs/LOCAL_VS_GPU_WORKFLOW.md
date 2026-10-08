# Local (development) vs GPU (training) workflow

| Machine | Role | Never do here |
|---|---|---|
| **Local laptop** (weak CPU, no CUDA) | write/debug code, small data subsets, smoke tests, unit tests, inference demos | full downloads, full training, full-dataset evaluation |
| **GPU machine** (teammate) | full downloads, full manifests, full training and evaluation | editing code that has not been committed |

Everything machine-specific lives in the YAML config (`data.*`, `train.*`, `output.*`) or a CLI override, so the
**same commit + same config** runs on both machines. Run all commands from the repository root.

## 1. Local laptop: what to run
```powershell
python -m pytest -q                                                    # all unit tests
python -m ml.training.train    --config ml/configs/smoke_cpu.yaml      # ~1 min: 16 train / 8 dev clips, 1 epoch, CPU
python -m ml.evaluation.evaluate --config ml/configs/smoke_cpu.yaml    # scores 8 test clips with the dev threshold
python -m ml.inference.predict --checkpoint runs/smoke/checkpoints/SMOKE_lfcc_cnn_cpu/best.pt --audio <file.wav>
python -m ml.training.train    --config ml/configs/smoke_cpu_xlsr.yaml # ~2 min incl. XLS-R load (needs 1.3 GB HF cache once)
```
Run the smoke test after every code change, **before** committing and handing over. Results of the smoke run are
meaningless (16 training clips); they only prove the pipeline works.

## 2. GPU machine: what to run
```bash
# once: environment (Linux lock; on Colab/Kaggle keep the image's CUDA torch, see README)
uv venv --python 3.12 .venv && uv pip sync --python .venv/bin/python environment/lock/py312-linux.txt
uv pip install --python .venv/bin/python --no-deps -e .
python scripts/environment_check.py          # must show "CUDA available: True"
python -m pytest -q
python -m ml.training.train --config ml/configs/smoke_cpu.yaml --device cuda   # 1-minute smoke test on the GPU

# data: download on the GPU machine itself (faster link, provenance recorded there)
export VAANI_DATA_ROOT=/path/with/space/vaanishield-data      # optional; default is <repo>/data
python scripts/acquire_asvspoof.py --full --confirm-large --accept-terms          # 7.6 GB LA.zip, verified
python -m ml.data.build_canonical_manifest --out data/manifests/vaanishield_v1_full.csv
python -m ml.data.check_manifest --manifest data/manifests/vaanishield_v1_full.csv  # must PASS before training

# full training + evaluation
python -m ml.training.train      --config ml/configs/full_gpu_aasist.yaml        # also: ml/configs/full_gpu_xlsr.yaml
# interrupted? run the same command with --resume (continues from <checkpoint_dir>/<exp_id>/last.pt;
# device, workers, data root and epochs may change, nothing else)
python -m ml.evaluation.evaluate --config ml/configs/full_gpu_aasist.yaml --split test
python -m ml.evaluation.evaluate --config ml/configs/full_gpu_aasist.yaml --split dev
```
Useful overrides (no YAML edit needed): `--device auto|cpu|cuda`, `--data-root`, `--num-workers`,
`--checkpoint-dir`, `--results-dir`, and for training `--epochs`.
To change anything else (batch size, subset size, lr, model), copy the config to a **new file with a new `exp_id`**.
Never edit a config after results were produced with it.

## 3. What to transfer
| Direction | What | How |
|---|---|---|
| laptop → GPU | code, configs, manifests (`data/manifests/*.csv`, `licence_registry.yaml`, `provenance/*.json`), `environment/` locks, docs | **git** (commit + push; the teammate pulls the exact commit) |
| GPU → laptop | `runs/full/checkpoints/<exp_id>/best.pt` | file copy (Drive/USB). ~1–4 MB per model |
| GPU → laptop | `runs/full/results/<exp_id>/` (`run_info.json`, `history.json`, `*_scores.csv`, `*_metrics.json`) | file copy, and commit the small result files you want to keep |
| GPU → laptop (optional) | `data/manifests/vaanishield_v1_full.csv` (+ its `_report.md/json`) | git; it is text and lets the laptop verify the manifest hash stored in the checkpoint |

Checkpoints are saved with CPU tensors, so a GPU-trained `best.pt` loads on the laptop with `--device cpu`.

## 4. What NOT to transfer
- `.venv/`: rebuild from the lock file on each machine.
- `data/raw/`, `data/processed/`: download on each machine with the scripts (provenance is recorded per download). Copy only if the GPU machine cannot download; then also copy the matching `data/manifests/provenance/*.json`.
- `data/raw/_logs/`, `__pycache__/`, `runs/smoke/`, `runs/compat/`: local scratch.
- Secrets (`HF_TOKEN`, Kaggle/Colab keys): set as environment variables on each machine; never in files.
- Source PDFs: project documents, not code; keep them out of Git.

## 5. Reproducing an experiment on another machine
1. Same **git commit** (`run_info.json` → `git_commit`) and same **config file** (`config_sha256` is in the checkpoint and `run_info.json`).
2. Same **environment lock** (`environment/lock/py312-*.txt`); record `python scripts/environment_check.py` output.
3. Same **manifest**: the checkpoint stores `manifest_sha256`; `python -m ml.evaluation.validate` (check 8) compares it with the file on disk.
4. Same **seed** (in the config). Data subsets (`max_clips_per_split`) are chosen deterministically from (manifest, split, seed), so both machines select the same clips.
5. Expect **bit-identical** results only on the same hardware type: CPU runs repeat exactly (verified), GPU (cuDNN) runs may differ slightly between GPUs/drivers. Compare metrics, not raw weights, across machines.
6. Never select thresholds, epochs or configs by looking at the test split; the dev split is for that.
