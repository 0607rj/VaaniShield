# VaaniShield

**Towards Code-Switch-Robust Audio Deepfake Detection for Hindi-English Speech under Messaging-App Compression**
B.Tech CSE research project (AKGEC / AKTU, Group-20).

> Status: pilot data downloaded; no models or results yet. The current state is always in [`PROJECT_STATE.md`](PROJECT_STATE.md).
> The tool will estimate whether a clip contains synthetic-speech artefacts. It is decision support, **not forensic proof**.

## Repository map

| Path | Contents |
|---|---|
| `ml/data/` | `manifest.py` (manifest CSV schema), `provenance.py` (download helpers, licence gate, provenance) |
| `ml/models/` | detector models (next step) |
| `ml/training/` | `config.py` (YAML config schema), `repro.py` (seeding, hashing); training loop (next) |
| `ml/evaluation/` | `metrics.py` (EER, AUC, ECE, ...), `evaluate.py` (metrics from a score CSV) |
| `ml/inference/` | score one audio file (later) |
| `ml/configs/` | plain YAML configs (none yet) |
| `data/raw/`, `data/processed/` | audio (gitignored); `data/raw/_logs/` = download logs |
| `data/manifests/` | licence registry, provenance JSON, licence texts; manifest CSVs (next) |
| `scripts/` | `acquire_*.py` dataset downloaders (inspect / pilot / full), `environment_check.py` + `env_info.py` |
| `client/`, `server/` | simple demo client and server (not built yet) |
| `research/` | literature matrix, gap analysis, references |
| `docs/` | earlier, larger design documents (reference only; the current plan is PROJECT_STATE.md §0) |
| `environment/` | dependency inputs (`*.in`) and hashed locks (`lock/`) |
| `tests/unit/` | unit tests |

`ml` is the Python package (`from ml.evaluation.metrics import eer`). Run commands from the repository root.

## Setup (Windows laptop, CPU)

Requires [uv](https://docs.astral.sh/uv/) and Git. Python 3.12 is fetched by uv if missing.

```powershell
git clone <repo-url> VaaniShield
cd VaaniShield
uv venv --python 3.12 .venv
uv pip sync --python .venv\Scripts\python.exe environment\lock\py312-windows.txt
uv pip install --python .venv\Scripts\python.exe --no-deps -e .
.venv\Scripts\activate
python scripts\environment_check.py
python -m pytest -q
```

ffmpeg (needed from Phase 2 for non-WAV decoding and codec augmentation). Install one pinned build, then re-run the environment check:
```powershell
winget install --id Gyan.FFmpeg -e        # or: choco install ffmpeg-full
python scripts\environment_check.py       # must show libopus + libmp3lame; libopencore_amrnb optional
```

Optional: set `VAANI_DATA_ROOT` to keep audio outside the repo (default: `data/`).

## Setup (Linux / Colab / Kaggle, GPU)

```bash
git clone <repo-url> VaaniShield && cd VaaniShield
pip install uv
# Option A (recommended on Colab/Kaggle): keep the image's CUDA torch; install the rest at locked versions.
grep -E '^[A-Za-z0-9_.-]+==' environment/lock/py312-linux.txt | awk '{print $1}' \
  | grep -v -E '^(torch|nvidia-|triton|cuda-)' > /tmp/vaani-constraints.txt
uv pip install --system -c /tmp/vaani-constraints.txt \
  -r environment/requirements-core.in -r environment/requirements-dev.in transformers
uv pip install --system --no-deps -e .
# Option B (own Linux GPU machine): exact hashed lock incl. CUDA torch (large download).
#   uv venv --python 3.12 .venv && uv pip sync --python .venv/bin/python environment/lock/py312-linux.txt && uv pip install --python .venv/bin/python --no-deps -e .
python scripts/environment_check.py
python -m pytest -q
```
On Colab/Kaggle the actual torch/CUDA versions differ from the lock. Record them with `environment_check.py`.

## Everyday commands

This laptop is for development and smoke tests; full training runs on the GPU machine. See [`docs/LOCAL_VS_GPU_WORKFLOW.md`](docs/LOCAL_VS_GPU_WORKFLOW.md).

```bash
python -m ml.training.train --config ml/configs/smoke_cpu.yaml
python -m ml.evaluation.evaluate --config ml/configs/smoke_cpu.yaml
```

```bash
python scripts/acquire_asvspoof.py --pilot 20 --accept-terms      # small licensed pilot (see --help of each acquire_*.py)
python -m ml.evaluation.evaluate --scores path/to/scores.csv [--threshold <dev-chosen>]
```

## Rules that are enforced, not just written down

- No datasets, audio, checkpoints, consent forms or secrets in Git (`.gitignore`).
- Configs reject unknown keys. Downloads record URL, version and sha256 in `data/manifests/provenance/`.
- Test sets are never used for tuning. Thresholds and calibration are chosen on dev only.
- Reported numbers must come from our own evaluation runs. Nothing is copied from papers.

## Licences

Our code licence is pending a team decision (PROJECT_STATE Q-L). Third-party datasets and models keep their own licences; see [`LICENSES_REGISTER.md`](LICENSES_REGISTER.md). Several are **non-commercial** (IndicSynth, XTTS-v2/CPML derivatives, SwitchLingua, CS-FLEURS).
