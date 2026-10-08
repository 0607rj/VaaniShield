# environment/

| File | Purpose |
|---|---|
| `requirements-core.in` | numerical / audio / config stack (lower bounds) |
| `requirements-torch.in` | torch + transformers (torchaudio intentionally excluded, DEC-14) |
| `requirements-dev.in` | pytest (ruff is pinned too but not configured) |
| `lock/py312-windows.txt` | hashed lock for Windows x86-64, Python 3.12 (CPU torch from PyPI) |
| `lock/py312-linux.txt` | hashed lock for Linux x86-64 (manylinux_2_28), Python 3.12. PyPI Linux torch = CUDA build (GPU machines) |
| `reports/` | output of `scripts/environment_check.py` (gitignored, machine-specific) |

Regenerate locks (only on purpose; commit the diff and note it in PROJECT_STATE.md):
```
uv pip compile environment/requirements-core.in environment/requirements-torch.in environment/requirements-dev.in \
  --python-version 3.12 --python-platform x86_64-pc-windows-msvc --generate-hashes -o environment/lock/py312-windows.txt
uv pip compile environment/requirements-core.in environment/requirements-torch.in environment/requirements-dev.in \
  --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes -o environment/lock/py312-linux.txt
```

Colab/Kaggle: their images ship their own CUDA torch. Installing `py312-linux.txt` would replace it (multi-GB download).
Preferred: keep the image's torch, install everything else, and let `environment_check.py` record the actual versions in
each run's `env.json`. See README "Colab / Kaggle".
