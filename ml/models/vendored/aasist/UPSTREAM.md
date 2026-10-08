# Vendored: AASIST (clovaai/aasist)

| Field | Value |
|---|---|
| Upstream | https://github.com/clovaai/aasist |
| Commit | `a04c9863f63d44471dde8a6abcb3b082b07cd1d1` (2022-01-18, "change model output") |
| Licence | MIT, Copyright (c) 2021-present NAVER Corp. (`LICENSE`); subcomponent notices in `NOTICE` (RawNet2 baseline: MIT, eurecom-asp; Jungjee/RawNet: MIT) |
| Files copied **unchanged** | `models/AASIST.py` → `AASIST.py` (sha256 `9e0d3e80937dd0577beea7883098465a479da23a198ebc0d712abcc59b0bec50`), `config/AASIST.conf`, `config/AASIST-L.conf`, `LICENSE`, `NOTICE` |
| Local modifications | none. All adaptation is in `ml/models/aasist.py` (wrapper). |
| Pretrained weights (not in Git) | `models/weights/AASIST.pth` sha256 `51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0`; `AASIST-L.pth` sha256 `814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a`. Downloaded on demand by the wrapper into `runs/upstream/aasist/`. |
| Upstream claims (README, not our result) | pretrained AASIST: EER 0.83 %, min t-DCF 0.0275 on ASVspoof 2019 LA eval; AASIST-L: EER 0.99 %, 0.0309 |
| Training data of the weights | ASVspoof 2019 LA **train** partition (README) |

## Conventions that differ from VaaniShield (handled in the wrapper)
- Upstream labels: bona fide = 1, spoof = 0; score = `output[:, 1]` (higher = bona fide). VaaniShield: spoof = 1. The wrapper swaps the two output columns.
- `Model.forward(x)` returns `(last_hidden, output)`; the wrapper returns `output` only.
- Input: raw 16 kHz waveform, 64,600 samples, no normalisation. Upstream eval takes the FIRST 64,600 samples; VaaniShield uses centre crop / windows (same for every model).
