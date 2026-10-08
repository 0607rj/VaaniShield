"""AASIST baseline (B1): thin wrapper around the unmodified upstream model in ml/models/vendored/aasist/.

Only conventions are adapted here (see vendored/aasist/UPSTREAM.md):
- input: wave [B, 64600] float32 at 16 kHz (same tensor every VaaniShield model receives)
- output: logits [B, 2] with index 1 = SPOOF (upstream index 1 = bona fide, so the columns are swapped)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch import nn

from ml.models.vendored.aasist.AASIST import Model as UpstreamAASIST

VENDOR_DIR = Path(__file__).resolve().parent / "vendored" / "aasist"
WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "runs" / "upstream" / "aasist"
COMMIT = "a04c9863f63d44471dde8a6abcb3b082b07cd1d1"
PRETRAINED = {
    "AASIST": ("models/weights/AASIST.pth", "51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0"),
    "AASIST-L": ("models/weights/AASIST-L.pth", "814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a"),
}


def upstream_model_config(variant: str = "AASIST") -> dict:
    return json.loads((VENDOR_DIR / f"{variant}.conf").read_text(encoding="utf-8"))["model_config"]


class AASISTDetector(nn.Module):
    def __init__(self, variant: str = "AASIST"):
        super().__init__()
        if variant not in PRETRAINED:
            raise ValueError(f"unknown AASIST variant {variant!r}")
        self.net = UpstreamAASIST(upstream_model_config(variant))
        self.config = {"variant": variant}

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        _, out = self.net(wave)  # upstream: (hidden, logits[bonafide=1, spoof=0])
        return out.flip(1)  # -> [bonafide, spoof]

    def load_upstream_pretrained(self) -> str:
        """Load the official checkpoint (trained on ASVspoof 2019 LA train). Returns its verified path."""
        path = fetch_pretrained(self.config["variant"])
        self.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return str(path)


def fetch_pretrained(variant: str = "AASIST") -> Path:
    """Download the official weights at the pinned commit (once) and verify their sha256."""
    from ml.data.provenance import download_file, sha256_of

    rel, sha = PRETRAINED[variant]
    dest = WEIGHTS_DIR / Path(rel).name
    if not dest.exists() or sha256_of(dest) != sha:
        url = f"https://raw.githubusercontent.com/clovaai/aasist/{COMMIT}/{rel}"
        download_file(url, dest, logging.getLogger("vaanishield.aasist"), expected_sha256=sha)
    return dest
