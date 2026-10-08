"""Baseline B0-lite: LFCC features + a small CNN. Deliberately simple and fast on CPU.

LFCC = log linear-frequency filterbank energies -> DCT (the classic ASVspoof front end), plus delta and
delta-delta, then per-utterance mean/variance normalisation. This is a simplified stand-in for LFCC-LCNN
(no max-feature-map layers); it is our own small implementation, not the ASVspoof reference code.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def linear_filterbank(n_filters: int, n_fft: int, sr: int) -> torch.Tensor:
    """Triangular filters equally spaced on a LINEAR frequency axis: [n_filters, n_fft//2+1]."""
    n_bins = n_fft // 2 + 1
    freqs = torch.linspace(0, sr / 2, n_bins)
    edges = torch.linspace(0, sr / 2, n_filters + 2)
    fb = torch.zeros(n_filters, n_bins)
    for m in range(n_filters):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (freqs - lo) / (mid - lo)
        down = (hi - freqs) / (hi - mid)
        fb[m] = torch.clamp(torch.minimum(up, down), min=0.0)
    return fb


def dct_matrix(n_in: int, n_out: int) -> torch.Tensor:
    """Orthonormal DCT-II as a matrix [n_out, n_in]."""
    n = torch.arange(n_in).float()
    k = torch.arange(n_out).float().unsqueeze(1)
    m = torch.cos(math.pi / n_in * (n + 0.5) * k) * math.sqrt(2.0 / n_in)
    m[0] /= math.sqrt(2.0)
    return m


def deltas(x: torch.Tensor, width: int = 2) -> torch.Tensor:
    """Regression deltas over the last (time) axis, same length as input."""
    denom = 2 * sum(i * i for i in range(1, width + 1))
    padded = F.pad(x, (width, width), mode="replicate")
    t = x.shape[-1]
    return sum(i * (padded[..., width + i : width + i + t] - padded[..., width - i : width - i + t]) for i in range(1, width + 1)) / denom


class LFCC(nn.Module):
    def __init__(self, sr: int = 16000, n_fft: int = 512, win: int = 320, hop: int = 160, n_filters: int = 20, n_lfcc: int = 20):
        super().__init__()
        self.n_fft, self.win, self.hop = n_fft, win, hop
        self.register_buffer("window", torch.hann_window(win), persistent=False)
        self.register_buffer("fb", linear_filterbank(n_filters, n_fft, sr), persistent=False)
        self.register_buffer("dct", dct_matrix(n_filters, n_lfcc), persistent=False)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:  # [B, L] -> [B, 3*n_lfcc, T]
        spec = torch.stft(wave, self.n_fft, self.hop, self.win, self.window, return_complex=True).abs() ** 2
        logfb = torch.log(torch.matmul(self.fb, spec) + 1e-6)  # [B, n_filters, T]
        c = torch.matmul(self.dct, logfb)  # [B, n_lfcc, T]
        d1 = deltas(c)
        feats = torch.cat([c, d1, deltas(d1)], dim=1)
        return (feats - feats.mean(-1, keepdim=True)) / (feats.std(-1, keepdim=True) + 1e-5)


class LFCCCNN(nn.Module):
    """wave [B, L] -> logits [B, 2] (index 1 = spoof)."""

    def __init__(self, channels: tuple[int, ...] = (16, 32, 32), dropout: float = 0.3):
        super().__init__()
        self.lfcc = LFCC()
        layers, c_in = [], 1
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c
        self.cnn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c_in, 2))
        self.config = {"channels": list(channels), "dropout": dropout}

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        x = self.lfcc(wave).unsqueeze(1)  # [B, 1, 60, T]
        x = self.cnn(x)
        x = x.mean(dim=(2, 3))  # global average pool
        return self.head(x)


def spoof_logit(logits: torch.Tensor) -> torch.Tensor:
    """Single score, higher = more likely spoof (DEC-1): logit_spoof - logit_bonafide."""
    return logits[:, 1] - logits[:, 0]
