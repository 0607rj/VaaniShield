"""PyTorch Dataset over a manifest split: returns (waveform [L] float32, label 0=bonafide/1=spoof, clip_id)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.data.audio import CROP_SAMPLES, crop_or_pad, preprocess
from ml.data.manifest import ManifestRow, read_manifest
from ml.training.repro import item_rng

LABELS = {"bonafide": 0, "spoof": 1}


def load_split(manifest: str | Path, split: str) -> list[ManifestRow]:
    rows = [r for r in read_manifest(Path(manifest)) if r.split == split]
    if not rows:
        raise ValueError(f"no rows with split={split!r} in {manifest}")
    return rows


def select_rows(manifest: str | Path, split: str, include_sources: list[str] | None = None,
                max_clips: int | None = None, seed: int = 0) -> list[ManifestRow]:
    """Rows of one split, optionally restricted to some sources and to a deterministic subset.

    The subset keeps the label balance of the split (stratified) and depends only on (manifest, split, seed),
    so the same config selects the same clips on every machine.
    """
    rows = load_split(manifest, split)
    if include_sources:
        rows = [r for r in rows if r.source_dataset in set(include_sources)]
        if not rows:
            raise ValueError(f"no {split} rows from sources {include_sources} in {manifest}")
    if max_clips and len(rows) > max_clips:
        rng = random.Random(f"{seed}|{split}")
        by_label: dict[str, list[ManifestRow]] = {}
        for r in sorted(rows, key=lambda r: r.clip_id):
            by_label.setdefault(r.label, []).append(r)
        picked = []
        for label, group in sorted(by_label.items()):
            k = max(1, round(max_clips * len(group) / len(rows)))
            picked += rng.sample(group, min(k, len(group)))
        rows = sorted(picked, key=lambda r: r.clip_id)[:max_clips] if len(picked) > max_clips else sorted(picked, key=lambda r: r.clip_id)
    return rows


class AudioDataset(Dataset):
    """preload=True keeps preprocessed audio in memory (small data); False reads each file on access (large data).

    train=True  -> random crop, different per epoch but reproducible (seed, epoch, clip_id)
    train=False -> centre crop, identical every time
    """

    def __init__(self, rows: list[ManifestRow], data_root: Path, train: bool, seed: int = 0, crop: int = CROP_SAMPLES,
                 preload: bool = True):
        self.rows = rows
        self.data_root = Path(data_root)
        self.train = train
        self.seed = seed
        self.crop = crop
        self.epoch = 0
        self.audio = [preprocess(self.data_root / r.path) for r in rows] if preload else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        wave = self.audio[i] if self.audio is not None else preprocess(self.data_root / r.path)
        rng = item_rng(self.seed, self.epoch, r.clip_id) if self.train else None
        wave = crop_or_pad(wave, self.crop, rng)
        return torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32)), LABELS[r.label], r.clip_id
