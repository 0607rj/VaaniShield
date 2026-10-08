"""Canonical manifest schema and I/O (docs/architecture.md §4.5, M05).

Phase 2A implements the schema, validation and writing. Speaker-graph splitting and leakage checks are Phase 2B (P2.7).
Manifests never contain absolute paths or third-party transcript text: ``transcript_id`` is a hash.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

COLUMNS = [
    "clip_id",
    "path",
    "label",
    "source_dataset",
    "subset",
    "speaker_id",
    "speaker_group",
    "gender",
    "language",
    "script",
    "transcript_id",
    "generator",
    "generator_family",
    "generator_version",
    "parent_clip_id",
    "transform",
    "codec",
    "sample_rate_orig",
    "duration_s",
    "sha256",
    "licence",
    "consent_status",
    "split",
    "notes",
]


class ManifestRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:\-]*$")
    path: str
    label: Literal["bonafide", "spoof"]
    source_dataset: str
    subset: str | None = None
    speaker_id: str
    speaker_group: str | None = None
    gender: Literal["f", "m", "u"] = "u"
    language: Literal["hi", "en", "hi-en"]
    script: str | None = None
    transcript_id: str | None = None
    generator: str
    generator_family: str
    generator_version: str | None = None
    parent_clip_id: str | None = None
    transform: str = "clean"
    codec: str | None = None  # codec of the stored file as decoded (e.g. flac, pcm_s16le, aac); None only in legacy manifests
    sample_rate_orig: int | None = Field(default=None, gt=0)
    duration_s: float | None = Field(default=None, ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    licence: str
    consent_status: Literal["public", "consented", "n/a"] = "public"
    split: Literal["train", "dev", "test"] | None = None
    notes: str | None = None

    @field_validator("path")
    @classmethod
    def _relative_posix(cls, v: str) -> str:
        if v.startswith(("/", "\\")) or ":" in v.split("/")[0] or "\\" in v or ".." in Path(v).parts:
            raise ValueError(f"path must be DATA_ROOT-relative POSIX without '..': {v!r}")
        return v

    @field_validator("transform")
    @classmethod
    def _transform(cls, v: str) -> str:
        if v != "clean" and not v.startswith(("sim:", "app:", "diag:")):
            raise ValueError("transform must be 'clean', 'sim:<...>', 'app:<...>' or 'diag:<...>' (diagnostic variants)")
        return v

    @field_validator("generator_family")
    @classmethod
    def _bonafide_family(cls, v: str, info) -> str:
        label = info.data.get("label")
        gen = info.data.get("generator")
        if label == "bonafide" and (gen != "none" or v != "none"):
            raise ValueError("bonafide rows must have generator='none' and generator_family='none'")
        if label == "spoof" and v == "none":
            raise ValueError("spoof rows need a generator_family")
        return v


def transcript_hash(text: str) -> str:
    """Stable id for a transcript without storing (possibly licensed) text in Git."""
    norm = " ".join(text.split())
    return "t_" + hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    ids = [r.clip_id for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate clip_id in manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r.clip_id):
            d = r.model_dump()
            w.writerow({k: "" if d[k] is None else d[k] for k in COLUMNS})
    tmp.replace(path)


def read_manifest(path: Path) -> list[ManifestRow]:
    """Read and validate every row (raises on the first invalid row, with its line number)."""
    rows: list[ManifestRow] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        legacy = [c for c in COLUMNS if c != "codec"]
        if reader.fieldnames not in (COLUMNS, legacy):  # legacy = pre-codec manifests (asvspoof_pilot.csv)
            raise ValueError(f"{path}: columns differ from the canonical schema")
        for i, raw in enumerate(reader, start=2):
            clean = {k: (v if v != "" else None) for k, v in raw.items()}
            try:
                rows.append(ManifestRow.model_validate(clean))
            except Exception as exc:  # noqa: BLE001 - re-raised with location
                raise ValueError(f"{path}:{i}: {exc}") from exc
    if len({r.clip_id for r in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate clip_id")
    return rows
