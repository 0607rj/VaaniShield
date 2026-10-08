"""Create simulated-channel variants of manifest clips (transform="sim:<condition>"), with per-sample metadata.

    python -m ml.data.build_codec_variants --config ml/configs/codec_pilot.yaml          # laptop: tiny pilot
    python -m ml.data.build_codec_variants --config ml/configs/codec_full_gpu.yaml       # GPU/large-CPU machine
    options: --data-root PATH  --workers N  --overwrite (re-encode even if the output file exists)

Outputs (paths from the config):
  audio     <data_root>/processed/codec_sim/<condition>-<params_hash>/<source>/<parent_clip_id>.wav  (PCM-16, 16 kHz)
  manifest  parents (clean) + variants; variant.parent_clip_id = parent, same split / speaker / label / source
  metadata  JSONL, one record per variant: parent, source, split, codec, bitrate (requested + actual), every
            transformation parameter, tool versions, delay, sha256, quality stats
  quality   per-condition summary (CSV + Markdown)
Held-out rule: conditions with role eval_heldout are never applied to train-split clips.
Deterministic: parent selection depends only on (manifest, split, seed); codecs run single-threaded on fixed input.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from pydantic import BaseModel, ConfigDict

from ml.data.audio import load_audio
from ml.data.codec import SR, Condition, CodecUnavailable, apply, available, quality, tool_versions
from ml.data.dataset import select_rows
from ml.data.manifest import ManifestRow, read_manifest, write_manifest
from ml.data.provenance import DataPaths
from ml.training.repro import sha256_file

SPLITS = ("train", "dev", "test")


class OutputCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio_subdir: str = "processed/codec_sim"  # relative to data_root
    manifest: str
    metadata: str
    quality: str  # CSV; a .md summary is written next to it


class CodecRunCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    input_manifests: list[str]
    data_root: str | None = None
    splits: list[str] = list(SPLITS)
    include_sources: list[str] = []
    max_clips_per_split: dict[str, int] = {}  # empty = every clip (full run)
    seed: int = 2026
    workers: int = 1
    conditions_file: str = "ml/configs/codec_conditions.yaml"
    conditions: list[str] = []                # empty = every condition in conditions_file
    skip_unavailable: bool = True             # False = stop if a listed condition cannot run on this machine
    output: OutputCfg


def load_cfg(path: str | Path) -> tuple[CodecRunCfg, list[Condition]]:
    cfg = CodecRunCfg(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    conds = [Condition(**c) for c in yaml.safe_load(Path(cfg.conditions_file).read_text(encoding="utf-8"))["conditions"]]
    if len({c.id for c in conds}) != len(conds):
        raise ValueError("duplicate condition ids")
    if cfg.conditions:
        unknown = set(cfg.conditions) - {c.id for c in conds}
        if unknown:
            raise ValueError(f"unknown conditions {sorted(unknown)}")
        conds = [c for c in conds if c.id in cfg.conditions]
    return cfg, conds


def parents(cfg: CodecRunCfg) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for m in cfg.input_manifests:
        for sp in cfg.splits:
            rows += [r for r in select_rows(m, sp, cfg.include_sources, cfg.max_clips_per_split.get(sp), cfg.seed)
                     if r.transform == "clean"]
    ids = [r.clip_id for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("clip_id collision across input manifests")
    return rows


def _one(job: tuple) -> list[dict]:
    """All conditions for one parent clip (one process; the parent is decoded once)."""
    parent, conds, root_s, audio_subdir, overwrite = job
    root = Path(root_s)
    p = ManifestRow(**parent)
    x = load_audio(root / p.path)  # mono, 16 kHz (soxr HQ), no trimming
    out = []
    for c in conds:
        if c.role == "eval_heldout" and p.split == "train":
            continue  # protocol: held-out codecs never touch training data
        dest = root / audio_subdir / f"{c.id}-{c.params_hash()}" / p.source_dataset / f"{p.clip_id}.wav"
        rec = {"variant_clip_id": f"{p.clip_id}__{c.id}", "parent_clip_id": p.clip_id, "parent_sha256": p.sha256,
               "source_dataset": p.source_dataset, "split": p.split, "label": p.label, "language": p.language,
               "condition": c.id, "role": c.role, "kind": c.kind, "backend": c.backend, "encoder": c.encoder,
               "container": c.container, "codec_sample_rate": c.codec_sample_rate, "bitrate_requested_bps": c.bitrate,
               "encoder_options": dict(c.options), "params_hash": c.params_hash(), "path": dest.relative_to(root).as_posix()}
        try:
            y, info = apply(x, c)
        except CodecUnavailable as e:
            out.append({**rec, "status": "unavailable", "reason": str(e)})
            continue
        if overwrite or not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp.wav")
            sf.write(tmp, y, SR, subtype="PCM_16")
            tmp.replace(dest)
        stored = sf.read(str(dest), dtype="float32")[0]  # quality of what is actually on disk (after PCM-16)
        out.append({**rec, "status": "ok", **info, "duration_s": round(len(stored) / SR, 3), "sha256": sha256_file(dest),
                    **quality(x, stored)})
    return out


def variant_row(p: ManifestRow, r: dict) -> ManifestRow:
    br = f" {r['bitrate_requested_bps']} b/s" if r["bitrate_requested_bps"] else ""
    return p.model_copy(update={
        "clip_id": r["variant_clip_id"], "path": r["path"], "parent_clip_id": p.clip_id, "transform": f"sim:{r['condition']}",
        "codec": "pcm_s16le", "sample_rate_orig": SR, "duration_s": r["duration_s"], "sha256": r["sha256"],
        "notes": f"sim {r['condition']} ({r['encoder'] or 'resample'}{br} @ {r['codec_sample_rate']} Hz, role {r['role']}, "
                 f"params {r['params_hash']}); parent {p.clip_id}"})


def summarise(meta: pd.DataFrame) -> pd.DataFrame:
    ok = meta[meta.status == "ok"]
    g = ok.groupby("condition", sort=False).agg(
        n=("variant_clip_id", "size"), role=("role", "first"), kbps_requested=("bitrate_requested_bps", lambda s: s.iloc[0] / 1000 if pd.notna(s.iloc[0]) else None),
        kbps_actual_mean=("bitrate_actual_bps", lambda s: round(s.mean() / 1000, 1) if s.notna().any() else None),
        delay_ms_median=("delay_samples_16k", lambda s: round(s.median() / 16, 1)), snr_db_median=("snr_db", "median"),
        lsd_db_median=("lsd_db", "median"), bw_out_hz_median=("bandwidth_hz_out", "median"),
        bw_ref_hz_median=("bandwidth_hz_ref", "median"), rms_change_db_median=("rms_change_db", "median"),
        clipped_frac_max=("clipped_frac_out", "max"))
    na = meta[meta.status != "ok"].groupby("condition", sort=False).agg(role_=("role", "first"), n_unavailable=("variant_clip_id", "size"),
                                                                  reason=("reason", "first"))
    out = g.join(na, how="outer")
    out["role"] = out["role"].fillna(out["role_"]) if "role_" in out else out["role"]
    return out.drop(columns=["role_"], errors="ignore").reset_index()


def _md_table(df: pd.DataFrame) -> str:
    cell = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build simulated codec/channel variants of manifest clips.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    cfg, conds = load_cfg(args.config)
    root = DataPaths.resolve(args.data_root or cfg.data_root).data_root
    workers = args.workers or cfg.workers
    status = {c.id: available(c) for c in conds}
    for cid, (ok, why) in status.items():
        print(f"  {cid:14s} {'ok  ' if ok else 'SKIP'} {why}")
    if not cfg.skip_unavailable and not all(ok for ok, _ in status.values()):
        raise SystemExit("some conditions are unavailable and skip_unavailable is false")
    par = parents(cfg)
    print(f"{cfg.run_id}: {len(par)} parent clips x {len(conds)} conditions, data_root {root}, workers {workers}")
    jobs = [(p.model_dump(), conds, str(root), cfg.output.audio_subdir, args.overwrite) for p in par]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_one, jobs, chunksize=4))
    else:
        results = [_one(j) for j in jobs]
    meta = [r for rs in results for r in rs]
    by_id = {p.clip_id: p for p in par}
    rows = list(par) + [variant_row(by_id[r["parent_clip_id"]], r) for r in meta if r["status"] == "ok"]
    out_m = Path(cfg.output.manifest)
    out_m.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(rows, out_m)
    read_manifest(out_m)  # schema round-trip
    header = {"run_id": cfg.run_id, "config": cfg.model_dump(), "conditions": [c.model_dump() | {"params_hash": c.params_hash()} for c in conds],
              "tools": tool_versions(), "parent_audio": "decoded with ml.data.audio.load_audio: mono, soxr HQ to 16 kHz, no trimming"}
    with open(cfg.output.metadata, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_header": header}, default=str) + "\n")
        for r in meta:
            fh.write(json.dumps(r, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)) + "\n")
    summ = summarise(pd.DataFrame(meta))
    summ.to_csv(cfg.output.quality, index=False)
    md = Path(cfg.output.quality).with_suffix(".md")
    md.write_text(f"# Codec variant quality: {cfg.run_id}\n\n{len(par)} parents; tools {header['tools']}\n\n"
                  + _md_table(summ) + "\n\nSNR/LSD are measured against the 16 kHz parent after delay compensation. "
                  "Low SNR for codecs is expected (perceptual codecs do not preserve waveform); LSD and bandwidth are the more "
                  "meaningful indicators.\n", encoding="utf-8")
    n_ok = sum(r["status"] == "ok" for r in meta)
    print(f"variants ok {n_ok}, unavailable {len(meta) - n_ok}\nmanifest {out_m} ({len(rows)} rows)\nmetadata {cfg.output.metadata}\nquality {cfg.output.quality} (+ .md)")
    print(summ.drop(columns=[c for c in ("reason",) if c in summ]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
