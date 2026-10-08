"""Repaired Hindi dataset v2 (research/hindi_repair_report.md): paired, balanced, identically preprocessed.

    python -m ml.data.build_hindi_v2 --config ml/configs/hindi_v2_pilot.yaml
    python -m ml.data.hindi_gate     --config ml/configs/hindi_gate.yaml --manifest data/manifests/hindi_v2_pilot.csv

Inputs (never modified): the IndicSynth Hindi clips of the v1 manifest and their metadata; the Kathbath *train*
recordings that IndicSynth used as references (fetched with `scripts/acquire_kathbath.py --fetch-ids
data/manifests/hindi_v2_reference_ids.txt --tars clean/train_audio.tar --allow-train --extra-per-speaker 10`);
Kathbath Hindi transcripts (`--transcripts`).

Steps:
  1. pair      every fake with its own reference recordings: target voice (XTTS, FreeVC) and source speech (FreeVC)
  2. real pool references first, then other recordings of the same speakers; per speaker group, real <= spoof count
  3. groups    speaker graph: target speaker <-> VC source speaker -> one group, one split
  4. splits    groups containing the held-out generator family -> test only; other groups split train/dev/test by
               clip share; dev never sees the held-out family (docs/data_flow.md rules 3-4)
  5. text      rows of train/dev whose transcript also occurs in test are dropped (data_flow.md rule 4)
  6. audio     identical v2 / v2.1 chain for both classes (ml/data/audio.py PREPROC_V2 / PREPROC_V2_1); optional matched AAC history for
               spoof (Kathbath's capture codec) so the last lossy stage is the same for both classes
  7. outputs   <data_root>/<audio_dir>/<label>/<generator>/<clip_id>.wav (PCM_16, 16 kHz), manifest, metadata JSONL
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from pydantic import BaseModel, ConfigDict

from ml.data.audio import PREPROC_V2, PREPROC_V2_1, SAMPLE_RATE, active_rms_dbfs, load_audio, preprocess_wave, remove_pauses
from ml.data.codec import tool_versions
from ml.data.manifest import ManifestRow, read_manifest, transcript_hash, write_manifest
from ml.data.provenance import DataPaths
from ml.evaluation.shortcut_probe import aac_roundtrip, native, probe
from ml.training.repro import sha256_file

FAMILY = {"xtts_v2": "xtts", "freevc24": "freevc"}
LICENCE = {"kathbath_hi": "CONFLICT:CC0-1.0|CC-BY-4.0", "indicsynth_hi": "CC-BY-NC-4.0"}
SPLITS = ("train", "dev", "test")


class CodecMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool
    encoder: str = "aac"
    sample_rate: int = 44100
    bitrate_kbps: float = 128.0


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audio_dir: str
    manifest: str
    metadata: str


class PauseDiagnostic(BaseModel):
    """Pause-removed copy of every row (transform diag:pauses_removed), for the pause-controlled experiment (D-H4)."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    audio_dir: str = ""
    manifest: str = ""


PREPROC = {"v2": (PREPROC_V2, "PCM_16"), "v2.1": (PREPROC_V2_1, "FLOAT")}


class HindiV2Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    build_id: str
    v1_manifest: str
    indicsynth_metadata_glob: str          # relative to data_root
    kathbath_train_audio_dir: str          # relative to data_root
    kathbath_transcripts_dir: str          # relative to data_root
    reference_ids: str
    codec_matching: CodecMatch
    holdout_family: str | None             # generator family held out for the unseen-family test fold
    split_shares: dict[str, float]
    min_duration_s: float = 1.0
    seed: int = 2026
    preprocessing: str = "v2"              # v2 (0.95 peak limit, PCM_16) or v2.1 (no limiter, float32 WAV)
    pause_diagnostic: PauseDiagnostic = PauseDiagnostic()
    output: Output


def preproc_hash(version: str = "v2") -> str:
    return hashlib.sha256(json.dumps({version: PREPROC[version][0]}, sort_keys=True).encode()).hexdigest()[:12]


def transcripts(root: Path, d: str) -> dict[str, tuple[str, str]]:
    out = {}
    for f in sorted((root / d).glob("*/transcription_n2w.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                name, text = line.split("\t", 1)
                out[name.rsplit(".", 1)[0]] = (f.parent.name, text)
    return out


def assign_splits(groups: dict[str, dict], holdout_family: str | None, shares: dict[str, float], seed: int) -> dict[str, str]:
    split_of = {g: "test" for g, v in groups.items() if holdout_family and holdout_family in v["families"]}
    rest = [g for g in groups if g not in split_of]
    rng = random.Random(seed)
    rng.shuffle(rest)
    rest.sort(key=lambda g: -groups[g]["n"])  # largest first; shuffled order breaks ties deterministically
    total = sum(groups[g]["n"] for g in rest)
    have = dict.fromkeys(SPLITS, 0)
    for g in rest:
        sp = max(SPLITS, key=lambda s: shares[s] * total - have[s])
        split_of[g] = sp
        have[sp] += groups[g]["n"]
    return split_of


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the repaired Hindi dataset v2.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root")
    args = ap.parse_args(argv)
    cfg = HindiV2Cfg(**yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    root = DataPaths.resolve(args.data_root).data_root
    rng_note = []

    # ---- 1. fakes (v1 IndicSynth clips) + their metadata
    v1 = [r for r in read_manifest(Path(cfg.v1_manifest)) if r.source_dataset == "indicsynth_hi"]
    meta = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(root / cfg.indicsynth_metadata_glob)))]).drop_duplicates("file")
    meta["base"] = meta.file.str.split("/").str[-1]
    meta = meta.set_index("base")
    tx = transcripts(root, cfg.kathbath_transcripts_dir)
    ref_ids = {ln.strip() for ln in open(cfg.reference_ids, encoding="utf-8") if ln.strip() and not ln.startswith("#")}
    real_dir = root / cfg.kathbath_train_audio_dir
    on_disk = {p.stem: p for p in sorted(real_dir.glob("*.m4a"))}

    fakes = []
    for r in v1:
        m = meta.loc[r.path.rsplit("/", 1)[-1]]
        gen = m["Generative Model"]
        tgt_ref = str(m["Target Reference Audio"]).rsplit(".", 1)[0]
        src_ref = str(m["Source Reference Audio"]).rsplit(".", 1)[0] if pd.notna(m["Source Reference Audio"]) else None
        tgt, src = f"kathbath:{int(m['Target Speaker ID'])}", (f"kathbath:{int(m['Source Speaker_ID'])}" if pd.notna(m["Source Speaker_ID"]) else None)
        if gen == "xtts_v2" and isinstance(m["TTS Transcript"], str) and m["TTS Transcript"].strip():
            tid = transcript_hash(m["TTS Transcript"])
        elif src_ref and src_ref in tx:
            tid = transcript_hash(tx[src_ref][1])  # FreeVC keeps the source recording's words
        else:
            tid = None
        fakes.append({"raw": r, "gen": gen, "family": FAMILY[gen], "target": tgt, "source": src, "target_ref": tgt_ref,
                      "source_ref": src_ref, "transcript_id": tid, "gender": str(m["Gender"])[:1].lower()})

    # ---- 2. real pool: references first, then same-speaker recordings
    def spk(stem: str) -> str:
        return f"kathbath:{stem.split('-')[1]}"

    needed_refs = {f["target_ref"] for f in fakes} | {f["source_ref"] for f in fakes if f["source_ref"]}
    real_cands = sorted(on_disk)
    missing_refs = sorted(needed_refs - set(on_disk))

    # ---- 3. speaker graph over fake speakers + real candidates
    links = [(f["target"], f["source"]) for f in fakes if f["source"]]
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for f in fakes:
        find(f["target"])
    for a, b in links:
        parent[find(a)] = find(b)
    group_of = lambda s: "hi_group:" + find(s)  # noqa: E731
    by_group_fake = defaultdict(list)
    for f in fakes:
        by_group_fake[group_of(f["target"])].append(f)
    fake_speakers = {f["target"] for f in fakes} | {f["source"] for f in fakes if f["source"]}
    real_by_group = defaultdict(list)
    for stem in real_cands:
        s = spk(stem)
        if s in fake_speakers:  # only speakers that occur on the spoof side (pairing)
            real_by_group[group_of(s)].append(stem)
    real_keep = {}
    for g, fs in by_group_fake.items():
        cands = sorted(real_by_group.get(g, []), key=lambda s: (s not in needed_refs, s))  # references first
        for stem in cands[: len(fs)]:
            real_keep[stem] = g

    # ---- 4. splits by group
    groups = {g: {"n": len(fs) + sum(1 for v in real_keep.values() if v == g), "families": {f["family"] for f in fs}}
              for g, fs in by_group_fake.items()}
    split_of = assign_splits(groups, cfg.holdout_family, cfg.split_shares, cfg.seed)

    # ---- 5. transcript disjointness (test text never in train/dev)
    real_tid = {s: (transcript_hash(tx[s][1]) if s in tx else None) for s in real_keep}
    test_tids = {f["transcript_id"] for f in fakes if split_of[group_of(f["target"])] == "test" and f["transcript_id"]}
    test_tids |= {real_tid[s] for s, g in real_keep.items() if split_of[g] == "test" and real_tid[s]}
    dropped_text = []
    fakes_kept = []
    for f in fakes:
        sp = split_of[group_of(f["target"])]
        if sp != "test" and f["transcript_id"] in test_tids:
            dropped_text.append(f["raw"].clip_id)
            continue
        fakes_kept.append(f)
    for s in list(real_keep):
        if split_of[real_keep[s]] != "test" and real_tid[s] in test_tids:
            dropped_text.append(s)
            del real_keep[s]

    # ---- 6. audio + rows
    out_dir = root / cfg.output.audio_dir
    params, subtype = PREPROC[cfg.preprocessing]
    ph = preproc_hash(cfg.preprocessing)
    rows, recs, dropped_short = [], [], []
    real_clip = {s: f"hv2_r_{s}" for s in real_keep}

    def emit(x16, label, gen, clip_id, base_rec, row_kw):
        y, info = preprocess_wave(x16, params)
        if y.size < cfg.min_duration_s * SAMPLE_RATE:
            dropped_short.append(clip_id)
            return
        dest = out_dir / label / gen / f"{clip_id}.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dest, y, SAMPLE_RATE, subtype=subtype)
        stored = sf.read(str(dest), dtype="float32")[0]
        rel = dest.relative_to(root).as_posix()
        sha = sha256_file(dest)
        rows.append(ManifestRow(clip_id=clip_id, path=rel, label=label, codec="pcm_s16le", sample_rate_orig=SAMPLE_RATE,
                                duration_s=round(stored.size / SAMPLE_RATE, 3), sha256=sha, language="hi", consent_status="public",
                                transform="clean", **row_kw))
        recs.append({**base_rec, "clip_id": clip_id, "label": label, "path": rel, "sha256": sha, "split": row_kw["split"],
                     "speaker_group": row_kw["speaker_group"], "preproc": params["id"], "preproc_hash": ph, **info,
                     "pre_gain_peak": round(float(np.abs(y).max() / 10 ** (info["gain_db"] / 20)), 5),
                     "would_clip_pcm16": bool(np.abs(y).max() >= 1.0),
                     "post_active_rms_dbfs": round(active_rms_dbfs(stored), 3), "post_peak": round(float(np.abs(stored).max()), 5),
                     "post_dc": float(stored.mean()), "post_duration_s": round(stored.size / SAMPLE_RATE, 3)})

    for s in sorted(real_keep):
        g = real_keep[s]
        p = on_disk[s]
        po = probe(p)
        emit(load_audio(p), "bonafide", "kathbath", real_clip[s],
             {"raw_path": p.relative_to(root).as_posix(), "raw_sha256": sha256_file(p), "orig_container": po["container"],
              "orig_codec": po["codec"], "orig_sample_rate": po["sr_native"], "orig_bitrate_kbps": po["bitrate_kbps"],
              "kathbath_split": tx.get(s, ("?",))[0], "role": "reference" if s in needed_refs else "same_speaker",
              "chain": ["upstream capture: AAC-LC (Kathbath release)", "decode", "soxr HQ -> 16 kHz", "v2"]},
             dict(source_dataset="kathbath_hi", subset="train_reference" if s in needed_refs else "train_same_speaker",
                  speaker_id=spk(s), speaker_group=g, gender=s.rsplit("-", 1)[-1][:1], transcript_id=real_tid[s],
                  generator="none", generator_family="none", licence=LICENCE["kathbath_hi"], split=split_of[g],
                  notes=f"preproc={params['id']}:{ph}; raw={p.name}"))
    cm = cfg.codec_matching
    for f in fakes_kept:
        r = f["raw"]
        g = group_of(f["target"])
        p = root / r.path
        po = probe(p)
        if cm.enabled:
            xn, sr = native(p)
            x16 = aac_roundtrip(xn, sr, cm.bitrate_kbps)
            chain = ["generator output (PCM 24 kHz)", f"matched codec: {cm.encoder} {cm.bitrate_kbps:g} kb/s @ {cm.sample_rate} Hz",
                     "decode", "soxr HQ -> 16 kHz", "v2"]
        else:
            x16 = load_audio(p)
            chain = ["generator output (PCM 24 kHz)", "soxr HQ -> 16 kHz", "v2"]
        parent_id = real_clip.get(f["target_ref"]) if f["target_ref"] in real_keep else None
        notes = f"preproc={params['id']}:{ph}; raw={p.name}; target_ref={f['target_ref']}" + (f"; vc_source={f['source']}; source_ref={f['source_ref']}" if f["source"] else "")
        emit(x16, "spoof", f["gen"], "hv2_s_" + r.clip_id.removeprefix("is_"),
             {"raw_path": r.path, "raw_sha256": r.sha256, "orig_container": po["container"], "orig_codec": po["codec"],
              "orig_sample_rate": po["sr_native"], "orig_bitrate_kbps": po["bitrate_kbps"], "target_ref": f["target_ref"],
              "source_ref": f["source_ref"], "target_ref_in_manifest": parent_id is not None,
              "source_ref_in_manifest": bool(f["source_ref"] and f["source_ref"] in real_keep), "codec_matching": cm.model_dump(),
              "chain": chain},
             dict(source_dataset="indicsynth_hi", subset="Hindi", speaker_id=f["target"], speaker_group=g, gender=f["gender"],
                  transcript_id=f["transcript_id"], generator=f["gen"], generator_family=f["family"], parent_clip_id=parent_id,
                  licence=LICENCE["indicsynth_hi"], split=split_of[g], notes=notes))

    write_manifest(rows, Path(cfg.output.manifest))
    read_manifest(Path(cfg.output.manifest))
    header = {"build_id": cfg.build_id, "config": cfg.model_dump(), "preproc_params": params, "preproc_hash": ph, "stored_subtype": subtype,
              "tools": tool_versions(), "n_fakes_in": len(fakes), "n_fakes_out": sum(r.label == "spoof" for r in rows),
              "n_real_out": sum(r.label == "bonafide" for r in rows), "references_needed": len(needed_refs),
              "references_missing_on_disk": missing_refs, "dropped_for_test_transcript_overlap": dropped_text,
              "dropped_too_short": dropped_short, "groups": {g: {"split": split_of[g], **{k: (sorted(v) if isinstance(v, set) else v) for k, v in d.items()}}
                                                                for g, d in groups.items()}, "notes": rng_note}
    with open(cfg.output.metadata, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_header": header}, ensure_ascii=False, default=str) + "\n")
        for rec in recs:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    if cfg.pause_diagnostic.enabled:  # pause-removed copies (diagnostic only; never the primary data)
        pd_rows = []
        for r in rows:
            x = sf.read(str(root / r.path), dtype="float32")[0]
            y = remove_pauses(x)
            dest = root / cfg.pause_diagnostic.audio_dir / r.label / f"{r.clip_id}__nopause.wav"
            dest.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dest, y, SAMPLE_RATE, subtype=subtype)
            pd_rows.append(r.model_copy(update={"clip_id": f"{r.clip_id}__nopause", "path": dest.relative_to(root).as_posix(),
                                                "parent_clip_id": r.clip_id, "transform": "diag:pauses_removed",
                                                "duration_s": round(y.size / SAMPLE_RATE, 3), "sha256": sha256_file(dest),
                                                "notes": (r.notes or "") + "; diag:pauses_removed (ml.data.audio.remove_pauses)"}))
        write_manifest(pd_rows, Path(cfg.pause_diagnostic.manifest))
        read_manifest(Path(cfg.pause_diagnostic.manifest))
        print(f"  pause diagnostic: {len(pd_rows)} pause-removed copies -> {cfg.pause_diagnostic.manifest}")
    c = Counter((r.split, r.label) for r in rows)
    print(f"{cfg.build_id}: {len(rows)} rows -> {cfg.output.manifest}")
    for sp in SPLITS:
        fam = dict(Counter(r.generator_family for r in rows if r.split == sp and r.label == "spoof"))
        print(f"  {sp:5s} real {c[(sp, 'bonafide')]:4d}  spoof {c[(sp, 'spoof')]:4d}  groups "
              f"{len({r.speaker_group for r in rows if r.split == sp}):3d}  spoof families {fam}")
    print(f"  references needed {len(needed_refs)}, missing on disk {len(missing_refs)}; dropped: text overlap {len(dropped_text)}, "
          f"too short {len(dropped_short)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
