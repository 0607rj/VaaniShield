"""200-clip Hindi-English (Hinglish) pilot: plan + manifest (dataset guide p6-7, adapted to available consented sources).

    python -m ml.data.build_hinglish_pilot plan     # deterministic selection -> data/manifests/hinglish_pilot_plan.csv
    python scripts/acquire_csfleurs.py --files data/manifests/hinglish_pilot_plan.csv --accept-terms
    python -m ml.data.build_hinglish_pilot build    # -> data/manifests/hinglish_pilot_v1.csv
    python -m ml.data.check_manifest --manifest data/manifests/hinglish_pilot_v1.csv

Composition (100 bona fide + 100 spoof, all hi-en):
  split  bona fide                                  spoof
  train  30 OpenSLR104 (speaker set A, 2/speaker)   30 CS-FLEURS XTTS voice hin-spk0
  dev    10 OpenSLR104 (speaker set B)              10 CS-FLEURS XTTS voice hin-spk1
  test   10 OpenSLR104 (speaker set C)              10 CS-FLEURS XTTS voice hin-spk2   (seen generator, unseen voice)
         50 CS-FLEURS read speaker AD (FLEURS text) 50 CS-FLEURS MMS concatenative     (HELD-OUT generator family)
Rules: speakers/voices are disjoint across splits; FLEURS sentence ids never cross splits; the MMS (concatenative)
generator family appears in test only. Transcripts are stored only as hashes (licensed text stays out of Git).
Team recordings (dataset guide) do not exist yet (consent pending), so the second real source is CS-FLEURS read speech.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from ml.data.build_canonical_manifest import _codec, cut_openslr_segment
from ml.data.manifest import ManifestRow, read_manifest, transcript_hash, write_manifest
from ml.data.provenance import DataPaths
from ml.training.repro import sha256_file

PLAN = Path("data/manifests/hinglish_pilot_plan.csv")
OUT = Path("data/manifests/hinglish_pilot_v1.csv")
PLAN_COLS = ["plan_id", "source", "subset", "file_name", "rec", "start", "end", "speaker", "label", "generator",
             "generator_family", "split", "sentence_id", "transcript_id"]
LICENCE = {"openslr104_hien": "CC-BY-SA-4.0", "cs_fleurs": "CC-BY-NC-4.0"}
XTTS_VOICE_SPLIT = {"hin-spk0": ("train", 30), "hin-spk1": ("dev", 10), "hin-spk2": ("test", 10)}
OPENSLR_SETS = (("train", 15), ("dev", 5), ("test", 5))  # speakers per split, 2 segments each
N_READ, N_MMS = 50, 50


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _fleurs_sid(row: dict) -> str | None:
    m = re.match(r"^(?:hin_)?(\d+)_", row["id"])
    return f"fleurs:{m.group(1)}" if m else None


def plan(paths: DataPaths, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    used_sentences: dict[str, str] = {}  # sentence_id -> split

    def take(sid: str | None, split: str) -> bool:
        if sid is None:
            return True
        if used_sentences.get(sid, split) != split:
            return False
        used_sentences[sid] = split
        return True

    # --- test: CS-FLEURS read (real, speaker AD) first, so its sentences are reserved for test
    read = [r for r in _jsonl(paths.raw("cs_fleurs") / "read/test/metadata.jsonl") if r["file_name"].startswith("audio/hin/")]
    for r in rng.sample(read, len(read)):
        if sum(o["subset"] == "read/test" for o in out) >= N_READ:
            break
        if take(_fleurs_sid(r), "test"):
            out.append({"source": "cs_fleurs", "subset": "read/test", "file_name": r["file_name"], "speaker": f"csfleurs:{r['speaker']}",
                        "label": "bonafide", "generator": "none", "generator_family": "none", "split": "test",
                        "sentence_id": _fleurs_sid(r), "transcript_id": transcript_hash(r["text"])})
    # --- XTTS voices, one split per voice, sentence-disjoint across splits
    xtts = _jsonl(paths.raw("cs_fleurs") / "xtts/train/metadata.jsonl")
    xtts = [r for r in xtts if "cs_hin_eng" in r["file_name"]]
    for voice, (split, n) in XTTS_VOICE_SPLIT.items():
        cand = [r for r in xtts if r["speaker"] == voice]
        got = 0
        for r in rng.sample(cand, len(cand)):
            if got >= n:
                break
            if take(_fleurs_sid(r), split):
                out.append({"source": "cs_fleurs", "subset": "xtts/train", "file_name": r["file_name"], "speaker": f"csfleurs:{voice}",
                            "label": "spoof", "generator": "xtts_csfleurs", "generator_family": "xtts", "split": split,
                            "sentence_id": _fleurs_sid(r), "transcript_id": transcript_hash(r["text"])})
                got += 1
        if got < n:
            raise SystemExit(f"only {got} sentence-disjoint XTTS clips for {voice}")
    # --- MMS concatenative: held-out generator family, test only
    mms = [r for r in _jsonl(paths.raw("cs_fleurs") / "mms/test/metadata.jsonl") if "cs_hin_eng" in r["file_name"]]
    for r in rng.sample(mms, N_MMS):
        out.append({"source": "cs_fleurs", "subset": "mms/test", "file_name": r["file_name"], "speaker": f"csfleurs:{r['speaker']}",
                    "label": "spoof", "generator": "mms_concat_csfleurs", "generator_family": "concatenative", "split": "test",
                    "sentence_id": None, "transcript_id": transcript_hash(r["text"])})
    # --- OpenSLR 104 real Hinglish lectures: speaker-disjoint sets, 2 segments per speaker (2-15 s)
    base = paths.raw("openslr104_hien") / "test" / "transcripts"
    utt2spk = dict(line.split() for line in (base / "utt2spk").read_text(encoding="utf-8").splitlines() if line.strip())
    texts = dict(line.split(maxsplit=1) for line in (base / "text").read_text(encoding="utf-8").splitlines() if " " in line)
    by_spk = defaultdict(list)
    for line in (base / "segments").read_text(encoding="utf-8").splitlines():
        utt, rec, s, e = line.split()
        if 2.0 <= float(e) - float(s) <= 15.0 and (paths.raw("openslr104_hien") / "test" / f"{rec}.wav").exists():
            by_spk[utt2spk[utt]].append((utt, rec, float(s), float(e)))
    speakers = sorted(s for s, segs in by_spk.items() if len(segs) >= 2)
    rng.shuffle(speakers)
    i = 0
    for split, n_spk in OPENSLR_SETS:
        for spk in speakers[i : i + n_spk]:
            for utt, rec, s, e in rng.sample(by_spk[spk], 2):
                out.append({"source": "openslr104_hien", "subset": "test", "file_name": utt, "rec": rec, "start": s, "end": e,
                            "speaker": f"openslr104:{spk}", "label": "bonafide", "generator": "none", "generator_family": "none",
                            "split": split, "sentence_id": None, "transcript_id": transcript_hash(texts.get(utt, utt))})
        i += n_spk
    if i > len(speakers):
        raise SystemExit(f"need {i} OpenSLR speakers, have {len(speakers)}")
    for k, row in enumerate(sorted(out, key=lambda r: (r["split"], r["source"], r["file_name"]))):
        row["plan_id"] = f"hp{k:03d}"
    return sorted(out, key=lambda r: r["plan_id"])


def build(paths: DataPaths, plan_rows: list[dict]) -> list[ManifestRow]:
    rows = []
    for p in plan_rows:
        if p["source"] == "openslr104_hien":
            f = cut_openslr_segment(paths, p["rec"], p["file_name"], float(p["start"]), float(p["end"]))
            parent, notes = f"os_rec_{p['rec']}", f"segment {float(p['start']):.2f}-{float(p['end']):.2f}s of recording {p['rec']}"
        else:
            f = paths.raw("cs_fleurs") / p["subset"] / re.sub(r"/+", "/", p["file_name"])
            parent, notes = None, f"cs_fleurs {p['subset']}" + (f"; {p['sentence_id']}" if p["sentence_id"] else "")
        if not f.exists():
            raise SystemExit(f"missing audio {f}; run scripts/acquire_csfleurs.py --files {PLAN} --accept-terms")
        codec, sr, dur = _codec(f)
        rows.append(ManifestRow(
            clip_id=p["plan_id"], path=paths.rel(f), label=p["label"], source_dataset=p["source"], subset=p["subset"],
            speaker_id=p["speaker"], speaker_group=p["speaker"], language="hi-en", transcript_id=p["transcript_id"],
            generator=p["generator"], generator_family=p["generator_family"], parent_clip_id=parent, codec=codec,
            sample_rate_orig=sr, duration_s=dur, sha256=sha256_file(f), licence=LICENCE[p["source"]], split=p["split"],
            notes=notes + ("; HELD-OUT generator family (test only)" if p["generator_family"] == "concatenative" else "")))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the 200-clip Hinglish pilot plan / manifest.")
    ap.add_argument("step", choices=["plan", "build"])
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)
    paths = DataPaths.resolve()
    if args.step == "plan":
        rows = plan(paths, args.seed)
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        with open(PLAN, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=PLAN_COLS, lineterminator="\n")
            w.writeheader()
            w.writerows({c: ("" if r.get(c) is None else r.get(c)) for c in PLAN_COLS} for r in rows)
        print(f"wrote {PLAN}: {len(rows)} planned clips")
    else:
        with open(PLAN, encoding="utf-8") as fh:
            plan_rows = [{k: (v if v != "" else None) for k, v in r.items()} for r in csv.DictReader(fh)]
        rows = build(paths, plan_rows)
        write_manifest(rows, OUT)
        read_manifest(OUT)
        print(f"wrote {OUT}: {len(rows)} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
