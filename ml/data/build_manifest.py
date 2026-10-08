"""Build the pilot manifest from ASVspoof 2019 LA clips on disk and assign a speaker-disjoint train/dev/test split.

Usage:
    python -m ml.data.build_manifest --out data/manifests/asvspoof_pilot.csv

Only clips that exist under data/raw/asvspoof2019_la/ are included (the pilot: 20 bona fide + 20 spoof).
Split rule: whole speakers go to one split (no speaker in two splits); each split must contain both classes.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import soundfile as sf

from ml.data.manifest import ManifestRow, read_manifest, write_manifest
from ml.data.provenance import DataPaths
from ml.training.repro import sha256_file

SOURCE = "asvspoof2019_la"
LICENCE = "ODC-By-1.0"
SPLITS = ("train", "dev", "test")


def asvspoof_rows(paths: DataPaths) -> list[ManifestRow]:
    root = paths.raw(SOURCE) / "LA"
    proto = root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
    flac_dir = root / "ASVspoof2019_LA_train" / "flac"
    on_disk = {p.stem: p for p in flac_dir.glob("*.flac")}
    if not on_disk:
        raise SystemExit(f"no FLAC files in {flac_dir}; run scripts/acquire_asvspoof.py --pilot 20 --accept-terms first")
    rows = []
    for line in proto.read_text(encoding="utf-8").splitlines():
        spk, utt, _, system, key = line.split()
        if utt not in on_disk:
            continue
        path = on_disk[utt]
        info = sf.info(str(path))
        spoof = key == "spoof"
        rows.append(
            ManifestRow(
                clip_id=utt,
                path=paths.rel(path),
                label=key,
                source_dataset=SOURCE,
                subset="train",  # ASVspoof partition the clip came from
                speaker_id=f"asvspoof19:{spk}",
                speaker_group=f"asvspoof19:{spk}",
                language="en",
                generator=system if spoof else "none",
                generator_family=f"asvspoof19_{system}" if spoof else "none",
                sample_rate_orig=info.samplerate,
                duration_s=round(info.duration, 3),
                sha256=sha256_file(path),
                licence=LICENCE,
                consent_status="public",
                notes="pilot",
            )
        )
    return rows


def split_by_speaker(rows: list[ManifestRow], fractions=(0.6, 0.2, 0.2), seed: int = 2026, tries: int = 1000) -> tuple[list[ManifestRow], int]:
    """Greedy speaker-group assignment towards the target clip fractions; retries seeds until every split has both classes."""
    groups: dict[str, list[ManifestRow]] = {}
    for r in rows:
        groups.setdefault(r.speaker_group, []).append(r)
    names = sorted(groups)
    total = len(rows)
    for s in range(seed, seed + tries):
        order = names[:]
        random.Random(s).shuffle(order)
        counts = dict.fromkeys(SPLITS, 0)
        assign = {}
        for g in order:
            deficit = {sp: fractions[i] * total - counts[sp] for i, sp in enumerate(SPLITS)}
            sp = max(SPLITS, key=lambda k: deficit[k])
            assign[g] = sp
            counts[sp] += len(groups[g])
        labels = {sp: {r.label for g, sp2 in assign.items() if sp2 == sp for r in groups[g]} for sp in SPLITS}
        if all(labels[sp] == {"bonafide", "spoof"} for sp in SPLITS):
            out = [r.model_copy(update={"split": assign[r.speaker_group]}) for r in rows]
            return out, s
    raise SystemExit("could not find a speaker-disjoint split with both classes in every split")


def check_split(rows: list[ManifestRow]) -> None:
    by_split: dict[str, set[str]] = {sp: set() for sp in SPLITS}
    for r in rows:
        if r.split not in SPLITS:
            raise ValueError(f"{r.clip_id}: missing split")
        by_split[r.split].add(r.speaker_group)
    for a in SPLITS:
        for b in SPLITS:
            if a < b and by_split[a] & by_split[b]:
                raise ValueError(f"speaker leakage between {a} and {b}: {sorted(by_split[a] & by_split[b])}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the ASVspoof pilot manifest with a speaker-disjoint split.")
    ap.add_argument("--out", default="data/manifests/asvspoof_pilot.csv")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args(argv)

    paths = DataPaths.resolve(args.data_root)
    rows, used_seed = split_by_speaker(asvspoof_rows(paths), seed=args.seed)
    check_split(rows)
    out = Path(args.out)
    write_manifest(rows, out)
    read_manifest(out)  # re-validate what was written
    print(f"wrote {out} ({len(rows)} clips, split seed {used_seed})")
    for sp in SPLITS:
        sub = [r for r in rows if r.split == sp]
        print(f"  {sp:5s}: {len(sub):2d} clips {dict(Counter(r.label for r in sub))}, {len({r.speaker_group for r in sub})} speakers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
