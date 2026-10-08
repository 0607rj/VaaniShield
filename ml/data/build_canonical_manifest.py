"""Build the canonical VaaniShield manifest (v1) from every source subset on disk.

Usage:
    python -m ml.data.build_canonical_manifest --out data/manifests/vaanishield_v1.csv

Sources and split rules (PROJECT_STATE §0.11):
  asvspoof2019_la  official partitions kept: train -> train, dev -> dev, eval -> test (speaker-disjoint by design;
                   eval attacks A07-A19 are unseen in train by design)
  kathbath_hi      Hindi bona fide  } split together by LINKED speaker group: an IndicSynth clip links its target
  indicsynth_hi    Hindi spoof      } speaker (cloned voice) and its VC source speaker; both are Kathbath speaker IDs
  openslr104_hien  Hinglish bona fide, TEST only (no Hinglish spoof exists yet; single-class source must not be trained on)

Every row records codec (as stored), parent_clip_id (for derived segments) and transform. Audio is not modified here,
except OpenSLR utterance segments, which are cut losslessly (PCM_16) from the original recordings into
data/processed/openslr104_hien/segments/ with parent_clip_id = recording id.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf

from ml.data.build_manifest import check_split, split_by_speaker
from ml.data.manifest import ManifestRow, read_manifest, transcript_hash, write_manifest
from ml.data.provenance import DataPaths
from ml.training.repro import sha256_file

LICENCES = {"asvspoof2019_la": "ODC-By-1.0", "indicsynth_hi": "CC-BY-NC-4.0", "kathbath_hi": "CONFLICT:CC0-1.0|CC-BY-4.0",
            "openslr104_hien": "CC-BY-SA-4.0"}
ASV_SPLIT = {"train": "train", "dev": "dev", "eval": "test"}
ASV_PROTO = {"train": "train.trn.txt", "dev": "dev.trl.txt", "eval": "eval.trl.txt"}
SUBTYPE_CODEC = {"PCM_16": "pcm_s16le", "PCM_24": "pcm_s24le", "FLOAT": "pcm_f32le"}


def _codec(path: Path) -> tuple[str, int, float]:
    """(codec name, sample rate, duration) of the stored file, without decoding the whole file."""
    if path.suffix.lower() == ".m4a":
        import av

        with av.open(str(path)) as c:
            s = c.streams.audio[0]
            dur = float(s.duration * s.time_base) if s.duration else float(c.duration / 1e6)
            return s.codec_context.name, int(s.rate), round(dur, 3)
    info = sf.info(str(path))
    codec = "flac" if info.format == "FLAC" else SUBTYPE_CODEC.get(info.subtype, info.subtype.lower())
    return codec, int(info.samplerate), round(info.duration, 3)


# ---------------------------------------------------------------------------------------------- per source
def asvspoof(paths: DataPaths) -> list[ManifestRow]:
    root = paths.raw("asvspoof2019_la") / "LA"
    rows = []
    for part, suffix in ASV_PROTO.items():
        flac_dir = root / f"ASVspoof2019_LA_{part}" / "flac"
        on_disk = {p.stem: p for p in flac_dir.glob("*.flac")}
        proto = root / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{suffix}"
        for line in proto.read_text(encoding="utf-8").splitlines():
            spk, utt, _, system, key = line.split()
            if utt not in on_disk:
                continue
            p = on_disk[utt]
            codec, sr, dur = _codec(p)
            spoof = key == "spoof"
            rows.append(ManifestRow(
                clip_id=utt, path=paths.rel(p), label=key, source_dataset="asvspoof2019_la", subset=part,
                speaker_id=f"asvspoof19:{spk}", speaker_group=f"asvspoof19:{spk}", language="en",
                generator=system if spoof else "none", generator_family=f"asvspoof19_{system}" if spoof else "none",
                codec=codec, sample_rate_orig=sr, duration_s=dur, sha256=sha256_file(p), licence=LICENCES["asvspoof2019_la"],
                split=ASV_SPLIT[part], notes="official partition"))
    return rows


def kathbath(paths: DataPaths) -> list[ManifestRow]:
    rows = []
    for p in sorted(paths.raw("kathbath_hi").rglob("*.m4a")):
        utt, spk, gender = p.stem.split("-")
        codec, sr, dur = _codec(p)
        rows.append(ManifestRow(
            clip_id=f"kb_{utt}", path=paths.rel(p), label="bonafide", source_dataset="kathbath_hi",
            subset=p.parent.parent.name, speaker_id=f"kathbath:{spk}", gender=gender if gender in ("f", "m") else "u",
            language="hi", generator="none", generator_family="none", codec=codec, sample_rate_orig=sr, duration_s=dur,
            sha256=sha256_file(p), licence=LICENCES["kathbath_hi"], consent_status="public"))
    return rows


def indicsynth(paths: DataPaths) -> tuple[list[ManifestRow], list[tuple[str, str]]]:
    """Rows + speaker links (target <-> VC source). Metadata from the pilot and subset CSVs written by the downloader."""
    meta: dict[str, dict] = {}
    for f in sorted((paths.raw("indicsynth_hi") / "Hindi").glob("*_metadata.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            meta[r["file"]] = r
    rows, links = [], []
    for rel, r in sorted(meta.items()):
        p = paths.data_root / rel
        if not p.exists():
            continue
        gen = r["Generative Model"]
        tgt = f"kathbath:{int(float(r['Target Speaker ID']))}"
        src = f"kathbath:{int(float(r['Source Speaker_ID']))}" if r.get("Source Speaker_ID") not in (None, "", "nan") else None
        if src:
            links.append((tgt, src))
        codec, sr, dur = _codec(p)
        g = (r.get("Gender") or "u")[0].lower()
        rows.append(ManifestRow(
            clip_id=f"is_{p.stem}", path=paths.rel(p), label="spoof", source_dataset="indicsynth_hi", subset="Hindi",
            speaker_id=tgt, gender=g if g in ("f", "m") else "u", language="hi",
            transcript_id=transcript_hash(r["TTS Transcript"]) if r.get("TTS Transcript") else None,
            generator=gen, generator_family={"xtts_v2": "xtts", "freevc24": "freevc", "vits": "vits"}.get(gen, gen),
            codec=codec, sample_rate_orig=sr, duration_s=dur, sha256=sha256_file(p), licence=LICENCES["indicsynth_hi"],
            notes=f"vc_source={src}" if src else None))
    return rows, links


def cut_openslr_segment(paths: DataPaths, rec: str, utt: str, start: float, end: float) -> Path:
    """Lossless PCM_16 cut of one Kaldi segment into data/processed/openslr104_hien/segments/<utt>.wav (idempotent)."""
    wav = paths.raw("openslr104_hien") / "test" / f"{rec}.wav"
    seg_path = paths.data_root / "processed" / "openslr104_hien" / "segments" / f"{utt}.wav"
    if not seg_path.exists():
        seg_path.parent.mkdir(parents=True, exist_ok=True)
        sr = sf.info(str(wav)).samplerate
        audio, sr = sf.read(str(wav), start=int(start * sr), stop=int(end * sr), dtype="int16")
        sf.write(seg_path, audio, sr, subtype="PCM_16")
    return seg_path


def openslr_segments(paths: DataPaths, per_recording: int, seed: int) -> list[ManifestRow]:
    """Cut up to `per_recording` utterances (Kaldi `segments` + `text`) from each test recording; TEST split only."""
    base = paths.raw("openslr104_hien") / "test"
    seg_file, text_file = base / "transcripts" / "segments", base / "transcripts" / "text"
    if not seg_file.exists():
        return []
    texts = dict(line.split(maxsplit=1) for line in text_file.read_text(encoding="utf-8").splitlines() if " " in line)
    utt2spk = dict(line.split() for line in (base / "transcripts" / "utt2spk").read_text(encoding="utf-8").splitlines() if line.strip())
    by_rec = defaultdict(list)
    for line in seg_file.read_text(encoding="utf-8").splitlines():
        utt, rec, start, end = line.split()
        by_rec[rec].append((utt, float(start), float(end)))
    out_dir = paths.data_root / "processed" / "openslr104_hien" / "segments"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, rng = [], random.Random(seed)
    for rec in sorted(by_rec):
        wav = base / f"{rec}.wav"
        if not wav.exists():
            continue
        segs = sorted(by_rec[rec], key=lambda s: s[1])
        chosen = sorted(rng.sample(segs, min(per_recording, len(segs))), key=lambda s: s[1])
        info = sf.info(str(wav))
        for utt, start, end in chosen:
            if end - start < 1.0:
                continue
            seg_path = cut_openslr_segment(paths, rec, utt, start, end)
            codec, sr, dur = _codec(seg_path)
            rows.append(ManifestRow(
                clip_id=f"os_{utt}", path=paths.rel(seg_path), label="bonafide", source_dataset="openslr104_hien", subset="test",
                speaker_id=f"openslr104:{utt2spk[utt]}", speaker_group=f"openslr104:{utt2spk[utt]}", language="hi-en",
                transcript_id=transcript_hash(texts[utt]) if utt in texts else None, generator="none", generator_family="none",
                parent_clip_id=f"os_rec_{rec}", codec=codec, sample_rate_orig=sr, duration_s=dur, sha256=sha256_file(seg_path),
                licence=LICENCES["openslr104_hien"], split="test",
                notes=f"segment {start:.2f}-{end:.2f}s of recording {rec}; speaker from transcripts/utt2spk; eval-only"))
    return rows


# ---------------------------------------------------------------------------------------------- speaker graph
def link_groups(rows: list[ManifestRow], links: list[tuple[str, str]]) -> list[ManifestRow]:
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in rows:
        find(r.speaker_id)
    for a, b in links:
        parent[find(a)] = find(b)
    return [r.model_copy(update={"speaker_group": "hi_group:" + find(r.speaker_id)}) for r in rows]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the canonical VaaniShield manifest.")
    ap.add_argument("--out", default="data/manifests/vaanishield_v1.csv")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--openslr-per-recording", type=int, default=10)
    args = ap.parse_args(argv)
    paths = DataPaths.resolve()

    asv = asvspoof(paths)
    is_rows, links = indicsynth(paths)
    hindi = link_groups(kathbath(paths) + is_rows, links)
    hindi, used_seed = split_by_speaker(hindi, seed=args.seed) if hindi else ([], None)
    osl = openslr_segments(paths, args.openslr_per_recording, args.seed)
    rows = asv + hindi + osl
    check_split(rows)
    out = Path(args.out)
    write_manifest(rows, out)
    read_manifest(out)
    print(f"wrote {out}: {len(rows)} clips (Hindi split seed {used_seed})")
    for src in sorted({r.source_dataset for r in rows}):
        sub = [r for r in rows if r.source_dataset == src]
        print(f"  {src:16s} {len(sub):5d}  " + "  ".join(f"{sp}={dict(Counter(r.label for r in sub if r.split == sp))}" for sp in ("train", "dev", "test")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
