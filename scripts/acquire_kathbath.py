#!/usr/bin/env python
"""Acquire Kathbath / IndicSUPERB Hindi bona fide speech (source_id: kathbath_hi).

Two official routes (both recorded in LICENSES_REGISTER.md; their licence statements CONFLICT):
  A. AI4Bharat objectstore tars (ungated): all 12 languages per tar (train 85 GB); m4a audio; Hindi not separable
     without streaming the whole tar.
  B. Hugging Face mirror ai4bharat/Kathbath (gated, auto-approval): per-language parquet shards.

Modes:
  --subset N         route A: N Hindi files from an objectstore tar (default clean/testkn_audio.tar), found by probing
                     tar headers at byte offsets; files are read individually by byte range (no full download)
  --fetch-ids FILE   route A: fetch exactly the Hindi recordings whose file stems are listed in FILE (one per line,
                     e.g. 844424930794018-1087-f) by walking the Hindi section of --tars header by header (512-byte
                     reads; no full download). Writes the full Hindi member listing of each walked tar to
                     raw/kathbath_hi/_listings/<tar>.csv. --extra-per-speaker K also fetches K further recordings
                     (first K by name) of every speaker that has a wanted recording. The 85 GB train tar is walked
                     only with --allow-train (about one request per member; slow).
  --transcripts      route A: the Hindi transcription_n2w.txt files of every split (train/valid/test/test_known,
                     ~18 MB text) from clean/transcripts_n2w.tar. Kept under raw/ (gitignored); only hashes enter manifests.
  --inspect          probe route-A tars (size, validators) and list the first members of the test-known tar;
                     report route-B gating and whether HF_TOKEN is set (the token value is never logged)
  --pilot N          route B only: N Hindi clips from one parquet row group. Needs: (1) you accepted the dataset terms
                     at https://huggingface.co/datasets/ai4bharat/Kathbath with your own account, (2) HF_TOKEN exported
                     in your shell (never written to any file in this repo).

Requires --accept-terms. Licence status is CONFLICT (CC0 vs CC-BY-4.0): we comply with the stricter reading.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tarfile
import urllib.parse

import pyarrow.parquet as pq

from ml.data.provenance import (
    AcquisitionError,
    DataPaths,
    HttpRangeReader,
    ProvenanceStore,
    add_path_args,
    file_record,
    open_stream,
    probe_url,
    require_source,
    run_cli,
    safe_member_path,
    setup_logging,
    summarize_files,
    write_atomic_bytes,
)

SOURCE = "kathbath_hi"
OBJECTSTORE = "https://objectstore.e2enetworks.net/indic-superb/kathbath"
TARS = {
    "clean/testkn_audio.tar": f"{OBJECTSTORE}/clean/testkn_audio.tar",
    "clean/testunk_audio.tar": f"{OBJECTSTORE}/clean/testunk_audio.tar",
    "clean/valid_audio.tar": f"{OBJECTSTORE}/clean/valid_audio.tar",
    "clean/train_audio.tar": f"{OBJECTSTORE}/clean/train_audio.tar",
    "clean/transcripts_n2w.tar": f"{OBJECTSTORE}/clean/transcripts_n2w.tar",
}
HF_REPO = "ai4bharat/Kathbath"
TOKEN_ENV = "HF_TOKEN"


def list_tar_head(url: str, max_members: int) -> list[dict]:
    reader = HttpRangeReader(url, block_size=64 * 1024, max_cached_blocks=8)
    out = []
    with tarfile.open(fileobj=reader, mode="r:") as tf:  # uncompressed tar: seekable header walk
        for m in tf:
            out.append({"name": m.name, "size": m.size, "type": "dir" if m.isdir() else "file"})
            if len(out) >= max_members:
                break
    return out


def _header_at(reader: HttpRangeReader, start: int, scan: int = 256 * 1024) -> tuple[int, tarfile.TarInfo] | None:
    """Find the first valid ustar header at a 512-aligned position in [start, start+scan)."""
    base = start - start % 512
    reader.seek(base)
    buf = reader.read(min(scan, reader.size - base))
    for pos in range(0, len(buf) - 512, 512):
        blk = buf[pos : pos + 512]
        if blk[257:262] != b"ustar":
            continue
        try:
            info = tarfile.TarInfo.frombuf(blk, tarfile.ENCODING, "surrogateescape")
        except tarfile.HeaderError:  # bad checksum: not a real header
            continue
        return base + pos, info
    return None


def _walk(reader: HttpRangeReader, header_off: int, max_members: int):
    """Yield (header_offset, TarInfo) for consecutive members starting at a known header."""
    off = header_off
    for _ in range(max_members):
        reader.seek(off)
        blk = reader.read(512)
        if len(blk) < 512 or blk == b"\0" * 512:
            return
        info = tarfile.TarInfo.frombuf(blk, tarfile.ENCODING, "surrogateescape")
        yield off, info
        off += 512 + ((info.size + 511) // 512) * 512


def _lang(name: str) -> str | None:
    parts = name.split("/")
    return parts[1] if len(parts) > 2 and parts[0].startswith("kb_data") else None


def subset_route_a(paths, logger, tar_key: str, n: int, n_probes_map: int, n_points: int, language: str = "hindi") -> dict:
    url = TARS[tar_key]
    reader = HttpRangeReader(url, block_size=64 * 1024, max_cached_blocks=16)
    # 1) map languages along the tar
    lang_map = []
    for i in range(n_probes_map):
        hit = _header_at(reader, int(reader.size * i / n_probes_map))
        if hit:
            lang_map.append((hit[0], _lang(hit[1].name)))
    offs = [o for o, lg in lang_map if lg == language]
    if not offs:
        raise AcquisitionError(f"no {language} members found in {tar_key} probes: {sorted({lg for _, lg in lang_map})}")
    step = reader.size / n_probes_map
    lo, hi = max(0, min(offs) - step), min(reader.size - 1, max(offs) + step)
    logger.info("language map (%d probes): %s; %s approx bytes %.0f-%.0f", len(lang_map), sorted({lg for _, lg in lang_map if lg}), language, lo, hi)
    # 2) sample n_points spread over the language range, take consecutive files at each
    per_point = max(1, -(-n // n_points))
    files, seen = [], set()
    base = paths.raw(SOURCE)
    for j in range(n_points):
        if len(files) >= n:
            break
        hit = _header_at(reader, int(lo + (hi - lo) * (j + 0.5) / n_points))
        if not hit:
            continue
        taken = 0
        for off, info in _walk(reader, hit[0], per_point * 3):
            if taken >= per_point or len(files) >= n:
                break
            if _lang(info.name) != language or not info.isfile() or not info.name.endswith(".m4a") or info.name in seen:
                continue
            reader.seek(off + 512)
            data = reader.read(info.size)
            if len(data) != info.size:
                raise AcquisitionError(f"short read for {info.name}")
            dest = safe_member_path(base, info.name)
            write_atomic_bytes(dest, data)
            rec = file_record(dest, paths)
            rec.update({"member": info.name, "tar_offset": off})
            files.append(rec)
            seen.add(info.name)
            taken += 1
    if len(files) < n:
        logger.warning("collected %d of %d requested files", len(files), n)
    speakers = {f["member"].rsplit("/", 1)[-1].split("-")[1] for f in files}
    logger.info("subset: %d %s files from %d speakers; %d bytes fetched in %d requests", len(files), language, len(speakers), reader.bytes_fetched, reader.requests)
    return {"mode": "subset", "partial": True, "route": "A (AI4Bharat objectstore tar, ungated)",
            "archive": {"name": tar_key, "url": url, "bytes": reader.size, "etag": reader.etag, "last_modified": reader.last_modified,
                        "sha256": None, "note": "members read by byte range from their own tar headers; archive not hashed"},
            "selection": {"language": language, "requested": n, "points": n_points, "per_point": per_point,
                          "rule": "evenly spaced points over the language region; consecutive members at each point"},
            "language_map": [{"offset": o, "language": lg} for o, lg in lang_map],
            "n_speakers": len(speakers), "transfer": {"bytes_fetched": reader.bytes_fetched, "range_requests": reader.requests},
            "files": files}


def walk_language(url: str, logger, language: str = "hindi", n_probes_map: int = 192, workers: int = 24) -> tuple[list[dict], dict]:
    """Every member of `language` in an objectstore tar: name, size, header offset (header-only reads).

    The language's byte range is cut into `workers` chunks that are walked in parallel; each chunk starts at the first
    valid ustar header at/after its start and stops at the first header at/after its end, so chunks neither overlap
    nor leave gaps.
    """
    from concurrent.futures import ThreadPoolExecutor

    mapper = HttpRangeReader(url, block_size=256 * 1024, max_cached_blocks=2)
    step = mapper.size / n_probes_map
    lang_map = []
    for i in range(n_probes_map):
        hit = _header_at(mapper, int(step * i))
        if hit:
            lang_map.append((hit[0], _lang(hit[1].name)))
    offs = [o for o, lg in lang_map if lg == language]
    meta = {"etag": mapper.etag, "last_modified": mapper.last_modified, "size": mapper.size}
    if not offs:
        return [], {**meta, "requests": mapper.requests, "bytes_fetched": mapper.bytes_fetched}
    before = [o for o, lg in lang_map if o < min(offs) and lg != language]
    after = [o for o, lg in lang_map if o > max(offs) and lg != language]
    lo, hi = (max(before) if before else 0), (min(after) if after else mapper.size)
    bounds = [(int(lo + (hi - lo) * k / workers) // 512) * 512 for k in range(workers + 1)]  # tar headers are 512-aligned

    def walk_chunk(k: int) -> tuple[list[dict], int, int]:
        scan = HttpRangeReader(url, block_size=256 * 1024, max_cached_blocks=2)
        hit = _header_at(scan, bounds[k])
        if not hit or hit[0] >= bounds[k + 1]:
            return [], scan.requests, scan.bytes_fetched
        hop = HttpRangeReader(url, block_size=1024, max_cached_blocks=2)
        out = []
        for off, info in _walk(hop, hit[0], 10**7):
            if off >= bounds[k + 1]:
                break
            if _lang(info.name) == language and info.isfile():
                out.append({"name": info.name, "size": info.size, "offset": off})
        return out, scan.requests + hop.requests, scan.bytes_fetched + hop.bytes_fetched

    with ThreadPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(walk_chunk, range(workers)))
    members = sorted((m for ms, _, _ in parts for m in ms), key=lambda m: m["offset"])
    if len({m["offset"] for m in members}) != len(members):
        raise AcquisitionError("overlapping chunks in tar walk")
    req = mapper.requests + sum(r for _, r, _ in parts)
    logger.info("  %s: %s region bytes %d-%d, %d members, %d requests", url.rsplit("/", 1)[-1], language, lo, hi, len(members), req)
    return members, {**meta, "requests": req, "bytes_fetched": mapper.bytes_fetched + sum(b for _, _, b in parts),
                     "region": [lo, hi], "workers": workers}


def fetch_ids(paths, logger, ids_file: str, tar_keys: list[str], extra_per_speaker: int, allow_train: bool) -> dict:
    with open(ids_file, encoding="utf-8") as fh:
        wanted = {ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")}
    if "clean/train_audio.tar" in tar_keys and not allow_train:
        raise AcquisitionError("walking clean/train_audio.tar needs --allow-train (about one request per member)")
    base = paths.raw(SOURCE)
    listing_dir = base / "_listings"
    listing_dir.mkdir(parents=True, exist_ok=True)
    found, archives, files = {}, {}, []
    for key in tar_keys:
        url = TARS[key]
        members, info = walk_language(url, logger, workers=48)
        archives[key] = {"url": url, **info, "hindi_members": len(members)}
        with open(listing_dir / (key.replace("/", "__") + ".csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["name", "size", "offset"], lineterminator="\n")
            w.writeheader()
            w.writerows(members)
        for m in members:
            found.setdefault(m["name"].rsplit("/", 1)[-1].rsplit(".", 1)[0], (key, m))
        logger.info("%s: %d Hindi members listed (%d requests)", key, len(members), info["requests"])
    hits = {s: found[s] for s in wanted if s in found}
    extra = {}
    if extra_per_speaker:
        for spk in sorted({s.split("-")[1] for s in hits}):
            cands = sorted(s for s in found if s.split("-")[1] == spk and s not in wanted)
            extra.update({s: found[s] for s in cands[:extra_per_speaker]})
    readers = {}
    for stem, (key, m) in sorted({**hits, **extra}.items()):
        r = readers.setdefault(key, HttpRangeReader(TARS[key], block_size=256 * 1024, max_cached_blocks=4))
        r.seek(m["offset"] + 512)
        data = r.read(m["size"])
        if len(data) != m["size"]:
            raise AcquisitionError(f"short read for {m['name']}")
        dest = safe_member_path(base, m["name"])
        write_atomic_bytes(dest, data)
        rec = file_record(dest, paths)
        rec.update({"member": m["name"], "tar": key, "tar_offset": m["offset"], "role": "wanted" if stem in wanted else "extra_same_speaker"})
        files.append(rec)
    missing = sorted(wanted - set(hits))
    logger.info("fetch-ids: %d of %d wanted found, %d extra same-speaker files; %d wanted not in %s",
                len(hits), len(wanted), len(extra), len(missing), tar_keys)
    return {"mode": "fetch_ids", "partial": True, "route": "A (AI4Bharat objectstore tar, ungated)", "ids_file": ids_file,
            "archives": archives, "n_wanted": len(wanted), "n_found": len(hits), "n_extra": len(extra),
            "missing": missing, "extra_per_speaker": extra_per_speaker, "files": files}


def fetch_transcripts(paths, logger, language: str = "hindi") -> dict:
    url = TARS["clean/transcripts_n2w.tar"]
    reader = HttpRangeReader(url, block_size=256 * 1024, max_cached_blocks=4)
    hit = _header_at(reader, 0)
    if not hit:
        raise AcquisitionError("no tar header at the start of transcripts_n2w.tar")
    files, seen_lang = [], False
    for off, info in _walk(reader, hit[0], 10**4):
        if _lang(info.name) == language and info.isfile():
            seen_lang = True
            reader.seek(off + 512)
            data = reader.read(info.size)
            if len(data) != info.size:
                raise AcquisitionError(f"short read for {info.name}")
            dest = safe_member_path(paths.raw(SOURCE), "_transcripts/" + info.name)
            write_atomic_bytes(dest, data)
            rec = file_record(dest, paths)
            rec.update({"member": info.name, "tar_offset": off})
            files.append(rec)
        elif seen_lang and _lang(info.name) not in (None, language):
            break
    logger.info("transcripts: %d %s files, %d bytes", len(files), language, sum(f["bytes"] for f in files))
    return {"mode": "transcripts", "partial": True, "route": "A (AI4Bharat objectstore tar, ungated)",
            "archive": {"name": "clean/transcripts_n2w.tar", "url": url, "bytes": reader.size, "etag": reader.etag,
                        "last_modified": reader.last_modified}, "files": files}


def hf_api(path: str, token: bool) -> tuple[dict | list, dict]:
    url = f"https://huggingface.co/api/datasets/{HF_REPO}{path}"
    with open_stream(url, token_env=TOKEN_ENV if token else None) as r:
        return json.loads(r.read().decode("utf-8")), dict(r.headers)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    add_path_args(ap)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--pilot", type=int, metavar="N")
    mode.add_argument("--subset", type=int, metavar="N", help="route A: N files of one language from an objectstore tar")
    mode.add_argument("--fetch-ids", metavar="FILE", help="route A: exact Hindi recordings listed in FILE")
    mode.add_argument("--transcripts", action="store_true", help="route A: Hindi transcripts of all splits (~18 MB)")
    ap.add_argument("--tars", nargs="+", default=["clean/valid_audio.tar", "clean/testkn_audio.tar", "clean/testunk_audio.tar"],
                    choices=sorted(k for k in TARS if "audio" in k), help="with --fetch-ids")
    ap.add_argument("--extra-per-speaker", type=int, default=0, help="with --fetch-ids")
    ap.add_argument("--allow-train", action="store_true", help="with --fetch-ids: permit walking the 85 GB train tar")
    ap.add_argument("--tar", default="clean/testkn_audio.tar", choices=sorted(TARS), help="with --subset")
    ap.add_argument("--points", type=int, default=30, help="with --subset: sampling points over the language region")
    ap.add_argument("--accept-terms", action="store_true")
    ap.add_argument("--max-row-group-mb", type=int, default=150)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    paths = DataPaths.resolve(args.data_root, args.meta_root).ensure()
    logger, log_path = setup_logging("acquire_kathbath", paths, args.verbose)
    entry = require_source(paths, SOURCE, args.accept_terms, logger)
    store = ProvenanceStore(paths, SOURCE)
    base = {"script": "scripts/acquire_kathbath.py", "source_url": entry["url"], "dataset_version": entry["version"],
            "licence": entry["licence_short"], "terms_acknowledged": True, "log_file": log_path.name}

    if args.inspect:
        probes = {}
        for name, url in TARS.items():
            try:
                probes[name] = probe_url(url)
            except AcquisitionError as exc:
                probes[name] = {"url": url, "error": str(exc)}
        head = list_tar_head(TARS["clean/testkn_audio.tar"], 40)
        meta, _ = hf_api("", token=False)
        token_set = bool(os.environ.get(TOKEN_ENV))
        logger.info("route A testkn head: %s", [m["name"] for m in head[:8]])
        logger.info("route B gated=%s sha=%s; %s set: %s", meta.get("gated"), meta.get("sha"), TOKEN_ENV, token_set)
        store.append({**base, "mode": "inspect", "route_a_probes": probes, "route_a_testkn_head": head,
                      "route_b": {"repo": HF_REPO, "gated": meta.get("gated"), "sha": meta.get("sha"),
                                  "card_license": (meta.get("cardData") or {}).get("license"), "token_present": token_set},
                      "files": []})
        return 0

    if args.transcripts:
        ev = fetch_transcripts(paths, logger)
        ev["summary"] = summarize_files(ev["files"])
        store.append({**base, **ev})
        return 0

    if args.fetch_ids:
        ev = fetch_ids(paths, logger, args.fetch_ids, args.tars, args.extra_per_speaker, args.allow_train)
        ev["summary"] = summarize_files(ev["files"])
        store.append({**base, **ev})
        return 0

    if args.subset is not None:
        if not 1 <= args.subset <= 2000:
            raise AcquisitionError("--subset N must be between 1 and 2000")
        ev = subset_route_a(paths, logger, args.tar, args.subset, n_probes_map=48, n_points=args.points)
        ev["summary"] = summarize_files(ev["files"])
        store.append({**base, **ev})
        logger.info("provenance updated: %s (%s)", store.path, ev["summary"])
        return 0

    # ---- pilot (route B) ----
    if not 1 <= args.pilot <= 100:
        raise AcquisitionError("--pilot N must be between 1 and 100")
    if not os.environ.get(TOKEN_ENV):
        raise AcquisitionError(
            f"Kathbath pilot uses the gated HF mirror: accept the terms at https://huggingface.co/datasets/{HF_REPO} "
            f"with your account, then export {TOKEN_ENV} in your shell (never in a file in this repo) and re-run."
        )
    meta, _ = hf_api("", token=True)
    rev = meta.get("sha")
    files_api, _ = hf_api(f"/tree/{rev}/hindi?expand=true", token=True)
    shards = sorted((f for f in files_api if f.get("path", "").endswith(".parquet")), key=lambda f: f.get("size", 0))
    if not shards:
        raise AcquisitionError("no hindi parquet shards visible (access not granted yet?)")
    prefer = [s for s in shards if re.search(r"test|valid", s["path"])] or shards
    shard = prefer[0]
    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{rev}/{urllib.parse.quote(shard['path'])}"
    reader = HttpRangeReader(url, block_size=512 * 1024, token_env=TOKEN_ENV)
    pf = pq.ParquetFile(reader)
    rg_bytes = pf.metadata.row_group(0).total_byte_size
    if rg_bytes / 1e6 > args.max_row_group_mb:
        raise AcquisitionError(f"row group 0 is {rg_bytes / 1e6:.0f} MB > --max-row-group-mb")
    rows = pf.read_row_group(0).to_pylist()[: args.pilot]
    audio_col = next((c for c in pf.schema_arrow.names if c.lower().startswith("audio")), None)
    if audio_col is None:
        raise AcquisitionError(f"no audio column in {pf.schema_arrow.names}")
    out_dir = paths.raw(SOURCE) / "hindi"
    files, meta_rows = [], []
    for i, row in enumerate(rows):
        a = row[audio_col] or {}
        if not a.get("bytes"):
            raise AcquisitionError(f"row {i}: no embedded audio bytes")
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", a.get("path") or f"{shard['path']}_rg0_{i}.bin")
        dest = safe_member_path(out_dir, f"audio/{name}")
        write_atomic_bytes(dest, a["bytes"])
        rec = file_record(dest, paths)
        rec.update({"shard": shard["path"], "row_group": 0, "row_in_group": i})
        files.append(rec)
        meta_rows.append({"file": rec["path"], **{k: v for k, v in row.items() if k != audio_col}})
    meta_path = out_dir / "pilot_metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(meta_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(meta_rows)
    files.append(file_record(meta_path, paths))
    ev = {"mode": "pilot", "partial": True, "route": "B (HF mirror, gated)", "hf_revision": rev, "shard": shard,
          "transfer": {"bytes_fetched": reader.bytes_fetched, "range_requests": reader.requests}, "files": files}
    ev["summary"] = summarize_files(files)
    store.append({**base, **ev})
    logger.info("provenance updated: %s (%s)", store.path, ev["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
