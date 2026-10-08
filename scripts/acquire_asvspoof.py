#!/usr/bin/env python
"""Acquire ASVspoof 2019 Logical Access (source_id: asvspoof2019_la).

Modes (exactly one):
  --inspect              probe LA.zip, save README/licence text, list the remote zip (no audio downloaded)
  --pilot N              extract the CM protocols + README.LA + N bona fide + N spoof clips of --partition (default train)
                         via HTTP range reads
  --full --confirm-large download the full 7.1 GB LA.zip (resumable, size-verified, sha256 recorded), then extract

All modes require --accept-terms (ODC-By 1.0, attribution; see LICENSES_REGISTER.md).
Example:  python scripts/acquire_asvspoof.py --pilot 20 --accept-terms
"""

from __future__ import annotations

import argparse
import random
import zipfile
from collections import Counter, defaultdict

from ml.data.provenance import (
    AcquisitionError,
    DataPaths,
    HttpRangeReader,
    ProvenanceStore,
    add_path_args,
    download_file,
    file_record,
    probe_url,
    require_source,
    run_cli,
    safe_member_path,
    setup_logging,
    summarize_files,
    write_atomic_bytes,
)

SOURCE = "asvspoof2019_la"
PROTOCOL_SUFFIX = ".txt"


def _save_docs(entry, paths, logger) -> list[dict]:
    out = []
    lic_dir = paths.licence_texts_dir / SOURCE
    for name in ("README.txt", "LICENSE_text.txt"):
        dest = lic_dir / name
        info = download_file(entry["archive_urls"][name], dest, logger, max_bytes=1_000_000)
        out.append({"path_meta": dest.relative_to(paths.meta_root).as_posix(), "bytes": info["bytes"], "sha256": info["sha256"], "url": info["url"]})
    return out


def _open_zip(entry, logger) -> tuple[zipfile.ZipFile, HttpRangeReader]:
    reader = HttpRangeReader(entry["archive_urls"]["LA.zip"], block_size=256 * 1024)
    expected = entry["archive_bytes"]["LA.zip"]
    if reader.size != expected:
        raise AcquisitionError(f"LA.zip size changed: {reader.size} != registry {expected}; re-verify the source")
    zf = zipfile.ZipFile(reader)
    logger.info("remote LA.zip opened: %d members, %d bytes", len(zf.infolist()), reader.size)
    return zf, reader


def _listing_summary(zf: zipfile.ZipFile) -> dict:
    tops = Counter()
    exts = Counter()
    for zi in zf.infolist():
        parts = zi.filename.split("/")
        tops["/".join(parts[:2])] += 1
        exts[zi.filename.rsplit(".", 1)[-1].lower() if "." in parts[-1] else "<dir>"] += 1
    return {"n_members": len(zf.infolist()), "by_top_dir": dict(sorted(tops.items())), "by_extension": dict(exts)}


def parse_cm_protocol(text: str) -> list[dict]:
    """ASVspoof 2019 LA CM protocol line: SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5 or parts[4] not in ("bonafide", "spoof"):
            raise AcquisitionError(f"unexpected protocol line: {line!r}")
        rows.append({"speaker_id": parts[0], "utt_id": parts[1], "system_id": parts[3], "key": parts[4]})
    return rows


def _extract(zf, member: str, paths: DataPaths) -> dict:
    dest = safe_member_path(paths.raw(SOURCE), member)
    data = zf.read(member)  # zipfile verifies the CRC-32 of every member it reads
    write_atomic_bytes(dest, data)
    return file_record(dest, paths)


def _extract_parallel(zf, members: list[str], url: str, paths: DataPaths, logger, workers: int = 8) -> tuple[list[dict], int]:
    """Fetch many members concurrently, ONE range request each, using offsets from the already-parsed central directory.

    DataShare is latency-bound (~seconds per request), so sequential zipfile reads are slow. Each member is verified
    exactly as zipfile does it: local header signature, size, and CRC-32 of the decompressed bytes.
    """
    import struct
    import threading
    import zlib
    from concurrent.futures import ThreadPoolExecutor

    local = threading.local()
    fetched = [0]
    lock = threading.Lock()

    def reader():
        if not hasattr(local, "r"):
            local.r = HttpRangeReader(url, block_size=64 * 1024)
        return local.r

    def one(member: str) -> dict:
        zi = zf.getinfo(member)
        dest = safe_member_path(paths.raw(SOURCE), member)
        if dest.exists() and zlib.crc32(dest.read_bytes()) & 0xFFFFFFFF == zi.CRC:
            return file_record(dest, paths)  # already on disk and identical
        span = 30 + len(zi.orig_filename.encode()) + 1024 + zi.compress_size  # local header + generous extra field
        raw = reader()._fetch(zi.header_offset, min(zi.header_offset + span, reader().size) - 1)
        sig, *_rest = struct.unpack("<4s5H3L2H", raw[:30])
        if sig != b"PK\x03\x04":
            raise AcquisitionError(f"{member}: bad local header signature")
        name_len, extra_len = struct.unpack("<2H", raw[26:30])
        start = 30 + name_len + extra_len
        comp = raw[start : start + zi.compress_size]
        if len(comp) != zi.compress_size:
            raise AcquisitionError(f"{member}: short read")
        if zi.compress_type == zipfile.ZIP_STORED:
            data = comp
        elif zi.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(comp, -15)
        else:
            raise AcquisitionError(f"{member}: unsupported compression {zi.compress_type}")
        if len(data) != zi.file_size or zlib.crc32(data) & 0xFFFFFFFF != zi.CRC:
            raise AcquisitionError(f"{member}: size/CRC-32 mismatch")
        write_atomic_bytes(dest, data)
        with lock:
            fetched[0] += len(raw)
        return file_record(dest, paths)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        records = list(ex.map(one, members))
    logger.info("parallel extraction: %d members, %d bytes fetched with %d workers", len(members), fetched[0], workers)
    return records, fetched[0]


PROTOCOL_FOR = {"train": "train.trn.txt", "dev": "dev.trl.txt", "eval": "eval.trl.txt"}


def pilot(entry, paths, logger, n: int, seed: int, partition: str = "train") -> dict:
    zf, reader = _open_zip(entry, logger)
    names = zf.namelist()
    protocols = [m for m in names if "cm_protocols" in m and m.endswith(PROTOCOL_SUFFIX)]
    readmes = [m for m in names if m.endswith("README.LA.txt")]
    if not protocols:
        raise AcquisitionError("no CM protocol files found in LA.zip")
    files = [_extract(zf, m, paths) for m in protocols + readmes]
    part_proto = [m for m in protocols if m.endswith(PROTOCOL_FOR[partition])]
    if len(part_proto) != 1:
        raise AcquisitionError(f"expected one {partition} protocol, found {part_proto}")
    rows = parse_cm_protocol(zf.read(part_proto[0]).decode("utf-8"))
    by_member = {m.rsplit("/", 1)[-1][: -len(".flac")]: m for m in names if f"LA_{partition}/flac/" in m and m.endswith(".flac")}
    rng = random.Random(seed)
    bona = [r for r in rows if r["key"] == "bonafide"]
    spoof_by_sys = defaultdict(list)
    for r in rows:
        if r["key"] == "spoof":
            spoof_by_sys[r["system_id"]].append(r)
    chosen = rng.sample(bona, n)
    systems = sorted(spoof_by_sys)
    for i in range(n):  # round-robin over attack systems so every attack in the partition is represented
        chosen.append(rng.choice(spoof_by_sys[systems[i % len(systems)]]))
    missing = [r["utt_id"] for r in chosen if r["utt_id"] not in by_member]
    if missing:
        raise AcquisitionError(f"protocol utterances without flac member in LA.zip: {missing[:5]}")
    records, parallel_bytes = _extract_parallel(zf, [by_member[r["utt_id"]] for r in chosen], entry["archive_urls"]["LA.zip"], paths, logger)
    files.extend(records)
    logger.info("pilot: %d files extracted; %d bytes fetched (central dir + %d parallel)", len(files), reader.bytes_fetched + parallel_bytes, parallel_bytes)
    return {
        "mode": "pilot",
        "partial": True,
        "selection": {"partition": partition, "n_bonafide": n, "n_spoof": n, "seed": seed, "spoof_systems": systems},
        "archive": {"name": "LA.zip", "bytes": reader.size, "etag": reader.etag, "last_modified": reader.last_modified,
                    "sha256": None, "note": "pilot reads members by range; whole-archive sha256 only in --full"},
        "transfer": {"bytes_fetched": reader.bytes_fetched + parallel_bytes, "central_dir_requests": reader.requests},
        "files": files,
    }


def full(entry, paths, logger) -> dict:
    archive = paths.raw(SOURCE) / "archives" / "LA.zip"
    info = download_file(entry["archive_urls"]["LA.zip"], archive, logger, expected_bytes=entry["archive_bytes"]["LA.zip"])
    files = []
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise AcquisitionError(f"CRC failure in {bad}")
        for zi in zf.infolist():
            if zi.is_dir():
                continue
            dest = safe_member_path(paths.raw(SOURCE), zi.filename)
            write_atomic_bytes(dest, zf.read(zi))
            files.append(file_record(dest, paths))
    return {"mode": "full", "partial": False, "archive": {"name": "LA.zip", **info}, "files": files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    add_path_args(ap)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--pilot", type=int, metavar="N")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--confirm-large", action="store_true", help="required with --full (7.1 GB)")
    ap.add_argument("--accept-terms", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--partition", choices=["train", "dev", "eval"], default="train", help="with --pilot: which LA partition to sample")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    paths = DataPaths.resolve(args.data_root, args.meta_root).ensure()
    logger, log_path = setup_logging("acquire_asvspoof", paths, args.verbose)
    entry = require_source(paths, SOURCE, args.accept_terms, logger)
    store = ProvenanceStore(paths, SOURCE)
    base = {"script": "scripts/acquire_asvspoof.py", "source_url": entry["url"], "doi": entry.get("doi"),
            "dataset_version": entry.get("version"), "licence": entry["licence_short"], "terms_acknowledged": True,
            "log_file": log_path.name}

    if args.inspect:
        docs = _save_docs(entry, paths, logger)
        probe = probe_url(entry["archive_urls"]["LA.zip"])
        zf, reader = _open_zip(entry, logger)
        summary = _listing_summary(zf)
        logger.info("listing summary: %s", summary["by_top_dir"])
        store.append({**base, "mode": "inspect", "probe": probe, "zip_listing": summary, "docs": docs,
                      "transfer": {"bytes_fetched": reader.bytes_fetched, "range_requests": reader.requests}, "files": []})
        return 0
    if args.pilot is not None:
        if not 1 <= args.pilot <= 1000:
            raise AcquisitionError("--pilot N must be between 1 and 1000 per class (larger pulls: use --full deliberately)")
        ev = pilot(entry, paths, logger, args.pilot, args.seed, args.partition)
    else:
        if not args.confirm_large:
            raise AcquisitionError("--full downloads 7.1 GB; add --confirm-large to proceed")
        ev = full(entry, paths, logger)
    ev["summary"] = summarize_files(ev["files"])
    store.append({**base, **ev})
    logger.info("provenance updated: %s (%s)", store.path, ev["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
