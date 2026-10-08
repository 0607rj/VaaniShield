#!/usr/bin/env python
"""Acquire the IndicSynth Hindi subset (source_id: indicsynth_hi) from Hugging Face at a pinned revision.

Modes (exactly one):
  --inspect              list the Hindi parquet shards (size + LFS sha256) and read a few parquet footers (no audio)
  --pilot N              N clips per generator (xtts_v2, freevc24): reads one parquet row group per generator
                         via HTTP range requests; audio bytes are written exactly as stored (no re-encoding)
  --subset K             middle row group of K evenly spaced shards, --rows-per-shard rows each (speaker/generator spread)
  --full --confirm-large download all Hindi shards (~52 GiB), verifying each against its LFS sha256

Requires --accept-terms (CC BY-NC 4.0: non-commercial research use, attribution).
Example:  python scripts/acquire_indicsynth.py --pilot 10 --accept-terms
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.parse

import pyarrow.parquet as pq

from ml.data.provenance import (
    AcquisitionError,
    DataPaths,
    HttpRangeReader,
    ProvenanceStore,
    add_path_args,
    download_file,
    file_record,
    open_stream,
    require_source,
    run_cli,
    safe_member_path,
    setup_logging,
    summarize_files,
    write_atomic_bytes,
)

SOURCE = "indicsynth_hi"
CONFIG = "Hindi"
GEN_COL = "Generative Model"
META_COLS = ["Generative Model", "Source Speaker_ID", "Target Speaker ID", "Gender", "Source Reference Audio",
             "Target Reference Audio", "TTS Transcript"]


def list_shards(entry) -> list[dict]:
    repo, rev = entry["hf_repo"], entry["hf_revision"]
    url: str | None = f"https://huggingface.co/api/datasets/{repo}/tree/{rev}/{CONFIG}?expand=true"
    items: list[dict] = []
    pages = 0
    while url:  # the tree API is paginated (Link: <...>; rel="next")
        with open_stream(url) as r:
            items.extend(json.loads(r.read().decode("utf-8")))
            link = r.headers.get("Link") or ""
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
        pages += 1
        if pages > 100:
            raise AcquisitionError("tree listing did not terminate after 100 pages")
    shards = []
    for it in items:
        if it.get("type") == "file" and it["path"].endswith(".parquet"):
            lfs = it.get("lfs") or {}
            shards.append({"path": it["path"], "bytes": it.get("size"), "lfs_sha256": lfs.get("oid")})
    if not shards:
        raise AcquisitionError(f"no parquet shards found under {CONFIG}/ at revision {rev}")
    shards.sort(key=lambda s: s["path"])
    if entry.get("hindi_shards") and len(shards) != entry["hindi_shards"]:
        raise AcquisitionError(f"shard count {len(shards)} != registry {entry['hindi_shards']}; dataset changed, re-verify")
    return shards


def resolve_url(entry, path: str) -> str:
    return f"https://huggingface.co/datasets/{entry['hf_repo']}/resolve/{entry['hf_revision']}/{urllib.parse.quote(path)}"


def footer(entry, shard: dict) -> tuple[pq.ParquetFile, HttpRangeReader, dict]:
    reader = HttpRangeReader(resolve_url(entry, shard["path"]), block_size=512 * 1024)
    if shard["bytes"] and reader.size != shard["bytes"]:
        raise AcquisitionError(f"{shard['path']}: size {reader.size} != listed {shard['bytes']}")
    pf = pq.ParquetFile(reader)
    md = pf.metadata
    info = {
        "path": shard["path"],
        "num_rows": md.num_rows,
        "num_row_groups": md.num_row_groups,
        "row_group_rows": [md.row_group(i).num_rows for i in range(md.num_row_groups)],
        "row_group_bytes": [md.row_group(i).total_byte_size for i in range(md.num_row_groups)],
        "columns": pf.schema_arrow.names,
    }
    return pf, reader, info


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def pilot(entry, paths, logger, n: int, max_rg_mb: int) -> dict:
    shards = list_shards(entry)
    want = {"xtts_v2": n, "freevc24": n}
    found: dict[str, list] = {g: [] for g in want}
    order = [0, len(shards) - 1] + list(range(1, len(shards) - 1))  # generators may be grouped by shard order
    files, shard_infos, fetched, requests = [], [], 0, 0
    out_dir = paths.raw(SOURCE) / CONFIG
    meta_rows = []
    for si in order[:12]:
        if all(len(found[g]) >= want[g] for g in want):
            break
        pf, reader, info = footer(entry, shards[si])
        gens = pf.read(columns=[GEN_COL]).column(GEN_COL).to_pylist()
        info["generators"] = sorted(set(gens))
        shard_infos.append(info)
        logger.info("%s: %d rows, %d row groups, generators=%s", info["path"], info["num_rows"], info["num_row_groups"], info["generators"])
        starts = [0]
        for k in info["row_group_rows"][:-1]:
            starts.append(starts[-1] + k)
        for g in want:
            if len(found[g]) >= want[g] or g not in gens:
                continue
            first = gens.index(g)
            rg = max(i for i, s in enumerate(starts) if s <= first)
            mb = info["row_group_bytes"][rg] / 1e6
            if mb > max_rg_mb:
                raise AcquisitionError(f"{info['path']} row group {rg} is {mb:.0f} MB > --max-row-group-mb {max_rg_mb}")
            table = pf.read_row_group(rg)
            rows = table.to_pylist()
            for local_idx, row in enumerate(rows):
                if row[GEN_COL] != g or len(found[g]) >= want[g]:
                    continue
                audio = row["audio"] or {}
                data = audio.get("bytes")
                if not data:
                    raise AcquisitionError(f"{info['path']} rg {rg} row {local_idx}: no embedded audio bytes")
                name = _safe_name(audio.get("path") or f"{info['path']}_rg{rg}_{local_idx}.wav")
                rel = f"audio/{_safe_name(g)}/{name}"
                dest = safe_member_path(out_dir, rel)
                write_atomic_bytes(dest, data)
                rec = file_record(dest, paths)
                rec.update({"shard": info["path"], "row_group": rg, "row_in_group": local_idx})
                files.append(rec)
                meta_rows.append({"file": rec["path"], **{c: row.get(c) for c in META_COLS}})
                found[g].append(rec["path"])
        fetched += reader.bytes_fetched
        requests += reader.requests
    missing = {g: want[g] - len(found[g]) for g in want if len(found[g]) < want[g]}
    if missing:
        raise AcquisitionError(f"could not find enough rows for {missing} in the scanned shards")
    # Third-party metadata incl. transcripts (CC BY-NC): kept next to the raw audio, never in Git.
    meta_path = out_dir / "pilot_metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", *META_COLS], lineterminator="\n")
        w.writeheader()
        w.writerows(meta_rows)
    files.append(file_record(meta_path, paths))
    logger.info("pilot: %d audio files + metadata; %d bytes fetched in %d range requests", len(files) - 1, fetched, requests)
    return {"mode": "pilot", "partial": True, "hf_revision": entry["hf_revision"], "n_shards_listed": len(shards),
            "total_listed_bytes": sum(s["bytes"] or 0 for s in shards), "shards_read": shard_infos,
            "selection": {"per_generator": n, "rule": "first N rows of each generator in scan order"},
            "transfer": {"bytes_fetched": fetched, "range_requests": requests}, "files": files}


def subset(entry, paths, logger, n_shards: int, rows_per_shard: int, max_rg_mb: int) -> dict:
    """One middle row group from each of n_shards evenly spaced shards; up to rows_per_shard rows from it.

    Spreads the sample over many target speakers and both generators (shards are grouped by generator).
    """
    shards = list_shards(entry)
    idx = sorted({round(i * (len(shards) - 1) / max(1, n_shards - 1)) for i in range(n_shards)})
    out_dir = paths.raw(SOURCE) / CONFIG
    files, meta_rows, infos, fetched, requests = [], [], [], 0, 0
    for si in idx:
        pf, reader, info = footer(entry, shards[si])
        rg = info["num_row_groups"] // 2
        mb = info["row_group_bytes"][rg] / 1e6
        if mb > max_rg_mb:
            raise AcquisitionError(f"{info['path']} row group {rg} is {mb:.0f} MB > --max-row-group-mb {max_rg_mb}")
        rows = pf.read_row_group(rg).to_pylist()[:rows_per_shard]
        info["row_group_used"] = rg
        info["generators_in_rows"] = sorted({r[GEN_COL] for r in rows})
        infos.append(info)
        for local_idx, row in enumerate(rows):
            audio = row["audio"] or {}
            data = audio.get("bytes")
            if not data:
                raise AcquisitionError(f"{info['path']} rg {rg} row {local_idx}: no embedded audio bytes")
            name = _safe_name(audio.get("path") or f"{info['path']}_rg{rg}_{local_idx}.wav")
            dest = safe_member_path(out_dir, f"audio/{_safe_name(row[GEN_COL])}/{name}")
            write_atomic_bytes(dest, data)
            rec = file_record(dest, paths)
            rec.update({"shard": info["path"], "row_group": rg, "row_in_group": local_idx})
            files.append(rec)
            meta_rows.append({"file": rec["path"], **{c: row.get(c) for c in META_COLS}})
        fetched += reader.bytes_fetched
        requests += reader.requests
        logger.info("%s rg %d: %d rows, generators %s", info["path"], rg, len(rows), info["generators_in_rows"])
    meta_path = out_dir / "subset_metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", *META_COLS], lineterminator="\n")
        w.writeheader()
        w.writerows(meta_rows)
    files.append(file_record(meta_path, paths))
    logger.info("subset: %d audio files; %d bytes fetched in %d range requests", len(files) - 1, fetched, requests)
    return {"mode": "subset", "partial": True, "hf_revision": entry["hf_revision"], "n_shards_listed": len(shards),
            "shards_read": infos, "selection": {"shard_indices": idx, "rows_per_shard": rows_per_shard,
            "rule": "middle row group of evenly spaced shards"},
            "transfer": {"bytes_fetched": fetched, "range_requests": requests}, "files": files}


def full(entry, paths, logger) -> dict:
    shards = list_shards(entry)
    files = []
    for s in shards:
        dest = safe_member_path(paths.raw(SOURCE), s["path"])
        if dest.exists() and dest.stat().st_size == s["bytes"]:
            logger.info("exists, re-verifying: %s", s["path"])
            rec = file_record(dest, paths)
            if s["lfs_sha256"] and rec["sha256"] != s["lfs_sha256"]:
                raise AcquisitionError(f"{s['path']}: sha256 mismatch with LFS oid")
        else:
            download_file(resolve_url(entry, s["path"]), dest, logger, expected_bytes=s["bytes"], expected_sha256=s["lfs_sha256"])
            rec = file_record(dest, paths)
        files.append(rec)
    return {"mode": "full", "partial": False, "hf_revision": entry["hf_revision"], "files": files}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    add_path_args(ap)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--pilot", type=int, metavar="N")
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--subset", type=int, metavar="K", help="K evenly spaced shards, one row group each")
    ap.add_argument("--rows-per-shard", type=int, default=25)
    ap.add_argument("--confirm-large", action="store_true", help="required with --full (~52 GiB)")
    ap.add_argument("--accept-terms", action="store_true")
    ap.add_argument("--max-row-group-mb", type=int, default=150)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    paths = DataPaths.resolve(args.data_root, args.meta_root).ensure()
    logger, log_path = setup_logging("acquire_indicsynth", paths, args.verbose)
    entry = require_source(paths, SOURCE, args.accept_terms, logger)
    store = ProvenanceStore(paths, SOURCE)
    base = {"script": "scripts/acquire_indicsynth.py", "source_url": entry["url"], "dataset_version": entry["version"],
            "licence": entry["licence_short"], "terms_acknowledged": True, "log_file": log_path.name, "config": CONFIG}
    if args.inspect:
        shards = list_shards(entry)
        infos, fetched = [], 0
        for i in sorted({0, len(shards) // 2, len(shards) - 1}):
            _, reader, info = footer(entry, shards[i])
            infos.append(info)
            fetched += reader.bytes_fetched
            logger.info("%s rows=%d row_groups=%d max_rg_MB=%.1f", info["path"], info["num_rows"], info["num_row_groups"], max(info["row_group_bytes"]) / 1e6)
        total = sum(s["bytes"] or 0 for s in shards)
        logger.info("Hindi: %d shards, %.1f GiB listed", len(shards), total / 1024**3)
        store.append({**base, "mode": "inspect", "shards": shards, "footers": infos, "total_listed_bytes": total,
                      "transfer": {"bytes_fetched": fetched}, "files": []})
        return 0
    if args.subset is not None:
        if not 1 <= args.subset <= 107:
            raise AcquisitionError("--subset K must be between 1 and 107 shards")
        ev = subset(entry, paths, logger, args.subset, args.rows_per_shard, args.max_row_group_mb)
    elif args.pilot is not None:
        if not 1 <= args.pilot <= 100:
            raise AcquisitionError("--pilot N must be between 1 and 100")
        ev = pilot(entry, paths, logger, args.pilot, args.max_row_group_mb)
    else:
        if not args.confirm_large:
            raise AcquisitionError("--full downloads ~52 GiB; add --confirm-large to proceed")
        ev = full(entry, paths, logger)
    ev["summary"] = summarize_files(ev["files"])
    store.append({**base, **ev})
    logger.info("provenance updated: %s (%s)", store.path, ev["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
