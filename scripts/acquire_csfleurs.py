#!/usr/bin/env python
"""Acquire selected CS-FLEURS Hindi-English files (source_id: cs_fleurs) at a pinned HF revision.

Modes (exactly one):
  --metadata            download metadata.jsonl of read/test, xtts/train, mms/test (~21 MB, text only)
  --files PLAN.csv      download exactly the CS-FLEURS files listed in a pilot plan
                        (columns source=cs_fleurs, subset, file_name; written by `python -m ml.data.build_hinglish_pilot plan`)

Requires --accept-terms (CC BY-NC 4.0: non-commercial research use, attribution).
Example:  python scripts/acquire_csfleurs.py --files data/manifests/hinglish_pilot_plan.csv --accept-terms
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from ml.data.provenance import (
    AcquisitionError,
    DataPaths,
    ProvenanceStore,
    add_path_args,
    download_file,
    file_record,
    require_source,
    run_cli,
    safe_member_path,
    setup_logging,
    sha256_of,
    summarize_files,
)

SOURCE = "cs_fleurs"
SUBSETS = ("read/test", "xtts/train", "mms/test")


def _url(entry, rel: str) -> str:
    return f"https://huggingface.co/datasets/{entry['hf_repo']}/resolve/{entry['hf_revision']}/{rel}"


def _norm(file_name: str) -> str:
    return re.sub(r"/+", "/", file_name)  # metadata has 'audio/x//y.wav'


def fetch(entry, paths, logger: logging.Logger, rels: list[str], workers: int = 6) -> list[dict]:
    def one(rel: str) -> dict:
        dest = safe_member_path(paths.raw(SOURCE), rel)
        if not dest.exists():
            download_file(_url(entry, rel), dest, logger, max_bytes=50_000_000)
        rec = file_record(dest, paths)
        rec["url"] = _url(entry, rel)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, rels))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_path_args(ap)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--metadata", action="store_true")
    mode.add_argument("--files", metavar="PLAN_CSV")
    ap.add_argument("--accept-terms", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    paths = DataPaths.resolve(args.data_root, args.meta_root).ensure()
    logger, log_path = setup_logging("acquire_csfleurs", paths, args.verbose)
    entry = require_source(paths, SOURCE, args.accept_terms, logger)
    if args.metadata:
        rels = [f"{s}/metadata.jsonl" for s in SUBSETS]
        mode = "metadata"
    else:
        with open(args.files, encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["source"] == SOURCE]
        if not rows:
            raise AcquisitionError(f"no cs_fleurs rows in {args.files}")
        rels = sorted({f"{r['subset']}/{_norm(r['file_name'])}" for r in rows})
        mode = "plan_files"
    files = fetch(entry, paths, logger, rels)
    for f in files:
        if sha256_of(paths.data_root / f["path"]) != f["sha256"]:
            raise AcquisitionError(f"hash changed during download: {f['path']}")
    ev = {"script": "scripts/acquire_csfleurs.py", "source_url": entry["url"], "dataset_version": entry["version"],
          "licence": entry["licence_short"], "terms_acknowledged": True, "log_file": log_path.name, "mode": mode,
          "partial": True, "plan": args.files, "files": files, "summary": summarize_files(files)}
    store = ProvenanceStore(paths, SOURCE)
    store.append(ev)
    logger.info("%s: %d files (%s); provenance %s", mode, len(files), ev["summary"], store.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
