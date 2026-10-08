#!/usr/bin/env python
"""In-the-Wild external evaluation set (source_id: in_the_wild). Metadata only in Phase 2A.

By protocol (docs/data_flow.md §5) In-the-Wild is never used for training or tuning, and audio is not touched until
the pipeline is frozen (Phase 7). This script therefore only:
  --inspect   probes the archive, lists the remote zip (HTTP range) and extracts the label file(s) and small text docs (*.csv, *.txt, *.md) only.
Full download is intentionally not implemented here; it will be added in Phase 7 with the frozen pipeline.
Requires --accept-terms (licence status: CONFLICT Apache-2.0 vs CC-BY-SA-4.0; evaluation use permitted under both).
"""

from __future__ import annotations

import argparse
import zipfile
from collections import Counter

from ml.data.provenance import (
    DataPaths,
    HttpRangeReader,
    ProvenanceStore,
    add_path_args,
    file_record,
    require_source,
    run_cli,
    safe_member_path,
    setup_logging,
    write_atomic_bytes,
)

SOURCE = "in_the_wild"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_path_args(ap)
    ap.add_argument("--inspect", action="store_true", required=True)
    ap.add_argument("--accept-terms", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    paths = DataPaths.resolve(args.data_root, args.meta_root).ensure()
    logger, log_path = setup_logging("acquire_inthewild", paths, args.verbose)
    entry = require_source(paths, SOURCE, args.accept_terms, logger)
    url = f"https://huggingface.co/datasets/{entry['hf_repo']}/resolve/{entry['hf_revision']}/release_in_the_wild.zip"
    reader = HttpRangeReader(url, block_size=512 * 1024)
    zf = zipfile.ZipFile(reader)
    infos = zf.infolist()
    exts = Counter(i.filename.rsplit(".", 1)[-1].lower() for i in infos if not i.is_dir())
    files = []
    for zi in infos:
        if zi.filename.lower().endswith((".csv", ".txt", ".md")) and zi.file_size < 20_000_000:
            dest = safe_member_path(paths.raw(SOURCE), zi.filename)
            write_atomic_bytes(dest, zf.read(zi))
            files.append(file_record(dest, paths))
    logger.info("zip members=%d by_ext=%s; label files=%s; fetched %d bytes", len(infos), dict(exts), [f["path"] for f in files], reader.bytes_fetched)
    ProvenanceStore(paths, SOURCE).append({
        "script": "scripts/acquire_inthewild.py", "source_url": entry["url"], "dataset_version": entry["version"],
        "licence": entry["licence_short"], "terms_acknowledged": True, "log_file": log_path.name, "mode": "inspect",
        "partial": True, "archive": {"name": "release_in_the_wild.zip", "bytes": reader.size, "etag": reader.etag},
        "zip_listing": {"n_members": len(infos), "by_extension": dict(exts)},
        "transfer": {"bytes_fetched": reader.bytes_fetched, "range_requests": reader.requests}, "files": files,
        "note": "label CSV only; no audio extracted (external test set untouched until Phase 7)",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
