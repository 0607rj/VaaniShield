#!/usr/bin/env python
"""Check that every audio file referenced by the given manifests exists under the data root and matches its sha256.

    python scripts/verify_data.py                                   # the committed manifests
    python scripts/verify_data.py data/manifests/vaanishield_v1_full.csv
Data root: $VAANI_DATA_ROOT, else <repo>/data. Exit code 1 if anything is missing or different.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ml.data.manifest import read_manifest
from ml.data.provenance import DataPaths
from ml.training.repro import sha256_file

DEFAULT = ["vaanishield_v1", "hinglish_pilot_v1", "codec_pilot_v1", "robust_smoke_codec"]


def main(argv: list[str]) -> int:
    root = DataPaths.resolve().data_root
    manifests = [Path(a) for a in argv] or [Path(f"data/manifests/{m}.csv") for m in DEFAULT]
    print(f"data root: {root}")
    failed = False
    for m in manifests:
        rows = read_manifest(m)
        missing = [r for r in rows if not (root / r.path).exists()]
        bad = [r for r in rows if r not in missing and sha256_file(root / r.path) != r.sha256]
        failed |= bool(missing or bad)
        print(f"{m.name:28s} {len(rows):6d} rows  missing {len(missing):5d}  hash mismatch {len(bad)}"
              + (f"  e.g. {(missing or bad)[0].path}" if missing or bad else ""))
    print("FAIL: copy/download the missing audio before training" if failed else "OK: all audio present and unchanged")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
