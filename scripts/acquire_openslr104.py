#!/usr/bin/env python
"""Acquire OpenSLR SLR104 (MUCS 2021) Hindi-English (source_id: openslr104_hien).

Modes (exactly one):
  --inspect              probe all mirrors (size, validators, range support); stream only the archive head to learn layout
  --pilot N              stream Hindi-English_test.tar.gz from the start, keep the first N .wav members plus every small
                         text file met on the way, then stop (download is truncated: the archive itself is NOT verified).
                         NOTE (inspect, 2026-10-07): members are whole lecture recordings (15-25 MB, ~8-13 min each) named
                         by random IDs; N counts RECORDINGS, so keep N small (1-10). Transcripts/segments were not in the
                         first 17 MB; if they sit at the end of the archive only --full can retrieve them.
  --full --confirm-large download the test archive (and with --include-train the 7.3 GB train archive), resumable and
                         size-verified, sha256 recorded (no upstream checksum is published), then extract safely

Requires --accept-terms (CC BY-SA 4.0).
Example:  python scripts/acquire_openslr104.py --pilot 2 --accept-terms
"""

from __future__ import annotations

import argparse
import tarfile

from ml.data.provenance import (
    AcquisitionError,
    CountingReader,
    DataPaths,
    ProvenanceStore,
    add_path_args,
    download_file,
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

SOURCE = "openslr104_hien"
TEST = "Hindi-English_test.tar.gz"
TRAIN = "Hindi-English_train.tar.gz"
TEXT_MAX_BYTES = 20_000_000


def first_working_mirror(urls: list[str], logger) -> tuple[str, dict]:
    errors = []
    for u in urls:
        try:
            p = probe_url(u)
            logger.info("mirror ok: %s (%s bytes)", u, p["total_bytes"])
            return u, p
        except AcquisitionError as exc:
            logger.warning("mirror failed: %s", exc)
            errors.append(str(exc))
    raise AcquisitionError(f"all mirrors failed: {errors}")


def stream_members(url: str, paths: DataPaths, logger, n_wav: int, byte_cap: int, keep: bool) -> dict:
    """Stream a .tar.gz sequentially; optionally keep the first n_wav wavs + small text files."""
    base = paths.raw(SOURCE)
    files, seen, n_saved = [], [], 0
    resp = open_stream(url)
    counter = CountingReader(resp)
    stopped_reason = "end_of_archive"
    try:
        with tarfile.open(fileobj=counter, mode="r|gz") as tf:
            for m in tf:
                if len(seen) < 400:
                    seen.append({"name": m.name, "size": m.size, "type": "dir" if m.isdir() else "file"})
                if m.isfile() and keep:
                    is_wav = m.name.lower().endswith(".wav")
                    is_text = not is_wav and m.size <= TEXT_MAX_BYTES
                    if (is_wav and n_saved < n_wav) or is_text:
                        f = tf.extractfile(m)
                        if f is None:
                            raise AcquisitionError(f"cannot read member {m.name}")
                        data = f.read()
                        if len(data) != m.size:
                            raise AcquisitionError(f"short member read {m.name}: {len(data)} != {m.size}")
                        dest = safe_member_path(base, m.name)
                        write_atomic_bytes(dest, data)
                        rec = file_record(dest, paths)
                        rec["member"] = m.name
                        files.append(rec)
                        n_saved += int(is_wav)
                if keep and n_saved >= n_wav and counter.bytes_read >= byte_cap:
                    stopped_reason = f"collected {n_wav} wav and passed byte cap {byte_cap}"
                    break
                if not keep and counter.bytes_read >= byte_cap:
                    stopped_reason = f"inspect byte cap {byte_cap}"
                    break
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise AcquisitionError(f"tar stream error after {counter.bytes_read} bytes: {exc}") from exc
    finally:
        resp.close()
    if keep and n_saved < n_wav:
        raise AcquisitionError(f"archive ended with only {n_saved} wav members (wanted {n_wav})")
    logger.info("streamed %d compressed bytes; kept %d files (%d wav); stop: %s", counter.bytes_read, len(files), n_saved, stopped_reason)
    return {"files": files, "members_seen_head": seen, "compressed_bytes_streamed": counter.bytes_read, "stopped": stopped_reason}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    add_path_args(ap)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--pilot", type=int, metavar="N")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--include-train", action="store_true", help="with --full: also the 7.3 GB train archive")
    ap.add_argument("--confirm-large", action="store_true")
    ap.add_argument("--byte-cap-mb", type=int, default=40, help="pilot/inspect: stop streaming after this many MB")
    ap.add_argument("--accept-terms", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    paths = DataPaths.resolve(args.data_root, args.meta_root).ensure()
    logger, log_path = setup_logging("acquire_openslr104", paths, args.verbose)
    entry = require_source(paths, SOURCE, args.accept_terms, logger)
    store = ProvenanceStore(paths, SOURCE)
    base = {"script": "scripts/acquire_openslr104.py", "source_url": entry["url"], "dataset_version": entry["version"],
            "licence": entry["licence_short"], "terms_acknowledged": True, "log_file": log_path.name}
    cap = args.byte_cap_mb * 1_000_000

    if args.inspect:
        probes = {}
        for name, urls in entry["archive_urls"].items():
            probes[name] = []
            for u in urls:
                try:
                    probes[name].append(probe_url(u))
                except AcquisitionError as exc:
                    probes[name].append({"url": u, "error": str(exc)})
        url, _ = first_working_mirror(entry["archive_urls"][TEST], logger)
        head = stream_members(url, paths, logger, 0, min(cap, 5_000_000), keep=False)
        store.append({**base, "mode": "inspect", "probes": probes, "archive": TEST, "mirror": url, **head, "files": []})
        return 0

    if args.pilot is not None:
        if not 1 <= args.pilot <= 10:
            raise AcquisitionError("--pilot N counts whole recordings (15-25 MB each); N must be between 1 and 10")
        url, probe = first_working_mirror(entry["archive_urls"][TEST], logger)
        if probe["total_bytes"] != entry["archive_bytes"][TEST]:
            raise AcquisitionError(f"{TEST} size {probe['total_bytes']} != registry {entry['archive_bytes'][TEST]}")
        res = stream_members(url, paths, logger, args.pilot, cap, keep=True)
        ev = {"mode": "pilot", "partial": True, "archive": {"name": TEST, "mirror": url, "bytes": probe["total_bytes"],
              "etag": probe["etag"], "last_modified": probe["last_modified"], "sha256": None,
              "note": "streamed prefix only; archive integrity NOT verified in pilot mode"},
              "selection": {"rule": f"first {args.pilot} wav members in archive order + all text files <= {TEXT_MAX_BYTES} bytes seen"},
              **res}
    else:
        if not args.confirm_large:
            raise AcquisitionError("--full downloads 443 MB (test) / 7.3 GB (train); add --confirm-large")
        files, archives = [], []
        for name in [TEST] + ([TRAIN] if args.include_train else []):
            url, probe = first_working_mirror(entry["archive_urls"][name], logger)
            dest = paths.raw(SOURCE) / "archives" / name
            info = download_file(url, dest, logger, expected_bytes=entry.get("archive_bytes", {}).get(name))
            archives.append({"name": name, "mirror": url, **info, "note": "no upstream checksum published; sha256 is ours"})
            with tarfile.open(dest, "r:gz") as tf:
                for m in tf.getmembers():
                    if m.isfile():
                        dst = safe_member_path(paths.raw(SOURCE), m.name)
                        write_atomic_bytes(dst, tf.extractfile(m).read())
                        files.append(file_record(dst, paths))
        ev = {"mode": "full", "partial": False, "archives": archives, "files": files}
    ev["summary"] = summarize_files(ev["files"])
    store.append({**base, **ev})
    logger.info("provenance updated: %s (%s)", store.path, ev["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
