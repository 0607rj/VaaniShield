"""Dataset acquisition and provenance primitives (Phase 2A).

Design rules (PROJECT_STATE Phase 2A):
- Paths are configurable: audio under ``DATA_ROOT`` (``--data-root`` > ``VAANI_DATA_ROOT`` > ``<repo>/data``);
  text artefacts (manifests, licence registry, provenance) under ``META_ROOT/manifests`` (``--meta-root`` > ``<repo>/data``).
- No credentials in code. Tokens are read from the environment (``HF_TOKEN``) only when a source needs them.
- Failed or partial transfers raise ``AcquisitionError``. They are never silently accepted.
- Every transfer is logged, and recorded in an append-only provenance file with URL, HTTP validators
  (ETag / Last-Modified), dataset version identifiers, sizes and SHA-256 of every stored file.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ml.training.repro import git_state

REPO_ROOT = Path(__file__).resolve().parents[2]
USER_AGENT = "VaaniShield-research-data-acquisition/0.1 (B.Tech research project; contact via repository)"
_CHUNK = 1 << 20

AUDIO_SUBDIRS = ("raw", "processed")


class AcquisitionError(RuntimeError):
    """Raised for any failed, partial or unverifiable acquisition step."""


# --------------------------------------------------------------------------------------------------- paths


@dataclass(frozen=True)
class DataPaths:
    data_root: Path  # raw / processed (audio; gitignored)
    meta_root: Path  # manifests/ (text; tracked in git)

    @classmethod
    def resolve(cls, data_root: str | os.PathLike | None = None, meta_root: str | os.PathLike | None = None) -> DataPaths:
        d = Path(data_root or os.environ.get("VAANI_DATA_ROOT") or (REPO_ROOT / "data")).expanduser().resolve()
        m = Path(meta_root or os.environ.get("VAANI_META_ROOT") or (REPO_ROOT / "data")).expanduser().resolve()
        return cls(d, m)

    def ensure(self) -> DataPaths:
        for sub in AUDIO_SUBDIRS:
            (self.data_root / sub).mkdir(parents=True, exist_ok=True)
        for d in (self.manifests_dir, self.provenance_dir, self.licence_texts_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def raw(self, source_id: str) -> Path:
        return self.data_root / "raw" / source_id

    def rel(self, path: Path) -> str:
        """Path relative to DATA_ROOT, POSIX style (the only form allowed in manifests)."""
        return Path(path).resolve().relative_to(self.data_root).as_posix()

    @property
    def manifests_dir(self) -> Path:
        return self.meta_root / "manifests"

    @property
    def provenance_dir(self) -> Path:
        return self.manifests_dir / "provenance"

    @property
    def licence_texts_dir(self) -> Path:
        return self.manifests_dir / "licence_texts"

    @property
    def logs_dir(self) -> Path:
        return self.data_root / "raw" / "_logs"  # local download logs (gitignored with the audio)

    @property
    def registry_path(self) -> Path:
        return self.manifests_dir / "licence_registry.yaml"


def add_path_args(parser) -> None:
    parser.add_argument("--data-root", default=None, help="audio root (default: $VAANI_DATA_ROOT or <repo>/data)")
    parser.add_argument("--meta-root", default=None, help="root holding manifests/ (default: <repo>/data)")


# --------------------------------------------------------------------------------------------------- logging


def setup_logging(script: str, paths: DataPaths, verbose: bool = False) -> tuple[logging.Logger, Path]:
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = paths.logs_dir / f"{script}_{ts}.log"
    parent = logging.getLogger("vaanishield.data")  # handlers live on the parent so run_cli errors are logged too
    parent.setLevel(logging.DEBUG)
    for h in list(parent.handlers):
        parent.removeHandler(h)
        h.close()
    logger = parent.getChild(script)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    parent.addHandler(fh)
    parent.addHandler(sh)
    logger.info("log file: %s", log_path)
    return logger, log_path


# --------------------------------------------------------------------------------------------------- hashing


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path, paths: DataPaths) -> dict[str, Any]:
    return {"path": paths.rel(path), "bytes": path.stat().st_size, "sha256": sha256_of(path)}


# --------------------------------------------------------------------------------------------------- HTTP


def _request(url: str, headers: dict[str, str] | None = None, timeout: float = 60.0):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 (https URLs from the licence registry)


def _auth_headers(token_env: str | None) -> dict[str, str]:
    if not token_env:
        return {}
    tok = os.environ.get(token_env)
    if not tok:
        raise AcquisitionError(f"this source needs a token in the environment variable {token_env} (never in code/config)")
    return {"Authorization": f"Bearer {tok}"}


def parse_content_range(value: str | None) -> tuple[int, int, int]:
    """'bytes 0-15/7640952520' -> (0, 15, 7640952520). Raises on anything else."""
    if not value or not value.startswith("bytes "):
        raise AcquisitionError(f"server did not return a Content-Range header (got {value!r})")
    span, total = value[6:].split("/", 1)
    start, end = span.split("-", 1)
    if total == "*":
        raise AcquisitionError("server did not report the total size")
    return int(start), int(end), int(total)


def probe_url(url: str, token_env: str | None = None, timeout: float = 60.0) -> dict[str, Any]:
    """GET bytes=0-0 (HEAD is unreliable on some hosts). Returns size, range support and validators."""
    try:
        with _request(url, {"Range": "bytes=0-0", **_auth_headers(token_env)}, timeout) as r:
            status = r.status
            h = r.headers
            final = r.geturl()
            r.read()
    except urllib.error.HTTPError as e:
        raise AcquisitionError(f"probe failed for {url}: HTTP {e.code} {e.reason}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise AcquisitionError(f"probe failed for {url}: {e}") from e
    info: dict[str, Any] = {
        "url": url,
        "final_url_host": urllib.request.urlparse(final).netloc,
        "status": status,
        "range_supported": status == 206,
        "etag": h.get("ETag"),
        "last_modified": h.get("Last-Modified"),
        "content_type": h.get("Content-Type"),
        "probed_at": datetime.now(UTC).isoformat(),
    }
    if status == 206:
        info["total_bytes"] = parse_content_range(h.get("Content-Range"))[2]
    else:
        cl = h.get("Content-Length")
        info["total_bytes"] = int(cl) if cl and cl.isdigit() else None
    return info


class HttpRangeReader(io.RawIOBase):
    """Seekable, read-only file object over HTTP range requests (block cache).

    Lets ``zipfile`` / ``pyarrow.parquet`` read single members / row groups of remote archives without
    downloading them. Every response is checked (206 + matching Content-Range); anything else raises.
    """

    def __init__(
        self,
        url: str,
        block_size: int = 1 << 20,
        max_cached_blocks: int = 64,
        retries: int = 4,
        timeout: float = 60.0,
        token_env: str | None = None,
    ) -> None:
        super().__init__()
        self.url = url
        self.block_size = block_size
        self.retries = retries
        self.timeout = timeout
        self._auth = _auth_headers(token_env)
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._max_cached = max_cached_blocks
        self._pos = 0
        self.bytes_fetched = 0
        self.requests = 0
        probe = probe_url(url, token_env, timeout)
        if not probe["range_supported"]:
            raise AcquisitionError(f"{url} does not support HTTP range requests")
        self.size: int = probe["total_bytes"]
        self.etag = probe["etag"]
        self.last_modified = probe["last_modified"]

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        new = base + offset
        if new < 0:
            raise ValueError("negative seek position")
        self._pos = new
        return new

    def _fetch(self, start: int, end: int) -> bytes:
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with _request(self.url, {"Range": f"bytes={start}-{end}", **self._auth}, self.timeout) as r:
                    if r.status != 206:
                        raise AcquisitionError(f"expected HTTP 206, got {r.status}")
                    s, e, total = parse_content_range(r.headers.get("Content-Range"))
                    if (s, e) != (start, end) or total != self.size:
                        raise AcquisitionError(f"Content-Range mismatch: asked {start}-{end}/{self.size}, got {s}-{e}/{total}")
                    data = r.read()
                if len(data) != end - start + 1:
                    raise AcquisitionError(f"short read: {len(data)} of {end - start + 1} bytes")
                self.bytes_fetched += len(data)
                self.requests += 1
                return data
            except (urllib.error.URLError, TimeoutError, OSError, AcquisitionError) as exc:
                last = exc
                time.sleep(min(2**attempt, 20))
        raise AcquisitionError(f"range request {start}-{end} failed after {self.retries} attempts: {last}")

    def _block(self, idx: int) -> bytes:
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        start = idx * self.block_size
        end = min(start + self.block_size, self.size) - 1
        data = self._fetch(start, end)
        self._cache[idx] = data
        if len(self._cache) > self._max_cached:
            self._cache.popitem(last=False)
        return data

    def readinto(self, b) -> int:
        if self._pos >= self.size:
            return 0
        n = min(len(b), self.size - self._pos)
        out = memoryview(b)
        if n > 2 * self.block_size:
            # Large sequential read (e.g. a zip central directory or a parquet column chunk): one request, no caching.
            data = self._fetch(self._pos, self._pos + n - 1)
            out[:n] = data
            self._pos += n
            return n
        done = 0
        while done < n:
            idx, off = divmod(self._pos, self.block_size)
            blk = self._block(idx)
            take = min(len(blk) - off, n - done)
            out[done : done + take] = blk[off : off + take]
            done += take
            self._pos += take
        return done


class CountingReader(io.RawIOBase):
    """Wraps a sequential stream and counts bytes (for streamed-pilot provenance)."""

    def __init__(self, raw) -> None:
        super().__init__()
        self._raw = raw
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        data = self._raw.read(len(b))
        n = len(data)
        b[:n] = data
        self.bytes_read += n
        return n


def open_stream(url: str, token_env: str | None = None, timeout: float = 120.0):
    try:
        return _request(url, _auth_headers(token_env), timeout)
    except urllib.error.HTTPError as e:
        raise AcquisitionError(f"GET {url}: HTTP {e.code} {e.reason}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise AcquisitionError(f"GET {url}: {e}") from e


def download_file(
    url: str,
    dest: Path,
    logger: logging.Logger,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    max_bytes: int | None = None,
    token_env: str | None = None,
    retries: int = 5,
) -> dict[str, Any]:
    """Resumable, verified download to ``dest`` (via ``dest.part``). Raises on any mismatch."""
    probe = probe_url(url, token_env)
    total = probe["total_bytes"]
    if total is None:
        raise AcquisitionError(f"{url}: unknown size; refusing an unverifiable download")
    if expected_bytes is not None and total != expected_bytes:
        raise AcquisitionError(f"{url}: server size {total} != expected {expected_bytes}")
    if max_bytes is not None and total > max_bytes:
        raise AcquisitionError(f"{url}: {total} bytes exceeds the allowed {max_bytes} (use the explicit full-download flag)")
    free = shutil.disk_usage(dest.parent if dest.parent.exists() else Path(dest.anchor)).free
    if total > free * 0.9:
        raise AcquisitionError(f"not enough disk space: need {total}, free {free}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > total:
            part.unlink()
            have = 0
        if have == total:
            break
        headers = {"Range": f"bytes={have}-", **_auth_headers(token_env)} if probe["range_supported"] else _auth_headers(token_env)
        if not probe["range_supported"] and have:
            part.unlink()
            have = 0
        try:
            with _request(url, headers, 120.0) as r:
                if have and r.status != 206:
                    raise AcquisitionError(f"resume requested but server answered {r.status}")
                with open(part, "ab" if have else "wb") as fh:
                    while True:
                        chunk = r.read(_CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
            logger.info("downloaded %s bytes of %s (attempt %d)", part.stat().st_size, total, attempt)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("download interrupted (attempt %d/%d): %s", attempt, retries, exc)
            time.sleep(min(2**attempt, 30))
    size = part.stat().st_size if part.exists() else 0
    if size != total:
        raise AcquisitionError(f"{url}: incomplete download ({size} of {total} bytes); .part kept for resume")
    digest = sha256_of(part)
    if expected_sha256 and digest != expected_sha256:
        raise AcquisitionError(f"{url}: sha256 mismatch ({digest} != {expected_sha256}); file kept as {part}")
    part.replace(dest)
    logger.info("verified %s sha256=%s", dest.name, digest)
    return {"url": url, "bytes": total, "sha256": digest, "etag": probe["etag"], "last_modified": probe["last_modified"]}


# --------------------------------------------------------------------------------------------------- registry


VALID_LICENCE_STATUS = {"verified", "conflict", "needs_verification"}


def load_registry(paths: DataPaths) -> dict[str, dict[str, Any]]:
    if not paths.registry_path.exists():
        raise AcquisitionError(f"licence registry missing: {paths.registry_path}")
    with open(paths.registry_path, encoding="utf-8") as fh:
        reg = yaml.safe_load(fh) or {}
    sources = reg.get("sources", {})
    if not isinstance(sources, dict) or not sources:
        raise AcquisitionError("licence registry has no sources")
    return sources


def require_source(paths: DataPaths, source_id: str, accept_terms: bool, logger: logging.Logger) -> dict[str, Any]:
    """Gate every acquisition on the licence registry (and an explicit, logged acknowledgement)."""
    reg = load_registry(paths)
    if source_id not in reg:
        raise AcquisitionError(f"{source_id} is not in the licence registry; add and verify it first")
    entry = reg[source_id]
    status = entry.get("licence_status")
    if status not in VALID_LICENCE_STATUS:
        raise AcquisitionError(f"{source_id}: invalid licence_status {status!r}")
    if entry.get("acquisition_allowed") is False:
        raise AcquisitionError(f"{source_id}: acquisition_allowed=false in the registry ({entry.get('blocker', '')})")
    if not accept_terms:
        raise AcquisitionError(
            f"{source_id}: re-run with --accept-terms after reading the terms in data/manifests/licence_registry.yaml and LICENSES_REGISTER.md "
            f"(licence: {entry.get('licence')}; status: {status})"
        )
    logger.info("terms acknowledged for %s: licence=%s status=%s", source_id, entry.get("licence"), status)
    if status != "verified":
        logger.warning("%s licence status is %s: %s", source_id, status, entry.get("licence_note", ""))
    return entry


# --------------------------------------------------------------------------------------------------- provenance


class ProvenanceStore:
    """Append-only JSON record per source: ``<meta_root>/manifests/provenance/<source_id>.json``."""

    def __init__(self, paths: DataPaths, source_id: str) -> None:
        self.path = paths.provenance_dir / f"{source_id}.json"
        self.source_id = source_id

    def load(self) -> dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"source_id": self.source_id, "events": []}

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        doc = self.load()
        git = git_state(REPO_ROOT)
        event = {
            "recorded_at": datetime.now(UTC).isoformat(),
            "git_commit": git.get("commit"),
            "git_dirty": git.get("dirty"),
            **event,
        }
        doc["events"].append(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.path)
        return event

    def stored_files(self) -> dict[str, dict[str, Any]]:
        """Latest record per stored file path across all events."""
        out: dict[str, dict[str, Any]] = {}
        for ev in self.load()["events"]:
            for f in ev.get("files", []):
                out[f["path"]] = f
        return out


def write_atomic_bytes(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)


def safe_member_path(base: Path, member: str) -> Path:
    """Join an archive member name under ``base``; refuses absolute paths and '..' (zip/tar slip)."""
    p = Path(member)
    if p.is_absolute() or any(part == ".." for part in p.parts) or member.startswith(("/", "\\")):
        raise AcquisitionError(f"unsafe archive member path: {member!r}")
    target = (base / p).resolve()
    if not str(target).startswith(str(base.resolve())):
        raise AcquisitionError(f"archive member escapes target dir: {member!r}")
    return target


def summarize_files(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    recs = list(records)
    return {"n_files": len(recs), "bytes": sum(r["bytes"] for r in recs)}


def run_cli(main_fn, argv=None) -> int:
    """Run a data-script ``main`` so that any AcquisitionError ends with a logged, non-zero exit."""
    try:
        return int(main_fn(argv) or 0)
    except AcquisitionError as exc:
        log = logging.getLogger("vaanishield.data")
        if log.handlers:
            log.error("FAILED: %s", exc)
        else:  # failure before logging was configured
            print(f"ERROR: {exc}")
        return 1
