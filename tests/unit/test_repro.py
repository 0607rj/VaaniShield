import numpy as np
import pytest

from ml.training.repro import (
    git_state,
    item_rng,
    item_seed,
    seed_everything,
    sha256_file,
    sha256_manifest,
)


def _write(path, text):
    path.write_text(text, encoding="utf-8", newline="")
    return path


def test_manifest_hash_invariant_to_row_and_column_order(tmp_path):
    a = _write(tmp_path / "a.csv", "clip_id,label,path\nc1,spoof,x/1.wav\nc2,bonafide,x/2.wav\n")
    b = _write(tmp_path / "b.csv", "path,clip_id,label\r\nx/2.wav,c2,bonafide\r\nx/1.wav,c1,spoof\r\n")
    assert sha256_manifest(a) == sha256_manifest(b)


def test_manifest_hash_detects_content_change(tmp_path):
    a = _write(tmp_path / "a.csv", "clip_id,label\nc1,spoof\n")
    b = _write(tmp_path / "b.csv", "clip_id,label\nc1,bonafide\n")
    assert sha256_manifest(a) != sha256_manifest(b)


def test_manifest_hash_requires_key(tmp_path):
    with pytest.raises(ValueError):
        sha256_manifest(_write(tmp_path / "a.csv", "id,label\n1,spoof\n"))


def test_sha256_file_known_value(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_item_seed_stable_and_distinct():
    assert item_seed(2026, 0, "clip_a") == item_seed(2026, 0, "clip_a")
    assert item_seed(2026, 0, "clip_a") != item_seed(2026, 1, "clip_a")
    assert item_seed(2026, 0, "clip_a") != item_seed(2027, 0, "clip_a")
    a = item_rng(1, 2, "x").integers(0, 1_000_000, 5)
    b = item_rng(1, 2, "x").integers(0, 1_000_000, 5)
    assert np.array_equal(a, b)


def test_seed_everything_makes_numpy_reproducible():
    seed_everything(123)
    a = np.random.rand(3)
    seed_everything(123)
    assert np.array_equal(a, np.random.rand(3))


def test_seed_everything_rejects_bad_seed():
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_git_state_outside_repo(tmp_path):
    st = git_state(tmp_path)
    assert st["available"] is False and st["commit"] is None
