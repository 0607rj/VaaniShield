"""Codec round-trips: same length, delay removed, deterministic, honest about missing encoders, held-out rule."""

import numpy as np
import pytest

from ml.data.codec import SR, CodecUnavailable, Condition, align, apply

OPUS = Condition(id="opus16", kind="codec", role="train_aug", encoder="libopus", container="ogg", bitrate=16000,
                 options={"application": "voip"})
AAC = Condition(id="aac24", kind="codec", role="eval_heldout", encoder="aac", container="adts", bitrate=24000)
NB = Condition(id="rs8k", kind="resample", role="train_aug", codec_sample_rate=8000)


def speechlike(seconds=1.5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # syllable-rate envelope, so cross-correlation has a unique peak
    return (0.2 * env * (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 1700 * t)) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def test_align_recovers_known_delay():
    x = speechlike()
    y = np.r_[np.zeros(500, np.float32), x][: len(x) + 300]
    out, lag = align(x, y, max_lag=2000)
    assert lag == 500 and len(out) == len(x)
    assert np.allclose(out[:-500], x[:-500])


@pytest.mark.parametrize("cond", [OPUS, AAC, NB])
def test_roundtrip_same_length_aligned_and_deterministic(cond):
    x = speechlike()
    y1, info = apply(x, cond)
    y2, _ = apply(x, cond)
    assert len(y1) == len(x) and y1.dtype == np.float32
    assert np.array_equal(y1, y2)
    assert np.corrcoef(x, y1)[0, 1] > 0.8  # delay compensated (an unaligned codec output correlates poorly)
    if cond.kind == "codec":
        assert info["bitrate_actual_bps"] == pytest.approx(cond.bitrate, rel=0.35)


def test_narrowband_removes_upper_band():
    x = speechlike() + 0.1 * np.sin(2 * np.pi * 6000 * np.arange(int(1.5 * SR)) / SR).astype(np.float32)
    y, _ = apply(x, NB)
    p = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    assert p[f > 4200].sum() < 1e-4 * p.sum()


def test_missing_encoder_is_reported_not_substituted():
    c = Condition(id="nope", kind="codec", role="train_aug", encoder="definitely_not_an_encoder", container="wav")
    with pytest.raises(CodecUnavailable):
        apply(speechlike(), c)


def test_params_hash_tracks_settings_not_labels():
    assert OPUS.params_hash() != OPUS.model_copy(update={"bitrate": 24000}).params_hash()
    assert OPUS.params_hash() == OPUS.model_copy(update={"description": "x", "role": "eval_heldout"}).params_hash()


def test_heldout_codecs_never_applied_to_train(tmp_path):
    import soundfile as sf

    from ml.data.build_codec_variants import _one
    from ml.data.manifest import ManifestRow

    sf.write(tmp_path / "a.wav", speechlike(), SR, subtype="PCM_16")
    for split, expected in (("train", {"opus16", "rs8k"}), ("test", {"opus16", "aac24", "rs8k"})):
        p = ManifestRow(clip_id=f"c_{split}", path="a.wav", label="bonafide", source_dataset="s", speaker_id="k",
                        speaker_group="k", language="hi-en", generator="none", generator_family="none", sha256="0" * 64,
                        licence="t", split=split)
        recs = _one((p.model_dump(), [OPUS, AAC, NB], str(tmp_path), "sim", False))
        assert {r["condition"] for r in recs} == expected
        assert all(r["parent_clip_id"] == p.clip_id and r["split"] == split and r["source_dataset"] == "s" for r in recs)
