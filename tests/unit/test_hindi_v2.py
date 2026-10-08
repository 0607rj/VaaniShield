"""Hindi repair: v2 preprocessing, split assignment, and the acceptance gate's verdict rules."""

import numpy as np
import soundfile as sf

from ml.data.audio import PREPROC_V2, active_rms_dbfs, preprocess_v2_wave
from ml.data.build_hindi_v2 import assign_splits
from ml.data.manifest import ManifestRow

SR = 16000


def tone(dc=0.0, amp=0.3, seconds=2.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    env = (t > 0.3) & (t < seconds - 0.3)
    return (dc + amp * env * np.sin(2 * np.pi * 220 * t) + 0.001 * rng.standard_normal(t.size)).astype(np.float32)


def test_v2_removes_dc_and_normalises_loudness_identically():
    quiet, loud = tone(dc=0.05, amp=0.05), tone(dc=-0.02, amp=0.9, seed=1)
    for x in (quiet, loud):
        y, info = preprocess_v2_wave(x)
        assert abs(float(y.mean())) < 1e-3
        assert abs(active_rms_dbfs(y) - PREPROC_V2["loudness_target_dbfs"]) < 0.5 or info["peak_limited"]
        assert np.abs(y).max() <= PREPROC_V2["peak_limit"] + 1e-6
    a, _ = preprocess_v2_wave(quiet)
    b, _ = preprocess_v2_wave(quiet.copy())
    assert np.array_equal(a, b)  # deterministic


def test_peak_limit_wins_over_loudness_target():
    x = np.zeros(SR * 2, np.float32)
    x[SR] = 1.0  # one click: reaching -26 dBFS RMS would need a huge gain
    x += 0.001 * np.random.default_rng(0).standard_normal(x.size).astype(np.float32)
    y, info = preprocess_v2_wave(x)
    assert info["peak_limited"] and np.abs(y).max() <= PREPROC_V2["peak_limit"] + 1e-6


def test_holdout_family_groups_go_to_test_only():
    groups = {f"g{i}": {"n": 10 + i, "families": {"xtts"}} for i in range(8)}
    groups["g_vc1"] = {"n": 30, "families": {"freevc", "xtts"}}
    groups["g_vc2"] = {"n": 5, "families": {"freevc"}}
    s = assign_splits(groups, "freevc", {"train": 0.6, "dev": 0.2, "test": 0.2}, seed=1)
    assert s["g_vc1"] == s["g_vc2"] == "test"
    assert {s[g] for g in groups if g.startswith("g") and not g.startswith("g_vc")} == {"train", "dev", "test"}
    assert s == assign_splits(groups, "freevc", {"train": 0.6, "dev": 0.2, "test": 0.2}, seed=1)


def _row(cid, label, split, group, path, fmt_family):
    return ManifestRow(clip_id=cid, path=path, label=label, source_dataset="kathbath_hi" if label == "bonafide" else "indicsynth_hi",
                       speaker_id=group, speaker_group=group, language="hi", transcript_id=f"t_{cid}",
                       generator="none" if label == "bonafide" else "xtts_v2", generator_family="none" if label == "bonafide" else fmt_family,
                       sha256=__import__("hashlib").sha256(cid.encode()).hexdigest(), licence="t", split=split)


def test_acceptance_gate_fails_on_format_split_and_missing_preprocessing_record(tmp_path):
    from ml.data.hindi_gate import _file_level, acceptance

    rows = []
    for i, sp in enumerate(["train"] * 4 + ["dev"] * 2 + ["test"] * 2):
        g = f"grp{i}"
        for lab, rate in (("bonafide", 24000), ("spoof", 16000)):  # deliberately different sample rates by class
            p = tmp_path / f"{lab}_{i}.wav"
            sf.write(p, tone(seed=i + (lab == "spoof") * 100, seconds=2.0)[: int(2.0 * rate)], rate, subtype="PCM_16")
            rows.append(_row(f"{lab}_{i}", lab, sp, g, p.name, "xtts"))
    cfg = {"acceptance": {"metadata_min_eer": 0.4, "duration_min_eer": 0.4, "silence_min_eer": 0.4, "level_min_eer": 0.4,
                          "loudness_tolerance_db": 0.5, "max_limiter_fraction_gap": 0.01, "max_abs_dc": 0.001, "class_ratio_range": [0.5, 2.0]},
           "protocol": {"holdout_family": None}}
    fl = _file_level(rows, tmp_path)
    acc = acceptance(rows, tmp_path, cfg, metadata=None, header=None, fl=fl)
    assert not acc["A2_format_samplerate_codec_bitrate_identical"]["ok"]
    assert not acc["A12_identical_preprocessing"]["ok"]  # no preprocessing record -> cannot be shown identical
    assert not all(c["ok"] for c in acc.values())  # -> ACCEPTANCE_FAIL; the gate has no READY verdict at all


def test_v2_1_has_no_limiter_and_one_linear_gain():
    from ml.data.audio import PREPROC_V2_1, preprocess_wave

    x = tone(amp=0.05)
    x[SR] = 0.99  # strong transient that the v2 limiter would have caught
    y, info = preprocess_wave(x, PREPROC_V2_1)
    assert PREPROC_V2_1["peak_limit"] is None and not info["peak_limited"]
    assert abs(active_rms_dbfs(y) - PREPROC_V2_1["loudness_target_dbfs"]) < 0.5
    assert np.abs(y).max() > 0.95  # allowed: stored as float32, never clipped
