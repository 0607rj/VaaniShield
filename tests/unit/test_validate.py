"""The leakage checks must FAIL on deliberately broken data (otherwise a PASS means nothing)."""

from ml.data.manifest import ManifestRow
from ml.evaluation.validate import check_descendants, check_source_leakage, check_speaker_leakage, check_split


def row(cid, spk, label, split, source="src", parent=None, transform="clean"):
    return ManifestRow(clip_id=cid, path=f"raw/x/{cid}.flac", label=label, source_dataset=source, speaker_id=spk,
                       speaker_group=spk, language="en", generator="none" if label == "bonafide" else "A01",
                       generator_family="none" if label == "bonafide" else "fam", sha256="0" * 64, licence="t",
                       split=split, parent_clip_id=parent, transform=transform)


GOOD = [row("a", "s1", "bonafide", "train"), row("b", "s1", "spoof", "train"),
        row("c", "s2", "bonafide", "dev"), row("d", "s2", "spoof", "dev"),
        row("e", "s3", "bonafide", "test"), row("f", "s3", "spoof", "test")]


def test_good_data_passes():
    assert check_split(GOOD)["ok"] and check_speaker_leakage(GOOD)["ok"]
    assert check_source_leakage(GOOD)["ok"] and check_descendants(GOOD)["ok"]


def test_speaker_leakage_detected():
    bad = GOOD + [row("g", "s1", "spoof", "test")]
    assert not check_speaker_leakage(bad)["ok"]


def test_single_class_source_detected():
    bad = GOOD + [row("g", "s4", "spoof", "train", source="tts_only"), row("h", "s5", "spoof", "test", source="tts_only")]
    assert not check_source_leakage(bad)["ok"]


def test_missing_class_in_split_detected():
    assert not check_split([r for r in GOOD if not (r.split == "dev" and r.label == "spoof")])["ok"]


def test_derived_row_crossing_split_detected():
    bad = GOOD + [row("a_opus", "s1", "bonafide", "test", parent="a", transform="sim:opus:16k")]
    assert not check_descendants(bad)["ok"]
    ok = GOOD + [row("a_opus", "s1", "bonafide", "train", parent="a", transform="sim:opus:16k")]
    assert check_descendants(ok)["ok"]


def test_generator_input_crossing_split_detected():
    from ml.data.check_manifest import generator_leakage

    real = row("r1", "kathbath:7", "bonafide", "train")
    fake_same_split = row("f1", "kathbath:7", "spoof", "train")
    fake_other_split = row("f2", "kathbath:7", "spoof", "test")
    assert generator_leakage([real, fake_same_split])["ok"]
    assert not generator_leakage([real, fake_other_split])["ok"]  # cloned voice's real audio is in train


def test_codec_shortcut_warning():
    from ml.data.check_manifest import shortcut_risks

    a = row("a", "s1", "bonafide", "train").model_copy(update={"codec": "aac", "language": "hi"})
    b = row("b", "s2", "spoof", "train").model_copy(update={"codec": "pcm_s16le", "language": "hi"})
    assert any("codec perfectly separates" in w for w in shortcut_risks([a, b])["warnings"])
    c = row("c", "s3", "spoof", "train").model_copy(update={"codec": "aac", "language": "hi"})
    assert not any("codec perfectly separates" in w for w in shortcut_risks([a, b, c])["warnings"])




def test_declared_eval_only_source_is_checked_when_used_for_training():
    from ml.data.check_manifest import shortcut_risks

    real = row("a", "s1", "bonafide", "train", source="openslr104_hien").model_copy(update={"codec": "pcm_s16le"})
    fake = row("b", "s2", "spoof", "train", source="tts").model_copy(update={"codec": "pcm_s16le"})
    w = shortcut_risks([real, fake])["warnings"]
    assert any("openslr104_hien is single-class (bonafide); USED FOR TRAINING" in x for x in w)
    assert any("source_dataset perfectly separates" in x for x in w)  # not hidden from the confound analysis
    test_only = real.model_copy(update={"split": "test"})
    assert any("declared eval-only" in x for x in shortcut_risks([test_only, fake])["warnings"])
def test_segments_of_external_recording_must_share_split():
    seg = lambda cid, sp: row(cid, "s9", "bonafide", sp, parent="rec_1")  # noqa: E731
    assert check_descendants(GOOD + [seg("x1", "test"), seg("x2", "test")])["ok"]
    assert not check_descendants(GOOD + [seg("x1", "test"), seg("x2", "train")])["ok"]
