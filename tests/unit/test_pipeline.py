"""Tests for the first vertical slice: audio preprocessing, split, model, checkpoint round-trip + inference."""

import numpy as np
import pytest
import soundfile as sf
import torch

from ml.data.audio import CROP_SAMPLES, crop_or_pad, load_audio, trim_silence, windows
from ml.data.build_manifest import check_split, split_by_speaker
from ml.data.manifest import ManifestRow
from ml.inference.predict import load_checkpoint, predict_file
from ml.models import build_model


def test_load_audio_makes_mono_16k(tmp_path):
    stereo = np.random.default_rng(0).uniform(-0.5, 0.5, size=(48000, 2)).astype(np.float32)
    path = tmp_path / "x.wav"
    sf.write(path, stereo, 48000)
    wave = load_audio(path)
    assert wave.ndim == 1 and wave.dtype == np.float32 and abs(wave.size - 16000) <= 1


def test_trim_silence_removes_edges():
    tone = np.sin(np.arange(16000) / 5).astype(np.float32)
    wave = np.concatenate([np.zeros(8000, np.float32), tone, np.zeros(8000, np.float32)])
    assert abs(trim_silence(wave).size - tone.size) <= 400  # within one 25 ms frame


def test_crop_or_pad_lengths_and_determinism():
    short = np.arange(1000, dtype=np.float32)
    assert crop_or_pad(short).size == CROP_SAMPLES and crop_or_pad(short)[1000] == 0  # tiled, not zero-padded
    long = np.arange(100000, dtype=np.float32)
    assert np.array_equal(crop_or_pad(long), crop_or_pad(long))  # centre crop is deterministic
    a = crop_or_pad(long, rng=np.random.default_rng(1))
    b = crop_or_pad(long, rng=np.random.default_rng(1))
    assert np.array_equal(a, b)


def test_windows_cover_whole_clip():
    wave = np.arange(150000, dtype=np.float32)
    ws = windows(wave)
    assert all(w.size == CROP_SAMPLES for w in ws) and ws[-1][-1] == wave[-1] and ws[0][0] == wave[0]


def _row(i, spk, label):
    return ManifestRow(clip_id=f"c{i}", path=f"raw/x/c{i}.flac", label=label, source_dataset="t", speaker_id=spk,
                       speaker_group=spk, language="en", generator="none" if label == "bonafide" else "A01",
                       generator_family="none" if label == "bonafide" else "fam", sha256="0" * 64, licence="t")


def test_split_is_speaker_disjoint_with_both_classes():
    rows = [_row(i, f"s{i % 10}", "bonafide" if i % 2 else "spoof") for i in range(40)]
    out, _ = split_by_speaker(rows)
    check_split(out)  # raises on speaker overlap
    for sp in ("train", "dev", "test"):
        assert {r.label for r in out if r.split == sp} == {"bonafide", "spoof"}


def test_check_split_detects_leakage():
    rows = [_row(0, "s1", "bonafide").model_copy(update={"split": "train"}),
            _row(1, "s1", "spoof").model_copy(update={"split": "test"})]
    with pytest.raises(ValueError):
        check_split(rows)


def test_model_shape():
    model = build_model("lfcc_cnn")
    assert model(torch.randn(2, CROP_SAMPLES)).shape == (2, 2)


def test_checkpoint_roundtrip_and_predict(tmp_path):
    torch.manual_seed(0)
    model = build_model("lfcc_cnn").eval()
    ckpt = {"model_kind": "lfcc_cnn", "model_config": model.config, "model_state": model.state_dict(),
            "sample_rate": 16000, "crop_samples": CROP_SAMPLES, "threshold_p_spoof": 0.5, "manifest": "test"}
    torch.save(ckpt, tmp_path / "best.pt")
    loaded, meta = load_checkpoint(tmp_path / "best.pt")
    x = torch.randn(1, CROP_SAMPLES)
    assert torch.allclose(model(x), loaded(x))
    audio = tmp_path / "a.wav"
    sf.write(audio, (0.1 * np.random.default_rng(0).standard_normal(40000)).astype(np.float32), 16000)
    out = predict_file(loaded, meta, audio)
    assert out["label"] in {"REAL", "SPOOF"} and 0 <= out["p_spoof"] <= 1
    assert out["label"] == ("SPOOF" if out["p_spoof"] >= 0.5 else "REAL")


def test_aasist_wrapper_swaps_upstream_label_order():
    """Upstream AASIST: column 1 = bona fide. Wrapper must expose column 1 = spoof."""
    torch.manual_seed(0)
    model = build_model("aasist").eval()
    x = torch.randn(2, CROP_SAMPLES)
    with torch.no_grad():
        _, upstream = model.net(x)
        ours = model(x)
    assert ours.shape == (2, 2)
    assert torch.equal(ours[:, 1], upstream[:, 0]) and torch.equal(ours[:, 0], upstream[:, 1])


def test_select_rows_subset_is_deterministic_stratified_and_source_filtered(tmp_path):
    from ml.data.dataset import select_rows
    from ml.data.manifest import write_manifest

    rows = [_row(i, f"s{i % 8}", "bonafide" if i % 4 else "spoof").model_copy(
        update={"split": "train", "source_dataset": "a" if i < 30 else "b", "codec": "flac"}) for i in range(40)]
    m = tmp_path / "m.csv"
    write_manifest(rows, m)
    a = select_rows(m, "train", ["a"], max_clips=12, seed=1)
    b = select_rows(m, "train", ["a"], max_clips=12, seed=1)
    assert [r.clip_id for r in a] == [r.clip_id for r in b] and len(a) == 12
    assert {r.source_dataset for r in a} == {"a"} and {r.label for r in a} == {"bonafide", "spoof"}
    assert [r.clip_id for r in select_rows(m, "train", ["a"], max_clips=12, seed=2)] != [r.clip_id for r in a]


def test_config_overrides(tmp_path):
    from ml.training.config import load_config

    p = tmp_path / "c.yaml"
    p.write_text("exp_id: X\nmodel: {kind: lfcc_cnn}\ntrain: {seed: 1, device: cuda}\n", encoding="utf-8")
    cfg = load_config(p, {"train.device": "cpu", "data.num_workers": 2, "output.results_dir": None})
    assert cfg.train.device == "cpu" and cfg.data.num_workers == 2 and cfg.output.results_dir == "runs"


TINY_W2V = {"hidden_size": 32, "num_hidden_layers": 3, "num_attention_heads": 2, "intermediate_size": 64,
            "conv_dim": [16] * 7, "do_stable_layer_norm": True, "feat_extract_norm": "layer"}


def test_xlsr_frozen_forward_gradients_and_compact_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = build_model("xlsr_mlp", {"encoder_config": TINY_W2V, "adapter_dim": 8})
    x = torch.randn(2, 16000)
    out = model(x)
    assert out.shape == (2, 2) and model.n_hidden == 4
    torch.nn.functional.cross_entropy(out, torch.tensor([0, 1])).backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert model.layer_logits.grad is not None and model.head[1].weight.grad.abs().sum() > 0
    state = model.checkpoint_state()
    assert not any(k.startswith("encoder.") for k in state)  # frozen encoder is not stored
    model.eval()
    ckpt = {"model_kind": "xlsr_mlp", "model_config": model.config, "model_state": state}
    torch.save(ckpt, tmp_path / "x.pt")
    from ml.inference.predict import load_checkpoint

    torch.manual_seed(0)  # same random tiny encoder as above (stands in for the pretrained one)
    loaded, _ = load_checkpoint(tmp_path / "x.pt")
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-6)


def test_xlsr_unfreeze_top_k_trains_only_top_layers():
    torch.manual_seed(0)
    model = build_model("xlsr_mlp", {"encoder_config": TINY_W2V, "freeze": False, "unfreeze_top_k": 1, "adapter_dim": 8})
    torch.nn.functional.cross_entropy(model(torch.randn(2, 16000)), torch.tensor([0, 1])).backward()
    layers = model.encoder.encoder.layers
    assert all(p.grad is not None for p in layers[-1].parameters())
    assert all(p.grad is None for p in layers[0].parameters()) and all(p.grad is None for p in model.encoder.feature_extractor.parameters())
    assert any(k.startswith("encoder.encoder.layers.2.") for k in model.checkpoint_state())  # unfrozen layer is stored
