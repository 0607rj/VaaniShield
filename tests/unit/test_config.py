import pytest
import yaml
from pydantic import ValidationError

from ml.training.config import ExperimentCfg, config_hash, dump_config, load_config

MINIMAL = {"exp_id": "E0_smoke", "model": {"kind": "dummy"}, "train": {"seed": 2026}}


def test_defaults_follow_decisions():
    cfg = ExperimentCfg.model_validate(MINIMAL)
    assert cfg.data.crop_samples == 64600 and cfg.data.sample_rate == 16000
    assert cfg.aug.enabled is False
    assert cfg.calib.method == "global_temp"


def test_hash_independent_of_key_order(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("exp_id: E0_smoke\nmodel: {kind: dummy}\ntrain: {seed: 2026, epochs: 3}\n", encoding="utf-8")
    b.write_text("train: {epochs: 3, seed: 2026}\nmodel: {kind: dummy}\nexp_id: E0_smoke\n", encoding="utf-8")
    assert config_hash(load_config(a)) == config_hash(load_config(b))


def test_hash_changes_with_content():
    other = {**MINIMAL, "train": {"seed": 2027}}
    assert config_hash(ExperimentCfg.model_validate(MINIMAL)) != config_hash(
        ExperimentCfg.model_validate(other)
    )


def test_dump_roundtrip(tmp_path):
    cfg = ExperimentCfg.model_validate(MINIMAL)
    out = tmp_path / "rt.yaml"
    dump_config(cfg, out)
    assert config_hash(load_config(out)) == config_hash(cfg)


@pytest.mark.parametrize(
    "patch",
    [
        {"unknown_key": 1},  # extra keys are forbidden (typo protection)
        {"model": {"kind": "xlsr_aasist_lite"}},  # SSL model without encoder_id
        {"model": {"kind": "dummy", "aggregation": "fixed_layer"}},  # missing fixed_layer
        {"model": {"kind": "dummy", "unfreeze_top_k": 2}},  # contradicts freeze_encoder=true
        {"aug": {"p_codec": 0.5}},  # codec prob without codecs
        {"aug": {"p_langmix": 0.5}},  # langmix prob without langmix section
        {"aug": {"rawboost_algo": [9]}},
        {"aug": {"p_codec": 1.5, "codecs": [{"name": "opus", "bitrates_kbps": [16]}]}},
        {"data": {"sample_rate": 22050}},
        {"exp_id": "bad id with spaces"},
        {"hypothesis_ids": ["Q1"]},
    ],
)
def test_invalid_configs_rejected(patch):
    raw = {**MINIMAL, **patch}
    with pytest.raises(ValidationError):
        ExperimentCfg.model_validate(raw)


def test_top_level_must_be_mapping(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text(yaml.safe_dump([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)
