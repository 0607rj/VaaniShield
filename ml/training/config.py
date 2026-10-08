"""Typed experiment configuration (docs/architecture.md M01).

YAML -> pydantic models with ``extra="forbid"``, so a typo in a config key fails loudly instead of being
silently ignored. ``config_hash`` is computed over canonical JSON and is independent of key order.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CodecSpec(_Strict):
    name: Literal["opus", "mp3", "amr_nb", "amr_wb", "speex_wb", "narrowband_sim"]
    bitrates_kbps: list[int] = Field(min_length=1)


class LangMixCfg(_Strict):
    crossfade_ms: tuple[int, int] = (10, 30)
    require_different_speaker_group: bool = True  # TD-12: symmetric, cross-voice for BOTH classes
    match_rms: bool = True


class DataCfg(_Strict):
    train_manifests: list[str] = []
    dev_manifests: list[str] = []
    test_manifests: list[str] = []
    crop_samples: int = Field(default=64600, gt=0)  # DEC-10 / TD-3
    sample_rate: Literal[16000] = 16000
    num_workers: int = Field(default=4, ge=0)
    sampler: Literal["none", "class_balanced", "class_source_balanced"] = "none"
    # Portability (PROJECT_STATE §0.12). Paths may use env vars, e.g. "${VAANI_DATA_ROOT}"; relative = repo root / CWD.
    data_root: str | None = None  # audio root; None -> $VAANI_DATA_ROOT or <repo>/data
    include_sources: list[str] = []  # empty = every source_dataset in the manifest
    max_clips_per_split: dict[Literal["train", "dev", "test"], int] = {}  # deterministic, label-stratified subset
    preload_audio: bool = True  # False = read each file when needed (large datasets)


class AugCfg(_Strict):
    enabled: bool = False
    p_codec: float = Field(default=0.0, ge=0.0, le=1.0)
    codecs: list[CodecSpec] = []
    p_rawboost: float = Field(default=0.0, ge=0.0, le=1.0)
    rawboost_algo: list[int] = []
    p_langmix: float = Field(default=0.0, ge=0.0, le=1.0)
    langmix: LangMixCfg | None = None
    p_resample: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _consistent(self) -> AugCfg:
        if self.p_codec > 0 and not self.codecs:
            raise ValueError("p_codec > 0 requires at least one codec in `codecs`")
        if self.p_langmix > 0 and self.langmix is None:
            raise ValueError("p_langmix > 0 requires a `langmix` section")
        if any(a < 1 or a > 8 for a in self.rawboost_algo):
            raise ValueError("rawboost_algo values must be in 1..8")
        return self


class ModelCfg(_Strict):
    kind: Literal["lfcc_cnn", "lfcc_lcnn", "aasist", "xlsr_mlp", "xlsr_aasist_lite", "w2v2en_aasist_lite", "dummy"]
    params: dict = {}  # keyword arguments for the model class (e.g. {variant: AASIST-L} or {channels: [16, 32, 32]})
    encoder_id: str | None = None
    encoder_revision: str | None = None
    freeze_encoder: bool = True
    unfreeze_top_k: int = Field(default=0, ge=0)
    aggregation: Literal["weighted", "last", "fixed_layer"] = "weighted"
    fixed_layer: int | None = None
    adapter_dim: int = Field(default=128, gt=0)
    truncate_after_layer: int | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ModelCfg:
        ssl = self.kind in {"xlsr_mlp", "xlsr_aasist_lite", "w2v2en_aasist_lite"}
        if ssl and not self.encoder_id:
            raise ValueError(f"model.kind={self.kind} requires encoder_id")
        if self.aggregation == "fixed_layer" and self.fixed_layer is None:
            raise ValueError("aggregation=fixed_layer requires fixed_layer")
        if self.freeze_encoder and self.unfreeze_top_k > 0:
            raise ValueError("unfreeze_top_k > 0 contradicts freeze_encoder=true")
        return self


class TrainCfg(_Strict):
    seed: int
    device: Literal["auto", "cpu", "cuda"] = "auto"
    epochs: int = Field(default=20, gt=0)
    batch_size: int = Field(default=8, gt=0)
    grad_accum: int = Field(default=4, gt=0)
    lr_head: float = Field(default=1e-4, gt=0)
    lr_encoder: float = Field(default=1e-6, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    amp: bool = True
    early_stop_patience: int = Field(default=5, gt=0)
    class_weights: tuple[float, float] | None = None
    max_grad_norm: float = Field(default=1.0, gt=0)


class OutputCfg(_Strict):
    checkpoint_dir: str = "runs"  # <checkpoint_dir>/<exp_id>/best.pt
    results_dir: str = "runs"  # <results_dir>/<exp_id>/{history.json, dev_scores.csv, <split>_metrics.json, ...}


class CalibCfg(_Strict):
    method: Literal["none", "global_temp", "conditional_temp"] = "global_temp"
    dev_conditions: list[str] = ["clean"]


class ExperimentCfg(_Strict):
    exp_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    hypothesis_ids: list[str] = []
    data: DataCfg = DataCfg()
    aug: AugCfg = AugCfg()
    model: ModelCfg
    train: TrainCfg
    calib: CalibCfg = CalibCfg()
    output: OutputCfg = OutputCfg()

    @field_validator("hypothesis_ids")
    @classmethod
    def _hyp_ids(cls, v: list[str]) -> list[str]:
        for h in v:
            if not (h.startswith("H") or h.startswith("E")):
                raise ValueError(f"hypothesis id {h!r} must look like H1..H13 or E1 (exploratory)")
        return v


def load_config(path: str | Path, overrides: dict[str, object] | None = None) -> ExperimentCfg:
    """Load YAML, apply optional dotted overrides (e.g. {"train.device": "cpu"}; None values are ignored), validate."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level of a config must be a mapping")
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        node = raw
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return ExperimentCfg.model_validate(raw)


def canonical_json(cfg: BaseModel) -> str:
    return json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(cfg: BaseModel) -> str:
    """SHA-256 of the canonical JSON form (key-order independent, defaults included)."""
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()


def dump_config(cfg: BaseModel, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.model_dump(mode="json"), fh, sort_keys=True, allow_unicode=True)
