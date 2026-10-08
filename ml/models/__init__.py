"""Model factory: one place that maps a config ``kind`` to a model class."""

from torch import nn


def build_model(kind: str, model_config: dict | None = None) -> nn.Module:
    if kind == "lfcc_cnn":
        from ml.models.lfcc_cnn import LFCCCNN

        return LFCCCNN(**(model_config or {}))
    if kind == "aasist":
        from ml.models.aasist import AASISTDetector

        return AASISTDetector(**(model_config or {}))
    if kind == "xlsr_mlp":
        from ml.models.xlsr import XLSRDetector

        return XLSRDetector(**(model_config or {}))
    raise ValueError(f"model kind {kind!r} is not implemented yet")


def model_params(model_cfg) -> dict:
    """Constructor kwargs from a ModelCfg: SSL encoder fields of the schema + free-form `params` (params win)."""
    if model_cfg.kind == "xlsr_mlp":
        base = {"hf_id": model_cfg.encoder_id, "revision": model_cfg.encoder_revision, "freeze": model_cfg.freeze_encoder,
                "unfreeze_top_k": model_cfg.unfreeze_top_k, "truncate_after_layer": model_cfg.truncate_after_layer,
                "adapter_dim": model_cfg.adapter_dim}
        return {**base, **model_cfg.params}
    return dict(model_cfg.params)


def checkpoint_state(model: nn.Module) -> dict:
    """Weights to store: compact if the model defines it (frozen XLS-R), else the full state dict (CPU tensors)."""
    if hasattr(model, "checkpoint_state"):
        return model.checkpoint_state()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_weights(model: nn.Module, state: dict) -> None:
    if hasattr(model, "load_checkpoint_state"):
        model.load_checkpoint_state(state)
    else:
        model.load_state_dict(state)
