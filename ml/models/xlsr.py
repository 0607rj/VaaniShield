"""XLS-R baseline (P1): wav2vec 2.0 XLS-R-300M encoder -> learned layer weighting -> attentive-stats pooling -> 2 logits.

    wave [B, L] 16 kHz -> per-utterance normalisation (as the checkpoint's feature extractor: do_normalize=True)
      -> XLS-R hidden states [B, K, T, 1024]   (K = CNN-projection output + transformer layers kept)
      -> softmax-weighted sum over K        [B, T, 1024]   (weights logged; architecture.md M10)
      -> Linear 1024->adapter_dim + SELU    [B, T, 128]
      -> attentive statistics pooling       [B, 256]
      -> Linear -> logits [B, 2] (index 1 = spoof)

Encoder options (config `model.params`): freeze (default), unfreeze_top_k transformer layers, truncate_after_layer.
The CNN feature extractor is always frozen. The encoder always runs in eval mode (no dropout / layerdrop / SpecAugment),
so a frozen run is deterministic; fine-tuned layers are trained without those regularisers (documented choice).

Checkpoints store only trainable parameters (head + unfrozen layers) and buffers, plus the HF id and revision; the frozen
encoder is reloaded from the Hugging Face cache (DEC-13). Weights: facebook/wav2vec2-xls-r-300m, Apache-2.0.
"""

from __future__ import annotations

import torch
from torch import nn

DEFAULT_ID = "facebook/wav2vec2-xls-r-300m"
DEFAULT_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"


class AttentiveStatsPool(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D] -> [B, 2D]
        w = torch.softmax(self.attn(x), dim=1)
        mean = (w * x).sum(1)
        std = torch.sqrt(((w * (x - mean.unsqueeze(1)) ** 2).sum(1)).clamp(min=1e-6))
        return torch.cat([mean, std], dim=-1)


class XLSRDetector(nn.Module):
    def __init__(self, hf_id: str = DEFAULT_ID, revision: str | None = DEFAULT_REVISION, freeze: bool = True,
                 unfreeze_top_k: int = 0, truncate_after_layer: int | None = None, adapter_dim: int = 128,
                 dropout: float = 0.2, encoder_config: dict | None = None):
        super().__init__()
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        regularisers_off = {"apply_spec_augment": False, "layerdrop": 0.0, "mask_time_prob": 0.0}
        if encoder_config is not None:  # tiny randomly initialised encoder: unit tests only (no download)
            self.encoder = Wav2Vec2Model(Wav2Vec2Config(**{**encoder_config, **regularisers_off}))
        else:
            self.encoder = Wav2Vec2Model.from_pretrained(hf_id, revision=revision, **regularisers_off)
        n_layers = len(self.encoder.encoder.layers)
        if truncate_after_layer is not None:
            if not 1 <= truncate_after_layer <= n_layers:
                raise ValueError(f"truncate_after_layer must be in 1..{n_layers}")
            self.encoder.encoder.layers = self.encoder.encoder.layers[:truncate_after_layer]
            n_layers = truncate_after_layer
        if unfreeze_top_k and freeze:
            raise ValueError("unfreeze_top_k > 0 requires freeze=False")
        if not 0 <= unfreeze_top_k <= n_layers:
            raise ValueError(f"unfreeze_top_k must be in 0..{n_layers}")
        self.encoder.requires_grad_(False)
        if not freeze:
            k = unfreeze_top_k or n_layers  # freeze=False and k=0 -> all transformer layers trainable
            for layer in self.encoder.encoder.layers[n_layers - k :]:
                layer.requires_grad_(True)
        self.n_hidden = n_layers + 1
        dim = self.encoder.config.hidden_size
        self.layer_logits = nn.Parameter(torch.zeros(self.n_hidden))
        self.adapter = nn.Sequential(nn.Linear(dim, adapter_dim), nn.SELU())
        self.pool = AttentiveStatsPool(adapter_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * adapter_dim, 2))
        self.config = {"hf_id": hf_id, "revision": revision, "freeze": freeze, "unfreeze_top_k": unfreeze_top_k,
                       "truncate_after_layer": truncate_after_layer, "adapter_dim": adapter_dim, "dropout": dropout,
                       "encoder_config": encoder_config}
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # always: no dropout / layerdrop / SpecAugment inside the encoder
        return self

    @property
    def encoder_trainable(self) -> bool:
        return any(p.requires_grad for p in self.encoder.parameters())

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits.detach(), dim=0)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        x = (wave - wave.mean(dim=1, keepdim=True)) / torch.sqrt(wave.var(dim=1, keepdim=True) + 1e-7)
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.encoder_trainable):
            hs = self.encoder(x, output_hidden_states=True).hidden_states  # tuple of K x [B, T, D]
        h = torch.stack(hs, dim=1)  # [B, K, T, D]
        w = torch.softmax(self.layer_logits, dim=0).view(1, -1, 1, 1)
        feats = self.adapter((w * h).sum(dim=1))
        return self.head(self.pool(feats))

    def param_groups(self, lr_head: float, lr_encoder: float, weight_decay: float) -> list[dict]:
        enc = [p for p in self.encoder.parameters() if p.requires_grad]
        head = [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
        groups = [{"params": head, "lr": lr_head, "weight_decay": weight_decay}]
        if enc:
            groups.append({"params": enc, "lr": lr_encoder, "weight_decay": weight_decay})
        return groups

    # --- compact checkpoints: trainable parameters (+ head buffers) only -----------------------------------------
    def checkpoint_state(self) -> dict[str, torch.Tensor]:
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if k in trainable or not k.startswith("encoder.")}

    def load_checkpoint_state(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys in XLS-R checkpoint: {unexpected[:5]}")
        not_encoder = [k for k in missing if not k.startswith("encoder.")]
        if not_encoder:
            raise RuntimeError(f"checkpoint lacks non-encoder keys: {not_encoder[:5]}")
        trainable_enc = {n for n, p in self.named_parameters() if p.requires_grad and n.startswith("encoder.")}
        if trainable_enc & set(missing):
            raise RuntimeError("checkpoint lacks weights of unfrozen encoder layers")
