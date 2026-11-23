"""NII AntiDeepfake detector (Yamagishi Lab), loaded WITHOUT fairseq.

Source: https://huggingface.co/nii-yamagishilab/mms-300m-anti-deepfake  (paper arXiv:2506.21090)
- SSL front-end (wav2vec 2.0 / MMS-300M, 24 layers, stable layer norm) post-trained on ~74k hours of real and
  fake speech in 100+ languages, + mean pooling + Linear(1024, 2).
- Upstream logits are [fake, real]; this wrapper returns VaaniShield's convention [bonafide, spoof] (index 1 = spoof).
- Input: 16 kHz mono waveform; each utterance is layer-normalised (zero mean, unit variance), as upstream does.
- Licence of the weights: CC BY-NC-SA 4.0 (non-commercial). Code: BSD-3-Clause.

The upstream checkpoint stores fairseq parameter names; ``fairseq_to_hf`` maps them onto transformers' Wav2Vec2Model.
"""

from __future__ import annotations

import re

import torch
from torch import nn

REPOS = {
    "mms-300m": "nii-yamagishilab/mms-300m-anti-deepfake",
    "wav2vec-large": "nii-yamagishilab/wav2vec-large-anti-deepfake",
    "wav2vec-small": "nii-yamagishilab/wav2vec-small-anti-deepfake",
}


def fairseq_to_hf(sd: dict, conv_norm_key: str) -> tuple[dict, dict]:
    """Return (encoder_state_dict for transformers Wav2Vec2Model, head_state_dict)."""
    enc, head = {}, {}
    for k, v in sd.items():
        if k.startswith("proj_fc."):
            head[k[len("proj_fc."):]] = v
            continue
        k = k.removeprefix("m_ssl.model.")
        if k.startswith(("quantizer.", "project_q.", "final_proj.")):
            continue  # pre-training heads, not used for detection
        m = re.match(r"feature_extractor\.conv_layers\.(\d+)\.0\.(weight|bias)", k)
        if m:
            enc[f"feature_extractor.conv_layers.{m[1]}.conv.{m[2]}"] = v
            continue
        m = re.match(r"feature_extractor\.conv_layers\.(\d+)\.2\.1\.(weight|bias)", k)
        if m:
            enc[f"feature_extractor.conv_layers.{m[1]}.layer_norm.{m[2]}"] = v
            continue
        if k.startswith("layer_norm."):
            enc["feature_projection." + k] = v
            continue
        if k.startswith("post_extract_proj."):
            enc["feature_projection.projection." + k.split(".", 1)[1]] = v
            continue
        if k == "mask_emb":
            enc["masked_spec_embed"] = v
            continue
        if k.startswith("encoder.pos_conv.0."):
            name = k.split(".")[-1]
            enc[{"bias": "encoder.pos_conv_embed.conv.bias", "weight_g": conv_norm_key + "0",
                 "weight_v": conv_norm_key + "1"}[name]] = v
            continue
        m = re.match(r"encoder\.layers\.(\d+)\.(.+)", k)
        if m:
            i, rest = m[1], m[2]
            rest = (rest.replace("self_attn_layer_norm", "layer_norm")
                        .replace("self_attn.", "attention.")
                        .replace("fc1.", "feed_forward.intermediate_dense.")
                        .replace("fc2.", "feed_forward.output_dense."))
            enc[f"encoder.layers.{i}.{rest}"] = v
            continue
        if k.startswith("encoder.layer_norm."):
            enc[k] = v
            continue
        raise KeyError(f"unmapped AntiDeepfake key: {k}")
    return enc, head


class AntiDeepfakeDetector(nn.Module):
    def __init__(self, variant: str = "mms-300m", load_pretrained: bool = True):
        super().__init__()
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        self.config = {"variant": variant}
        repo = REPOS[variant]
        from huggingface_hub import hf_hub_download

        cfg = Wav2Vec2Config.from_pretrained(repo)
        # match upstream fine-tuning: no layer drop, no dropout, no SpecAugment masking
        for a in ("hidden_dropout", "attention_dropout", "activation_dropout", "feat_proj_dropout", "layerdrop",
                  "final_dropout", "mask_time_prob"):
            setattr(cfg, a, 0.0)
        cfg.apply_spec_augment = False
        self.encoder = Wav2Vec2Model(cfg)
        self.head = nn.Linear(cfg.hidden_size, 2)
        if load_pretrained:
            from safetensors.torch import load_file

            sd = load_file(hf_hub_download(repo, "model.safetensors"))
            own = self.encoder.state_dict()
            g_key = next((k[:-1] for k in own if k.startswith("encoder.pos_conv_embed.conv.parametrizations.weight.original")),
                         None)
            conv_norm_key = g_key or "encoder.pos_conv_embed.conv.weight_"
            if g_key is None:  # old-style weight_norm names: weight_g / weight_v
                enc, head = fairseq_to_hf(sd, "encoder.pos_conv_embed.conv.weight_")
                enc = {k.replace("weight_0", "weight_g").replace("weight_1", "weight_v"): v for k, v in enc.items()}
            else:
                enc, head = fairseq_to_hf(sd, conv_norm_key)
            missing, unexpected = self.encoder.load_state_dict(enc, strict=False)
            missing = [m for m in missing if "masked_spec_embed" not in m]
            unexpected = [u for u in unexpected if "masked_spec_embed" not in u]
            if missing or unexpected:
                raise RuntimeError(f"AntiDeepfake weight mapping failed. missing={missing[:5]} unexpected={unexpected[:5]}")
            self.head.load_state_dict(head)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        wave = nn.functional.layer_norm(wave, wave.shape[-1:])  # per-utterance normalisation (upstream)
        h = self.encoder(wave).last_hidden_state  # [B, T, 1024], after the final encoder layer norm
        logits = self.head(h.mean(dim=1))  # upstream: [fake, real]
        return logits.flip(1)  # -> [bonafide, spoof]
