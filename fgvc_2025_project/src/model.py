from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn as nn
import timm


@dataclass
class LoraConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_filter: Optional[Callable[[str, nn.Module], bool]] = None  # decide which Linear to wrap


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.use_bias = base.bias is not None
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        # Freeze base weights to train only LoRA adapters
        for p in self.base.parameters():
            p.requires_grad = False

        if r > 0:
            # Initialize A low rank with small std, B zeros
            self.lora_A = nn.Parameter(torch.zeros(self.in_features, r))
            self.lora_B = nn.Parameter(torch.zeros(r, self.out_features))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r and self.lora_A is not None and self.lora_B is not None:
            x_ = self.dropout(x)
            delta = x_.matmul(self.lora_A).matmul(self.lora_B)
            out = out + self.scaling * delta
        return out


def _default_target_filter(name: str, m: nn.Module) -> bool:
    # Target common Linear layers in ViTs (qkv/proj/mlp), but keep generic
    if not isinstance(m, nn.Linear):
        return False
    lname = name.lower()
    if any(k in lname for k in ["qkv", "attn", "proj", "fc", "mlp", "head"]):
        return True
    return True  # fallback: all Linear layers


import math

def apply_lora(model: nn.Module, cfg: LoraConfig) -> nn.Module:
    if cfg.r <= 0:
        return model
    target_filter = cfg.target_filter or _default_target_filter
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and target_filter(name, module):
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = model
            if parent_name:
                for p in parent_name.split("."):
                    parent = getattr(parent, p)
            child_name = name.split(".")[-1]
            wrapped = LoRALinear(module, r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout)
            setattr(parent, child_name, wrapped)
    return model


def build_model(num_classes: int, backbone: str = "vit_base_patch16_224", pretrained: bool = True,
                use_lora: bool = False, lora_r: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.0) -> nn.Module:
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
    if use_lora:
        cfg = LoraConfig(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        model = apply_lora(model, cfg)
    return model
