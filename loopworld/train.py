"""Training loop utilities for LoopWorld R1."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from torch.optim import AdamW

from .flow import sample_xt, velocity_target
from .losses import exit_losses
from .model import LoopWorldModel


def build_optimizer(model: nn.Module, lr: float = 3e-4, weight_decay: float = 0.01) -> AdamW:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower() or "adaLN" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )


class EMA:
    def __init__(self, model: nn.Module, decay_cap: float = 0.999):
        self.decay_cap = decay_cap
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.n = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.n += 1
        decay = min(self.decay_cap, (1 + self.n) / (10 + self.n))
        msd = model.state_dict()
        for k, v in self.shadow.items():
            v.mul_(decay).add_(msd[k].detach(), alpha=1.0 - decay)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)


def train_step(
    model: LoopWorldModel,
    batch: dict,
    optimizer: AdamW,
    *,
    train_loops: int = 4,
    exit_weights: Sequence[float] = (1 / 6, 1 / 6, 1 / 6, 1 / 2),
    grad_clip: float = 1.0,
    use_bf16: bool = True,
    skip_nonfinite: bool = False,
) -> Dict[str, float]:
    model.train()
    x_clean = batch["target_rgbd"]
    B = x_clean.shape[0]
    device = x_clean.device
    t = torch.rand(B, device=device)
    eps = torch.randn_like(x_clean)
    xt = sample_xt(x_clean, t, eps)
    v_tgt = velocity_target(x_clean, eps)

    optimizer.zero_grad(set_to_none=True)
    amp = use_bf16 and device.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
        exits = model.forward_exits(
            xt, t,
            batch["target_plucker"],
            batch["context_rgbd"],
            batch["context_plucker"],
            batch["context_valid"],
            num_loops=train_loops,
            return_all_exits=True,
        )
        losses = exit_losses(exits, v_tgt, exit_weights)
        loss = losses["L_total"]
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    skipped = 0.0
    if skip_nonfinite and not (torch.isfinite(loss.detach()) and torch.isfinite(grad_norm)):
        optimizer.zero_grad(set_to_none=True)  # drop this update entirely
        skipped = 1.0
    else:
        optimizer.step()
    out = {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in losses.items()}
    out["grad_norm"] = float(grad_norm)
    out["skipped"] = skipped
    return out
