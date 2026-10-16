"""Evaluation helpers (Stage E stub; sampling entry for Stage B)."""

from __future__ import annotations

from typing import Optional

import torch

from .flow import euler_sample
from .model import LoopWorldModel
from .encoding import decode_rgbd


@torch.no_grad()
def sample_view(
    model: LoopWorldModel,
    context_rgbd: torch.Tensor,
    context_plucker: torch.Tensor,
    context_valid: torch.Tensor,
    target_plucker: torch.Tensor,
    *,
    num_loops: int = 4,
    steps: int = 20,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate encoded RGB-D for one target view. No scene/renderer handles."""
    B = context_rgbd.shape[0]
    H = W = model.image_size
    device = context_rgbd.device
    x = noise if noise is not None else torch.randn(B, H, W, 4, device=device)

    def model_fn(x_t, t, **kw):
        return model.forward_velocity(
            x_t, t, target_plucker, context_rgbd, context_plucker, context_valid, num_loops=num_loops
        )

    return euler_sample(model_fn, x, steps=steps)


@torch.no_grad()
def sample_and_decode(model: LoopWorldModel, **kwargs):
    x = sample_view(model, **kwargs)
    rgb, depth_info, oob = decode_rgbd(x)
    return rgb, depth_info, oob, x
