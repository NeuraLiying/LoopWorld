"""Multi-exit rectified-flow losses (R1 §5)."""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch


def exit_losses(
    velocities: List[torch.Tensor],
    v_target: torch.Tensor,
    weights: Sequence[float],
    rgb_weight: float = 0.5,
    depth_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """velocities: list of (B,H,W,4); v_target same shape."""
    assert len(velocities) == len(weights)
    total = torch.zeros((), device=v_target.device, dtype=torch.float32)
    out: Dict[str, torch.Tensor] = {}
    for k, (v, w) in enumerate(zip(velocities, weights), start=1):
        err = (v.float() - v_target.float()) ** 2
        l_rgb = err[..., :3].mean()
        l_depth = err[..., 3].mean()
        l = rgb_weight * l_rgb + depth_weight * l_depth
        out[f"L_rgb_{k}"] = l_rgb.detach()
        out[f"L_depth_{k}"] = l_depth.detach()
        out[f"L_{k}"] = l.detach()
        total = total + float(w) * l
    out["L_total"] = total
    return out
