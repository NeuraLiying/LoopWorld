"""RGB-D encode/decode (R1 plan §4.2)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

Z_NEAR = 0.5
Z_FAR = 12.0


def encode_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """[0,1] -> [-1,1]."""
    return rgb * 2.0 - 1.0


def decode_rgb(x: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """[-1,1]-ish -> [0,1] clipped; also return pre-clip OOB fraction."""
    rgb01 = (x + 1.0) * 0.5
    oob = ((rgb01 < 0) | (rgb01 > 1)).any(dim=-1).float().mean().item() if rgb01.numel() else 0.0
    return rgb01.clamp(0, 1), oob


def encode_depth(
    z: torch.Tensor,
    mask: torch.Tensor,
    z_near: float = Z_NEAR,
) -> torch.Tensor:
    """z-depth + hit mask -> q in [-1,1]. Background u=0 -> q=-1."""
    u = torch.zeros_like(z)
    u = torch.where(mask, z_near / z.clamp_min(z_near * 1e-3), torch.zeros_like(z))
    q = 2.0 * u - 1.0
    return q


def decode_depth(
    q: torch.Tensor,
    z_near: float = Z_NEAR,
    z_far: float = Z_FAR,
) -> Dict[str, torch.Tensor]:
    """Decode network depth channel.

    Returns dict with z_pred, valid_surface, below_range, above_range, nonfinite.
    """
    u_raw = (q + 1.0) * 0.5
    nonfinite = ~torch.isfinite(u_raw)
    u_min = z_near / z_far
    valid = torch.isfinite(u_raw) & (u_raw >= u_min) & (u_raw <= 1.0)
    below = torch.isfinite(u_raw) & (u_raw < u_min)
    above = torch.isfinite(u_raw) & (u_raw > 1.0)
    z = torch.full_like(u_raw, z_far)
    z = torch.where(valid, z_near / u_raw.clamp_min(1e-8), z)
    # for metrics: below -> z_far (already), above -> z_near
    z_metric = z.clone()
    z_metric = torch.where(above, torch.full_like(z, z_near), z_metric)
    z_metric = torch.where(below, torch.full_like(z, z_far), z_metric)
    z_metric = torch.where(nonfinite, torch.full_like(z, z_far), z_metric)
    return {
        "z_pred": z,
        "z_metric": z_metric,
        "valid_surface": valid,
        "below_range": below,
        "above_range": above,
        "nonfinite": nonfinite,
        "u_raw": u_raw,
    }


def encode_rgbd(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    mask: torch.Tensor,
    z_near: float = Z_NEAR,
) -> torch.Tensor:
    """rgb (...,H,W,3) [0,1], depth, mask -> (...,H,W,4) in approx [-1,1]."""
    e_rgb = encode_rgb(rgb)
    e_d = encode_depth(depth, mask, z_near=z_near).unsqueeze(-1)
    return torch.cat([e_rgb, e_d], dim=-1)


def decode_rgbd(x: torch.Tensor, z_near: float = Z_NEAR, z_far: float = Z_FAR):
    rgb, oob = decode_rgb(x[..., :3])
    depth_info = decode_depth(x[..., 3], z_near=z_near, z_far=z_far)
    return rgb, depth_info, oob
