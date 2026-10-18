"""Analytic ray-casting renderer for simple scenes (ground, boxes, spheres).

Returns RGB in [0,1], camera z-depth, and hit mask. Background has no points.
All geometry math in FP32.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from .cameras import invert_c2w, pixel_rays_cam


@dataclass
class Sphere:
    center: Tuple[float, float, float]
    radius: float
    color: Tuple[float, float, float]


@dataclass
class Box:
    """Axis-aligned box: center + half-extents (hx, hy, hz)."""
    center: Tuple[float, float, float]
    half: Tuple[float, float, float]
    color: Tuple[float, float, float]


@dataclass
class Scene:
    spheres: List[Sphere] = field(default_factory=list)
    boxes: List[Box] = field(default_factory=list)
    ground_z: float = 0.0
    ground_half: float = 2.0  # x/y in [-half, half]
    ground_color: Tuple[float, float, float] = (0.55, 0.55, 0.58)
    bg_color: Tuple[float, float, float] = (0.08, 0.09, 0.12)


def _ray_sphere(o, d, center, radius):
    """Ray-sphere intersection. o,d: (...,3). Returns t (>=0) or inf."""
    oc = o - center
    b = (oc * d).sum(-1)
    c = (oc * oc).sum(-1) - radius * radius
    disc = b * b - c
    t = torch.full_like(b, float("inf"))
    ok = disc >= 0
    sqrt_disc = torch.sqrt(disc.clamp_min(0))
    t0 = -b - sqrt_disc
    t1 = -b + sqrt_disc
    # smallest positive
    cand = torch.where(t0 > 1e-4, t0, t1)
    t = torch.where(ok & (cand > 1e-4), cand, t)
    return t


def _ray_aabb(o, d, center, half):
    """Ray vs axis-aligned box. Returns t or inf."""
    mn = center - half
    mx = center + half
    # avoid div0
    inv = 1.0 / d.clamp(min=-1e8).where(d.abs() > 1e-8, torch.full_like(d, 1e-8).copysign(d + 1e-12))
    # Better:
    eps = 1e-8
    d_safe = torch.where(d.abs() < eps, torch.full_like(d, eps).copysign(torch.ones_like(d)), d)
    t0 = (mn - o) / d_safe
    t1 = (mx - o) / d_safe
    tmin = torch.minimum(t0, t1).max(dim=-1).values
    tmax = torch.maximum(t0, t1).min(dim=-1).values
    hit = (tmax >= tmin) & (tmax > 1e-4)
    t = torch.where(hit, torch.where(tmin > 1e-4, tmin, tmax), torch.full_like(tmin, float("inf")))
    # still need tmax > 0 and tmin < tmax
    t = torch.where(hit & (t > 1e-4), t, torch.full_like(t, float("inf")))
    return t


def _ray_ground(o, d, z0, half):
    """Ground plane z=z0, limited to |x|,|y| <= half."""
    t = torch.full(o.shape[:-1], float("inf"), device=o.device, dtype=o.dtype)
    dz = d[..., 2]
    ok = dz.abs() > 1e-8
    t_plane = (z0 - o[..., 2]) / dz.clamp(min=-1e8).where(ok, torch.ones_like(dz))
    p = o + t_plane.unsqueeze(-1) * d
    in_b = (p[..., 0].abs() <= half) & (p[..., 1].abs() <= half) & (t_plane > 1e-4) & ok
    t = torch.where(in_b, t_plane, t)
    return t


def render_scene(
    scene: Scene,
    c2w: torch.Tensor,
    K: torch.Tensor,
    image_size: int,
    z_near: float = 0.5,
    z_far: float = 12.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render RGB, z-depth, hit_mask.

    c2w: (4,4) or (B,4,4)
    Returns rgb (...,H,W,3), depth (...,H,W), mask (...,H,W) bool.
    Background pixels: rgb=bg_color, depth=z_far, mask=False.
    """
    single = c2w.ndim == 2
    if single:
        c2w = c2w.unsqueeze(0)
    B = c2w.shape[0]
    device, dtype = c2w.device, torch.float32
    c2w = c2w.to(dtype=dtype)
    K = K.to(device=device, dtype=dtype)

    dirs_cam = pixel_rays_cam(image_size, K, device=device, dtype=dtype)  # H,W,3
    dirs_cam_n = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    H = W = image_size

    R = c2w[:, :3, :3]  # B,3,3
    o = c2w[:, :3, 3]  # B,3
    # world dirs: R @ d_cam
    d_flat = dirs_cam_n.reshape(-1, 3)  # HW,3
    dw = torch.einsum("bij,pj->bpi", R, d_flat).reshape(B, H, W, 3)
    ow = o[:, None, None, :].expand(B, H, W, 3)

    best_t = torch.full((B, H, W), float("inf"), device=device, dtype=dtype)
    best_rgb = torch.tensor(scene.bg_color, device=device, dtype=dtype).view(1, 1, 1, 3).expand(B, H, W, 3).clone()

    # ground
    tg = _ray_ground(ow, dw, scene.ground_z, scene.ground_half)
    closer = tg < best_t
    best_t = torch.where(closer, tg, best_t)
    gc = torch.tensor(scene.ground_color, device=device, dtype=dtype)
    best_rgb = torch.where(closer.unsqueeze(-1), gc.view(1, 1, 1, 3).expand_as(best_rgb), best_rgb)

    for sp in scene.spheres:
        c = torch.tensor(sp.center, device=device, dtype=dtype)
        ts = _ray_sphere(ow, dw, c, sp.radius)
        closer = ts < best_t
        best_t = torch.where(closer, ts, best_t)
        col = torch.tensor(sp.color, device=device, dtype=dtype)
        best_rgb = torch.where(closer.unsqueeze(-1), col.view(1, 1, 1, 3).expand_as(best_rgb), best_rgb)

    for bx in scene.boxes:
        c = torch.tensor(bx.center, device=device, dtype=dtype)
        h = torch.tensor(bx.half, device=device, dtype=dtype)
        tb = _ray_aabb(ow, dw, c, h)
        closer = tb < best_t
        best_t = torch.where(closer, tb, best_t)
        col = torch.tensor(bx.color, device=device, dtype=dtype)
        best_rgb = torch.where(closer.unsqueeze(-1), col.view(1, 1, 1, 3).expand_as(best_rgb), best_rgb)

    hit = torch.isfinite(best_t) & (best_t < 1e6)
    # z-depth = t * d_cam_z  (since ||d_world||=1 and d_cam is unit, t is Euclidean;
    # camera z = (R^T (p-o))_z = t * (R^T d_w)_z = t * d_cam_z)
    d_cam_z = dirs_cam_n[..., 2].view(1, H, W).expand(B, H, W)
    z_depth = best_t * d_cam_z
    z_depth = torch.where(hit, z_depth, torch.full_like(z_depth, z_far))
    # simple lambertian-ish shading from +z light for a bit of structure
    # (optional visual; colors already set)
    rgb = best_rgb.clamp(0, 1)
    # fill bg color where miss
    bg = torch.tensor(scene.bg_color, device=device, dtype=dtype).view(1, 1, 1, 3)
    rgb = torch.where(hit.unsqueeze(-1), rgb, bg.expand_as(rgb))

    if single:
        return rgb[0], z_depth[0], hit[0]
    return rgb, z_depth, hit


def make_debug_scene_two_occluders() -> Scene:
    """Two spheres along a line of sight for occlusion check."""
    return Scene(
        spheres=[
            Sphere(center=(0.0, 0.0, 0.4), radius=0.35, color=(0.9, 0.2, 0.2)),
            Sphere(center=(0.0, 0.0, 0.4), radius=0.0, color=(0.0, 0.0, 0.0)),  # placeholder removed below
        ],
        boxes=[],
        ground_z=0.0,
        ground_half=2.0,
    )


def make_two_sphere_occlusion_scene() -> Scene:
    """Near red sphere and far blue sphere; camera along +x looking at origin."""
    return Scene(
        spheres=[
            Sphere(center=(0.6, 0.0, 0.4), radius=0.3, color=(0.95, 0.15, 0.1)),
            Sphere(center=(-0.6, 0.0, 0.4), radius=0.35, color=(0.1, 0.25, 0.95)),
        ],
        boxes=[],
        ground_z=0.0,
        ground_half=2.0,
    )
