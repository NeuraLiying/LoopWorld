"""Camera geometry for LoopWorld.

Conventions (documented explicitly per R1 plan §4.1):
- World frame: right-handed, **z-up**. Origin near scene center; look-at target (0,0,0.4).
- Camera frame: OpenCV-style — **x right, y down, z forward** (into the scene).
- Extrinsics: ``c2w`` is a 4x4 camera-to-world matrix. Camera center is ``c2w[:3, 3]``.
  Columns of ``c2w[:3, :3]`` are camera axes expressed in world: right, down, forward.
- Intrinsics ``K`` (3x3): pixel coordinates with origin at top-left; principal point at
  image center ``((W-1)/2, (H-1)/2)`` so that pixel-center rays are used
  (ray through pixel (u,v) uses continuous coords (u+0.5, v+0.5) when sampling,
  equivalently fx/fy with cx=(W-1)/2, cy=(H-1)/2 and integer pixel indices as centers).
- Depth is **camera z-depth** (forward-axis distance), not ray Euclidean length.
- FOV is horizontal field-of-view in radians for square images.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def make_K(
    image_size: int,
    fov_deg: float = 60.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Square-image pinhole K with principal point at pixel centers."""
    fov = math.radians(fov_deg)
    fx = (image_size / 2.0) / math.tan(fov / 2.0)
    fy = fx
    cx = (image_size - 1) / 2.0
    cy = (image_size - 1) / 2.0
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    return K


def look_at_c2w(
    eye: torch.Tensor,
    target: torch.Tensor,
    up: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build c2w from eye position looking at target (OpenCV camera axes).

    eye/target: (..., 3). Returns (..., 4, 4).
    """
    if up is None:
        up = torch.tensor([0.0, 0.0, 1.0], device=eye.device, dtype=eye.dtype)
        while up.ndim < eye.ndim:
            up = up.unsqueeze(0)
        up = up.expand_as(eye)

    forward = target - eye
    forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # right = forward × up (OpenCV x-right, y-down, z-forward with z-up world).
    # Using up × forward flips right/down (180° roll). Nudge if parallel.
    right = torch.cross(forward, up, dim=-1)
    right_norm = right.norm(dim=-1, keepdim=True)
    alt_up = torch.tensor([0.0, 1.0, 0.0], device=eye.device, dtype=eye.dtype)
    while alt_up.ndim < eye.ndim:
        alt_up = alt_up.unsqueeze(0)
    alt_up = alt_up.expand_as(eye)
    bad = right_norm.squeeze(-1) < 1e-6
    if bad.any():
        right = torch.where(bad.unsqueeze(-1), torch.cross(forward, alt_up, dim=-1), right)
        right_norm = right.norm(dim=-1, keepdim=True)
    right = right / right_norm.clamp_min(1e-8)
    down = torch.cross(forward, right, dim=-1)  # y down
    # columns: right, down, forward
    R = torch.stack([right, down, forward], dim=-1)  # (..., 3, 3)
    c2w = torch.zeros(*eye.shape[:-1], 4, 4, device=eye.device, dtype=eye.dtype)
    c2w[..., :3, :3] = R
    c2w[..., :3, 3] = eye
    c2w[..., 3, 3] = 1.0
    return c2w


def orbit_c2w(
    azimuth_deg: float | torch.Tensor,
    elevation_deg: float | torch.Tensor,
    radius: float = 3.5,
    target: Tuple[float, float, float] = (0.0, 0.0, 0.4),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Orbit camera: azimuth around +z, elevation from xy-plane upward."""
    if not torch.is_tensor(azimuth_deg):
        azimuth_deg = torch.tensor(azimuth_deg, device=device, dtype=dtype)
    if not torch.is_tensor(elevation_deg):
        elevation_deg = torch.tensor(elevation_deg, device=device, dtype=dtype)
    az = torch.deg2rad(azimuth_deg.to(dtype=dtype))
    el = torch.deg2rad(elevation_deg.to(dtype=dtype))
    # eye in world: z-up spherical
    x = radius * torch.cos(el) * torch.cos(az)
    y = radius * torch.cos(el) * torch.sin(az)
    z = radius * torch.sin(el) + target[2] * 0.0  # elevation from origin plane; add target z below
    # Place eye so look-at hits target; elevation measured from horizontal through target
    eye = torch.stack(
        [
            radius * torch.cos(el) * torch.cos(az) + target[0],
            radius * torch.cos(el) * torch.sin(az) + target[1],
            radius * torch.sin(el) + target[2],
        ],
        dim=-1,
    )
    tgt = torch.tensor(target, device=eye.device, dtype=dtype)
    while tgt.ndim < eye.ndim:
        tgt = tgt.unsqueeze(0)
    tgt = tgt.expand_as(eye)
    return look_at_c2w(eye, tgt)


def invert_c2w(c2w: torch.Tensor) -> torch.Tensor:
    """World-to-camera from c2w."""
    R = c2w[..., :3, :3]
    t = c2w[..., :3, 3]
    R_inv = R.transpose(-1, -2)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)
    w2c = torch.zeros_like(c2w)
    w2c[..., :3, :3] = R_inv
    w2c[..., :3, 3] = t_inv
    w2c[..., 3, 3] = 1.0
    return w2c


def project_points(
    points_world: torch.Tensor,
    c2w: torch.Tensor,
    K: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world points to pixels.

    Returns:
        uv: (..., 2) pixel coordinates (u horizontal, v vertical)
        z: (...,) camera z-depth
        valid: (...,) bool — in front of camera (z > 0)
    """
    w2c = invert_c2w(c2w)
    ones = torch.ones(*points_world.shape[:-1], 1, device=points_world.device, dtype=points_world.dtype)
    pw = torch.cat([points_world, ones], dim=-1)  # (..., 4)
    # Support broadcasting: c2w may be (..., 4,4) matching points or batched differently
    pc = (w2c @ pw.unsqueeze(-1)).squeeze(-1)[..., :3]
    z = pc[..., 2]
    valid = z > 1e-6
    u = K[..., 0, 0] * (pc[..., 0] / z.clamp_min(1e-8)) + K[..., 0, 2]
    v = K[..., 1, 1] * (pc[..., 1] / z.clamp_min(1e-8)) + K[..., 1, 2]
    uv = torch.stack([u, v], dim=-1)
    return uv, z, valid


def unproject_depth(
    depth: torch.Tensor,
    c2w: torch.Tensor,
    K: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unproject z-depth image to world points.

    depth: (..., H, W) camera z-depth
    Returns points (..., H, W, 3). Invalid/masked pixels are NaN.
    """
    *batch, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    # pixel centers
    u = xs
    v = ys
    fx, fy = K[..., 0, 0], K[..., 1, 1]
    cx, cy = K[..., 0, 2], K[..., 1, 2]
    # broadcast K scalars
    while fx.ndim < depth.ndim:
        fx = fx.unsqueeze(-1)
        fy = fy.unsqueeze(-1)
        cx = cx.unsqueeze(-1)
        cy = cy.unsqueeze(-1)
    x_cam = (u - cx) / fx * depth
    y_cam = (v - cy) / fy * depth
    z_cam = depth
    pc = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (..., H, W, 3)
    # Explicit spatial dims so batch (B,H,W) + c2w (B,4,4) broadcasts:
    # R (...,1,1,3,3), t (...,1,1,3).
    R = c2w[..., None, None, :3, :3]
    t = c2w[..., None, None, :3, 3]
    pw = (R @ pc.unsqueeze(-1)).squeeze(-1) + t
    if mask is not None:
        bad = ~mask
        pw = pw.clone()
        pw[bad] = float("nan")
    return pw


def pixel_rays_cam(
    image_size: int,
    K: torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unit-ish camera-space ray directions for each pixel (not normalized).

    Returns (H, W, 3) directions in camera frame (z=1 plane directions).
    """
    H = W = image_size
    device = device or K.device
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    dirs = torch.stack([(xs - cx) / fx, (ys - cy) / fy, torch.ones_like(xs)], dim=-1)
    return dirs


def plucker_rays(
    c2w: torch.Tensor,
    K: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """Plücker ray encoding per pixel: (d, o × d) in world, shape (..., H, W, 6).

    d is normalized world-space direction; o is camera center.
    """
    dirs_cam = pixel_rays_cam(image_size, K if K.ndim == 2 else K.reshape(-1, 3, 3)[0], device=c2w.device, dtype=c2w.dtype)
    # normalize in camera then rotate
    dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    R = c2w[..., :3, :3]  # (..., 3, 3)
    o = c2w[..., :3, 3]  # (..., 3)
    # dirs_world = R @ dirs_cam
    # Broadcast: dirs_cam is (H,W,3)
    H = W = image_size
    dirs_flat = dirs_cam.reshape(-1, 3)  # (HW, 3)
    # R: (..., 3, 3) -> (..., 1, 3, 3) @ (HW, 3, 1)
    Rw = R.unsqueeze(-3)  # (..., 1, 3, 3)
    dw = (Rw @ dirs_flat.unsqueeze(-1)).squeeze(-1)  # (..., HW, 3)
    dw = dw / dw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    ow = o.unsqueeze(-2).expand_as(dw)  # (..., HW, 3)
    m = torch.cross(ow, dw, dim=-1)
    plucker = torch.cat([dw, m], dim=-1).reshape(*c2w.shape[:-2], H, W, 6)
    return plucker


def z_depth_to_ray_distance(z: torch.Tensor, ray_dir_cam: torch.Tensor) -> torch.Tensor:
    """Convert camera z-depth to Euclidean ray distance.

    ray_dir_cam: directions with z-component (unnormalized ok); distance = z / dir_z_unit
    where dir is normalized so ||d||=1, then dist = z / d_z.
    """
    d = ray_dir_cam / ray_dir_cam.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return z / d[..., 2].clamp_min(1e-8)


def ray_distance_to_z_depth(dist: torch.Tensor, ray_dir_cam: torch.Tensor) -> torch.Tensor:
    d = ray_dir_cam / ray_dir_cam.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return dist * d[..., 2]
