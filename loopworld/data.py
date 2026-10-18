"""Procedural scenes, trajectories, and training sample construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from .cameras import make_K, orbit_c2w, plucker_rays
from .encoding import encode_rgbd, Z_NEAR, Z_FAR
from .renderer import Box, Scene, Sphere, render_scene


def split_seeds(namespace: str = "r1") -> Dict[str, List[int]]:
    """Independent seed namespaces for splits (R1 §4.4)."""
    base = {"train": 100000, "val": 200000, "test": 300000, "overfit": 400000}
    return {
        "overfit": [base["overfit"] + i for i in range(4)],
        "val": [base["val"] + i for i in range(8)],
        "test": [base["test"] + i for i in range(24)],
        "train": [base["train"] + i for i in range(100000)],  # online pool
    }


def _rand_color(g: torch.Generator, device) -> Tuple[float, float, float]:
    c = torch.rand(3, generator=g, device=device) * 0.7 + 0.25
    return tuple(c.tolist())


def sample_scene(seed: int, device: torch.device = torch.device("cpu")) -> Scene:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    n_obj = int(torch.randint(2, 5, (1,), generator=g).item())  # 2–4
    spheres: List[Sphere] = []
    boxes: List[Box] = []
    placed = []
    for _ in range(n_obj):
        for _try in range(40):
            xy = (torch.rand(2, generator=g) * 2 - 1).tolist()
            size = float(torch.rand(1, generator=g).item() * 0.6 + 0.2)
            kind = int(torch.randint(0, 2, (1,), generator=g).item())
            ok = True
            for px, py, ps in placed:
                if (xy[0] - px) ** 2 + (xy[1] - py) ** 2 < (0.5 * (size + ps)) ** 2:
                    ok = False
                    break
            if not ok:
                continue
            placed.append((xy[0], xy[1], size))
            col = _rand_color(g, "cpu")
            if kind == 0:
                spheres.append(Sphere(center=(xy[0], xy[1], size), radius=size, color=col))
            else:
                hz = size
                boxes.append(Box(center=(xy[0], xy[1], hz), half=(size * 0.8, size * 0.8, hz), color=col))
            break
    ground_col = tuple((torch.rand(3, generator=g) * 0.15 + 0.45).tolist())
    return Scene(spheres=spheres, boxes=boxes, ground_z=0.0, ground_half=2.0, ground_color=ground_col)


def sample_trajectory(
    seed: int,
    n_gen_views: int = 24,
    step_deg: float = 15.0,
    elev_range: Tuple[float, float] = (20.0, 40.0),
    radius: float = 3.5,
) -> torch.Tensor:
    """Return c2w stack (N, 4, 4) with N = n_gen_views + 1 (default 25).

    Index 0 is the conditioning 首帧. Indices 1..n_gen_views are generation
    targets; index n_gen_views completes a full circle (n_gen_views * step_deg
    degrees) and revisits the same pose as index 0 (closed-loop / 回访).
    """
    assert n_gen_views * step_deg == 360.0, (
        f"closed loop requires n_gen_views*step_deg=360, got {n_gen_views}*{step_deg}"
    )
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + 17)
    az0 = float(torch.rand(1, generator=g).item() * 360.0)
    sign = 1.0 if torch.rand(1, generator=g).item() > 0.5 else -1.0
    elev = float(
        torch.rand(1, generator=g).item() * (elev_range[1] - elev_range[0]) + elev_range[0]
    )
    n_poses = n_gen_views + 1  # 首帧 + 24 gen (last == first pose)
    c2ws = []
    for i in range(n_poses):
        az = az0 + sign * step_deg * i
        c2ws.append(orbit_c2w(az, elev, radius=radius))
    out = torch.stack(c2ws, dim=0)
    # Closed-loop: last pose must match first (360° orbit).
    eye0, eyeN = out[0, :3, 3], out[-1, :3, 3]
    R0, RN = out[0, :3, :3], out[-1, :3, :3]
    assert torch.allclose(eye0, eyeN, atol=1e-5), (
        f"closed-loop eye mismatch: {(eye0 - eyeN).norm().item()}"
    )
    assert torch.allclose(R0, RN, atol=1e-5), (
        f"closed-loop R mismatch: {(R0 - RN).abs().max().item()}"
    )
    # Adjacent steps ~ step_deg on azimuth (chord length check via eye).
    if n_poses >= 3:
        d01 = (out[1, :3, 3] - out[0, :3, 3]).norm().item()
        d12 = (out[2, :3, 3] - out[1, :3, 3]).norm().item()
        assert abs(d01 - d12) / max(d01, 1e-6) < 1e-3, "non-uniform trajectory steps"
    return out


@dataclass
class SampleBatch:
    target_rgbd: torch.Tensor      # B,H,W,4 encoded
    target_plucker: torch.Tensor   # B,H,W,6
    context_rgbd: torch.Tensor     # B,S,H,W,4
    context_plucker: torch.Tensor  # B,S,H,W,6
    context_valid: torch.Tensor    # B,S bool
    # raw for checks / viz
    target_rgb: torch.Tensor
    target_depth: torch.Tensor
    target_mask: torch.Tensor
    meta: dict


def render_views(
    scene: Scene,
    c2ws: torch.Tensor,
    image_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render one or more cameras; uses renderer batch path when N>1."""
    K = make_K(image_size, device=device)
    c2ws = c2ws.to(device)
    if c2ws.ndim == 2:
        c2ws = c2ws.unsqueeze(0)
    rgb, depth, mask = render_scene(scene, c2ws, K, image_size)
    return rgb, depth, mask


def make_training_sample(
    scene_seed: int,
    traj_seed: int,
    image_size: int,
    device: torch.device,
    single_context_prob: float = 0.5,
    context_slots: int = 2,
    g_extra: Optional[torch.Generator] = None,
) -> SampleBatch:
    """One training sample (batch size 1 fields without batch dim — caller stacks).

    R1 protocol: trajectory has 25 poses (index 0 = 首帧, 1..24 = gen targets;
    index 24 revisits pose 0). Single-context: j ~ U{1..24}, ctx=[0].
    Double-context: j ~ U{2..24}, ctx=[0, j-1]. Only needed views are rendered.
    """
    scene = sample_scene(scene_seed)
    c2ws = sample_trajectory(traj_seed).to(device)
    n_poses = c2ws.shape[0]
    n_gen = n_poses - 1  # 24
    assert n_poses == 25 and n_gen == 24, f"expected 25 poses, got {n_poses}"

    g = g_extra or torch.Generator(device="cpu")
    if g_extra is None:
        g.manual_seed(scene_seed * 1009 + traj_seed)

    use_single = torch.rand(1, generator=g).item() < single_context_prob
    if use_single:
        # j in 1..24 inclusive (array index == j); never target index 0 as "novel"
        j = int(torch.randint(1, n_gen + 1, (1,), generator=g).item())
        target_idx = j
        ctx_indices = [0]
    else:
        # j in 2..24 inclusive; recent context is j-1 (>=1), distinct from 首帧
        j = int(torch.randint(2, n_gen + 1, (1,), generator=g).item())
        target_idx = j
        ctx_indices = [0, j - 1]

    assert 1 <= target_idx <= n_gen, target_idx
    assert ctx_indices[0] == 0
    if not use_single:
        assert len(ctx_indices) == 2 and ctx_indices[1] == target_idx - 1
        assert ctx_indices[0] != ctx_indices[1], f"duplicate ctx {ctx_indices}"
        assert ctx_indices[1] >= 1

    # A6: render only unique cameras needed for this sample
    need = sorted(set([target_idx] + ctx_indices))
    K = make_K(image_size, device=device)
    rgb, depth, mask = render_views(scene, c2ws[need], image_size, device)
    slot = {orig: i for i, orig in enumerate(need)}

    ti = slot[target_idx]
    target_rgb = rgb[ti]
    target_depth = depth[ti]
    target_mask = mask[ti]
    target_rgbd = encode_rgbd(target_rgb, target_depth, target_mask)
    target_plucker = plucker_rays(c2ws[target_idx], K, image_size)

    S = context_slots
    context_rgbd = torch.zeros(S, image_size, image_size, 4, device=device)
    context_plucker = torch.zeros(S, image_size, image_size, 6, device=device)
    context_valid = torch.zeros(S, dtype=torch.bool, device=device)
    for s_i, vi in enumerate(ctx_indices):
        if s_i >= S:
            break
        ri = slot[vi]
        context_rgbd[s_i] = encode_rgbd(rgb[ri], depth[ri], mask[ri])
        context_plucker[s_i] = plucker_rays(c2ws[vi], K, image_size)
        context_valid[s_i] = True

    return SampleBatch(
        target_rgbd=target_rgbd,
        target_plucker=target_plucker,
        context_rgbd=context_rgbd,
        context_plucker=context_plucker,
        context_valid=context_valid,
        target_rgb=target_rgb,
        target_depth=target_depth,
        target_mask=target_mask,
        meta={
            "scene_seed": scene_seed,
            "traj_seed": traj_seed,
            "target_idx": target_idx,
            "ctx": ctx_indices,
            "n_poses": n_poses,
            "rendered_views": need,
            "is_closed_loop_revisit": bool(target_idx == n_gen),
        },
    )


def collate_samples(samples: List[SampleBatch]) -> SampleBatch:
    return SampleBatch(
        target_rgbd=torch.stack([s.target_rgbd for s in samples]),
        target_plucker=torch.stack([s.target_plucker for s in samples]),
        context_rgbd=torch.stack([s.context_rgbd for s in samples]),
        context_plucker=torch.stack([s.context_plucker for s in samples]),
        context_valid=torch.stack([s.context_valid for s in samples]),
        target_rgb=torch.stack([s.target_rgb for s in samples]),
        target_depth=torch.stack([s.target_depth for s in samples]),
        target_mask=torch.stack([s.target_mask for s in samples]),
        meta={"n": len(samples)},
    )
