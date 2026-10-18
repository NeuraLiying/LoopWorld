"""Long-horizon geometric memory: accumulate RGB-D as world points, reproject by z-buffer.

Train: integrate GT frames. Eval AR: integrate the model's own predictions.
No new model parameters — memory is just a different context slot content.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import torch
from .cameras import make_K, unproject_depth, invert_c2w


class PointMemory:
    """Per-episode point buffer. XYZ + RGB in world frame; reproject via z-buffer splat."""

    def __init__(self, max_points: int = 120_000, subsample_stride: int = 1):
        self.max_points = max_points
        self.subsample_stride = subsample_stride  # every Nth pixel when integrating
        self.xyz: Optional[torch.Tensor] = None  # (N,3)
        self.rgb: Optional[torch.Tensor] = None  # (N,3)

    def clear(self):
        self.xyz = self.rgb = None

    @torch.no_grad()
    def integrate(self, rgb, depth, mask, c2w, K):
        """rgb (H,W,3) [0,1], depth (H,W) z, mask (H,W) bool, c2w (4,4), K (3,3)."""
        s = self.subsample_stride
        rgb_s, depth_s, mask_s = rgb[::s, ::s], depth[::s, ::s], mask[::s, ::s]
        # Scale K for subsampled grid: fx,fy,cx,cy all / s roughly — use same K with integer coords
        # unproject_depth uses arange(H)*same K; for subsampled image we need adjusted K
        Hs, Ws = depth_s.shape
        Ks = K.clone()
        Ks[0, 0] = K[0, 0] / s; Ks[1, 1] = K[1, 1] / s
        Ks[0, 2] = (K[0, 2] + 0.5) / s - 0.5; Ks[1, 2] = (K[1, 2] + 0.5) / s - 0.5
        pw = unproject_depth(depth_s, c2w, Ks, mask_s)  # Hs,Ws,3
        ok = mask_s & torch.isfinite(pw[..., 0])
        pts = pw[ok]; cols = rgb_s[ok]
        if pts.numel() == 0:
            return
        if self.xyz is None:
            self.xyz, self.rgb = pts, cols
        else:
            self.xyz = torch.cat([self.xyz, pts], 0)
            self.rgb = torch.cat([self.rgb, cols], 0)
        if self.xyz.shape[0] > self.max_points:
            # keep most recent + random older
            n = self.max_points
            keep_new = n // 2
            older = self.xyz.shape[0] - keep_new
            idx = torch.randperm(older, device=self.xyz.device)[: n - keep_new]
            self.xyz = torch.cat([self.xyz[idx], self.xyz[-keep_new:]], 0)
            self.rgb = torch.cat([self.rgb[idx], self.rgb[-keep_new:]], 0)

    @torch.no_grad()
    def reproject(self, c2w, K, H: int, W: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return rgb (H,W,3), z-depth (H,W), mask (H,W). Empty memory → black / z=12 / False."""
        W = W or H
        device = c2w.device
        rgb_out = torch.zeros(H, W, 3, device=device)
        z_out = torch.full((H, W), 12.0, device=device)
        mask = torch.zeros(H, W, dtype=torch.bool, device=device)
        if self.xyz is None or self.xyz.shape[0] == 0:
            return rgb_out, z_out, mask
        w2c = invert_c2w(c2w)
        pc = self.xyz @ w2c[:3, :3].T + w2c[:3, 3]  # N,3 camera
        z = pc[:, 2]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u = fx * pc[:, 0] / z.clamp_min(1e-4) + cx
        v = fy * pc[:, 1] / z.clamp_min(1e-4) + cy
        ui = u.round().long(); vi = v.round().long()
        inb = (z > 0.05) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not inb.any():
            return rgb_out, z_out, mask
        ui, vi, z, cols = ui[inb], vi[inb], z[inb], self.rgb[inb]
        # z-buffer via scatter_reduce amin, then gather winners
        flat = vi * W + ui
        zbuf = torch.full((H * W,), float("inf"), device=device)
        zbuf.scatter_reduce_(0, flat, z, reduce="amin", include_self=True)
        win = (z <= zbuf[flat] + 1e-4)  # points that hit the near surface
        # last-wins among winners for color (stable enough)
        flat_w, z_w, c_w = flat[win], z[win], cols[win]
        # sort by descending z so nearest writes last
        order = z_w.argsort(descending=True)
        flat_w, c_w, z_w = flat_w[order], c_w[order], z_w[order]
        rgb_flat = rgb_out.view(-1, 3)
        z_flat = z_out.view(-1)
        m_flat = mask.view(-1)
        rgb_flat[flat_w] = c_w
        z_flat[flat_w] = z_w
        m_flat[flat_w] = True
        return rgb_out, z_out, mask


@torch.no_grad()
def build_memory_from_frames(rgb, depth, mask, c2w, K, upto: int, stride: int = 4, max_points=120_000) -> PointMemory:
    """Integrate frames [0, upto) with temporal stride. rgb/depth/mask/c2w are (T,...)."""
    mem = PointMemory(max_points=max_points, subsample_stride=2)
    idxs = list(range(0, upto, stride))
    if upto - 1 not in idxs and upto > 0:
        idxs.append(upto - 1)  # always include most recent
    for i in idxs:
        mem.integrate(rgb[i], depth[i], mask[i], c2w[i], K)
    return mem


@torch.no_grad()
def reproject_batch(memories: List[PointMemory], c2w, K, H: int):
    """c2w (B,4,4) → stacked rgb,z,mask (B,H,H,*)"""
    outs = [m.reproject(c2w[i], K, H) for i, m in enumerate(memories)]
    return (torch.stack([o[0] for o in outs]), torch.stack([o[1] for o in outs]),
            torch.stack([o[2] for o in outs]))
