"""Geometric memory v2: 2x2 splat + 3x3 hole filling + higher point cap. New file (v1 untouched)."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .cameras import unproject_depth, invert_c2w


class PointMemory2:
    def __init__(self, max_points: int = 300_000, splat: int = 2, hole_fill: bool = False):
        self.max_points, self.splat, self.hole_fill = max_points, splat, hole_fill
        self.xyz = self.rgb = None

    def clear(self):
        self.xyz = self.rgb = None

    @torch.no_grad()
    def integrate(self, rgb, depth, mask, c2w, K):
        pw = unproject_depth(depth, c2w, K, mask)
        ok = mask & torch.isfinite(pw[..., 0])
        pts, cols = pw[ok], rgb[ok]
        if pts.numel() == 0: return
        if self.xyz is None: self.xyz, self.rgb = pts, cols
        else: self.xyz = torch.cat([self.xyz, pts]); self.rgb = torch.cat([self.rgb, cols])
        n = self.xyz.shape[0]
        if n > self.max_points:
            keep_new = self.max_points // 2
            idx = torch.randperm(n - keep_new, device=self.xyz.device)[: self.max_points - keep_new]
            self.xyz = torch.cat([self.xyz[idx], self.xyz[-keep_new:]]); self.rgb = torch.cat([self.rgb[idx], self.rgb[-keep_new:]])

    @torch.no_grad()
    def reproject(self, c2w, K, H):
        W = H; dev = c2w.device
        rgb = torch.zeros(H * W, 3, device=dev); zb = torch.full((H * W,), float("inf"), device=dev)
        if self.xyz is not None and self.xyz.shape[0]:
            w2c = invert_c2w(c2w); pc = self.xyz @ w2c[:3, :3].T + w2c[:3, 3]; z = pc[:, 2]
            ok = z > 0.05; pc, z, col = pc[ok], z[ok], self.rgb[ok]
            u = K[0, 0] * pc[:, 0] / z + K[0, 2]; v = K[1, 1] * pc[:, 1] / z + K[1, 2]
            def splat_into(us, vs, zz, cc):
                inb = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
                us, vs, zz, cc = us[inb], vs[inb], zz[inb], cc[inb]
                zbuf = torch.full((H * W,), float("inf"), device=dev); out = torch.zeros(H * W, 3, device=dev)
                if us.numel():
                    flat = vs * W + us
                    zbuf.scatter_reduce_(0, flat, zz, reduce="amin", include_self=True)
                    win = zz <= zbuf[flat] * 1.0001
                    cnt = torch.zeros(H * W, device=dev).index_add_(0, flat[win], torch.ones(int(win.sum()), device=dev))
                    out.index_add_(0, flat[win], cc[win]); out = out / cnt.clamp_min(1).unsqueeze(-1)
                return out, zbuf
            rgb, zb = splat_into(u.round().long(), v.round().long(), z, col)  # primary: exact 1x1
            if self.splat >= 2:  # secondary 2x2 footprint only fills primary holes, and only if not behind a nearer surface
                u0, v0 = torch.floor(u).long(), torch.floor(v).long()
                r2, z2 = splat_into(torch.cat([u0, u0 + 1, u0, u0 + 1]), torch.cat([v0, v0, v0 + 1, v0 + 1]), z.repeat(4), col.repeat(4, 1))
                hole = ~torch.isfinite(zb) & torch.isfinite(z2)
                rgb = torch.where(hole[:, None], r2, rgb); zb = torch.where(hole, z2, zb)
        m = torch.isfinite(zb)
        rgb, zb, m = rgb.view(H, W, 3), zb.view(H, W), m.view(H, W)
        if self.hole_fill:
            zi = torch.where(m, zb, torch.full_like(zb, 1e6))
            pz = F.unfold(zi[None, None], 3, padding=1).view(9, H, W)
            pr = F.unfold(rgb.permute(2, 0, 1)[None], 3, padding=1).view(3, 9, H, W)
            zmin, arg = pz.min(0)
            nb_cnt = (pz < 1e5).sum(0)
            fill = (~m) & (nb_cnt >= 6)
            cfill = torch.gather(pr, 1, arg[None, None].expand(3, 1, H, W)).squeeze(1).permute(1, 2, 0)
            rgb = torch.where(fill[..., None], cfill, rgb); zb = torch.where(fill, zmin, zb); m = m | fill
        zb = torch.where(m, zb, torch.full_like(zb, 12.0))
        return rgb, zb, m
