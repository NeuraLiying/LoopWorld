"""Maze v2 + geometric memory training samples (new file)."""
from __future__ import annotations
import numpy as np, torch
from .cameras import make_K
from .encoding import encode_rgbd
from .maze_v2 import render_maze_multi
from .maze_v2_data import MazeEpisode, maze_plucker, poses_to_actions
from .geom_memory import PointMemory


def make_maze_mem_batch(seeds, H, device, rng, mem_stride=4, use_pose=True, n_frames=128, ep_cache=None, max_points=120_000):
    """Each sample: target j, context=[memory_reproj_to_j, prev], action j-1→j.
    Memory integrates GT frames [0,j) with temporal stride (teacher forcing)."""
    K = make_K(H, device=device)
    B = len(seeds)
    # Pass 1: pick j per seed, collect all frames we need to render
    meta = []
    scenes_r, cams_r = [], []  # for multi render of memory+prev+target
    for sd in seeds:
        ep = ep_cache.get(sd) if ep_cache is not None else None
        if ep is None:
            ep = MazeEpisode(sd, n_frames, device)
            if ep_cache is not None:
                ep_cache[sd] = ep
        j = int(rng.integers(1, n_frames))
        mem_idx = list(range(0, j, mem_stride))
        if j - 1 not in mem_idx:
            mem_idx.append(j - 1)
        meta.append((ep, j, mem_idx))
        for i in mem_idx + [j]:  # prev is in mem_idx; target last
            scenes_r.append(ep.scene); cams_r.append(ep.c2w[i])
    c2w = torch.stack(cams_r)
    rgb, z, m = render_maze_multi(scenes_r, c2w, H)
    # Split back
    tr, tp, cr, cp, cv, act = [], [], [], [], [], []
    cursor = 0
    for ep, j, mem_idx in meta:
        n_m = len(mem_idx)
        # frames: mem_idx (n_m) then target (1)
        fr = rgb[cursor:cursor + n_m + 1]; fz = z[cursor:cursor + n_m + 1]; fm = m[cursor:cursor + n_m + 1]
        fc = torch.stack([ep.c2w[i] for i in mem_idx] + [ep.c2w[j]])
        cursor += n_m + 1
        # build memory from mem frames (all before target; includes prev)
        mem = PointMemory(max_points=max_points, subsample_stride=1)
        for k in range(n_m):
            mem.integrate(fr[k], fz[k], fm[k], fc[k], K)
        mr, mz, mm = mem.reproject(ep.c2w[j], K, H)
        # prev = last of mem_idx (== j-1 always after our construction)
        prev_i = n_m - 1
        enc_t = encode_rgbd(fr[-1], fz[-1], fm[-1])
        enc_p = encode_rgbd(fr[prev_i], fz[prev_i], fm[prev_i])
        enc_m = encode_rgbd(mr, mz, mm)
        pl_t = maze_plucker(ep.c2w[j], K, H, use_pose)
        pl_p = maze_plucker(ep.c2w[j - 1], K, H, use_pose)
        pl_m = maze_plucker(ep.c2w[j], K, H, use_pose)  # same camera as target — memory is in target view
        # context slots: [memory, prev]
        c_r = torch.stack([enc_m, enc_p], 0)
        c_p = torch.stack([pl_m, pl_p], 0)
        # memory valid if any hit
        valid = torch.tensor([mm.any().item(), True], device=device)
        tr.append(enc_t); tp.append(pl_t); cr.append(c_r); cp.append(c_p); cv.append(valid)
        act.append(int(ep.actions[j]))
    return {"target_rgbd": torch.stack(tr), "target_plucker": torch.stack(tp), "context_rgbd": torch.stack(cr),
            "context_plucker": torch.stack(cp), "context_valid": torch.stack(cv),
            "action": torch.tensor(act, device=device)}
