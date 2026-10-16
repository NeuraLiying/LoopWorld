"""Maze v2 training samples / action sequences (new file)."""
from __future__ import annotations
import math
import numpy as np, torch
from .cameras import make_K, plucker_rays
from .encoding import encode_rgbd
from .maze_v2 import sample_maze, sample_maze_trajectory, render_maze, revisit_pairs

PLUCKER_MOMENT_SCALE = 0.6  # maze |o x d| <= ~5.7 vs orbit ~3.5 -> rescale moment channels


def poses_to_actions(poses):
    """actions[j] = action taking pose j-1 -> j (actions[0]=noop)."""
    acts = [0]
    for (x0, y0, a0), (x1, y1, a1) in zip(poses[:-1], poses[1:]):
        dy = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        if abs(dy) > 1e-3:
            acts.append(2 if dy > 0 else 3)
        elif math.hypot(x1 - x0, y1 - y0) > 1e-3:
            acts.append(1)
        else:
            acts.append(0)
    return np.array(acts, np.int64)


def maze_plucker(c2w, K, H, use_pose=True):
    pl = plucker_rays(c2w, K, H)
    if not use_pose:
        return torch.zeros_like(pl)
    return torch.cat([pl[..., :3], pl[..., 3:] * PLUCKER_MOMENT_SCALE], -1)


class MazeEpisode:
    def __init__(self, seed, n_frames=128, device="cpu"):
        self.scene = sample_maze(seed)
        self.c2w, self.poses, _, self.kind = sample_maze_trajectory(self.scene, seed, n_frames)
        self.c2w = self.c2w.to(device)
        self.actions = poses_to_actions(self.poses)
        self.revisit = dict(revisit_pairs(self.poses))  # j -> first-visit i


def make_maze_batch(seeds, H, device, rng, p_revisit=0.3, kmax=32, use_pose=True, n_frames=128, ep_cache=None):
    """Batched: all 3*B views rendered in one render_maze_multi call."""
    from .maze_v2 import render_maze_multi
    K = make_K(H, device=device); B = len(seeds)
    scenes, cams, act, ev = [], [], [], []
    for sd in seeds:
        ep = ep_cache.get(sd) if ep_cache is not None else None
        if ep is None:
            ep = MazeEpisode(sd, n_frames, device)
            if ep_cache is not None:
                ep_cache[sd] = ep
        j = int(rng.integers(1, n_frames))
        if j in ep.revisit and rng.random() < p_revisit:
            e = ep.revisit[j]
        else:
            e = j - int(rng.integers(2, kmax + 1))
        ev.append(e >= 0)
        scenes += [ep.scene] * 3; cams += [ep.c2w[j], ep.c2w[j - 1], ep.c2w[max(e, 0)]]
        act.append(int(ep.actions[j]))
    c2w = torch.stack(cams)
    rgb, z, m = render_maze_multi(scenes, c2w, H)
    enc = encode_rgbd(rgb, z, m).view(B, 3, H, H, 4); pl = maze_plucker(c2w, K, H, use_pose).view(B, 3, H, H, 6)
    valid = torch.tensor(ev, device=device)
    cr = torch.stack([enc[:, 2] * valid.view(B, 1, 1, 1), enc[:, 1]], 1)
    cp = torch.stack([pl[:, 2] * valid.view(B, 1, 1, 1), pl[:, 1]], 1)
    cv = torch.stack([valid, torch.ones_like(valid)], 1)
    return {"target_rgbd": enc[:, 0], "target_plucker": pl[:, 0], "context_rgbd": cr, "context_plucker": cp,
            "context_valid": cv, "action": torch.tensor(act, device=device)}
