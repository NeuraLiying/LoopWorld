"""LoopWorld scene generator v2: procedural grid maze with winding corridors.

New, self-contained module (does not change v1 data path). Output format matches v1:
RGB in [0,1], camera z-depth, hit mask (sky = miss), c2w (OpenCV cam axes, z-up world),
make_K(96, 60deg), so encode_rgbd / plucker_rays work unchanged.

Scene: W x H grid of cells (cell size CELL), maze carved by randomized DFS, then
"braided" (extra walls removed) so that cycles exist -> same-heading revisits.
Walls are thin AABBs with procedural world-space textures (consistent across views),
regions get distinct wall palettes, landmarks (tall colored pillars / spheres) sit in
selected cells, small objects (boxes) add occlusion.
Maze is centered at world origin to keep Plücker moments (o x d) bounded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch

from .cameras import make_K, look_at_c2w, pixel_rays_cam

CELL = 1.6
WALL_T = 0.08
WALL_H = 1.0
CAM_H = 0.45
PITCH_DEG = -6.0
Z_FAR_SCENE = 11.5  # max visible distance must stay < encoding Z_FAR=12

PALETTES = [  # region wall base colors
    (0.80, 0.42, 0.30), (0.30, 0.55, 0.80), (0.45, 0.72, 0.38), (0.85, 0.75, 0.35),
    (0.62, 0.40, 0.75), (0.35, 0.72, 0.70), (0.80, 0.50, 0.65), (0.60, 0.60, 0.62),
]
LANDMARK_COLORS = [(0.95, 0.1, 0.1), (0.1, 0.3, 0.95), (0.98, 0.85, 0.05), (0.05, 0.85, 0.3),
                   (0.95, 0.45, 0.0), (0.85, 0.1, 0.85)]


@dataclass
class MazeScene:
    nx: int
    ny: int
    # open[(cx,cy)] -> set of neighbor cells connected
    adj: Dict[Tuple[int, int], set]
    boxes: np.ndarray        # (N,6) center xyz, half xyz
    box_col: np.ndarray      # (N,3)
    box_tex: np.ndarray      # (N,) int texture id (0 wall stripes,1 bricks,2 plain object)
    spheres: np.ndarray      # (M,4) center, r
    sph_col: np.ndarray      # (M,3)
    floor_cols: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    sky_col: Tuple[float, float, float] = (0.08, 0.09, 0.12)
    landmarks: List[Tuple[int, int, int]] = field(default_factory=list)  # (cx,cy,color_id)

    def cell_center(self, c):
        x0 = -self.nx * CELL / 2
        y0 = -self.ny * CELL / 2
        return (x0 + (c[0] + 0.5) * CELL, y0 + (c[1] + 0.5) * CELL)


def _carve(nx, ny, rng, braid):
    adj = {(i, j): set() for i in range(nx) for j in range(ny)}
    start = (int(rng.integers(nx)), int(rng.integers(ny)))
    stack, seen = [start], {start}
    while stack:
        c = stack[-1]
        nb = [(c[0] + dx, c[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        nb = [n for n in nb if n in adj and n not in seen]
        if not nb:
            stack.pop(); continue
        n = nb[int(rng.integers(len(nb)))]
        adj[c].add(n); adj[n].add(c); seen.add(n); stack.append(n)
    # braid: open extra walls to create cycles
    walls = [((i, j), (i + 1, j)) for i in range(nx - 1) for j in range(ny)] + \
            [((i, j), (i, j + 1)) for i in range(nx) for j in range(ny - 1)]
    closed = [w for w in walls if w[1] not in adj[w[0]]]
    rng.shuffle(closed)
    for a, b in closed[: int(len(closed) * braid)]:
        adj[a].add(b); adj[b].add(a)
    return adj


def sample_maze(seed: int, nx: int = 6, ny: int = 6, braid: float = 0.15) -> MazeScene:
    rng = np.random.default_rng(int(seed))
    adj = _carve(nx, ny, rng, braid)
    x0, y0 = -nx * CELL / 2, -ny * CELL / 2
    n_reg = 4
    reg_col = [PALETTES[k] for k in rng.permutation(len(PALETTES))[:n_reg]]
    reg_tex = rng.integers(0, 2, n_reg)

    def region(cx, cy):
        return (2 if cx >= nx / 2 else 0) + (1 if cy >= ny / 2 else 0)

    boxes, cols, texs = [], [], []

    def add_wall(cx, cy, xmin, xmax, ymin, ymax):
        r = region(cx, cy)
        jitter = rng.uniform(-0.05, 0.05, 3)
        boxes.append([(xmin + xmax) / 2, (ymin + ymax) / 2, WALL_H / 2, (xmax - xmin) / 2, (ymax - ymin) / 2, WALL_H / 2])
        cols.append(np.clip(np.array(reg_col[r]) + jitter, 0, 1)); texs.append(int(reg_tex[r]))

    h = WALL_T / 2
    for i in range(nx):
        for j in range(ny):
            xa, xb = x0 + i * CELL, x0 + (i + 1) * CELL
            ya, yb = y0 + j * CELL, y0 + (j + 1) * CELL
            if (i + 1, j) not in adj[(i, j)]:  # east wall
                add_wall(i, j, xb - h, xb + h, ya - h, yb + h)
            if (i, j + 1) not in adj[(i, j)]:  # north wall
                add_wall(i, j, xa - h, xb + h, yb - h, yb + h)
            if i == 0:
                add_wall(i, j, xa - h, xa + h, ya - h, yb + h)
            if j == 0:
                add_wall(i, j, xa - h, xb + h, ya - h, ya + h)
    # landmarks: tall pillars in a few cells (corner of cell, not blocking center line)
    cells = list(adj.keys())
    order = rng.permutation(len(cells))
    landmarks = []
    spheres, sph_col = [], []
    lm_ids = rng.permutation(len(LANDMARK_COLORS))
    for k in range(4):
        c = cells[order[k]]
        cx, cy = x0 + (c[0] + 0.5) * CELL, y0 + (c[1] + 0.5) * CELL
        ox, oy = rng.choice([-1, 1], 2) * CELL * 0.32
        col = LANDMARK_COLORS[lm_ids[k]]
        if k % 2 == 0:
            boxes.append([cx + ox, cy + oy, 0.7, 0.1, 0.1, 0.7]); cols.append(np.array(col)); texs.append(2)
            spheres.append([cx + ox, cy + oy, 1.55, 0.18]); sph_col.append(col)
        else:
            spheres.append([cx + ox, cy + oy, 0.18, 0.18]); sph_col.append(col)
        landmarks.append((c[0], c[1], int(lm_ids[k])))
    # small occluding objects
    for k in range(4, 4 + 6):
        c = cells[order[k]]
        cx, cy = x0 + (c[0] + 0.5) * CELL, y0 + (c[1] + 0.5) * CELL
        ox, oy = rng.choice([-1, 1], 2) * CELL * 0.32
        s = rng.uniform(0.08, 0.13)
        hz = rng.uniform(0.1, 0.3)
        boxes.append([cx + ox, cy + oy, hz, s, s, hz]); cols.append(rng.uniform(0.25, 0.95, 3)); texs.append(2)
    fc = tuple(rng.uniform(0.35, 0.55, 3).tolist())
    fc2 = tuple((np.array(fc) * 0.75).tolist())
    return MazeScene(nx, ny, adj, np.array(boxes, np.float32), np.array(cols, np.float32), np.array(texs),
                     np.array(spheres, np.float32).reshape(-1, 4), np.array(sph_col, np.float32).reshape(-1, 3),
                     (fc, fc2), landmarks=landmarks)


# ---------------------------------------------------------------- rendering
@torch.no_grad()
def render_maze(scene: MazeScene, c2w: torch.Tensor, image_size: int = 96, fov_deg: float = 60.0, chunk: int = 32):
    """Returns rgb (B,H,W,3), z-depth (B,H,W), mask (B,H,W). Miss (sky) -> mask False, z=12."""
    single = c2w.ndim == 2
    if single:
        c2w = c2w[None]
    outs = [_render_chunk(scene, c2w[i:i + chunk], image_size, fov_deg) for i in range(0, c2w.shape[0], chunk)]
    rgb, z, m = [torch.cat(x) for x in zip(*outs)]
    return (rgb[0], z[0], m[0]) if single else (rgb, z, m)


def _render_chunk(scene, c2w, S, fov_deg):
    dev = c2w.device
    c2w = c2w.float()
    K = make_K(S, fov_deg, device=dev)
    dc = pixel_rays_cam(S, K, device=dev)
    dc = dc / dc.norm(dim=-1, keepdim=True)
    B = c2w.shape[0]
    d = torch.einsum("bij,hwj->bhwi", c2w[:, :3, :3], dc)  # B,H,W,3
    o = c2w[:, None, None, :3, 3].expand_as(d)
    inf = torch.tensor(float("inf"), device=dev)
    # floor z=0 within maze bounds
    hx, hy = scene.nx * CELL / 2, scene.ny * CELL / 2
    tf = -o[..., 2] / torch.where(d[..., 2].abs() < 1e-8, torch.full_like(d[..., 2], -1e-8), d[..., 2])
    pf = o + tf[..., None] * d
    okf = (tf > 1e-4) & (pf[..., 0].abs() <= hx) & (pf[..., 1].abs() <= hy)
    best = torch.where(okf, tf, inf)
    kind = torch.zeros_like(best, dtype=torch.long)  # 0 floor
    idx = torch.zeros_like(kind)
    # boxes (all at once)
    bx = torch.as_tensor(scene.boxes, device=dev)
    if len(bx):
        mn, mx = bx[:, :3] - bx[:, 3:], bx[:, :3] + bx[:, 3:]
        ds = torch.where(d.abs() < 1e-8, torch.full_like(d, 1e-8), d)[..., None, :]
        t0 = (mn - o[..., None, :]) / ds
        t1 = (mx - o[..., None, :]) / ds
        tmin = torch.minimum(t0, t1).amax(-1)
        tmax = torch.maximum(t0, t1).amin(-1)
        tb = torch.where((tmax >= tmin) & (tmin > 1e-4), tmin, inf)
        tbm, bi = tb.min(-1)
        c = tbm < best
        best = torch.where(c, tbm, best); kind = torch.where(c, 1, kind); idx = torch.where(c, bi, idx)
    sp = torch.as_tensor(scene.spheres, device=dev)
    if len(sp):
        oc = o[..., None, :] - sp[:, :3]
        b = (oc * d[..., None, :]).sum(-1)
        cc = (oc * oc).sum(-1) - sp[:, 3] ** 2
        disc = b * b - cc
        ts = -b - disc.clamp_min(0).sqrt()
        ts = torch.where((disc >= 0) & (ts > 1e-4), ts, inf)
        tsm, si = ts.min(-1)
        c = tsm < best
        best = torch.where(c, tsm, best); kind = torch.where(c, 2, kind); idx = torch.where(c, si, idx)
    hit = torch.isfinite(best)
    p = o + torch.where(hit, best, torch.zeros_like(best))[..., None] * d
    rgb = torch.empty(B, S, S, 3, device=dev)
    rgb[:] = torch.tensor(scene.sky_col, device=dev)
    # floor: checker of 0.4 + darkening with distance-free shading
    fc = torch.tensor(scene.floor_cols, device=dev)
    chk = ((torch.floor(p[..., 0] / 0.4) + torch.floor(p[..., 1] / 0.4)) % 2).long()
    m0 = hit & (kind == 0)
    rgb = torch.where(m0[..., None], fc[chk], rgb)
    # boxes: normal from closest face, world-space texture
    if len(bx):
        m1 = hit & (kind == 1)
        idx_b = idx.clamp(max=len(bx) - 1)
        cen, hal = bx[idx_b, :3], bx[idx_b, 3:]
        q = (p - cen) / hal
        ax = q.abs().argmax(-1)  # 0:x face,1:y face,2:top
        col = torch.as_tensor(scene.box_col, device=dev)[idx_b]
        tex = torch.as_tensor(scene.box_tex, device=dev)[idx_b]
        shade = torch.tensor([0.78, 1.0, 1.15], device=dev)[ax]
        u = torch.where(ax == 0, p[..., 1], p[..., 0])
        stripe = 0.82 + 0.18 * (torch.floor(u / 0.2) % 2)
        band = torch.where((p[..., 2] > 0.62) & (p[..., 2] < 0.7), 0.6, 1.0)
        brick_off = (torch.floor(p[..., 2] / 0.125) % 2) * 0.15
        mortar = ((((u + brick_off) / 0.3) % 1) < 0.08) | (((p[..., 2] / 0.125) % 1) < 0.12)
        brick = torch.where(mortar, 0.55, 1.0)
        f = torch.where(tex == 0, stripe * band, torch.where(tex == 1, brick, torch.ones_like(stripe)))
        rgb = torch.where(m1[..., None], (col * (f * shade)[..., None]).clamp(0, 1), rgb)
    if len(sp):
        m2 = hit & (kind == 2)
        idx = idx.clamp(max=len(sp) - 1)
        n = (p - sp[idx, :3]) / sp[idx, 3:4]
        lam = 0.55 + 0.45 * (n * torch.tensor([0.3, 0.4, 0.866], device=dev)).sum(-1).clamp(0, 1)
        rgb = torch.where(m2[..., None], (torch.as_tensor(scene.sph_col, device=dev)[idx] * lam[..., None]).clamp(0, 1), rgb)
    z = best * dc[..., 2]
    hit = hit & (z < Z_FAR_SCENE)
    z = torch.where(hit, z, torch.full_like(z, 12.0))
    rgb = torch.where(hit[..., None], rgb, torch.tensor(scene.sky_col, device=dev))
    return rgb, z, hit


# ---------------------------------------------------------------- trajectories
def _bfs(adj, a, b):
    prev, q = {a: None}, [a]
    while q:
        c = q.pop(0)
        if c == b:
            break
        for n in adj[c]:
            if n not in prev:
                prev[n] = c; q.append(n)
    if b not in prev:
        return [a]
    path = [b]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return path[::-1]


def _find_cycle(adj, rng, min_len=4, max_len=11):
    """A simple cycle through the braided maze: pick an extra edge (a,b), path a->b avoiding that edge."""
    edges = [(a, b) for a in adj for b in adj[a] if a < b]
    rng.shuffle(edges)
    for a, b in edges:
        adj2 = {k: set(v) for k, v in adj.items()}
        adj2[a].discard(b); adj2[b].discard(a)
        p = _bfs(adj2, a, b)
        if p[0] == a and p[-1] == b and min_len <= len(p) <= max_len:
            return p + [a]  # closed loop
    return None


def cells_to_poses(scene, cells, step=0.4, turn_deg=22.5):
    """Smooth walk along cell centers: translate in `step` increments, rotate in place at turns."""
    pts = [np.array(scene.cell_center(c)) for c in cells]
    poses = []  # (x,y,yaw)
    yaw = math.atan2(*(pts[1] - pts[0])[::-1])
    pos = pts[0].copy()
    poses.append((pos[0], pos[1], yaw))
    for k in range(1, len(pts)):
        dv = pts[k] - pts[k - 1]
        tgt = math.atan2(dv[1], dv[0])
        dyaw = (tgt - yaw + math.pi) % (2 * math.pi) - math.pi
        n_t = int(round(abs(dyaw) / math.radians(turn_deg)))
        for i in range(1, n_t + 1):
            poses.append((pos[0], pos[1], yaw + dyaw * i / n_t))
        yaw = tgt
        n_s = int(round(np.linalg.norm(dv) / step))
        for i in range(1, n_s + 1):
            q = pts[k - 1] + dv * i / n_s
            poses.append((q[0], q[1], yaw))
        pos = pts[k].copy()
    return poses


def poses_to_c2w(poses, device="cpu"):
    out = []
    pr = math.radians(PITCH_DEG)
    for x, y, yaw in poses:
        eye = torch.tensor([x, y, CAM_H])
        fwd = torch.tensor([math.cos(yaw) * math.cos(pr), math.sin(yaw) * math.cos(pr), math.sin(pr)])
        out.append(look_at_c2w(eye, eye + fwd))
    return torch.stack(out).float().to(device)


def sample_maze_trajectory(scene: MazeScene, seed: int, n_frames: int = 128, kind: str = "auto"):
    """Walk-through trajectory with guaranteed revisits.

    kinds:
      loop    : start -> walk to a cycle -> go around the cycle twice (same-heading revisit)
      outback : A -> B -> A -> B (revisit A..B corridor in both directions; 3rd leg = same heading as 1st)
    Returns c2w (n_frames,4,4), poses list, cells list, kind.
    """
    rng = np.random.default_rng(int(seed) + 7919)
    if kind == "auto":
        kind = "loop" if rng.random() < 0.6 else "outback"
    cells = None
    if kind == "loop":
        cyc = _find_cycle(scene.adj, rng)
        if cyc is None:
            kind = "outback"
        else:
            near = [c for c in scene.adj if len(_bfs(scene.adj, c, cyc[0])) <= 3]
            start = near[int(rng.integers(len(near)))]
            lead = _bfs(scene.adj, start, cyc[0])
            cells = lead + cyc[1:] * 4
    if kind == "outback":
        ks = list(scene.adj)
        a = ks[int(rng.integers(len(ks)))]
        cand = [c for c in ks if 6 <= len(_bfs(scene.adj, a, c)) <= 9] or [max(ks, key=lambda c: len(_bfs(scene.adj, a, c)))]
        b = cand[int(rng.integers(len(cand)))]
        ab = _bfs(scene.adj, a, b)
        cells = ab + ab[::-1][1:] + ab[1:] + ab[::-1][1:]
    poses = cells_to_poses(scene, cells)
    while len(poses) < n_frames:  # extend by bouncing
        poses = poses + poses[::-1][1:]
    poses = poses[:n_frames]
    return poses_to_c2w(poses), poses, cells, kind


def revisit_pairs(poses, min_gap=24, pos_tol=0.25, yaw_tol_deg=12.0):
    """For each frame j, earliest frame i<j-min_gap with ~same pose (first visit). Returns list of (j,i)."""
    P = np.array(poses)
    out = []
    for j in range(len(P)):
        for i in range(0, j - min_gap):
            dp = np.hypot(*(P[j, :2] - P[i, :2]))
            dy = abs((P[j, 2] - P[i, 2] + math.pi) % (2 * math.pi) - math.pi)
            if dp < pos_tol and dy < math.radians(yaw_tol_deg):
                out.append((j, i)); break
    return out


# ---------------------------------------------------------------- multi-scene batched rendering
_PACK_CACHE = {}


def _pack(scene, dev):
    key = (id(scene), str(dev))
    if key not in _PACK_CACHE:
        _PACK_CACHE[key] = dict(bx=torch.as_tensor(scene.boxes, device=dev), bc=torch.as_tensor(scene.box_col, device=dev),
                                bt=torch.as_tensor(scene.box_tex, device=dev), sp=torch.as_tensor(scene.spheres, device=dev),
                                sc=torch.as_tensor(scene.sph_col, device=dev), fc=torch.tensor(scene.floor_cols, device=dev))
        if len(_PACK_CACHE) > 50000: _PACK_CACHE.clear()
    return _PACK_CACHE[key]


@torch.no_grad()
def render_maze_multi(scenes, c2w, image_size=96, fov_deg=60.0):
    """scenes: list (len M) of MazeScene, one per camera; c2w (M,4,4). Same output as render_maze.
    Pixel-identical to render_maze (same math), but one kernel set for all cameras."""
    dev = c2w.device; M = c2w.shape[0]; S = image_size
    P = [_pack(s, dev) for s in scenes]
    nb = max(p["bx"].shape[0] for p in P); ns = max(max(p["sp"].shape[0] for p in P), 1)
    bx = torch.zeros(M, nb, 6, device=dev); bv = torch.zeros(M, nb, dtype=torch.bool, device=dev)
    bc = torch.zeros(M, nb, 3, device=dev); bt = torch.zeros(M, nb, dtype=torch.long, device=dev)
    sp = torch.zeros(M, ns, 4, device=dev); sv = torch.zeros(M, ns, dtype=torch.bool, device=dev); sc = torch.zeros(M, ns, 3, device=dev)
    fc = torch.stack([p["fc"] for p in P])
    for i, p in enumerate(P):
        n = p["bx"].shape[0]; bx[i, :n] = p["bx"]; bv[i, :n] = True; bc[i, :n] = p["bc"]; bt[i, :n] = p["bt"]
        n = p["sp"].shape[0]; sp[i, :n] = p["sp"]; sv[i, :n] = True; sc[i, :n] = p["sc"]
    bx[~bv] = torch.tensor([0, 0, 0, 1e-3, 1e-3, 1e-3], device=dev); sp[~sv, 3] = 1e-3
    c2w = c2w.float(); K = make_K(S, fov_deg, device=dev)
    dc = pixel_rays_cam(S, K, device=dev); dc = dc / dc.norm(dim=-1, keepdim=True)
    d = torch.einsum("bij,hwj->bhwi", c2w[:, :3, :3], dc); o = c2w[:, None, None, :3, 3].expand_as(d)
    inf = torch.tensor(float("inf"), device=dev)
    hx, hy = scenes[0].nx * CELL / 2, scenes[0].ny * CELL / 2
    tf = -o[..., 2] / torch.where(d[..., 2].abs() < 1e-8, torch.full_like(d[..., 2], -1e-8), d[..., 2])
    pf = o + tf[..., None] * d
    best = torch.where((tf > 1e-4) & (pf[..., 0].abs() <= hx) & (pf[..., 1].abs() <= hy), tf, inf)
    kind = torch.zeros_like(best, dtype=torch.long); idx = torch.zeros_like(kind)
    B4 = lambda x: x[:, None, None]  # (M,N,..)->(M,1,1,N,..)
    mn, mx = B4(bx[..., :3] - bx[..., 3:]), B4(bx[..., :3] + bx[..., 3:])
    ds = torch.where(d.abs() < 1e-8, torch.full_like(d, 1e-8), d)[..., None, :]
    t0 = (mn - o[..., None, :]) / ds; t1 = (mx - o[..., None, :]) / ds
    tmin = torch.minimum(t0, t1).amax(-1); tmax = torch.maximum(t0, t1).amin(-1)
    tb = torch.where((tmax >= tmin) & (tmin > 1e-4) & B4(bv), tmin, inf)
    tbm, bi = tb.min(-1); c = tbm < best
    best = torch.where(c, tbm, best); kind = torch.where(c, 1, kind); idx = torch.where(c, bi, idx)
    oc = o[..., None, :] - B4(sp[..., :3]); b = (oc * d[..., None, :]).sum(-1)
    disc = b * b - ((oc * oc).sum(-1) - B4(sp[..., 3]) ** 2); ts = -b - disc.clamp_min(0).sqrt()
    ts = torch.where((disc >= 0) & (ts > 1e-4) & B4(sv), ts, inf)
    tsm, si = ts.min(-1); c = tsm < best
    best = torch.where(c, tsm, best); kind = torch.where(c, 2, kind); idx = torch.where(c, si, idx)
    hit = torch.isfinite(best)
    p = o + torch.where(hit, best, torch.zeros_like(best))[..., None] * d
    g = lambda T, ix: torch.gather(T, 1, ix.reshape(M, -1, *([1] * (T.ndim - 2))).expand(M, S * S, *T.shape[2:])).reshape(M, S, S, *T.shape[2:])
    sky = torch.tensor(scenes[0].sky_col, device=dev)
    rgb = sky.expand(M, S, S, 3).clone()
    chk = ((torch.floor(p[..., 0] / 0.4) + torch.floor(p[..., 1] / 0.4)) % 2).long()
    fcol = torch.gather(fc, 1, chk.reshape(M, -1, 1).expand(M, S * S, 3)).reshape(M, S, S, 3)
    rgb = torch.where((hit & (kind == 0))[..., None], fcol, rgb)
    ib = idx.clamp(max=nb - 1); bxx = g(bx, ib)
    cen, hal = bxx[..., :3], bxx[..., 3:]
    ax = ((p - cen) / hal).abs().argmax(-1)
    col = g(bc, ib); tex = g(bt, ib)
    shade = torch.tensor([0.78, 1.0, 1.15], device=dev)[ax]
    u = torch.where(ax == 0, p[..., 1], p[..., 0])
    stripe = 0.82 + 0.18 * (torch.floor(u / 0.2) % 2)
    band = torch.where((p[..., 2] > 0.62) & (p[..., 2] < 0.7), 0.6, 1.0)
    brick_off = (torch.floor(p[..., 2] / 0.125) % 2) * 0.15
    mortar = ((((u + brick_off) / 0.3) % 1) < 0.08) | (((p[..., 2] / 0.125) % 1) < 0.12)
    f = torch.where(tex == 0, stripe * band, torch.where(tex == 1, torch.where(mortar, 0.55, 1.0), torch.ones_like(stripe)))
    rgb = torch.where((hit & (kind == 1))[..., None], (col * (f * shade)[..., None]).clamp(0, 1), rgb)
    isp = idx.clamp(max=ns - 1); spp = g(sp, isp)
    n = (p - spp[..., :3]) / spp[..., 3:4]
    lam = 0.55 + 0.45 * (n * torch.tensor([0.3, 0.4, 0.866], device=dev)).sum(-1).clamp(0, 1)
    rgb = torch.where((hit & (kind == 2))[..., None], (g(sc, isp) * lam[..., None]).clamp(0, 1), rgb)
    z = best * dc[..., 2]
    hit = hit & (z < Z_FAR_SCENE)
    z = torch.where(hit, z, torch.full_like(z, 12.0))
    rgb = torch.where(hit[..., None], rgb, sky)
    return rgb, z, hit
