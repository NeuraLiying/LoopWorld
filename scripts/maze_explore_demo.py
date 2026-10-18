#!/usr/bin/env python3
"""Explore-feel demo: pure DFS covering tour (no multi-lap cycles). Demo-only path design.
Prefer new cells; turn into unexplored branches at junctions; backtrack only when stuck;
optional late return to start. Layout best ckpt, same video panels."""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch, cv2
from loopworld.cameras import make_K
from loopworld.encoding import encode_rgbd, decode_rgbd
from loopworld.flow import euler_sample
from loopworld.maze_v2 import sample_maze, render_maze, cells_to_poses, poses_to_c2w, revisit_pairs, _bfs, CELL
from loopworld.maze_v2_data import maze_plucker, poses_to_actions
from loopworld.geom_memory2 import PointMemory2
sys.path.insert(0, os.path.dirname(__file__))
from maze_v2_mem_eval import load_model, psnr
from maze_v2_preview import topdown, write_h264, dcol


def explore_dfs(scene, seed, prefer_far=True):
    """Single-pass recursive-style DFS: at each cell, prefer unexplored neighbors (random among them).
    Backtrack when dead-end. Returns cell tour including backtrack steps."""
    rng = np.random.default_rng(int(seed) + 7777)
    # start at a corner / border cell for "enter the maze" feel
    border = [c for c in scene.adj if c[0] in (0, scene.nx - 1) or c[1] in (0, scene.ny - 1)]
    start = border[int(rng.integers(len(border)))]
    visited = {start}
    stack = [start]
    tour = [start]
    while stack:
        c = stack[-1]
        unseen = [n for n in scene.adj[c] if n not in visited]
        if not unseen:
            stack.pop()
            if stack:
                tour.append(stack[-1])  # backtrack
            continue
        # prefer neighbor that leads toward less-visited region (degree of unexplored)
        if prefer_far and len(unseen) > 1:
            scores = []
            for n in unseen:
                # how many unseen cells reachable from n without going through visited (approx: unseen neighbors of n)
                bonus = sum(1 for m in scene.adj[n] if m not in visited)
                scores.append(bonus + rng.random() * 0.1)
            n = unseen[int(np.argmax(scores))]
        else:
            n = unseen[int(rng.integers(len(unseen)))]
        visited.add(n)
        stack.append(n)
        tour.append(n)
    return tour, start


def explore_trajectory(scene, seed, n_frames=160, late_return=True):
    tour, start = explore_dfs(scene, seed)
    if late_return and tour[-1] != start:
        back = _bfs(scene.adj, tour[-1], start)
        tour = tour + back[1:]
    poses = cells_to_poses(scene, tour, step=0.55, turn_deg=30.0)  # demo: faster explore coverage
    # if too short, bounce once along the DFS tree (still exploration-like: re-walk then stop)
    if len(poses) < n_frames // 2:
        poses = poses + poses[::-1][1:]
    # pad by continuing idle at end (noop) rather than re-circling — freeze last pose
    if len(poses) < n_frames:
        poses = poses + [poses[-1]] * (n_frames - len(poses))
    poses = poses[:n_frames]
    acts = poses_to_actions(poses)
    # cell index along frames: map each pose to nearest tour cell by walking progress
    # approximate unique cells over time by resampling tour onto frames
    n_tour = len(tour)
    # poses length may differ from tour due to turn-in-place frames
    # build per-pose cell by projecting xy to grid
    x0, y0 = -scene.nx * CELL / 2, -scene.ny * CELL / 2
    def cell_of(x, y):
        return (int(np.clip((x - x0) / CELL, 0, scene.nx - 1)), int(np.clip((y - y0) / CELL, 0, scene.ny - 1)))
    cells_at = [cell_of(x, y) for x, y, _ in poses]
    def uniq_by(f):
        return len(set(cells_at[:f]))
    rv = revisit_pairs(poses, min_gap=16)
    early_rv = sum(1 for j, i in rv if j < n_frames // 2)
    late_rv = sum(1 for j, i in rv if j >= n_frames // 2)
    turns = int((np.asarray(acts) >= 2).sum())
    # junction turns: turn while at a cell with deg>=3
    deg = {k: len(v) for k, v in scene.adj.items()}
    junc_turns = 0
    for t in range(1, len(poses)):
        if acts[t] >= 2:
            c = cells_at[t]
            if deg.get(c, 0) >= 3:
                junc_turns += 1
    stats = dict(
        uniq_total=len(set(cells_at)), uniq_f32=uniq_by(min(32, n_frames)), uniq_f64=uniq_by(min(64, n_frames)),
        uniq_f128=uniq_by(min(128, n_frames)), turns=turns, junction_turns=junc_turns,
        n_revisit=len(rv), early_revisits=early_rv, late_revisits=late_rv,
        tour_cells=len(tour), tour_unique=len(set(tour)), start=list(start), n_frames=n_frames,
    )
    return poses_to_c2w(poses), poses, tour, acts, rv, stats, cells_at


def pick_seeds(cands, n=3, n_frames=160):
    scored = []
    for s in cands:
        sc = sample_maze(s)
        _, poses, tour, acts, rv, st, _ = explore_trajectory(sc, s, n_frames)
        # want: high early unique, low early revisits, many junction turns, full coverage
        score = (st["uniq_f32"], st["uniq_f64"], st["junction_turns"], st["uniq_total"], -st["early_revisits"], st["turns"])
        scored.append((score, s, st))
    scored.sort(reverse=True)
    return scored[:n]


@torch.no_grad()
def rollout(model, scene, c2w, poses, acts, H=96, R=4, steps=5, stride=4, device="cuda"):
    K = make_K(H, device=device); T = c2w.shape[0]; c2w = c2w.to(device)
    rgb, z, m = render_maze(scene, c2w, H)
    PL = maze_plucker(c2w, K, H, True)
    ACT = torch.tensor(acts, device=device)
    preds = torch.zeros(T, H, H, 4, device=device); preds[0] = encode_rgbd(rgb[0], z[0], m[0])
    mem = PointMemory2(); mem.integrate(rgb[0], z[0], m[0], c2w[0], K)
    mem_rgb = torch.zeros(T, H, H, 3, device=device)
    gen = torch.Generator(device=device).manual_seed(1234)
    for j in range(1, T):
        mr, mz, mm = mem.reproject(c2w[j], K, H); mem_rgb[j] = mr * mm[..., None]
        cr = torch.stack([encode_rgbd(mr, mz, mm), preds[j - 1]], 0)[None]
        cp = torch.stack([PL[j], PL[j - 1]], 0)[None]
        cv = torch.tensor([[bool(mm.any().item()), True]], device=device)
        model.cur_action = ACT[j:j + 1]
        noise = torch.randn(1, H, H, 4, device=device, generator=gen)
        fn = lambda x, t: model.forward_velocity(x, t, PL[j:j + 1], cr, cp, cv, num_loops=R)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds[j] = euler_sample(fn, noise, steps=steps).float().clamp(-1, 1)[0]
        if j % stride == 0 or j == T - 1:
            pr, di, _ = decode_rgbd(preds[j:j + 1])
            mem.integrate(pr[0], di["z_metric"][0], di["valid_surface"][0], c2w[j], K)
    model.cur_action = None
    P, di, _ = decode_rgbd(preds)
    return dict(rgb=rgb.cpu(), z=z.cpu(), m=m.cpu(), pr=P.cpu(), pz=di["z_pred"].cpu(), pv=di["valid_surface"].cpu(),
                mem=mem_rgb.cpu(), ps=psnr(P, rgb).cpu().numpy())


def make_video(items, path, fps=12):
    frames = []; S = 192
    up = lambda x: cv2.resize(x, (S, S), interpolation=cv2.INTER_NEAREST)
    for it in items:
        scene, poses, rv, st, v = it["scene"], it["poses"], it["rv"], it["stats"], it["vis"]
        rvj = {j for j, _ in rv}
        for j in range(v["rgb"].shape[0]):
            g = (v["rgb"][j].numpy() * 255).astype(np.uint8)
            p = (v["pr"][j].numpy() * 255).astype(np.uint8)
            mimg = (v["mem"][j].numpy() * 255).astype(np.uint8)
            gd = dcol(v["z"][j].numpy(), v["m"][j].numpy()); pd = dcol(v["pz"][j].numpy(), v["pv"][j].numpy())
            top = np.concatenate([up(g), up(p), up(mimg)], 1)
            bot = np.concatenate([up(gd), up(pd), np.zeros((S, S, 3), np.uint8)], 1)
            grid = np.concatenate([top, bot], 0)
            td = topdown(scene, poses, j, rvj, S=2 * S)
            f = np.concatenate([grid, td], 1)
            hdr = np.zeros((52, f.shape[1], 3), np.uint8)
            lab = (f"layout-v3 best | seed {it['seed']} EXPLORE DFS | "
                   f"uniq@32/64/128={st['uniq_f32']}/{st['uniq_f64']}/{st['uniq_f128']} "
                   f"turns {st['turns']} (junc {st['junction_turns']}) "
                   f"rv early/late {st['early_revisits']}/{st['late_revisits']} | "
                   f"f{j}/{v['rgb'].shape[0]-1} PSNR {v['ps'][j]:.1f}" + (" REVISIT" if j in rvj else ""))
            cv2.putText(hdr, lab, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cv2.putText(hdr, "top: GT | pred | memory hint   bottom: GT depth | pred depth      right: map + explore path (backtrack only)",
                        (4, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
            frames.append(np.concatenate([hdr, f], 0))
    write_h264(frames, path, fps=fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/maze_layout.pt")
    ap.add_argument("--out", default="exp_records/maze_explore_demo")
    ap.add_argument("--n", type=int, default=3); ap.add_argument("--frames", type=int, default=160)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    cands = list(range(600000, 600032)) + list(range(700000, 700150)) + list(range(800000, 800100))
    picked = pick_seeds(cands, a.n, a.frames)
    print("PICKED", [(s, st) for _, s, st in picked], flush=True)
    model, _ = load_model(a.ckpt, "cuda")
    items = []; all_stats = []
    for score, seed, st0 in picked:
        sc = sample_maze(seed)
        c2w, poses, tour, acts, rv, st, cells_at = explore_trajectory(sc, seed, a.frames)
        print("ROLL", seed, st, flush=True)
        vis = rollout(model, sc, c2w, poses, acts)
        st["mean_psnr"] = float(vis["ps"][1:].mean())
        st["seed"] = int(seed)
        items.append(dict(seed=seed, scene=sc, poses=poses, rv=rv, stats=st, vis=vis))
        all_stats.append(st)
    make_video(items, os.path.join(a.out, "layout_explore_h264.mp4"))
    json.dump(all_stats, open(os.path.join(a.out, "path_stats.json"), "w"), indent=1)
    print("DONE", all_stats, flush=True)

if __name__ == "__main__":
    main()
