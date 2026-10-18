#!/usr/bin/env python3
"""Complex-route demo for layout best ckpt: full-coverage winding + multi-loop + outback.
Picks 3 mazes with max unique cells / turns / revisits. New script."""
import argparse, json, math, os, sys, time, subprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch, cv2
from loopworld.cameras import make_K
from loopworld.encoding import encode_rgbd, decode_rgbd
from loopworld.flow import euler_sample
from loopworld.maze_v2 import (sample_maze, render_maze, cells_to_poses, poses_to_c2w, revisit_pairs,
                               _bfs, _find_cycle, CELL)
from loopworld.maze_v2_data import maze_plucker, poses_to_actions
from loopworld.geom_memory2 import PointMemory2
sys.path.insert(0, os.path.dirname(__file__))
from maze_v2_mem_eval import load_model, psnr
from maze_v2_preview import topdown, write_h264, dcol


def covering_tour(scene, rng):
    start = list(scene.adj)[int(rng.integers(len(scene.adj)))]
    visited = {start}; stack = [start]; tour = [start]
    while stack:
        c = stack[-1]
        nb = [n for n in scene.adj[c] if n not in visited]
        if not nb:
            stack.pop()
            if stack: tour.append(stack[-1])
            continue
        n = nb[int(rng.integers(len(nb)))]
        visited.add(n); stack.append(n); tour.append(n)
    return tour


def complex_trajectory(scene, seed, n_frames=160):
    """Multi-loop first (revisits), then covering winding tour, then out-and-back."""
    rng = np.random.default_rng(int(seed) + 4242)
    cells = list(scene.adj.keys())
    cyc = _find_cycle(scene.adj, rng, min_len=6, max_len=12)
    if cyc is None:
        cyc = _find_cycle(scene.adj, rng, min_len=4, max_len=16)
    start = cells[int(rng.integers(len(cells)))]
    if cyc is not None:
        lead = _bfs(scene.adj, start, cyc[0])
        # 1 lead-in + 3 full laps of the cycle (same heading revisits)
        tour = lead + cyc[1:] * 3
    else:
        tour = [start]
    # then a covering DFS from current end (adds corridors / junctions not on the loop)
    cov = covering_tour(scene, rng)
    connect = _bfs(scene.adj, tour[-1], cov[0])
    tour = tour + connect[1:] + cov[1:]
    # outback to farthest cell
    far = max(cells, key=lambda c: len(_bfs(scene.adj, tour[-1], c)))
    ab = _bfs(scene.adj, tour[-1], far)
    tour = tour + ab[1:] + ab[::-1][1:]
    poses = cells_to_poses(scene, tour)
    while len(poses) < n_frames:
        poses = poses + poses[::-1][1:]
    poses = poses[:n_frames]
    acts = poses_to_actions(poses)
    rv = revisit_pairs(poses, min_gap=20)
    uniq = len(set(tour))
    turns = int((np.asarray(acts) >= 2).sum())
    return poses_to_c2w(poses), poses, tour, acts, rv, uniq, turns


def pick_seeds(candidates, n=3, n_frames=160):
    scored = []
    for s in candidates:
        sc = sample_maze(s)
        _, poses, tour, acts, rv, uniq, turns = complex_trajectory(sc, s, n_frames)
        # prefer: many unique cells, many turns, many revisits, and not tiny cycles
        scored.append((len(rv), uniq, turns, len(set(tour[:50])), s))
    scored.sort(reverse=True)
    picked = []
    for row in scored:
        if row[0] >= 20 and row[1] >= 40 and row[2] >= 12:
            picked.append(row)
        if len(picked) >= n: break
    if len(picked) < n:
        picked = scored[:n]
    return picked


@torch.no_grad()
def rollout(model, scene, c2w, poses, acts, rv, H=96, R=4, steps=5, stride=4, device="cuda"):
    K = make_K(H, device=device); T = c2w.shape[0]
    c2w = c2w.to(device)
    rgb, z, m = render_maze(scene, c2w, H)
    PL = maze_plucker(c2w, K, H, True)
    ACT = torch.tensor(acts, device=device)
    preds = torch.zeros(T, H, H, 4, device=device)
    preds[0] = encode_rgbd(rgb[0], z[0], m[0])
    mem = PointMemory2(); mem.integrate(rgb[0], z[0], m[0], c2w[0], K)
    mem_rgb = torch.zeros(T, H, H, 3, device=device)
    gen = torch.Generator(device=device).manual_seed(1234)
    for j in range(1, T):
        mr, mz, mm = mem.reproject(c2w[j], K, H)
        mem_rgb[j] = mr * mm[..., None]
        cr = torch.stack([encode_rgbd(mr, mz, mm), preds[j - 1]], 0)[None]
        cp = torch.stack([PL[j], PL[j - 1]], 0)[None]
        cv = torch.tensor([[mm.any().item(), True]], device=device)
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
        e, poses, rv, stats, v = it["ep"], it["poses"], it["rv"], it["stats"], it["vis"]
        rvj = {j for j, _ in rv}
        scene = e  # actually MazeScene
        for j in range(v["rgb"].shape[0]):
            g = (v["rgb"][j].numpy() * 255).astype(np.uint8)
            p = (v["pr"][j].numpy() * 255).astype(np.uint8)
            m = (v["mem"][j].numpy() * 255).astype(np.uint8)
            gd = dcol(v["z"][j].numpy(), v["m"][j].numpy()); pd = dcol(v["pz"][j].numpy(), v["pv"][j].numpy())
            top = np.concatenate([up(g), up(p), up(m)], 1)
            bot = np.concatenate([up(gd), up(pd), np.zeros((S, S, 3), np.uint8)], 1)
            grid = np.concatenate([top, bot], 0)
            td = topdown(scene, poses, j, rvj, S=2 * S)
            f = np.concatenate([grid, td], 1)
            hdr = np.zeros((48, f.shape[1], 3), np.uint8)
            lab = (f"layout-v3 best | seed {it['seed']} COMPLEX winding | "
                   f"cells {stats['uniq_cells']}/36 turns {stats['turns']} revisits {stats['n_revisit']} | "
                   f"frame {j}/{v['rgb'].shape[0]-1} PSNR {v['ps'][j]:.1f}" + ("  REVISIT" if j in rvj else ""))
            cv2.putText(hdr, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(hdr, "top: GT | pred (own-memory AR) | memory hint   bottom: GT depth | pred depth      right: map + path",
                        (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            frames.append(np.concatenate([hdr, f], 0))
    write_h264(frames, path, fps=fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/maze_layout.pt")
    ap.add_argument("--out", default="exp_records/maze_complex_demo")
    ap.add_argument("--n", type=int, default=3); ap.add_argument("--frames", type=int, default=160)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    cand = list(range(600000, 600032)) + list(range(700000, 700100)) + list(range(800000, 800050))
    picked = pick_seeds(cand, a.n, a.frames)
    print("PICKED", picked, flush=True)
    model, ck = load_model(a.ckpt, "cuda")
    items = []; stats_all = []
    for uniq, turns, nrv, early, seed in picked:
        sc = sample_maze(seed)
        c2w, poses, tour, acts, rv, uniq, turns = complex_trajectory(sc, seed, a.frames)
        st = dict(seed=int(seed), uniq_cells=int(uniq), turns=int(turns), n_revisit=len(rv),
                  first_revisit=rv[0] if rv else None, path_cell_steps=len(tour), n_frames=a.frames,
                  mean_psnr=None)
        print("ROLL", st, flush=True)
        vis = rollout(model, sc, c2w, poses, acts, rv)
        st["mean_psnr"] = float(vis["ps"][1:].mean())
        st["psnr_seg"] = {"1-8": float(vis["ps"][1:9].mean()), "9-32": float(vis["ps"][9:33].mean()),
                          "33-64": float(vis["ps"][33:65].mean()), "65+": float(vis["ps"][65:].mean())}
        items.append(dict(seed=seed, ep=sc, poses=poses, rv=rv, stats=st, vis=vis))
        stats_all.append(st)
    make_video(items, os.path.join(a.out, "layout_complex_h264.mp4"))
    json.dump(stats_all, open(os.path.join(a.out, "path_stats.json"), "w"), indent=1)
    print("DONE", stats_all, flush=True)

if __name__ == "__main__":
    main()
