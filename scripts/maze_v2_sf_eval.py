#!/usr/bin/env python3
"""Own-memory AR eval with PointMemory2 (self-forcing run). Modes own / oracle_mem."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch
from loopworld.cameras import make_K
from loopworld.encoding import encode_rgbd, decode_rgbd
from loopworld.flow import euler_sample
from loopworld.maze_v2 import render_maze
from loopworld.maze_v2_data import MazeEpisode, maze_plucker
from loopworld.geom_memory2 import PointMemory2
sys.path.insert(0, os.path.dirname(__file__))
from maze_v2_mem_eval import load_model, TEST_SEEDS, SEGS, psnr


@torch.no_grad()
def run_sf_eval(model, mode="own", n_scenes=8, T=128, R=4, steps=5, stride=4, chunk=16, dev="cuda", mem_kw=None):
    H = model.image_size; K = make_K(H, device=dev); was = model.training; model.eval()
    eps = [MazeEpisode(s, T, dev) for s in TEST_SEEDS[:n_scenes]]; N = len(eps)
    gt = [render_maze(e.scene, e.c2w, H) for e in eps]
    Gr = torch.stack([g[0] for g in gt]); Gz = torch.stack([g[1] for g in gt]); Gm = torch.stack([g[2] for g in gt])
    Genc = encode_rgbd(Gr, Gz, Gm)
    PL = torch.stack([maze_plucker(e.c2w, K, H, True) for e in eps])
    ACT = torch.tensor(np.stack([e.actions for e in eps]), device=dev)
    preds = torch.zeros(N, T, H, H, 4, device=dev); preds[:, 0] = Genc[:, 0]
    mems = [PointMemory2(**(mem_kw or {})) for _ in range(N)]
    for n in range(N): mems[n].integrate(Gr[n, 0], Gz[n, 0], Gm[n, 0], eps[n].c2w[0], K)
    cov = np.zeros((N, T)); hint = []
    gen = torch.Generator(device=dev).manual_seed(1234)
    for j in range(1, T):
        rs = [mems[n].reproject(eps[n].c2w[j], K, H) for n in range(N)]
        mr = torch.stack([r[0] for r in rs]); mz = torch.stack([r[1] for r in rs]); mm = torch.stack([r[2] for r in rs])
        cov[:, j] = mm.float().mean((1, 2)).cpu().numpy()
        for n, e in enumerate(eps):
            if j in e.revisit:
                k = mm[n] & Gm[n, j]
                if k.any(): hint.append(float(10 * torch.log10(1 / ((mr[n][k] - Gr[n, j][k]) ** 2).mean().clamp_min(1e-10))))
        prev = Genc[:, j - 1] if mode == "oracle_mem_gtprev" else preds[:, j - 1]
        cr = torch.stack([encode_rgbd(mr, mz, mm), prev], 1); cp = torch.stack([PL[:, j], PL[:, j - 1]], 1)
        cv = torch.stack([mm.flatten(1).any(1), torch.ones(N, dtype=torch.bool, device=dev)], 1)
        noise = torch.randn(N, H, H, 4, device=dev, generator=gen); outs = []
        for s in range(0, N, chunk):
            sl = slice(s, s + chunk); model.cur_action = ACT[sl, j]
            fn = lambda x, t, sl=sl: model.forward_velocity(x, t, PL[sl, j], cr[sl], cp[sl], cv[sl], num_loops=R)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs.append(euler_sample(fn, noise[sl], steps=steps).float())
        preds[:, j] = torch.cat(outs).clamp(-1, 1)
        if j % stride == 0 or j == T - 1:
            if mode == "own":
                pr, di, _ = decode_rgbd(preds[:, j])
                for n in range(N): mems[n].integrate(pr[n], di["z_metric"][n], di["valid_surface"][n], eps[n].c2w[j], K)
            else:
                for n in range(N): mems[n].integrate(Gr[n, j], Gz[n, j], Gm[n, j], eps[n].c2w[j], K)
    model.cur_action = None; model.train(was)
    P, di, _ = decode_rgbd(preds); pz = di["z_metric"]; ps = psnr(P, Gr).cpu().numpy()
    ar = ((((pz - Gz).abs() / Gz) * Gm).sum((-2, -1)) / Gm.sum((-2, -1)).clamp_min(1)).cpu().numpy()
    selfc, gtr = [], []
    for n, e in enumerate(eps):
        for j, i in e.revisit.items(): selfc.append(float(psnr(P[n, j], P[n, i]))); gtr.append(float(ps[n, j]))
    return {"mode": mode, "psnr_all": float(ps[:, 1:].mean()), "psnr_seg": {k: float(ps[:, a:b].mean()) for k, (a, b) in SEGS.items()},
            "absrel_all": float(ar[:, 1:].mean()), "revisit": {"psnr_pred_vs_gt_at_revisit": float(np.mean(gtr)), "psnr_pred_vs_own_first": float(np.mean(selfc)), "n_pairs": len(gtr)},
            "mem_cov": float(cov[:, 1:].mean()), "hint_psnr_revisit": float(np.mean(hint)) if hint else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="own,oracle_mem"); ap.add_argument("--n-scenes", type=int, default=32)
    ap.add_argument("--splat", type=int, default=2); ap.add_argument("--hole-fill", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    m, ck = load_model(a.ckpt, "cuda"); res = {"ckpt": a.ckpt, "step": ck.get("step"), "results": []}
    for md in a.modes.split(","):
        t0 = time.time(); r = run_sf_eval(m, md, a.n_scenes, mem_kw=dict(splat=a.splat, hole_fill=a.hole_fill)); r["sec"] = time.time() - t0
        res["results"].append(r); print(json.dumps(r), flush=True); json.dump(res, open(os.path.join(a.out, "sf_eval.json"), "w"), indent=1)
