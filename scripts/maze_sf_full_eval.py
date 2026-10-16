#!/usr/bin/env python3
"""Full 32-maze diag for SF run (PointMemory2): modes own/oracle_mem/oracle_mem_gtprev/nomem,
AbsRel breakdown, optional H.264 video (RGB | depth | memory hint | top-down map). New script."""
import argparse, json, os, sys, time, math, subprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch, cv2
from loopworld.cameras import make_K
from loopworld.encoding import encode_rgbd, decode_rgbd
from loopworld.flow import euler_sample
from loopworld.maze_v2 import render_maze, CELL
from loopworld.maze_v2_data import MazeEpisode, maze_plucker
from loopworld.geom_memory2 import PointMemory2
sys.path.insert(0, os.path.dirname(__file__))
from maze_v2_mem_eval import load_model, TEST_SEEDS, SEGS, psnr
from maze_v2_preview import topdown, write_h264, dcol


@torch.no_grad()
def run(model, mode, n_scenes=32, T=128, R=4, steps=5, stride=4, chunk=16, dev="cuda", video=0):
    H = model.image_size; K = make_K(H, device=dev); model.eval()
    eps = [MazeEpisode(s, T, dev) for s in TEST_SEEDS[:n_scenes]]; N = len(eps)
    gt = [render_maze(e.scene, e.c2w, H) for e in eps]
    Gr = torch.stack([g[0] for g in gt]); Gz = torch.stack([g[1] for g in gt]); Gm = torch.stack([g[2] for g in gt])
    Genc = encode_rgbd(Gr, Gz, Gm)
    PL = torch.stack([maze_plucker(e.c2w, K, H, True) for e in eps])
    ACT = torch.tensor(np.stack([e.actions for e in eps]), device=dev)
    preds = torch.zeros(N, T, H, H, 4, device=dev); preds[:, 0] = Genc[:, 0]
    mems = [PointMemory2() for _ in range(N)]
    for n in range(N): mems[n].integrate(Gr[n, 0], Gz[n, 0], Gm[n, 0], eps[n].c2w[0], K)
    cov = np.zeros((N, T)); hint = []; vmem = torch.zeros(min(video, N), T, H, H, 3)
    gen = torch.Generator(device=dev).manual_seed(1234)
    for j in range(1, T):
        rs = [mems[n].reproject(eps[n].c2w[j], K, H) for n in range(N)]
        mr = torch.stack([r[0] for r in rs]); mz = torch.stack([r[1] for r in rs]); mm = torch.stack([r[2] for r in rs])
        if video: vmem[:, j] = (mr[:video] * mm[:video, ..., None]).cpu()
        cov[:, j] = mm.float().mean((1, 2)).cpu().numpy()
        for n, e in enumerate(eps):
            if j in e.revisit:
                k = mm[n] & Gm[n, j]
                if k.any(): hint.append(float(10 * torch.log10(1 / ((mr[n][k] - Gr[n, j][k]) ** 2).mean().clamp_min(1e-10))))
        prev = Genc[:, j - 1] if mode == "oracle_mem_gtprev" else preds[:, j - 1]
        cr = torch.stack([encode_rgbd(mr, mz, mm), prev], 1); cp = torch.stack([PL[:, j], PL[:, j - 1]], 1)
        mv = mm.flatten(1).any(1)
        if mode == "nomem":
            cr[:, 0] = 0; cp[:, 0] = 0; mv = torch.zeros_like(mv)
        cv = torch.stack([mv, torch.ones(N, dtype=torch.bool, device=dev)], 1)
        noise = torch.randn(N, H, H, 4, device=dev, generator=gen); outs = []
        for s in range(0, N, chunk):
            sl = slice(s, s + chunk); model.cur_action = ACT[sl, j]
            fn = lambda x, t, sl=sl: model.forward_velocity(x, t, PL[sl, j], cr[sl], cp[sl], cv[sl], num_loops=R)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs.append(euler_sample(fn, noise[sl], steps=steps).float())
        preds[:, j] = torch.cat(outs).clamp(-1, 1)
        if j % stride == 0 or j == T - 1:
            if mode in ("own", "nomem"):
                pr, di, _ = decode_rgbd(preds[:, j])
                for n in range(N): mems[n].integrate(pr[n], di["z_metric"][n], di["valid_surface"][n], eps[n].c2w[j], K)
            else:
                for n in range(N): mems[n].integrate(Gr[n, j], Gz[n, j], Gm[n, j], eps[n].c2w[j], K)
    model.cur_action = None
    P, di, _ = decode_rgbd(preds); pz = di["z_metric"]; pv = di["valid_surface"]; ps = psnr(P, Gr).cpu().numpy()
    rel = (pz - Gz).abs() / Gz
    ar = (((rel * Gm).sum((-2, -1)) / Gm.sum((-2, -1)).clamp_min(1))).cpu().numpy()
    sel = Gm[:, 1:]; relv = rel[:, 1:][sel]; gzv = Gz[:, 1:][sel]; pvv = pv[:, 1:][sel]
    bins = {"z<2": gzv < 2, "2-5": (gzv >= 2) & (gzv < 5), "z>=5": gzv >= 5}
    absd = {"mean": float(relv.mean()), "median": float(relv.median()), "p90": float(relv.quantile(0.9)) if relv.numel() < 16_000_000 else None,
            "frac_rel_gt_1": float((relv > 1).float().mean()), "contrib_rel_gt_1": float((relv * (relv > 1)).sum() / relv.sum()),
            "pred_invalid_on_gt_hit": float((~pvv).float().mean()), "mean_rel_excl_pred_invalid": float(relv[pvv].mean()),
            "by_gt_depth": {k: {"frac": float(b.float().mean()), "mean": float(relv[b].mean()), "median": float(relv[b].median())} for k, b in bins.items()},
            "pred_hit_on_gt_sky": float(pv[:, 1:][~Gm[:, 1:]].float().mean()),
            "scale_median_pred_over_gt": float((pz[:, 1:][sel & pv[:, 1:]] / Gz[:, 1:][sel & pv[:, 1:]]).median())}
    selfc, gtr = [], []
    for n, e in enumerate(eps):
        for j, i in e.revisit.items(): selfc.append(float(psnr(P[n, j], P[n, i]))); gtr.append(float(ps[n, j]))
    res = {"mode": mode, "psnr_all": float(ps[:, 1:].mean()), "psnr_seg": {k: float(ps[:, a:b].mean()) for k, (a, b) in SEGS.items()},
           "absrel": float(ar[:, 1:].mean()), "revisit_vs_gt": float(np.mean(gtr)), "revisit_vs_own": float(np.mean(selfc)), "n_revisit": len(gtr),
           "mem_cov_all": float(cov[:, 1:].mean()), "hint_psnr_revisit": float(np.mean(hint)) if hint else None, "absrel_diag": absd}
    vid = None
    if video:
        vid = dict(eps=eps[:video], gr=Gr[:video].cpu(), gz=Gz[:video].cpu(), gm=Gm[:video].cpu(), pr=P[:video].cpu(),
                   pz=di["z_pred"][:video].cpu(), pv=pv[:video].cpu(), mem=vmem, ps=ps[:video])
    model.train(); return res, vid


def make_video(vid, path, label):
    frames = []
    S = 192
    up = lambda x: cv2.resize(x, (S, S), interpolation=cv2.INTER_NEAREST)
    for s, e in enumerate(vid["eps"]):
        rvj = set(e.revisit.keys())
        for j in range(vid["gr"].shape[1]):
            g = (vid["gr"][s, j].numpy() * 255).astype(np.uint8); p = (vid["pr"][s, j].numpy() * 255).astype(np.uint8)
            m = (vid["mem"][s, j].numpy() * 255).astype(np.uint8)
            gd = dcol(vid["gz"][s, j].numpy(), vid["gm"][s, j].numpy()); pd = dcol(vid["pz"][s, j].numpy(), vid["pv"][s, j].numpy())
            top = np.concatenate([up(g), up(p), up(m)], 1); bot = np.concatenate([up(gd), up(pd), np.zeros((S, S, 3), np.uint8)], 1)
            grid = np.concatenate([top, bot], 0)
            td = topdown(e.scene, e.poses, j, rvj, S=2 * S)
            f = np.concatenate([grid, td], 1)
            hdr = np.zeros((40, f.shape[1], 3), np.uint8)
            cv2.putText(hdr, f"{label} | maze {TEST_SEEDS[s]} frame {j}/127 | PSNR {vid['ps'][s, j]:.1f}" + ("  REVISIT" if j in rvj else ""),
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(hdr, "top: GT | pred (own-memory AR) | memory hint   bottom: GT depth | pred depth      right: GT map + path",
                        (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
            frames.append(np.concatenate([hdr, f], 0))
    write_h264(frames, path, fps=12)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="own,oracle_mem,oracle_mem_gtprev,nomem"); ap.add_argument("--n-scenes", type=int, default=32)
    ap.add_argument("--video", type=int, default=0); ap.add_argument("--label", default="")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    m, ck = load_model(a.ckpt, "cuda"); res = {"ckpt": a.ckpt, "step": ck.get("step"), "results": []}
    for md in a.modes.split(","):
        t0 = time.time(); r, vid = run(m, md, a.n_scenes, video=a.video if md == "own" else 0); r["sec"] = time.time() - t0
        res["results"].append(r); print(json.dumps(r), flush=True); json.dump(res, open(os.path.join(a.out, "full_diag.json"), "w"), indent=1)
        if vid is not None: make_video(vid, os.path.join(a.out, "sf_ar_own_memory_h264.mp4"), a.label or "self-forcing")
