#!/usr/bin/env python3
"""Maze v2 + geometric memory AR eval. Memory integrates OWN predictions (and GT frame 0)."""
import argparse, json, os, sys, time, subprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch, cv2
from loopworld.cameras import make_K
from loopworld.encoding import encode_rgbd, decode_rgbd
from loopworld.flow import euler_sample
from loopworld.maze_v2 import render_maze
from loopworld.maze_v2_data import MazeEpisode, maze_plucker
from loopworld.geom_memory import PointMemory
from loopworld.model_action import ActionLoopWorldModel

TEST_SEEDS = list(range(600000, 600032))
SEGS = {"1-8": (1, 9), "9-32": (9, 33), "33-64": (33, 65), "65-127": (65, 128)}
FF = "/tmp/iiof/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"


def psnr(a, b):
    return 10 * torch.log10(1 / ((a - b) ** 2).mean(dim=(-3, -2, -1)).clamp_min(1e-10))


@torch.no_grad()
def run_ar_eval_mem(model, n_scenes=8, n_frames=128, R=4, steps=5, use_pose=True, device="cuda",
                    keep_video=0, mem_stride=4, max_points=120_000, chunk=8):
    model.eval(); H = model.image_size; K = make_K(H, device=device)
    eps = [MazeEpisode(s, n_frames, device) for s in TEST_SEEDS[:n_scenes]]
    N = len(eps)
    gt = [render_maze(e.scene, e.c2w, H) for e in eps]
    G_rgb = torch.stack([g[0] for g in gt]); G_z = torch.stack([g[1] for g in gt]); G_m = torch.stack([g[2] for g in gt])
    PL = torch.stack([maze_plucker(e.c2w, K, H, use_pose) for e in eps])
    ACT = torch.tensor(np.stack([e.actions for e in eps]), device=device)
    preds = torch.zeros(N, n_frames, H, H, 4, device=device)
    preds[:, 0] = encode_rgbd(G_rgb[:, 0], G_z[:, 0], G_m[:, 0])
    # per-scene memories, seed with GT frame 0
    mems = [PointMemory(max_points=max_points, subsample_stride=1) for _ in range(N)]
    for n in range(N):
        mems[n].integrate(G_rgb[n, 0], G_z[n, 0], G_m[n, 0], eps[n].c2w[0], K)
    gen = torch.Generator(device=device).manual_seed(1234)
    P_rgb_live = G_rgb[:, 0].clone()  # for memory integrate of predictions
    P_z_live = G_z[:, 0].clone(); P_m_live = G_m[:, 0].clone()
    for j in range(1, n_frames):
        # reproject memories to camera j
        mrs, mzs, mms = [], [], []
        for n in range(N):
            r, z, m = mems[n].reproject(eps[n].c2w[j], K, H)
            mrs.append(r); mzs.append(z); mms.append(m)
        mr, mz, mm = torch.stack(mrs), torch.stack(mzs), torch.stack(mms)
        enc_m = encode_rgbd(mr, mz, mm)
        cr = torch.stack([enc_m, preds[:, j - 1]], 1)
        cp = torch.stack([PL[:, j], PL[:, j - 1]], 1)  # memory uses target pose Plücker
        cv = torch.stack([mm.flatten(1).any(1), torch.ones(N, dtype=torch.bool, device=device)], 1)
        noise = torch.randn(N, H, H, 4, device=device, generator=gen)
        outs = []
        for s in range(0, N, chunk):
            sl = slice(s, s + chunk)
            model.cur_action = ACT[sl, j]
            fn = lambda x, t, sl=sl: model.forward_velocity(x, t, PL[sl, j], cr[sl], cp[sl], cv[sl], num_loops=R)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs.append(euler_sample(fn, noise[sl], steps=steps).float())
        preds[:, j] = torch.cat(outs).clamp(-1, 1)
        pr, di, _ = decode_rgbd(preds[:, j])
        pz, pv = di["z_metric"], di["valid_surface"]
        # integrate prediction into memory (every mem_stride frames, always last)
        for n in range(N):
            if (j % mem_stride == 0) or (j == n_frames - 1):
                mems[n].integrate(pr[n], pz[n], pv[n], eps[n].c2w[j], K)
        P_rgb_live, P_z_live, P_m_live = pr, pz, pv
    model.cur_action = None
    P_rgb, di, _ = decode_rgbd(preds)
    pz = di["z_metric"]
    ps = psnr(P_rgb, G_rgb).cpu().numpy()
    absrel = (((pz - G_z).abs() / G_z) * G_m).sum((-2, -1)) / G_m.sum((-2, -1)).clamp_min(1)
    absrel = absrel.cpu().numpy()
    res = {"R": R, "steps": steps, "n_scenes": N, "use_pose": use_pose, "memory": True,
           "psnr_all": float(ps[:, 1:].mean()), "absrel_all": float(absrel[:, 1:].mean()),
           "psnr_seg": {k: float(ps[:, a:b].mean()) for k, (a, b) in SEGS.items()},
           "absrel_seg": {k: float(absrel[:, a:b].mean()) for k, (a, b) in SEGS.items()},
           "psnr_per_j": [float(x) for x in ps.mean(0)]}
    self_c, acc_rv, acc_first, nrv = [], [], [], 0
    for n, e in enumerate(eps):
        for j, i in e.revisit.items():
            self_c.append(float(psnr(P_rgb[n, j], P_rgb[n, i])))
            acc_rv.append(float(ps[n, j])); acc_first.append(float(ps[n, i])); nrv += 1
    res["revisit"] = {"n_pairs": nrv,
                      "psnr_pred_vs_own_first": float(np.mean(self_c)) if nrv else None,
                      "psnr_pred_vs_gt_at_revisit": float(np.mean(acc_rv)) if nrv else None,
                      "psnr_pred_vs_gt_at_first_visit": float(np.mean(acc_first)) if nrv else None}
    vid = None
    if keep_video:
        # also store memory reprojections for last kept scenes at a few frames — skip for size
        vid = (G_rgb[:keep_video].cpu(), G_z[:keep_video].cpu(), G_m[:keep_video].cpu(),
               P_rgb[:keep_video].cpu(), di["z_pred"][:keep_video].cpu(), di["valid_surface"][:keep_video].cpu(),
               [eps[k].revisit for k in range(keep_video)])
    model.train()
    return res, vid


def write_video(vid, path, fps=25):
    gr, gz, gm, pr, pz, pv, rvs = vid
    def dcol(z, m):
        u = ((8.0 - z.numpy()) / 7.5).clip(0, 1) * m.numpy()
        c = cv2.applyColorMap((u * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[:, :, ::-1].copy(); c[~m.numpy()] = 0
        return c
    frames = []
    for s in range(gr.shape[0]):
        for j in range(gr.shape[1]):
            top = np.concatenate([(gr[s, j].numpy() * 255).astype(np.uint8), (pr[s, j].numpy() * 255).astype(np.uint8)], 1)
            bot = np.concatenate([dcol(gz[s, j], gm[s, j]), dcol(pz[s, j], pv[s, j])], 1)
            f = cv2.resize(np.concatenate([top, bot], 0), (768, 768), interpolation=cv2.INTER_NEAREST)
            f = np.concatenate([np.zeros((24, 768, 3), np.uint8), f], 0)
            cv2.putText(f, f"mem AR s{s} f{j}" + (" REVISIT" if j in rvs[s] else ""), (6, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            frames.append(f)
    h, w = frames[0].shape[:2]
    p = subprocess.Popen([FF, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)],
                         stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for f in frames: p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close(); p.wait()


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False); cfg = ck["config"]
    m = ActionLoopWorldModel(image_size=int(cfg["image_size"]), patch_size=int(cfg.get("patch_size", 4)),
                             width=int(cfg["width"]), heads=int(cfg["heads"])).to(device)
    m.load_state_dict(ck["ema"]); return m, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-scenes", type=int, default=32); ap.add_argument("--Rs", default="4"); ap.add_argument("--steps", default="5")
    ap.add_argument("--video-scenes", type=int, default=2); ap.add_argument("--no-pose", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    m, ck = load_model(a.ckpt, "cuda"); allr = {"ckpt": a.ckpt, "step": ck.get("step"), "results": {}}
    for R in map(int, a.Rs.split(",")):
        for S in map(int, a.steps.split(",")):
            t0 = time.time(); kv = a.video_scenes if (R == 4 and S == 5) else 0
            r, vid = run_ar_eval_mem(m, a.n_scenes, R=R, steps=S, use_pose=not a.no_pose, keep_video=kv)
            r["sec"] = time.time() - t0; allr["results"][f"R{R}_S{S}"] = r
            print(f"R{R} S{S}: psnr={r['psnr_all']:.2f} seg={r['psnr_seg']} rv={r['revisit']}", flush=True)
            if vid is not None: write_video(vid, os.path.join(a.out, f"ar_mem_R{R}_S{S}_h264.mp4"))
            json.dump(allr, open(os.path.join(a.out, "maze_v2_mem_eval.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
