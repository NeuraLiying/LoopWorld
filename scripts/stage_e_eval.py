#!/usr/bin/env python3
"""Stage E eval: held-out test scenes, modes S/T/AR, R x flow-steps sweep, H.264 videos. No training."""
import argparse, json, os, subprocess, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch, torch.nn.functional as F, cv2
from loopworld.cameras import make_K, plucker_rays
from loopworld.data import sample_scene, sample_trajectory, split_seeds
from loopworld.encoding import encode_rgbd, decode_rgbd
from loopworld.eval import sample_view as _sv

def sample_view(m, cr, cp, cv, tp, num_loops, steps, noise, chunk=8):
    return torch.cat([_sv(m, cr[i:i+chunk], cp[i:i+chunk], cv[i:i+chunk], tp[i:i+chunk], num_loops=num_loops, steps=steps, noise=noise[i:i+chunk])
                      for i in range(0, cr.shape[0], chunk)])
from loopworld.model import LoopWorldModel
from loopworld.renderer import render_scene

import shutil
FF = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"  # needs libx264
BUCKETS = {"near(1-4)": range(0, 4), "mid(5-12)": range(4, 12), "far(13-20)": range(12, 20),
           "back(21-23)": range(20, 23), "revisit(24)": [23]}

def ssim(a, b):
    a = a.permute(0, 3, 1, 2); b = b.permute(0, 3, 1, 2)
    g = torch.exp(-((torch.arange(11, device=a.device) - 5.0) ** 2) / (2 * 1.5 ** 2)); g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(3, 1, 11, 11)
    f = lambda x: F.conv2d(x, w, padding=5, groups=3)
    ma, mb = f(a), f(b)
    saa = f(a * a) - ma ** 2; sbb = f(b * b) - mb ** 2; sab = f(a * b) - ma * mb
    s = ((2 * ma * mb + 1e-4) * (2 * sab + 9e-4)) / ((ma ** 2 + mb ** 2 + 1e-4) * (saa + sbb + 9e-4))
    return s.mean(dim=(1, 2, 3))

def per_img(pr, pz, pv, gr, gz, gm):
    mse = ((pr - gr) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    out = {"psnr": (10 * torch.log10(1 / mse)).tolist(), "ssim": ssim(pr, gr).tolist(), "absrel": [], "valid": []}
    for i in range(gr.shape[0]):
        mk = gm[i]
        out["absrel"].append(((pz[i][mk] - gz[i][mk]).abs() / gz[i][mk]).mean().item())
        out["valid"].append(pv[i][mk].float().mean().item())
    return out

def dcol(z, mk):
    u = ((7.0 - z.float()) / 5.0).clamp(0, 1); u = torch.where(mk, u, torch.zeros_like(u))
    c = cv2.applyColorMap((u.cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[:, :, ::-1].copy()
    c[~mk.cpu().numpy()] = 0
    return c

def write_h264(frames, path, fps=6):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen([FF, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)],
                         stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for f in frames: p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close(); p.wait()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-scenes", type=int, default=24)
    ap.add_argument("--Rs", default="1,2,4"); ap.add_argument("--steps", default="5,10,20")
    ap.add_argument("--modes", default="S,T,AR"); ap.add_argument("--video-R", type=int, default=4)
    ap.add_argument("--video-steps", type=int, default=5); ap.add_argument("--video-scenes", type=int, default=3); ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda"); out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); cfg = ck["config"]
    H = int(cfg["image_size"])
    m = LoopWorldModel(image_size=H, patch_size=int(cfg.get("patch_size", 4)), width=int(cfg["width"]), heads=int(cfg["heads"])).to(dev)
    m.load_state_dict(ck["ema"]); m.eval()
    K = make_K(H, device=dev)
    seeds = split_seeds()["test"][: a.n_scenes]
    Rs = [int(x) for x in a.Rs.split(",")]; Ss = [int(x) for x in a.steps.split(",")]; modes = a.modes.split(",")
    # pre-render GT for all scenes (25 poses)
    G = []
    with torch.no_grad():
        for sd in seeds:
            sc = sample_scene(sd); c2 = sample_trajectory(sd).to(dev)
            rgb, z, mk = render_scene(sc, c2, K, H)
            G.append(dict(rgb=rgb, z=z, mk=mk, enc=encode_rgbd(rgb, z, mk), pl=plucker_rays(c2, K, H)))
    N = len(G)
    def noise_for(sd, j, R, S):
        g = torch.Generator(device=dev).manual_seed(sd * 1000 + j)  # same noise across R/S configs
        return torch.randn(1, H, H, 4, device=dev, generator=g)

    results = {"ckpt": a.ckpt, "ckpt_step": ck.get("step"), "seeds": seeds, "configs": {}}
    t0 = time.time(); keep_ar = {}
    with torch.no_grad():
        for R in Rs:
            for S in Ss:
                for mode in modes:
                    agg = {"psnr": [], "ssim": [], "absrel": [], "valid": []}  # [scene][24]
                    if mode in ("S", "T"):
                        for si, (sd, g) in enumerate(zip(seeds, G)):
                            B = 24
                            cr = torch.zeros(B, 2, H, H, 4, device=dev); cp = torch.zeros(B, 2, H, H, 6, device=dev)
                            cv = torch.zeros(B, 2, dtype=torch.bool, device=dev)
                            cr[:, 0] = g["enc"][0]; cp[:, 0] = g["pl"][0]; cv[:, 0] = True
                            if mode == "T":
                                cr[1:, 1] = g["enc"][1:24]; cp[1:, 1] = g["pl"][1:24]; cv[1:, 1] = True
                            nz = torch.cat([noise_for(sd, j, R, S) for j in range(1, 25)])
                            x = sample_view(m, cr, cp, cv, g["pl"][1:], num_loops=R, steps=S, noise=nz)
                            pr, di, _ = decode_rgbd(x)
                            r = per_img(pr, di["z_metric"], di["valid_surface"], g["rgb"][1:], g["z"][1:], g["mk"][1:])
                            for k in agg: agg[k].append(r[k])
                    else:  # AR: batch over scenes, sequential over j
                        enc0 = torch.stack([g["enc"][0] for g in G]); pl = torch.stack([g["pl"] for g in G])  # N,25,...
                        prev = None; preds = []
                        per = {k: [[] for _ in range(N)] for k in agg}
                        for j in range(1, 25):
                            cr = torch.zeros(N, 2, H, H, 4, device=dev); cp = torch.zeros(N, 2, H, H, 6, device=dev)
                            cv = torch.zeros(N, 2, dtype=torch.bool, device=dev)
                            cr[:, 0] = enc0; cp[:, 0] = pl[:, 0]; cv[:, 0] = True
                            if prev is not None:
                                cr[:, 1] = prev; cp[:, 1] = pl[:, j - 1]; cv[:, 1] = True
                            nz = torch.cat([noise_for(sd, j, R, S) for sd in seeds])
                            x = sample_view(m, cr, cp, cv, pl[:, j], num_loops=R, steps=S, noise=nz)
                            prev = x.clamp(-1, 1)
                            pr, di, _ = decode_rgbd(x)
                            gr = torch.stack([g["rgb"][j] for g in G]); gz = torch.stack([g["z"][j] for g in G]); gm = torch.stack([g["mk"][j] for g in G])
                            r = per_img(pr, di["z_metric"], di["valid_surface"], gr, gz, gm)
                            for k in agg:
                                for i in range(N): per[k][i].append(r[k][i])
                            if (not a.no_video) and R == a.video_R and S == a.video_steps:
                                preds.append((pr[: a.video_scenes].cpu(), di["z_pred"][: a.video_scenes].cpu(), di["valid_surface"][: a.video_scenes].cpu()))
                        agg = per
                        if preds: keep_ar = preds
                    summ = {}
                    for k, v in agg.items():
                        arr = np.array(v, dtype=float)
                        summ[k] = {"all": float(np.nanmean(arr)), **{b: float(np.nanmean(arr[:, list(ix)])) for b, ix in BUCKETS.items()},
                                   "per_j": [float(x) for x in np.nanmean(arr, axis=0)]}
                    results["configs"][f"{mode}_R{R}_S{S}"] = summ
                    print(f"{mode} R{R} S{S}: psnr={summ['psnr']['all']:.2f} far={summ['psnr']['far(13-20)']:.2f} "
                          f"absrel={summ['absrel']['all']:.3f} t={time.time()-t0:.0f}s", flush=True)
                    json.dump(results, open(os.path.join(out, "stage_e_results.json"), "w"), indent=1)
        # videos
        def frame(gt_rgb, gt_z, gt_m, p_rgb, p_z, p_v, label):
            top = np.concatenate([(gt_rgb.cpu().numpy() * 255).astype(np.uint8), (p_rgb.numpy() * 255).astype(np.uint8)], 1)
            bot = np.concatenate([dcol(gt_z, gt_m), dcol(p_z, p_v)], 1)
            f = np.concatenate([top, bot], 0); f = cv2.resize(f, (f.shape[1] * 4, f.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
            return cv2.putText(np.ascontiguousarray(f), label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if keep_ar:
            fr = []
            for s in range(len(keep_ar[0][0])):
                for j in range(24):
                    p = keep_ar[j]
                    fr.append(frame(G[s]["rgb"][j + 1], G[s]["z"][j + 1], G[s]["mk"][j + 1], p[0][s], p[1][s], p[2][s],
                                    f"AR rollout seed {seeds[s]} j={j+1} R{a.video_R} S{a.video_steps}  L:GT R:Pred"))
            write_h264(fr, os.path.join(out, "ar_rollout_h264.mp4"))
        # S-mode orbit video
        fr = []
        for s in range(0 if a.no_video else a.video_scenes):
            g = G[s]; B = 24
            cr = torch.zeros(B, 2, H, H, 4, device=dev); cp = torch.zeros(B, 2, H, H, 6, device=dev); cv = torch.zeros(B, 2, dtype=torch.bool, device=dev)
            cr[:, 0] = g["enc"][0]; cp[:, 0] = g["pl"][0]; cv[:, 0] = True
            nz = torch.cat([noise_for(seeds[s], j, 0, 0) for j in range(1, 25)])
            x = sample_view(m, cr, cp, cv, g["pl"][1:], num_loops=a.video_R, steps=a.video_steps, noise=nz)
            pr, di, _ = decode_rgbd(x)
            for j in range(24):
                fr.append(frame(g["rgb"][j + 1], g["z"][j + 1], g["mk"][j + 1], pr[j].cpu(), di["z_pred"][j].cpu(), di["valid_surface"][j].cpu(),
                                f"mode S seed {seeds[s]} j={j+1} R{a.video_R} S{a.video_steps}  L:GT R:Pred"))
        if fr: write_h264(fr, os.path.join(out, "heldout_orbit_S_h264.mp4"))
    print("DONE", time.time() - t0, out, flush=True)

if __name__ == "__main__":
    main()
