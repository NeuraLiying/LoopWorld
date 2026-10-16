#!/usr/bin/env python3
"""LoopWorld offline demos D1/D2/D3/D4/D7. Inference only; reuses Stage E eval protocol (same seeds, same noise)."""
import argparse, json, os, subprocess, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from loopworld.cameras import make_K, plucker_rays, orbit_c2w, project_points, unproject_depth
from loopworld.data import sample_scene, sample_trajectory, split_seeds
from loopworld.encoding import encode_rgbd, decode_rgbd
from loopworld.model import LoopWorldModel
from loopworld.renderer import render_scene
from stage_e_eval import sample_view, dcol, FF

FONT = cv2.FONT_HERSHEY_SIMPLEX
def u8(x): return (np.clip(x.cpu().numpy() if torch.is_tensor(x) else x, 0, 1) * 255).astype(np.uint8)
def up(img, s): return cv2.resize(img, (img.shape[1] * s, img.shape[0] * s), interpolation=cv2.INTER_NEAREST)
def txt(img, s, xy, sc=0.5, col=(255, 255, 255)):
    img = np.ascontiguousarray(img)
    cv2.putText(img, s, xy, FONT, sc, (0, 0, 0), 3, cv2.LINE_AA); cv2.putText(img, s, xy, FONT, sc, col, 1, cv2.LINE_AA)
    return img
def bar(w, s, h=28, sc=0.55):
    b = np.full((h, w, 3), 24, np.uint8); return txt(b, s, (8, h - 9), sc)
def even(f):
    h, w = f.shape[:2]; return np.ascontiguousarray(f[: h - h % 2, : w - w % 2])
def psnr(a, b): return float(10 * np.log10(1 / max(float(((a - b) ** 2).mean()), 1e-10)))

def write_mp4(frames, path, fps=25, hold=1):
    frames = [even(f) for f in frames]; h, w = frames[0].shape[:2]
    p = subprocess.Popen([FF, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(path)],
                         stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for f in frames:
        for _ in range(hold): p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait(); print("wrote", path, flush=True)
def write_gif(mp4, gif, width, fps=12):
    vf = f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer"
    subprocess.run([FF, "-y", "-i", str(mp4), "-vf", vf, "-loop", "0", str(gif)], check=True, stderr=subprocess.DEVNULL); print("wrote", gif, flush=True)

def splat(P, C, c2w, K, S, r=1, bg=16):
    """z-buffered point splatting. P,C (N,3) torch on device."""
    uv, z, v = project_points(P, c2w, K)
    img = torch.full((S * S, 3), bg / 255.0, device=P.device)
    zb = torch.full((S * S,), float("inf"), device=P.device)
    us, vs = uv[:, 0].round().long(), uv[:, 1].round().long()
    cand = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            uu, vv = us + dx, vs + dy
            ok = v & (uu >= 0) & (uu < S) & (vv >= 0) & (vv < S)
            cand.append((vv[ok] * S + uu[ok], z[ok], C[ok]))
    idx = torch.cat([c[0] for c in cand]); zz = torch.cat([c[1] for c in cand]); cc = torch.cat([c[2] for c in cand])
    zb.scatter_reduce_(0, idx, zz, reduce="amin")
    win = zz <= zb[idx] + 1e-6
    img[idx[win]] = cc[win]
    return u8(img.reshape(S, S, 3))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-scenes", type=int, default=24); ap.add_argument("--steps", type=int, default=5)
    a = ap.parse_args()
    dev = torch.device("cuda"); out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); cfg = ck["config"]; H = int(cfg["image_size"])
    m = LoopWorldModel(image_size=H, patch_size=int(cfg.get("patch_size", 4)), width=int(cfg["width"]), heads=int(cfg["heads"])).to(dev)
    m.load_state_dict(ck["ema"]); m.eval()
    K = make_K(H, device=dev); seeds = split_seeds()["test"][: a.n_scenes]; N = len(seeds); ST = a.steps
    G = []; C2 = []
    with torch.no_grad():
        for sd in seeds:
            sc = sample_scene(sd); c2 = sample_trajectory(sd).to(dev); C2.append(c2)
            rgb, z, mk = render_scene(sc, c2, K, H)
            G.append(dict(rgb=rgb, z=z, mk=mk, enc=encode_rgbd(rgb, z, mk), pl=plucker_rays(c2, K, H)))
    def noise_for(sd, j):
        g = torch.Generator(device=dev).manual_seed(sd * 1000 + j); return torch.randn(1, H, H, 4, device=dev, generator=g)
    def dec(x):
        pr, di, _ = decode_rgbd(x); return dict(rgb=pr.float().cpu(), z=di["z_metric"].float().cpu(), zp=di["z_pred"].float().cpu(), v=di["valid_surface"].cpu())
    def run_ST(mode, R):  # returns list[N] of dict with (24,...) tensors
        res = []
        for sd, g in zip(seeds, G):
            B = 24; cr = torch.zeros(B, 2, H, H, 4, device=dev); cp = torch.zeros(B, 2, H, H, 6, device=dev); cv = torch.zeros(B, 2, dtype=torch.bool, device=dev)
            cr[:, 0] = g["enc"][0]; cp[:, 0] = g["pl"][0]; cv[:, 0] = True
            if mode == "T": cr[1:, 1] = g["enc"][1:24]; cp[1:, 1] = g["pl"][1:24]; cv[1:, 1] = True
            nz = torch.cat([noise_for(sd, j) for j in range(1, 25)])
            res.append(dec(sample_view(m, cr, cp, cv, g["pl"][1:], num_loops=R, steps=ST, noise=nz)))
        return res
    def run_AR(R):
        enc0 = torch.stack([g["enc"][0] for g in G]); pl = torch.stack([g["pl"] for g in G]); prev = None; steps = []
        for j in range(1, 25):
            cr = torch.zeros(N, 2, H, H, 4, device=dev); cp = torch.zeros(N, 2, H, H, 6, device=dev); cv = torch.zeros(N, 2, dtype=torch.bool, device=dev)
            cr[:, 0] = enc0; cp[:, 0] = pl[:, 0]; cv[:, 0] = True
            if prev is not None: cr[:, 1] = prev; cp[:, 1] = pl[:, j - 1]; cv[:, 1] = True
            nz = torch.cat([noise_for(sd, j) for sd in seeds])
            x = sample_view(m, cr, cp, cv, pl[:, j], num_loops=R, steps=ST, noise=nz); prev = x.clamp(-1, 1); steps.append(dec(x))
        return [{k: torch.stack([s[k][i] for s in steps]) for k in steps[0]} for i in range(N)]
    t0 = time.time()
    with torch.no_grad():
        P = {("S", R): run_ST("S", R) for R in (1, 2, 4)}
        P[("T", 4)] = run_ST("T", 4); P[("AR", 4)] = run_AR(4)
    print(f"inference done {time.time()-t0:.0f}s", flush=True)
    gt = lambda i, k: G[i][k][1:].float().cpu()
    def ps(key, i): return [psnr(P[key][i]["rgb"][j].numpy(), gt(i, "rgb")[j].numpy()) for j in range(24)]
    stats = {f"{k[0]}_R{k[1]}": np.array([ps(k, i) for i in range(N)]) for k in P}  # N x 24
    summary = {k: {"mean": float(v.mean()), "per_j": v.mean(0).round(3).tolist()} for k, v in stats.items()}
    SC = 4; T = H * SC
    # ---------------- D1 split screen S-mode GT|R1|R2|R4 ----------------
    fr = []
    for i in range(3):
        for j in range(24):
            cols = []
            g_rgb = up(u8(gt(i, "rgb")[j]), SC); g_d = up(dcol(gt(i, "z")[j], G[i]["mk"][j + 1].cpu()), SC)
            cols.append(np.concatenate([txt(g_rgb, "GT", (8, 22)), g_d], 0))
            for R in (1, 2, 4):
                p = P[("S", R)][i]; pv = psnr(p["rgb"][j].numpy(), gt(i, "rgb")[j].numpy())
                cols.append(np.concatenate([txt(up(u8(p["rgb"][j]), SC), f"R{R}  {pv:.1f} dB", (8, 22)), up(dcol(p["zp"][j], p["v"][j]), SC)], 0))
            body = np.concatenate(cols, 1)
            fr.append(np.concatenate([bar(body.shape[1], f"D1  loop iterations R=1/2/4 (same weights, same noise) | mode S: first frame only | scene {seeds[i]}  view {j+1}/24"),
                                      body, bar(body.shape[1], f"held-out 24 scenes mean PSNR  R1 {summary['S_R1']['mean']:.2f}  R2 {summary['S_R2']['mean']:.2f}  R4 {summary['S_R4']['mean']:.2f} dB   (top RGB, bottom depth)")], 0))
    write_mp4(fr, f"{out}/D1_loop_R1_R2_R4_split_h264.mp4", hold=6)
    # ---------------- D2 accumulated point cloud ----------------
    S2 = 448; K2 = make_K(S2, fov_deg=50.0, device=dev)
    for i in range(2):
        c2 = C2[i]
        def cloud(rgbs, zs, ms, n):
            pts = unproject_depth(zs[:n].to(dev), c2[1:n + 1], K, mask=ms[:n].to(dev)); col = rgbs[:n].to(dev)
            ok = ~torch.isnan(pts[..., 0]); return pts[ok], col[ok]
        g_rgb, g_z, g_m = gt(i, "rgb"), gt(i, "z"), G[i]["mk"][1:].cpu()
        p = P[("AR", 4)][i]
        nF = 200; fr = []
        for f in range(nF):
            n = min(24, 1 + f * 24 // 120)  # accumulate during first 120 frames
            az = 360.0 * f / nF + 30; cam = orbit_c2w(az, 30.0 + 10 * np.sin(2 * np.pi * f / nF), radius=5.5, device=dev)
            a1 = splat(*cloud(g_rgb, g_z, g_m, n), cam, K2, S2); a2 = splat(*cloud(p["rgb"], p["z"], p["v"], n), cam, K2, S2)
            body = np.concatenate([txt(a1, "GT RGB-D", (10, 26), 0.7), txt(a2, "LoopWorld AR prediction (R4)", (10, 26), 0.7)], 1)
            fr.append(np.concatenate([bar(body.shape[1], f"D2  point cloud fused from {n}/24 generated RGB-D views + known poses | scene {seeds[i]}", 32, 0.6), body], 0))
        write_mp4(fr, f"{out}/D2_pointcloud_orbit_scene{seeds[i]}_h264.mp4")
    # ---------------- D3 AR drift ----------------
    ar, tt, ss = stats["AR_R4"].mean(0), stats["T_R4"].mean(0), stats["S_R4"].mean(0)
    plt.figure(figsize=(7, 3.6), dpi=110)
    for y, l, c in ((tt, "T (true prev frame)", "tab:green"), (ss, "S (first frame only)", "tab:blue"), (ar, "AR (own prediction)", "tab:red")): plt.plot(range(1, 25), y, "o-", ms=3, label=l, color=c)
    plt.xlabel("view j (15 deg / step, 24 = back at start)"); plt.ylabel("PSNR (dB), 24 held-out scenes"); plt.grid(alpha=.3); plt.legend(); plt.title("D3  autoregressive drift, R4"); plt.tight_layout()
    plt.savefig(f"{out}/D3_psnr_curve.png"); plt.close()
    fr = []
    for i in range(3):
        for j in range(24):
            g_ = gt(i, "rgb")[j].numpy(); pa = P[("AR", 4)][i]["rgb"][j].numpy(); pt = P[("T", 4)][i]["rgb"][j].numpy()
            def heat(p):
                e = np.abs(p - g_).mean(-1); return cv2.applyColorMap((np.clip(e / 0.25, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1]
            row = np.concatenate([txt(up(u8(g_), SC), "GT", (8, 22)), txt(up(u8(pa), SC), f"AR {psnr(pa, g_):.1f} dB", (8, 22)),
                                  txt(up(heat(pa), SC), "AR |err|", (8, 22)), txt(up(heat(pt), SC), f"T |err| {psnr(pt, g_):.1f} dB", (8, 22))], 1)
            fig = plt.figure(figsize=(row.shape[1] / 100, 2.4), dpi=100); ax = fig.add_subplot(111)
            ax.plot(range(1, 25), tt, color="tab:green", label="T mean"); ax.plot(range(1, 25), ar, color="tab:red", label="AR mean (24 scenes)")
            ax.plot(range(1, j + 2), stats["AR_R4"][i, : j + 1], "o-", ms=3, color="k", label="AR this scene"); ax.axvline(j + 1, color="gray", ls=":")
            ax.set_xlim(0.5, 24.5); ax.set_ylim(min(stats["AR_R4"][:3].min(), ar.min()) - 1, max(tt.max(), stats["T_R4"][:3].max()) + 1)
            ax.set_ylabel("PSNR dB"); ax.legend(loc="lower left", fontsize=8, ncol=3); ax.grid(alpha=.3); fig.tight_layout(); fig.canvas.draw()
            cimg = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(); plt.close(fig)
            cimg = cv2.resize(cimg, (row.shape[1], cimg.shape[0]))
            fr.append(np.concatenate([bar(row.shape[1], f"D3  AR drift: model feeds its own prediction back | scene {seeds[i]}  view {j+1}/24 | heat scale 0-0.25"), row, cimg], 0))
    write_mp4(fr, f"{out}/D3_ar_drift_h264.mp4", hold=6)
    # ---------------- D4 return to start ----------------
    rev = {k: stats[k][:, 23] for k in ("AR_R4", "T_R4", "S_R4")}
    fr = []; tiles = []
    for i in range(4):
        start = up(u8(G[i]["rgb"][0].float().cpu()), SC)
        for j in range(24):
            pa = P[("AR", 4)][i]["rgb"][j]; pt = P[("T", 4)][i]["rgb"][j]
            lab = "returned" if j == 23 else f"view {j+1}/24"
            row = np.concatenate([txt(start.copy(), "START (given)", (8, 22)), txt(up(u8(pa), SC), f"AR {lab}", (8, 22)), txt(up(u8(pt), SC), f"T {lab}", (8, 22))], 1)
            if j == 23:
                row = txt(row, f"vs start: AR {stats['AR_R4'][i,23]:.1f} dB   T {stats['T_R4'][i,23]:.1f} dB", (8, T - 12), 0.7, (255, 230, 80))
                tiles.append(row)
            f = np.concatenate([bar(row.shape[1], f"D4  full 360 deg orbit back to the start pose | scene {seeds[i]}"), row,
                                bar(row.shape[1], f"24-scene mean PSNR at return: AR {rev['AR_R4'].mean():.2f}  S {rev['S_R4'].mean():.2f}  T {rev['T_R4'].mean():.2f} dB")], 0)
            fr += [f] * (40 if j == 23 else 4)
    write_mp4(fr, f"{out}/D4_return_to_start_h264.mp4")
    cv2.imwrite(f"{out}/D4_return_to_start_grid.png", np.concatenate(tiles, 0)[:, :, ::-1])
    # ---------------- D7 wall ----------------
    fr = []
    for j in range(24):
        tl = [txt(up(u8(P[("AR", 4)][i]["rgb"][j]), 2), str(seeds[i]), (4, 14), 0.4) for i in range(N)]
        rows = [np.concatenate(tl[r * 6:(r + 1) * 6], 1) for r in range(N // 6)]
        body = np.concatenate(rows, 0)
        fr.append(np.concatenate([bar(body.shape[1], f"D7  {N} held-out scenes, AR rollout R4, from one start frame | view {j+1}/24"), body], 0))
    write_mp4(fr, f"{out}/D7_scene_wall_h264.mp4", hold=4)
    json.dump({"ckpt": a.ckpt, "step": ck.get("step"), "seeds": seeds, "flow_steps": ST, "psnr": summary,
               "revisit_psnr_mean": {k: float(v.mean()) for k, v in rev.items()}}, open(f"{out}/demo_numbers.json", "w"), indent=1)
    for nm, w in (("D1_loop_R1_R2_R4_split", 768), ("D3_ar_drift", 768), ("D4_return_to_start", 576), ("D7_scene_wall", 576)):
        write_gif(f"{out}/{nm}_h264.mp4", f"{out}/{nm}.gif", w)
    write_gif(f"{out}/D2_pointcloud_orbit_scene{seeds[0]}_h264.mp4", f"{out}/D2_pointcloud_orbit.gif", 640, fps=15)
    print("DONE", time.time() - t0, flush=True)

if __name__ == "__main__":
    main()
