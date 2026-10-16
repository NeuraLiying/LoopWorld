#!/usr/bin/env python3
"""Maze v2: generate small sample set (npz), geometry checks, H.264 preview. No training."""
import argparse, json, os, subprocess, sys, math, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch, cv2
from loopworld.cameras import make_K, plucker_rays, unproject_depth, invert_c2w
from loopworld.encoding import encode_rgbd, Z_NEAR
from loopworld.maze_v2 import (sample_maze, sample_maze_trajectory, render_maze, revisit_pairs, CELL)

FF = "/tmp/iiof/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"

def write_h264(frames, path, fps=25):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen([FF, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for f in frames: p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close(); p.wait()

def dcol(z, m):
    u = ((8.0 - z) / 7.5).clip(0, 1) * m
    c = cv2.applyColorMap((u * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)[:, :, ::-1].copy(); c[~m] = 0
    return c

def topdown(sc, poses, upto, revisit_js, S=288):
    img = np.full((S, S, 3), 245, np.uint8)
    W = sc.nx * CELL; s = (S - 16) / W
    tp = lambda x, y: (int(8 + (x + W / 2) * s), int(S - 8 - (y + sc.ny * CELL / 2) * s))
    for b, col in zip(sc.boxes, sc.box_col):
        a = tp(b[0] - b[3], b[1] - b[4]); c = tp(b[0] + b[3], b[1] + b[4])
        cv2.rectangle(img, (a[0], c[1]), (c[0], a[1]), tuple(int(v * 255) for v in col), -1)
    for sp, col in zip(sc.spheres, sc.sph_col):
        cv2.circle(img, tp(sp[0], sp[1]), max(2, int(sp[3] * s)), tuple(int(v * 255) for v in col), -1)
    P = [tp(x, y) for x, y, _ in poses]
    for k in range(1, len(P)): cv2.line(img, P[k - 1], P[k], (200, 200, 200), 1)
    for k in range(1, upto + 1): cv2.line(img, P[k - 1], P[k], (30, 30, 30), 2)
    x, y, yaw = poses[upto]
    c = P[upto]; L = 22
    for da in (-math.pi / 6, math.pi / 6):
        e = (int(c[0] + L * math.cos(yaw + da)), int(c[1] - L * math.sin(yaw + da))); cv2.line(img, c, e, (220, 30, 30), 1)
    cv2.circle(img, c, 4, (220, 30, 30) if upto not in revisit_js else (20, 170, 20), -1)
    return img

def reproj_check(rgb, z, m, c2w, K, pairs):
    """Unproject frame a, project into b, compare to b's depth where visible. Returns stats."""
    H = rgb.shape[1]; errs = []; cerr = []; covs = []
    for a, b in pairs:
        pw = unproject_depth(z[a], c2w[a], K, m[a]).reshape(-1, 3)
        ok = torch.isfinite(pw[:, 0]); pw = pw[ok]; ca = rgb[a].reshape(-1, 3)[ok]
        w2c = invert_c2w(c2w[b]); pc = pw @ w2c[:3, :3].T + w2c[:3, 3]
        zz = pc[:, 2]; u = K[0, 0] * pc[:, 0] / zz + K[0, 2]; v = K[1, 1] * pc[:, 1] / zz + K[1, 2]
        ui, vi = u.round().long(), v.round().long()
        inb = (zz > 0.1) & (ui >= 0) & (ui < H) & (vi >= 0) & (vi < H)
        zb = z[b][vi[inb], ui[inb]]; mb = m[b][vi[inb], ui[inb]]
        rel = (zz[inb] - zb).abs() / zb
        vis = mb & (rel < 0.05)  # visible (not occluded) at b
        covs.append(vis.float().mean().item() if inb.any() else 0)
        # sub-pixel error: compare on visible points only
        errs.append(rel[mb].median().item() if mb.any() else float("nan"))
        cerr.append((ca[inb][vis] - rgb[b][vi[inb][vis], ui[inb][vis]]).abs().mean().item() if vis.any() else float("nan"))
    return dict(median_rel_depth_err=float(np.nanmedian(errs)), visible_frac=float(np.mean(covs)), color_l1_on_visible=float(np.nanmean(cerr)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--frames", type=int, default=128); ap.add_argument("--video-n", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=500000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); H = 96; K = make_K(H, device=dev)
    stats = []; frames_all = []; t0 = time.time()
    for k in range(a.n):
        sd = a.seed0 + k
        sc = sample_maze(sd); c2w, poses, cells, kind = sample_maze_trajectory(sc, sd, a.frames)
        c2w = c2w.to(dev); rgb, z, m = render_maze(sc, c2w, H)
        pl = plucker_rays(c2w, K, H); enc = encode_rgbd(rgb, z, m)
        rv = revisit_pairs(poses)
        adj = reproj_check(rgb, z, m, c2w, K, [(i, i + 1) for i in range(len(poses) - 1)])
        far = reproj_check(rgb, z, m, c2w, K, [(i, i + 4) for i in range(0, len(poses) - 4, 4)])
        rvc = reproj_check(rgb, z, m, c2w, K, [(i, j) for j, i in rv]) if rv else {}
        rv_psnr = [float(10 * torch.log10(1 / ((rgb[j] - rgb[i]) ** 2).mean().clamp_min(1e-10))) for j, i in rv]
        st = dict(seed=sd, kind=kind, n_frames=len(poses), n_cells_path=len(cells), n_unique_cells=len(set(cells)),
                  n_revisit_frames=len(rv), first_revisit=rv[0][0] if rv else None,
                  gt_revisit_psnr_mean=float(np.mean(rv_psnr)) if rv else None,
                  hit_frac=m.float().mean().item(), near_violation_frac=(m & (z < Z_NEAR)).float().mean().item(),
                  zmax=z[m].max().item(), plucker_moment_absmax=pl[..., 3:].abs().max().item(),
                  enc_range=[enc.min().item(), enc.max().item()],
                  reproj_adjacent=adj, reproj_gap4=far, reproj_revisit=rvc)
        stats.append(st); print(json.dumps(st), flush=True)
        np.savez_compressed(os.path.join(a.out, f"maze_{sd}.npz"), rgb=(rgb.cpu().numpy() * 255).astype(np.uint8),
                            depth=z.cpu().numpy().astype(np.float16), mask=m.cpu().numpy(), c2w=c2w.cpu().numpy(),
                            K=K.cpu().numpy(), poses=np.array(poses, np.float32), revisit=np.array(rv, np.int32).reshape(-1, 2))
        if k < a.video_n:
            rvj = {j for j, _ in rv}; rgbn = rgb.cpu().numpy(); zn = z.cpu().numpy(); mn = m.cpu().numpy()
            for f in range(len(poses)):
                r = cv2.resize((rgbn[f] * 255).astype(np.uint8), (288, 288), interpolation=cv2.INTER_NEAREST)
                d = cv2.resize(dcol(zn[f], mn[f]), (288, 288), interpolation=cv2.INTER_NEAREST)
                fr = np.concatenate([r, d, topdown(sc, poses, f, rvj)], 1)
                fr = np.concatenate([np.zeros((24, fr.shape[1], 3), np.uint8), fr], 0)
                lab = f"maze v2 seed {sd} [{kind}] frame {f+1}/{len(poses)}" + ("  REVISIT" if f in rvj else "")
                cv2.putText(fr, lab, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(fr, "RGB 96x96 | depth | top-down map", (520, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                frames_all.append(fr)
    write_h264(frames_all, os.path.join(a.out, "maze_v2_preview_h264.mp4"), fps=25)
    json.dump(stats, open(os.path.join(a.out, "maze_v2_checks.json"), "w"), indent=1)
    print("DONE", time.time() - t0, flush=True)

if __name__ == "__main__":
    main()
