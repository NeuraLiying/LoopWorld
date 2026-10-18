#!/usr/bin/env python3
"""Generate short orbit RGB(+depth) demo video from a Stage D checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import torch

from loopworld.cameras import make_K, orbit_c2w, plucker_rays
from loopworld.data import sample_scene, sample_trajectory
from loopworld.encoding import decode_rgbd, encode_rgbd
from loopworld.eval import sample_view
from loopworld.model import LoopWorldModel
from loopworld.renderer import render_scene


def depth_to_uint8(z, mask=None, z_near=0.5, z_far=12.0):
    z = z.detach().float().cpu()
    u = (z_near / z.clamp(min=z_near, max=z_far)).clamp(0, 1)
    if mask is not None:
        u = torch.where(mask.cpu(), u, torch.zeros_like(u))
    g = (u.numpy() * 255.0 + 0.5).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def rgb_to_uint8(rgb):
    x = rgb.detach().float().cpu().clamp(0, 1).numpy()
    return (x * 255.0 + 0.5).astype(np.uint8)


def load_ema_model(ckpt_path: Path, device: torch.device) -> tuple[LoopWorldModel, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config") or {}
    args = ckpt.get("args") or {}
    # prefer config, fall back to explicit
    image_size = int(cfg.get("image_size", 96))
    width = int(cfg.get("width", 768))
    heads = int(cfg.get("heads", 12))
    patch_size = int(cfg.get("patch_size", 4))
    # allow override from resolved if stored oddly
    model = LoopWorldModel(
        image_size=image_size, patch_size=patch_size, width=width, heads=heads
    ).to(device)
    if "ema" in ckpt and ckpt["ema"] is not None:
        model.load_state_dict(ckpt["ema"], strict=True)
        src = "ema"
    else:
        model.load_state_dict(ckpt["model"], strict=True)
        src = "model"
    model.eval()
    meta = {
        "image_size": image_size,
        "width": width,
        "heads": heads,
        "step": ckpt.get("step"),
        "weights": src,
        "train_loops": int(cfg.get("train_loops", 4)),
        "sample_steps": int(cfg.get("default_sampling_steps", 20)),
    }
    return model, meta


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True, help="mp4 path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--scene-seed", type=int, default=300007)
    p.add_argument("--traj-seed", type=int, default=300007)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--scale", type=int, default=4, help="nearest upsample for visibility")
    p.add_argument("--num-loops", type=int, default=None)
    p.add_argument("--sample-steps", type=int, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    model, meta = load_ema_model(Path(args.ckpt), device)
    H = W = meta["image_size"]
    R = args.num_loops or meta["train_loops"]
    S = args.sample_steps or meta["sample_steps"]
    print(f"model {meta} R={R} S={S}")

    scene = sample_scene(args.scene_seed)
    c2ws = sample_trajectory(args.traj_seed).to(device)  # 25 poses, closed loop
    K = make_K(H, device=device)
    n = c2ws.shape[0]
    print(f"trajectory poses={n}")

    # GT first frame as sole context (autoregressive-style conditioning for demo)
    rgb0, depth0, mask0 = render_scene(scene, c2ws[0], K, H)
    ctx_rgbd = encode_rgbd(rgb0, depth0, mask0).unsqueeze(0).unsqueeze(0)  # 1,1,H,W,4
    # pad second slot as invalid
    ctx_rgbd = torch.cat(
        [ctx_rgbd, torch.zeros(1, 1, H, W, 4, device=device)], dim=1
    )
    ctx_pl = plucker_rays(c2ws[0], K, H).unsqueeze(0).unsqueeze(0)
    ctx_pl = torch.cat([ctx_pl, torch.zeros(1, 1, H, W, 6, device=device)], dim=1)
    ctx_valid = torch.tensor([[True, False]], device=device)

    torch.manual_seed(20261012)
    frames = []
    # Frame 0: GT context (mark as GT)
    gt0 = rgb_to_uint8(rgb0)
    d0 = depth_to_uint8(depth0, mask0)
    label0 = np.concatenate([gt0, d0], axis=1)
    frames.append(("gt", label0))

    # Generate targets j=1..24 (include closed-loop revisit)
    for j in range(1, n):
        tgt_pl = plucker_rays(c2ws[j], K, H).unsqueeze(0)
        noise = torch.randn(1, H, H, 4, device=device)
        x = sample_view(
            model,
            ctx_rgbd,
            ctx_pl,
            ctx_valid,
            tgt_pl,
            num_loops=R,
            steps=S,
            noise=noise,
        )
        pred_rgb, dinfo, _ = decode_rgbd(x)
        # also render GT for side-by-side
        gt_rgb, gt_d, gt_m = render_scene(scene, c2ws[j], K, H)
        left = rgb_to_uint8(gt_rgb)
        right = rgb_to_uint8(pred_rgb[0])
        d_gt = depth_to_uint8(gt_d, gt_m)
        d_pr = depth_to_uint8(dinfo["z_pred"][0], dinfo["valid_surface"][0])
        # layout: GT-RGB | Pred-RGB | GT-Depth | Pred-Depth
        row = np.concatenate([left, right, d_gt, d_pr], axis=1)
        frames.append(("gen", row))
        if j % 4 == 0 or j == n - 1:
            print(f"  generated view {j}/{n-1}")

    # Write mp4 via OpenCV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # all frames same width: frame0 is 2 panels, others 4 — pad frame0
    w4 = frames[1][1].shape[1]
    h0, w0 = frames[0][1].shape[:2]
    pad = np.zeros((h0, w4 - w0, 3), dtype=np.uint8)
    frames[0] = ("gt", np.concatenate([frames[0][1], pad], axis=1))

    scale = max(1, int(args.scale))
    h, w = frames[0][1].shape[:2]
    Hs, Ws = h * scale, w * scale
    # mp4v then fallback
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(args.fps), (Ws, Hs))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(out_path), fourcc, float(args.fps), (Ws, Hs))
    assert writer.isOpened(), f"failed to open VideoWriter for {out_path}"

    for kind, fr in frames:
        if scale > 1:
            fr = cv2.resize(fr, (Ws, Hs), interpolation=cv2.INTER_NEAREST)
        # OpenCV expects BGR
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()

    meta_out = {
        "ckpt": str(args.ckpt),
        "out": str(out_path),
        "scene_seed": args.scene_seed,
        "traj_seed": args.traj_seed,
        "n_frames": len(frames),
        "fps": args.fps,
        "layout": "GT-RGB | Pred-RGB | GT-Depth | Pred-Depth (frame0: GT only padded)",
        "conditioning": "single GT first-frame context; each target independent (teacher-forced context, not autoregressive rollout)",
        "model_meta": meta,
        "num_loops": R,
        "sample_steps": S,
    }
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes) frames={len(frames)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
