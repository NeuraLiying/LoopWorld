#!/usr/bin/env python3
"""Stage B: 32x32 overfit on 4 scenes (smoke width-256). Max 1000 steps or 10 min."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from PIL import Image

from loopworld.data import collate_samples, make_training_sample, split_seeds
from loopworld.encoding import decode_rgbd
from loopworld.eval import sample_view
from loopworld.model import LoopWorldModel
from loopworld.train import EMA, build_optimizer, train_step


def _to_uint8_rgb(rgb01: torch.Tensor) -> np.ndarray:
    x = rgb01.detach().float().cpu().clamp(0, 1).numpy()
    return (x * 255.0 + 0.5).astype(np.uint8)


def _depth_to_uint8(
    z: torch.Tensor, mask: torch.Tensor | None = None, z_near=0.5, z_far=12.0
) -> np.ndarray:
    z = z.detach().float().cpu()
    u = (z_near / z.clamp(min=z_near, max=z_far)).clamp(0, 1)
    if mask is not None:
        u = torch.where(mask.cpu(), u, torch.zeros_like(u))
    g = (u.numpy() * 255.0 + 0.5).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def save_preview_grid(
    out_path: Path,
    gt_rgb: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_rgb: torch.Tensor,
    pred_z: torch.Tensor,
    pred_valid: torch.Tensor,
    n: int = 4,
    scale: int = 8,
):
    tiles = []
    for i in range(min(n, gt_rgb.shape[0])):
        col = np.concatenate(
            [
                _to_uint8_rgb(gt_rgb[i]),
                _depth_to_uint8(gt_depth[i], gt_mask[i]),
                _to_uint8_rgb(pred_rgb[i]),
                _depth_to_uint8(pred_z[i], pred_valid[i]),
            ],
            axis=0,
        )
        tiles.append(col)
    grid = np.concatenate(tiles, axis=1)
    H, W = grid.shape[:2]
    img = Image.fromarray(grid).resize((W * scale, H * scale), Image.NEAREST)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"  saved preview {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--max-minutes", type=float, default=10.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out-dir", type=str, default="")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else repo / "exp_records" / f"stage_b_overfit_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Stage B overfit: size={args.image_size} width={args.width} device={device}")
    print(f"  out_dir={out_dir}")
    if device.type == "cuda":
        print(
            f"  GPU: {torch.cuda.get_device_name(0)} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )

    torch.manual_seed(20261012)
    model = LoopWorldModel(
        image_size=args.image_size, patch_size=4, width=args.width, heads=args.heads
    ).to(device)
    nparams = model.count_parameters()
    print(f"  params={nparams:,}")
    opt = build_optimizer(model, lr=args.lr)
    ema = EMA(model)

    seeds = split_seeds()["overfit"]
    pool = []
    for sid in seeds:
        for k in range(8):
            pool.append(make_training_sample(sid, sid * 10 + k, args.image_size, device))

    eval_batch = collate_samples(pool[: args.batch_size])
    eval_dict = {
        "target_rgbd": eval_batch.target_rgbd,
        "target_plucker": eval_batch.target_plucker,
        "context_rgbd": eval_batch.context_rgbd,
        "context_plucker": eval_batch.context_plucker,
        "context_valid": eval_batch.context_valid,
    }
    torch.manual_seed(12345)
    t_fixed = torch.rand(args.batch_size, device=device)
    eps_fixed = torch.randn_like(eval_dict["target_rgbd"])

    def eval_loss(m):
        from loopworld.flow import sample_xt, velocity_target
        from loopworld.losses import exit_losses

        m.eval()
        with torch.no_grad():
            xt = sample_xt(eval_dict["target_rgbd"], t_fixed, eps_fixed)
            v = velocity_target(eval_dict["target_rgbd"], eps_fixed)
            exits = m.forward_exits(
                xt,
                t_fixed,
                eval_dict["target_plucker"],
                eval_dict["context_rgbd"],
                eval_dict["context_plucker"],
                eval_dict["context_valid"],
                num_loops=4,
                return_all_exits=True,
            )
            L = exit_losses(exits, v, [1 / 6, 1 / 6, 1 / 6, 1 / 2])["L_total"]
        m.train()
        return float(L)

    torch.manual_seed(999)
    preview_noise = torch.randn(
        eval_batch.target_rgbd.shape[0], args.image_size, args.image_size, 4, device=device
    )

    def previews_with_noise(m):
        m.eval()
        with torch.no_grad():
            x = sample_view(
                m,
                eval_batch.context_rgbd,
                eval_batch.context_plucker,
                eval_batch.context_valid,
                eval_batch.target_plucker,
                num_loops=4,
                steps=20,
                noise=preview_noise.clone(),
            )
            rgb, depth_info, oob = decode_rgbd(x)
        return rgb, depth_info, oob, x

    rgb0, dinfo0, oob0, _ = previews_with_noise(model)
    save_preview_grid(
        preview_dir / "step0000_init.png",
        eval_batch.target_rgb,
        eval_batch.target_depth,
        eval_batch.target_mask,
        rgb0,
        dinfo0["z_pred"],
        dinfo0["valid_surface"],
    )

    L0 = eval_loss(model)
    print(f"  initial_eval_loss={L0:.6f}")
    t0 = time.time()
    step = 0
    saw_nan = False
    last_metrics = {}
    while step < args.steps and (time.time() - t0) < args.max_minutes * 60:
        idx = torch.randint(0, len(pool), (args.batch_size,)).tolist()
        batch = collate_samples([pool[i] for i in idx])
        bd = {
            "target_rgbd": batch.target_rgbd,
            "target_plucker": batch.target_plucker,
            "context_rgbd": batch.context_rgbd,
            "context_plucker": batch.context_plucker,
            "context_valid": batch.context_valid,
        }
        metrics = train_step(model, bd, opt, train_loops=4)
        ema.update(model)
        step += 1
        last_metrics = metrics
        if (not np.isfinite(metrics["L_total"])) or (
            not np.isfinite(metrics.get("grad_norm", 0.0))
        ):
            saw_nan = True
            print(f"  NaN/Inf at step={step} metrics={metrics}")
            break
        if step % 50 == 0 or step == 1:
            lt = metrics["L_total"]
            gn = metrics["grad_norm"]
            print(f"  step={step} loss={lt:.4f} grad={gn:.3f}")

    elapsed = time.time() - t0
    L1 = eval_loss(model)
    ema_model = LoopWorldModel(
        image_size=args.image_size, patch_size=4, width=args.width, heads=args.heads
    ).to(device)
    ema.copy_to(ema_model)
    L1_ema = eval_loss(ema_model)

    rgb1, dinfo1, oob1, _ = previews_with_noise(ema_model)
    save_preview_grid(
        preview_dir / f"step{step:04d}_ema.png",
        eval_batch.target_rgb,
        eval_batch.target_depth,
        eval_batch.target_mask,
        rgb1,
        dinfo1["z_pred"],
        dinfo1["valid_surface"],
    )
    rgb_m, dinfo_m, _, _ = previews_with_noise(model)
    save_preview_grid(
        preview_dir / f"step{step:04d}_model.png",
        eval_batch.target_rgb,
        eval_batch.target_depth,
        eval_batch.target_mask,
        rgb_m,
        dinfo_m["z_pred"],
        dinfo_m["valid_surface"],
    )

    drop = (L0 - L1) / max(L0, 1e-8)
    drop_ema = (L0 - L1_ema) / max(L0, 1e-8)
    finite = np.isfinite(L0) and np.isfinite(L1) and np.isfinite(L1_ema) and (not saw_nan)

    pred = rgb1.float()
    gt = eval_batch.target_rgb.float()
    pred_std = float(pred.std())
    cors = []
    for i in range(pred.shape[0]):
        a = pred[i].reshape(-1)
        b = gt[i].reshape(-1)
        a = a - a.mean()
        b = b - b.mean()
        denom = (a.norm() * b.norm()).clamp_min(1e-8)
        cors.append(float((a * b).sum() / denom))
    mean_corr = float(np.mean(cors))
    gt_z = eval_batch.target_depth
    gt_m = eval_batch.target_mask
    z_pred = dinfo1["z_metric"]
    if gt_m.any():
        absrel = float(
            ((z_pred[gt_m] - gt_z[gt_m]).abs() / gt_z[gt_m].clamp_min(1e-3)).mean()
        )
    else:
        absrel = float("nan")

    # Encoded RGB-D finite check on eval targets/contexts
    enc_finite = bool(
        torch.isfinite(eval_batch.target_rgbd).all()
        and torch.isfinite(eval_batch.context_rgbd).all()
    )
    # Depth gates (EMA preview vs GT)
    valid_pred = dinfo1["valid_surface"]
    nonfinite_pred = dinfo1["nonfinite"]
    if gt_m.any():
        depth_valid_on_gt = float(valid_pred[gt_m].float().mean())
    else:
        depth_valid_on_gt = float("nan")
    if (~gt_m).any():
        bg_false_surface = float(valid_pred[~gt_m].float().mean())
    else:
        bg_false_surface = 0.0
    depth_nonfinite_frac = float(nonfinite_pred.float().mean())
    depth_absrel_ok = bool(np.isfinite(absrel) and absrel < 0.50)
    depth_valid_ok = bool(np.isfinite(depth_valid_on_gt) and depth_valid_on_gt >= 0.30)
    depth_bg_ok = bool(bg_false_surface <= 0.50)
    depth_finite_ok = bool(depth_nonfinite_frac < 1e-3)
    depth_screen = depth_absrel_ok and depth_valid_ok and depth_bg_ok and depth_finite_ok

    loss_screen = (L1 < L0) and (drop >= 0.7)
    recognizable = (pred_std > 0.05) and (mean_corr > 0.15)
    finite = bool(finite and enc_finite)
    passed = bool(finite and loss_screen and recognizable and depth_screen)

    summary = {
        "stage": "B",
        "image_size": args.image_size,
        "width": args.width,
        "heads": args.heads,
        "params": nparams,
        "steps": step,
        "elapsed_s": elapsed,
        "initial_eval_loss": L0,
        "final_eval_loss": L1,
        "final_ema_eval_loss": L1_ema,
        "drop_frac": drop,
        "drop_ema_frac": drop_ema,
        "saw_nan": saw_nan,
        "finite": finite,
        "enc_finite": enc_finite,
        "pred_rgb_std": pred_std,
        "pred_gt_corr": mean_corr,
        "depth_absrel_ema": absrel,
        "depth_valid_on_gt": depth_valid_on_gt,
        "bg_false_surface_rate": bg_false_surface,
        "depth_nonfinite_frac": depth_nonfinite_frac,
        "depth_absrel_ok": depth_absrel_ok,
        "depth_valid_ok": depth_valid_ok,
        "depth_bg_ok": depth_bg_ok,
        "depth_finite_ok": depth_finite_ok,
        "depth_screen": depth_screen,
        "rgb_oob_frac": float(oob1),
        "loss_screen_70pct": loss_screen,
        "recognizable_layout": recognizable,
        "passed": passed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "previews": [
            str(preview_dir / "step0000_init.png"),
            str(preview_dir / f"step{step:04d}_ema.png"),
            str(preview_dir / f"step{step:04d}_model.png"),
        ],
        "last_train_metrics": {k: float(v) for k, v in last_metrics.items()},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(
        {"model": model.state_dict(), "ema": ema.shadow, "step": step, "args": vars(args)},
        out_dir / "overfit_ckpt.pt",
    )

    print(
        f"  final_eval_loss={L1:.6f} ema={L1_ema:.6f} "
        f"drop={drop*100:.1f}% drop_ema={drop_ema*100:.1f}%"
    )
    print(
        f"  steps={step} elapsed_s={elapsed:.1f} nan={saw_nan} "
        f"corr={mean_corr:.3f} absrel={absrel:.3f} std={pred_std:.3f} "
        f"depth_screen={depth_screen} valid_on_gt={depth_valid_on_gt:.3f} "
        f"bg_false={bg_false_surface:.3f}"
    )
    print(f"  summary={out_dir / 'summary.json'}")
    print("=== Stage B: {} ===".format("PASS" if passed else "FAIL/CHECK"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
