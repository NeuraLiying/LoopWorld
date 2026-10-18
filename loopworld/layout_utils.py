"""Layout-error round helpers (new file): hard-frame detection, confident-pixel mask, weighted train step, depth metrics."""
from __future__ import annotations
import numpy as np, torch
from .flow import sample_xt, velocity_target


def hard_frames(ep, gz, gm, wall_z=1.2):
    """(T,) bool: wall-facing (central-region median GT depth < wall_z, ~31% of frames) OR a turn taken at a junction cell (deg>=3)."""
    T = gz.shape[0]; H = gz.shape[1]; c = slice(H // 3, 2 * H // 3)
    zc = torch.where(gm[:, c, c], gz[:, c, c], torch.full_like(gz[:, c, c], 12.0)).flatten(1).median(1).values.cpu().numpy()
    acts = np.asarray(ep.actions)
    hard = zc < wall_z
    junc = [ep.scene.cell_center(k) for k, v in ep.scene.adj.items() if len(v) >= 3]
    for t, (x, y, _) in enumerate(ep.poses):
        if acts[t] >= 2 and any(abs(cx - x) < 0.25 and abs(cy - y) < 0.25 for cx, cy in junc): hard[t] = True
    return torch.tensor(hard)


def confident_mask(x1, x2, rgb_tol=0.08, depth_rel_tol=0.05):
    """x1,x2 encoded RGB-D samples (B,H,W,4) from two noises. Returns (B,H,W) bool."""
    from .encoding import decode_rgbd
    r1, d1, _ = decode_rgbd(x1); r2, d2, _ = decode_rgbd(x2)
    rgb_ok = (r1 - r2).abs().amax(-1) < rgb_tol
    z1, z2 = d1["z_metric"], d2["z_metric"]
    dep_ok = (z1 - z2).abs() / torch.minimum(z1, z2).clamp_min(1e-3) < depth_rel_tol
    return rgb_ok & dep_ok & d1["valid_surface"] & d2["valid_surface"]


def weighted_train_step(model, batch, opt, weights, train_loops, exit_weights, grad_clip=1.0):
    """Like loopworld.train.train_step but per-sample loss weights (B,), normalized to mean 1."""
    model.train(); x = batch["target_rgbd"]; B = x.shape[0]; dev = x.device
    t = torch.rand(B, device=dev); eps = torch.randn_like(x); xt = sample_xt(x, t, eps); v = velocity_target(x, eps)
    w = weights / weights.mean()
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        exits = model.forward_exits(xt, t, batch["target_plucker"], batch["context_rgbd"], batch["context_plucker"],
                                    batch["context_valid"], num_loops=train_loops, return_all_exits=True)
    total = torch.zeros((), device=dev)
    for vk, ew in zip(exits, exit_weights):
        err = (vk.float() - v.float()) ** 2
        per = 0.5 * err[..., :3].mean((1, 2, 3)) + 0.5 * err[..., 3].mean((1, 2))
        total = total + ew * (per * w).sum() / B
    total.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    skipped = 0.0
    if not (torch.isfinite(total) and torch.isfinite(gn)):
        opt.zero_grad(set_to_none=True); skipped = 1.0
    else:
        opt.step()
    return {"L_total": float(total.detach()), "grad_norm": float(gn), "skipped": skipped}


def depth_metrics(pz, pv, gz, gm):
    """pz,gz (...,H,W) z; pv pred valid surface, gm GT hit. Returns dict."""
    sel = gm; rel = ((pz - gz).abs() / gz)[sel]
    ratio = torch.maximum(pz[sel] / gz[sel], gz[sel] / pz[sel].clamp_min(1e-3))
    inter = (pv & gm).sum().float(); union = (pv | gm).sum().float()
    oi = (~pv & ~gm).sum().float(); ou = (~pv | ~gm).sum().float()
    return {"absrel_mean": float(rel.mean()), "absrel_median": float(rel.median()) if rel.numel() < 16_000_000 else float(rel[::4].median()),
            "delta1": float((ratio < 1.25).float().mean()), "iou_wall": float(inter / union.clamp_min(1)), "iou_open": float(oi / ou.clamp_min(1))}
