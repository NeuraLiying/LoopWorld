"""Stage A geometry & interface checks (R1 plan §6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from .cameras import (
    invert_c2w,
    look_at_c2w,
    make_K,
    orbit_c2w,
    pixel_rays_cam,
    project_points,
    ray_distance_to_z_depth,
    unproject_depth,
    z_depth_to_ray_distance,
)
from .data import make_training_sample, sample_trajectory
from .model import LoopWorldModel
from .renderer import Scene, Sphere, render_scene, make_two_sphere_occlusion_scene


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def check_project_unproject(device: torch.device) -> CheckResult:
    """Analytic uv+z roundtrip + sparse image unproject (no stub pass)."""
    K = make_K(64, device=device)
    c2w = orbit_c2w(30.0, 25.0, device=device)
    pts = torch.tensor(
        [
            [0.0, 0.0, 0.4],
            [0.5, -0.3, 0.2],
            [-0.4, 0.6, 0.8],
            [0.2, 0.2, 0.1],
            [0.0, 0.0, 0.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    uv, z, valid = project_points(pts, c2w, K)
    H = W = 64
    inside = (
        valid
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < W)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < H)
        & (z > 0.5)
        & (z < 12)
    )
    if inside.sum() < 2:
        return CheckResult(
            "project_unproject", False, f"too few valid points: {inside.sum().item()}"
        )

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (uv[:, 0] - cx) / fx * z
    y = (uv[:, 1] - cy) / fy * z
    pc_rec = torch.stack([x, y, z], dim=-1)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    pw_rec = (R @ pc_rec.unsqueeze(-1)).squeeze(-1) + t
    cam_o = c2w[:3, 3]
    dist = (pts[inside] - cam_o).norm(dim=-1).clamp_min(1e-6)
    err = (pw_rec[inside] - pts[inside]).norm(dim=-1) / dist
    max_rel = err.max().item()

    # Sparse depth-map unproject: compare recovered points at integer pixels
    depth_img = torch.ones(64, 64, device=device)
    mask_img = torch.zeros(64, 64, dtype=torch.bool, device=device)
    expected = {}
    for i in range(pts.shape[0]):
        if not inside[i]:
            continue
        uu = int(uv[i, 0].round().item())
        vv = int(uv[i, 1].round().item())
        if 0 <= uu < 64 and 0 <= vv < 64:
            depth_img[vv, uu] = z[i]
            mask_img[vv, uu] = True
            expected[(vv, uu)] = pts[i]
    if not mask_img.any():
        return CheckResult("project_unproject", False, "no integer pixels for image unproject")
    pw_img = unproject_depth(depth_img, c2w, K, mask_img)
    img_rels = []
    for (vv, uu), pt in expected.items():
        # Quantization to integer pixel shifts the ray; allow looser tol than analytic
        pred = pw_img[vv, uu]
        d = (pt - cam_o).norm().clamp_min(1e-6)
        img_rels.append(((pred - pt).norm() / d).item())
    max_img_rel = max(img_rels) if img_rels else float("inf")
    ok = max_rel < 1e-4 and max_img_rel < 5e-2 and mask_img.sum().item() >= 2
    return CheckResult(
        "project_unproject",
        ok,
        f"max_rel_err={max_rel:.3e} img_unproj_rel={max_img_rel:.3e} n={inside.sum().item()} "
        f"n_img={mask_img.sum().item()}",
    )


def check_batch_unproject(device: torch.device) -> CheckResult:
    """Batch depth (B,H,W) + c2w (B,4,4) must unproject without shape errors."""
    B, H, W = 2, 8, 8
    K = make_K(H, device=device)
    c2w = orbit_c2w(
        torch.tensor([0.0, 40.0], device=device),
        torch.tensor([25.0, 30.0], device=device),
        radius=3.5,
        device=device,
    )
    assert c2w.shape == (B, 4, 4)
    depth = torch.full((B, H, W), 3.0, device=device)
    mask = torch.ones(B, H, W, dtype=torch.bool, device=device)
    try:
        pw = unproject_depth(depth, c2w, K, mask)
    except Exception as e:
        return CheckResult("batch_unproject", False, f"raised {type(e).__name__}: {e}")
    if pw.shape != (B, H, W, 3):
        return CheckResult("batch_unproject", False, f"bad shape {tuple(pw.shape)}")
    # Single-camera path still works
    pw1 = unproject_depth(depth[0], c2w[0], K, mask[0])
    if pw1.shape != (H, W, 3):
        return CheckResult("batch_unproject", False, f"single shape {tuple(pw1.shape)}")
    match = torch.allclose(pw[0], pw1, atol=1e-5)
    finite = torch.isfinite(pw).all().item()
    ok = match and finite
    return CheckResult(
        "batch_unproject",
        ok,
        f"shape={tuple(pw.shape)} match_single={match} finite={finite}",
    )


def check_camera_upright(device: torch.device) -> CheckResult:
    """OpenCV+z-up: looking along +Y from (0,0,1), right=+X, down=-Z (not 180° roll)."""
    eye = torch.tensor([0.0, 0.0, 1.0], device=device)
    target = torch.tensor([0.0, 1.0, 1.0], device=device)
    c2w = look_at_c2w(eye, target)
    right, down, forward = c2w[:3, 0], c2w[:3, 1], c2w[:3, 2]
    exp_r = torch.tensor([1.0, 0.0, 0.0], device=device)
    exp_d = torch.tensor([0.0, 0.0, -1.0], device=device)
    exp_f = torch.tensor([0.0, 1.0, 0.0], device=device)
    ok_axes = (
        torch.allclose(right, exp_r, atol=1e-5)
        and torch.allclose(down, exp_d, atol=1e-5)
        and torch.allclose(forward, exp_f, atol=1e-5)
    )
    det = torch.det(c2w[:3, :3]).item()
    # Orbit camera: world +Z should have negative image-v component (up on screen)
    c2w_o = orbit_c2w(0.0, 30.0, device=device)
    down_o = c2w_o[:3, 1]
    world_up = torch.tensor([0.0, 0.0, 1.0], device=device)
    # camera down · world_up < 0 means world up points toward image up
    upright_orbit = float(torch.dot(down_o, world_up)) < -0.5
    ok = ok_axes and abs(det - 1.0) < 1e-5 and upright_orbit
    return CheckResult(
        "camera_upright",
        ok,
        f"right={right.tolist()} down={down.tolist()} forward={forward.tolist()} "
        f"det={det:.6f} upright_orbit={upright_orbit} down·up={float(torch.dot(down_o, world_up)):.3f}",
    )


def check_z_vs_ray_distance(device: torch.device) -> CheckResult:
    """z↔ray conversion + independent ground-plane world-z assert."""
    K = make_K(32, device=device)
    c2w = orbit_c2w(0.0, 30.0, radius=3.5, device=device)
    scene = Scene(
        spheres=[], boxes=[], ground_z=0.0, ground_half=2.0, ground_color=(0.5, 0.5, 0.5)
    )
    rgb, depth, mask = render_scene(scene, c2w, K, 32)
    dirs = pixel_rays_cam(32, K, device=device)
    pts = unproject_depth(depth, c2w, K, mask)
    o = c2w[:3, 3]
    dist_eucl = (pts - o).norm(dim=-1)
    dist_from_z = z_depth_to_ray_distance(depth, dirs)
    rel = (
        (dist_from_z[mask] - dist_eucl[mask]).abs() / dist_eucl[mask].clamp_min(1e-6)
    ).max().item()
    z_back = ray_distance_to_z_depth(dist_from_z, dirs)
    z_rel = (
        (z_back[mask] - depth[mask]).abs() / depth[mask].clamp_min(1e-6)
    ).max().item()
    # Independent ground truth: hits on empty scene must lie on z=0 plane
    if mask.any():
        z_err = pts[mask][:, 2].abs().max().item()
    else:
        z_err = float("inf")
    center = depth[16, 16]
    dist_c = dist_from_z[16, 16]
    ok = rel < 1e-4 and z_rel < 1e-4 and z_err < 1e-4 and mask.sum().item() > 10
    return CheckResult(
        "z_vs_ray_distance",
        ok,
        f"dist_rel={rel:.3e} z_rel={z_rel:.3e} ground_z_err={z_err:.3e} "
        f"center_z={center.item():.4f} center_dist={dist_c.item():.4f} n_hits={mask.sum().item()}",
    )


def check_occlusion_nearest(device: torch.device) -> CheckResult:
    """Two spheres on one ray: both intersect, renderer picks nearer (red)."""
    scene = make_two_sphere_occlusion_scene()
    K = make_K(64, device=device)
    # elev=0 keeps eye, near, far collinear on horizontal through target z
    c2w = orbit_c2w(0.0, 0.0, radius=3.5, device=device)
    near_c = torch.tensor([0.6, 0.0, 0.4], device=device)
    far_c = torch.tensor([-0.6, 0.0, 0.4], device=device)
    eye = c2w[:3, 3]
    dirn = near_c - eye
    dirn = dirn / dirn.norm().clamp_min(1e-8)

    def sphere_hits(center, radius):
        oc = eye - center
        b = torch.dot(dirn, oc)
        c = torch.dot(oc, oc) - radius * radius
        disc = b * b - c
        return disc.item() >= 0.0, disc.item()

    near_hit, disc_n = sphere_hits(near_c, 0.3)
    far_hit, disc_f = sphere_hits(far_c, 0.35)
    if not (near_hit and far_hit):
        return CheckResult(
            "occlusion_nearest",
            False,
            f"ray through near must hit both spheres; near_disc={disc_n:.3f} far_disc={disc_f:.3f}",
        )

    rgb, depth, mask = render_scene(scene, c2w, K, 64)
    uv_n, z_n, v_n = project_points(near_c.unsqueeze(0), c2w, K)
    uv_f, z_f, v_f = project_points(far_c.unsqueeze(0), c2w, K)
    if not v_n[0]:
        return CheckResult("occlusion_nearest", False, "near sphere not in front of camera")
    u = int(uv_n[0, 0].round().item())
    v = int(uv_n[0, 1].round().item())
    if not (0 <= u < 64 and 0 <= v < 64):
        return CheckResult("occlusion_nearest", False, f"near proj out of frame uv=({u},{v})")
    # Same pixel for far (collinear setup)
    uf = int(uv_f[0, 0].round().item())
    vf = int(uv_f[0, 1].round().item())
    same_pixel = (u == uf) and (v == vf)
    if not mask[v, u]:
        return CheckResult("occlusion_nearest", False, "expected hit at near projection")
    hit_rgb = rgb[v, u]
    redish = hit_rgb[0] > hit_rgb[2] + 0.2
    z_hit = depth[v, u].item()
    closer_to_near = abs(z_hit - z_n[0].item()) < abs(z_hit - z_f[0].item())
    pts = unproject_depth(depth, c2w, K, mask)
    n_bg_finite = torch.isfinite(pts[~mask]).any().item() if (~mask).any() else False
    n_fg = mask.sum().item()
    ok = (
        redish
        and closer_to_near
        and (not n_bg_finite)
        and n_fg > 0
        and same_pixel
        and near_hit
        and far_hit
    )
    return CheckResult(
        "occlusion_nearest",
        ok,
        f"rgb={hit_rgb.tolist()} z_hit={z_hit:.3f} z_near={z_n[0].item():.3f} "
        f"z_far={z_f[0].item():.3f} redish={redish} closer={closer_to_near} "
        f"same_pixel={same_pixel} dual_hit=True bg_pts={n_bg_finite} n_fg={n_fg}",
    )


def check_trajectory_protocol(device: torch.device) -> CheckResult:
    """25 poses, closed loop, correct j ranges, no self-recon / dup ctx."""
    traj = sample_trajectory(7)
    n = traj.shape[0]
    closed = torch.allclose(traj[0], traj[-1], atol=1e-5)
    # angular step: eyes of adjacent frames (excluding last==first pair as step)
    d_adj = (traj[1, :3, 3] - traj[0, :3, 3]).norm().item()
    d_last_open = (traj[-2, :3, 3] - traj[-3, :3, 3]).norm().item()
    uniform = abs(d_adj - d_last_open) / max(d_adj, 1e-6) < 1e-3

    bad = []
    zeros = 0
    dup = 0
    revisit = 0
    for i in range(80):
        s = make_training_sample(
            1000 + i, 2000 + i, 8, device, single_context_prob=1.0
        )
        if s.meta["target_idx"] == 0:
            zeros += 1
            bad.append("single_target0")
        if s.meta["target_idx"] < 1 or s.meta["target_idx"] > 24:
            bad.append(f"single_j={s.meta['target_idx']}")
        if s.meta["ctx"] != [0]:
            bad.append(f"single_ctx={s.meta['ctx']}")
        if s.meta.get("is_closed_loop_revisit"):
            revisit += 1
        # rendered views subset
        need = s.meta["rendered_views"]
        if len(need) > 2:
            bad.append(f"single_over_render={need}")

    for i in range(80):
        s = make_training_sample(
            3000 + i, 4000 + i, 8, device, single_context_prob=0.0
        )
        j = s.meta["target_idx"]
        ctx = s.meta["ctx"]
        if j < 2 or j > 24:
            bad.append(f"double_j={j}")
        if ctx != [0, j - 1]:
            bad.append(f"double_ctx={ctx}")
        if ctx[0] == ctx[1]:
            dup += 1
            bad.append("dup_ctx")
        if len(s.meta["rendered_views"]) > 3:
            bad.append(f"double_over_render={s.meta['rendered_views']}")

    ok = (
        n == 25
        and closed
        and uniform
        and zeros == 0
        and dup == 0
        and len(bad) == 0
    )
    return CheckResult(
        "trajectory_protocol",
        ok,
        f"n={n} closed={closed} uniform={uniform} single_target0={zeros} "
        f"dup_ctx={dup} revisit_in_single_80={revisit} bad_n={len(bad)} "
        f"bad_sample={bad[:3]}",
    )


def check_model_ignores_evaluator_gt(device: torch.device) -> CheckResult:
    """Fixed inputs/noise: model API must not accept scene/renderer/GT handles."""
    torch.manual_seed(0)
    model = LoopWorldModel(image_size=32, patch_size=4, width=64, heads=2).to(device)
    model.eval()
    B, H, W, S = 2, 32, 32, 2
    xt = torch.randn(B, H, W, 4, device=device)
    t = torch.rand(B, device=device)
    tpl = torch.randn(B, H, W, 6, device=device)
    cr = torch.randn(B, S, H, W, 4, device=device)
    cp = torch.randn(B, S, H, W, 6, device=device)
    cv = torch.tensor([[True, False], [True, True]], device=device)
    with torch.no_grad():
        out1 = model.forward_velocity(xt, t, tpl, cr, cp, cv, num_loops=2)
        out2 = model.forward_velocity(xt, t, tpl, cr, cp, cv, num_loops=2)
    import inspect

    sig = inspect.signature(model.forward_velocity)
    banned = {"scene", "renderer", "gt", "target_gt", "target_rgb"}
    has_banned = bool(banned & set(sig.parameters))
    same = torch.allclose(out1, out2, atol=0, rtol=0)
    # sample_view path: only context + cameras + noise
    from loopworld.eval import sample_view

    sig_s = inspect.signature(sample_view)
    has_banned_s = bool(banned & set(sig_s.parameters))
    ok = same and not has_banned and not has_banned_s
    return CheckResult(
        "model_ignores_evaluator_gt",
        ok,
        f"identical={same} banned_fwd={has_banned} banned_sample={has_banned_s} "
        f"max_diff={(out1 - out2).abs().max().item():.3e}",
    )


def check_loops_and_masking(device: torch.device) -> CheckResult:
    """Param count independent of R; shapes OK; missing slots masked; hidden state carried."""
    torch.manual_seed(1)
    model = LoopWorldModel(image_size=32, patch_size=4, width=64, heads=2).to(device)
    model.eval()
    n0 = model.count_parameters()
    B, H, W, S = 2, 32, 32, 2
    xt = torch.randn(B, H, W, 4, device=device)
    t = torch.tensor([0.3, 0.7], device=device)
    tpl = torch.randn(B, H, W, 6, device=device)
    cr = torch.randn(B, S, H, W, 4, device=device)
    cp = torch.randn(B, S, H, W, 6, device=device)
    cv = torch.tensor([[True, False], [True, True]], device=device)

    shapes_ok = True
    details = [f"params={n0}"]
    with torch.no_grad():
        for R in (1, 2, 4):
            outs = model.forward_exits(
                xt, t, tpl, cr, cp, cv, num_loops=R, return_all_exits=True
            )
            if len(outs) != R:
                shapes_ok = False
            for o in outs:
                if o.shape != (B, H, W, 4):
                    shapes_ok = False
            details.append(f"R={R}:n_exits={len(outs)},shape={tuple(outs[-1].shape)}")

    with torch.no_grad():
        for blk in list(model.core) + [model.pre, model.post]:
            D = model.width
            bias = blk.adaLN[-1].bias
            bias.zero_()
            bias[2 * D : 3 * D] = 1.0
            bias[5 * D : 6 * D] = 1.0
        model.head.proj.weight.normal_(0, 0.02)
        model.head.proj.bias.normal_(0, 0.02)
        model.head.adaLN[-1].bias.zero_()

        tokens, pad = model._embed_views(xt, tpl, cr, cp, cv)
        t_emb = model.t_embed(t)
        h0 = model.pre(tokens, t_emb, key_padding_mask=pad)
        h1 = h0
        for block in model.core:
            h1 = block(h1, t_emb, key_padding_mask=pad)
        h2 = h1
        for block in model.core:
            h2 = block(h2, t_emb, key_padding_mask=pad)
        hidden_loops_change = not torch.allclose(h1, h2, atol=1e-5)
        core_changes = not torch.allclose(h0, h1, atol=1e-5)

        e1 = model.forward_exits(
            xt, t, tpl, cr, cp, cv, num_loops=1, return_all_exits=True
        )[-1]
        e2_all = model.forward_exits(
            xt, t, tpl, cr, cp, cv, num_loops=2, return_all_exits=True
        )
        mid_matches = torch.allclose(e1, e2_all[0], atol=1e-5)
        exits_change = not torch.allclose(e1, e2_all[-1], atol=1e-5)

        h_reinit = model.pre(tokens, t_emb, key_padding_mask=pad)
        for block in model.core:
            h_reinit = block(h_reinit, t_emb, key_padding_mask=pad)
        carried_not_reinit = torch.allclose(h_reinit, h1, atol=1e-5) and hidden_loops_change

    details.append(
        f"core_changes={core_changes} hidden_loops_change={hidden_loops_change} "
        f"mid_matches_R1={mid_matches} exits_change={exits_change} "
        f"carried_not_reinit={carried_not_reinit}"
    )
    loops_change = (
        core_changes
        and hidden_loops_change
        and mid_matches
        and exits_change
        and carried_not_reinit
    )

    cr_zero = cr.clone()
    cr_zero[:, 1] = 0
    cr_junk = cr.clone()
    cr_junk[:, 1] = torch.randn_like(cr_junk[:, 1]) * 10
    cv_mask = torch.tensor([[True, False], [True, False]], device=device)
    with torch.no_grad():
        o_z = model.forward_velocity(xt, t, tpl, cr_zero, cp, cv_mask, num_loops=2)
        o_j = model.forward_velocity(xt, t, tpl, cr_junk, cp, cv_mask, num_loops=2)
    masked_ok = torch.allclose(o_z, o_j, atol=1e-5)
    details.append(f"masked_invariant={masked_ok} diff={(o_z - o_j).abs().max().item():.3e}")

    n1 = model.count_parameters()
    params_ok = n0 == n1
    ok = shapes_ok and loops_change and mid_matches and masked_ok and params_ok
    return CheckResult("loops_and_masking", ok, "; ".join(details))


def run_all_checks(device: torch.device | None = None) -> Tuple[bool, List[CheckResult]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [
        check_project_unproject(device),
        check_batch_unproject(device),
        check_camera_upright(device),
        check_z_vs_ray_distance(device),
        check_occlusion_nearest(device),
        check_trajectory_protocol(device),
        check_model_ignores_evaluator_gt(device),
        check_loops_and_masking(device),
    ]
    return all(r.passed for r in results), results
