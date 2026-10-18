#!/usr/bin/env python3
"""Stage D: ~1-hour main train (R1 width-640). Engineering demo run.

Loads configs/r1_main_640.yaml. Stops on wall-clock OR update-budget
(whichever comes first). Saves mid/final checkpoints, curve JSONL, summary.
Does NOT run Stage E eval matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml

from loopworld.data import collate_samples, make_training_sample, split_seeds
from loopworld.encoding import decode_rgbd
from loopworld.eval import sample_view
from loopworld.model import LoopWorldModel
from loopworld.train import EMA, build_optimizer, train_step


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_online_batch(batch_size, image_size, device, step_idx, train_seeds):
    samples = []
    for i in range(batch_size):
        sid = train_seeds[(step_idx * batch_size + i) % len(train_seeds)]
        traj = sid * 1009 + step_idx * 17 + i
        samples.append(make_training_sample(sid, traj, image_size, device))
    batch = collate_samples(samples)
    return batch, {
        "target_rgbd": batch.target_rgbd,
        "target_plucker": batch.target_plucker,
        "context_rgbd": batch.context_rgbd,
        "context_plucker": batch.context_plucker,
        "context_valid": batch.context_valid,
    }


def set_lr(opt, lr: float):
    for g in opt.param_groups:
        g["lr"] = lr


LR_MIN_FRAC = 0.1


def lr_at_step(step: int, total: int, base_lr: float, warmup: int) -> float:
    """Linear warmup then cosine decay to 0.1 * base_lr."""
    if total <= 0:
        return base_lr
    if step < warmup:
        return base_lr * float(step + 1) / float(max(warmup, 1))
    t = (step - warmup) / max(total - warmup, 1)
    t = min(max(t, 0.0), 1.0)
    # cosine from 1 -> 0.1
    return base_lr * (LR_MIN_FRAC + (1.0 - LR_MIN_FRAC) * 0.5 * (1.0 + math.cos(math.pi * t)))


MIN_FREE_BYTES = 5 * 1024**3
CKPT_EST_BYTES = 800 * 1024**2
BEST_CKPT_PATH = None  # set by held-out logic; never deleted by disk guard


def _ensure_space(dirpath: Path) -> bool:
    """If free < 5GB + one ckpt, delete oldest ckpt_step*.pt in THIS run dir only
    (never newest, never BEST_CKPT_PATH). Returns False if still low -> caller skips save."""
    import shutil
    need = MIN_FREE_BYTES + CKPT_EST_BYTES
    while shutil.disk_usage(dirpath).free < need:
        cands = sorted(dirpath.glob("ckpt_step*.pt"), key=lambda q: q.stat().st_mtime)
        cands = [c for c in cands[:-1] if BEST_CKPT_PATH is None or c.resolve() != Path(BEST_CKPT_PATH).resolve()]
        if not cands:
            print(f"  [diskguard] free={shutil.disk_usage(dirpath).free/1e9:.1f}GB low, nothing deletable -> SKIP save", flush=True)
            return False
        print(f"  [diskguard] free={shutil.disk_usage(dirpath).free/1e9:.1f}GB low; deleting {cands[0]}", flush=True)
        cands[0].unlink()
    return True


def save_ckpt(path: Path, model, ema, opt, step, args_dict, cfg, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not _ensure_space(path.parent):
        return False
    payload = {
        "model": model.state_dict(),
        "ema": ema.shadow,
        "optimizer": opt.state_dict(),
        "step": step,
        "ema_n": ema.n,
        "args": args_dict,
        "config": cfg,
        "rng_torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["rng_cuda"] = torch.cuda.get_rng_state_all()
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic: no truncated ckpts
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="",
        help="YAML config (default: configs/r1_main_640.yaml)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="")
    p.add_argument("--wall-budget-seconds", type=float, default=None)
    p.add_argument("--update-budget-seconds", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-minutes", type=str, default="15,30")
    p.add_argument("--batch-size", type=int, default=None, help="override yaml batch_size")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--tag", type=str, default="", help="optional out-dir tag suffix")
    p.add_argument("--resume", type=str, default="", help="ckpt to resume model/EMA/optimizer/RNG/step from")
    p.add_argument("--heldout-every", type=int, default=500)
    p.add_argument("--lr-peak", type=float, default=None, help="override yaml learning_rate (peak)")
    p.add_argument("--lr-min-frac", type=float, default=0.1, help="cosine floor as fraction of peak")
    p.add_argument("--grad-clip", type=float, default=None, help="override yaml grad_clip_norm")
    p.add_argument("--skip-nonfinite", action="store_true", help="skip optimizer step on non-finite loss/grad")
    p.add_argument("--max-consec-skips", type=int, default=50)
    p.add_argument("--reset-schedule", action="store_true", help="new warmup+cosine from resume step to max_steps")
    p.add_argument("--ckpt-every", type=int, default=0, help="save ckpt every N global steps (keep all)")
    p.add_argument("--ckpt-eval", action="store_true", help="spawn async S/T/AR R4 eval at each ckpt-every ckpt")
    p.add_argument("--logit-every", type=int, default=0,
                   help="every N steps record/log max attention logit (pre+core); 0=off")
    args = p.parse_args()
    global LR_MIN_FRAC
    LR_MIN_FRAC = float(args.lr_min_frac)

    repo = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config) if args.config else repo / "configs" / "r1_main_640.yaml"
    cfg = load_yaml(cfg_path)

    image_size = int(args.image_size if args.image_size is not None else cfg["image_size"])
    width = int(args.width if args.width is not None else cfg["width"])
    heads = int(args.heads if args.heads is not None else cfg["heads"])
    batch_size = int(args.batch_size if args.batch_size is not None else cfg["batch_size"])
    train_loops = int(cfg["train_loops"])
    lr = float(args.lr_peak if args.lr_peak is not None else cfg["learning_rate"])
    grad_clip_v = float(args.grad_clip if args.grad_clip is not None else cfg.get("grad_clip_norm", 1.0))
    assert width % heads == 0, f"width {width} not divisible by heads {heads}"
    assert image_size % int(cfg.get("patch_size", 4)) == 0
    exit_weights = list(cfg["exit_loss_weights"])
    wall_budget = float(
        args.wall_budget_seconds
        if args.wall_budget_seconds is not None
        else cfg.get("main_wall_budget_seconds", 3600)
    )
    update_budget = float(
        args.update_budget_seconds
        if args.update_budget_seconds is not None
        else cfg.get("main_update_budget_seconds", 3000)
    )
    # From Stage C: N_floor ~13364 @ 3000s; use as soft step cap
    max_steps = int(args.max_steps) if args.max_steps is not None else 14000

    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"w{width}_s{image_size}_b{batch_size}"
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else repo / "exp_records" / f"stage_d_scaled_{tag}_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(exist_ok=True)
    curve_path = out_dir / "train_curve.jsonl"

    device = torch.device(args.device)
    env = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "platform": platform.platform(),
        "config_path": str(cfg_path),
    }
    print("=== Stage D main train ===")
    print(f"  out_dir={out_dir}")
    print(f"  config={cfg_path}")
    print(
        f"  width={width} heads={heads} size={image_size} batch={batch_size} "
        f"R={train_loops} lr={lr}"
    )
    print(
        f"  wall_budget={wall_budget}s update_budget={update_budget}s "
        f"max_steps={max_steps} warmup={args.warmup_steps}"
    )
    print(f"  env={json.dumps(env)}")

    with open(out_dir / "config_resolved.json", "w") as f:
        json.dump({"cfg": cfg, "env": env, "cli": vars(args)}, f, indent=2)

    torch.manual_seed(int(cfg.get("train_seed", 20261012)))
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = LoopWorldModel(
        image_size=image_size,
        patch_size=int(cfg.get("patch_size", 4)),
        width=width,
        heads=heads,
    ).to(device)
    nparams = model.count_parameters()
    print(f"  params={nparams:,}")
    opt = build_optimizer(model, lr=lr, weight_decay=float(cfg.get("weight_decay", 0.01)))
    ema = EMA(model, decay_cap=float(cfg.get("ema_decay_cap", 0.999)))
    train_seeds = split_seeds()["train"][:8192]

    start_step = 0
    if args.resume:
        rk = torch.load(args.resume, map_location="cpu", weights_only=False)
        rc = rk.get("config") or {}
        for key in ("image_size", "width", "heads", "patch_size", "train_loops"):
            if key in rc:
                assert int(rc[key]) == int(cfg[key]), f"resume config mismatch on {key}: {rc[key]} vs {cfg[key]}"
        # QK-norm adds attn.q_norm.weight / attn.k_norm.weight; older ckpts omit them.
        missing, unexpected = model.load_state_dict(rk["model"], strict=False)
        print(f"  resume model: missing={list(missing)} unexpected={list(unexpected)}")
        if unexpected:
            raise RuntimeError(f"unexpected keys loading resume ckpt: {unexpected}")
        for mk in missing:
            if "q_norm.weight" not in mk and "k_norm.weight" not in mk:
                raise RuntimeError(f"unexpected missing key (not QK-norm): {mk}")
        # EMA: copy matching keys; init missing (new QK-norm gains) from current model
        new_shadow = {}
        ema_src = rk["ema"]
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in ema_src:
                    new_shadow[k] = ema_src[k].to(device)
                else:
                    new_shadow[k] = v.detach().clone()
                    print(f"  EMA init from model for new key: {k}")
        ema.shadow = new_shadow
        ema.n = int(rk.get("ema_n", rk["step"]))
        if missing:
            # New params have no Adam state; rebuild optimizer (keeps loaded weights).
            opt = build_optimizer(model, lr=lr, weight_decay=float(cfg.get("weight_decay", 0.01)))
            print("  rebuilt AdamW (QK-norm params added; prior Adam moments dropped)")
        else:
            opt.load_state_dict(rk["optimizer"])
        torch.set_rng_state(rk["rng_torch"])
        if device.type == "cuda" and "rng_cuda" in rk:
            torch.cuda.set_rng_state_all(rk["rng_cuda"])
        start_step = int(rk["step"])
        print(f"  RESUMED from {args.resume} at step={start_step} ema_n={ema.n} (LR schedule continues on same horizon max_steps={max_steps})")
        del rk

    # Held-out PSNR probe: val seeds (not train/test), mode S (first-frame ctx), R=train_loops, 5 Euler steps, fixed noise
    from loopworld.cameras import make_K as _mk, plucker_rays as _pl
    from loopworld.data import sample_scene as _ss, sample_trajectory as _st
    from loopworld.encoding import encode_rgbd as _enc
    from loopworld.renderer import render_scene as _rs
    _K = _mk(image_size, device=device)
    _hj = [4, 8, 12, 16, 20, 24]
    _hc, _hp, _hv, _ht, _hg = [], [], [], [], []
    with torch.no_grad():
        for _sd in split_seeds()["val"][:4]:
            _sc = _ss(_sd); _c2 = _st(_sd).to(device)
            _r, _z, _m = _rs(_sc, _c2, _K, image_size)
            _e = _enc(_r, _z, _m); _pls = _pl(_c2, _K, image_size)
            for _j in _hj:
                c = torch.zeros(2, image_size, image_size, 4, device=device); c[0] = _e[0]
                pp = torch.zeros(2, image_size, image_size, 6, device=device); pp[0] = _pls[0]
                _hc.append(c); _hp.append(pp); _hv.append(torch.tensor([True, False], device=device))
                _ht.append(_pls[_j]); _hg.append(_r[_j])
    _hc, _hp, _hv, _ht, _hg = map(torch.stack, (_hc, _hp, _hv, _ht, _hg))
    _hn = torch.randn(_hc.shape[0], image_size, image_size, 4, device=device, generator=torch.Generator(device=device).manual_seed(777))
    heldout_path = out_dir / "heldout_curve.jsonl"
    heldout_log = []

    def heldout_psnr(m_, R_=None):
        R_ = R_ or train_loops
        m_.eval()
        with torch.no_grad():
            xs = [sample_view(m_, _hc[i:i+8], _hp[i:i+8], _hv[i:i+8], _ht[i:i+8], num_loops=R_, steps=5, noise=_hn[i:i+8].clone())
                  for i in range(0, _hc.shape[0], 8)]
            rgb_, _, _ = decode_rgbd(torch.cat(xs))
            mse = ((rgb_ - _hg) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
        m_.train()
        return float((10 * torch.log10(1 / mse)).mean())

    _ema_probe = LoopWorldModel(image_size=image_size, patch_size=int(cfg.get("patch_size", 4)), width=width, heads=heads).to(device)

    def log_heldout(step_):
        ema.copy_to(_ema_probe)
        rec = {"step": step_, "wall_s": time.perf_counter() - t_wall0 if "t_wall0" in dir() else None,
               "heldout_psnr_model": heldout_psnr(model), "heldout_psnr_ema": heldout_psnr(_ema_probe),
               "heldout_psnr_ema_R1": heldout_psnr(_ema_probe, 1)}
        heldout_log.append(rec)
        global BEST_CKPT_PATH
        if rec["heldout_psnr_ema"] >= max(r["heldout_psnr_ema"] for r in heldout_log):
            cp_ = ckpt_dir / f"ckpt_step{step_:06d}.pt"
            if cp_.exists():
                BEST_CKPT_PATH = str(cp_)
        with open(heldout_path, "a") as hf:
            hf.write(json.dumps(rec) + "\n")
        print(f"  [heldout] step={step_} psnr_model={rec['heldout_psnr_model']:.2f} psnr_ema_R4={rec['heldout_psnr_ema']:.2f} psnr_ema_R1={rec['heldout_psnr_ema_R1']:.2f}", flush=True)

    ckpt_minute_marks = [float(x) for x in args.ckpt_minutes.split(",") if x.strip()]
    ckpt_saved_marks = set()

    t_wall0 = time.perf_counter()
    update_seconds = 0.0
    step = start_step
    n_skips = 0
    consec_skips = 0
    saw_nan = False
    stop_reason = "max_steps"
    last_metrics = {}
    losses_window = []

    # Fixed tiny eval batch for curve (not for model selection)
    eval_batch_raw, eval_bd = make_online_batch(4, image_size, device, 0, train_seeds[:64])
    torch.manual_seed(4242)
    eval_t = torch.rand(4, device=device)
    eval_eps = torch.randn_like(eval_bd["target_rgbd"])

    def eval_loss(m):
        from loopworld.flow import sample_xt, velocity_target
        from loopworld.losses import exit_losses

        m.eval()
        with torch.no_grad():
            xt = sample_xt(eval_bd["target_rgbd"], eval_t, eval_eps)
            v = velocity_target(eval_bd["target_rgbd"], eval_eps)
            exits = m.forward_exits(
                xt,
                eval_t,
                eval_bd["target_plucker"],
                eval_bd["context_rgbd"],
                eval_bd["context_plucker"],
                eval_bd["context_valid"],
                num_loops=train_loops,
                return_all_exits=True,
            )
            L = exit_losses(exits, v, exit_weights)["L_total"]
        m.train()
        return float(L)

    L0 = eval_loss(model)
    log_heldout(step)
    print(f"  initial_eval_loss={L0:.6f}")

    with open(curve_path, "w") as curve_f:
        curve_f.write(
            json.dumps(
                {
                    "step": 0,
                    "wall_s": 0.0,
                    "update_s": 0.0,
                    "eval_loss": L0,
                    "phase": "init",
                }
            )
            + "\n"
        )
        curve_f.flush()

        while step < max_steps:
            wall_elapsed = time.perf_counter() - t_wall0
            if wall_elapsed >= wall_budget:
                stop_reason = "wall_budget"
                break
            if update_seconds >= update_budget:
                stop_reason = "update_budget"
                break

            if args.reset_schedule:
                cur_lr = lr_at_step(step - start_step, max_steps - start_step, lr, args.warmup_steps)
            else:
                cur_lr = lr_at_step(step, max_steps, lr, args.warmup_steps)
            set_lr(opt, cur_lr)

            t0 = time.perf_counter()
            _, bd = make_online_batch(batch_size, image_size, device, step + 1, train_seeds)
            do_logit = args.logit_every > 0 and ((step + 1) % args.logit_every == 0)
            model.record_attn_logit = do_logit
            metrics = train_step(
                model,
                bd,
                opt,
                train_loops=train_loops,
                exit_weights=exit_weights,
                grad_clip=grad_clip_v,
                use_bf16=True,
                skip_nonfinite=args.skip_nonfinite,
            )
            model.record_attn_logit = False
            if do_logit:
                metrics.update({f"logit_{k}": v for k, v in model.max_attn_logits().items()})
            if metrics.get("skipped", 0.0):
                n_skips += 1; consec_skips += 1
                print(f"  SKIP non-finite update at step={step+1} loss={metrics['L_total']} grad={metrics.get('grad_norm')} total_skips={n_skips} consec={consec_skips}", flush=True)
            else:
                consec_skips = 0
                ema.update(model)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            update_seconds += dt
            step += 1
            last_metrics = metrics
            losses_window.append(metrics["L_total"])
            if len(losses_window) > 100:
                losses_window.pop(0)

            gn = metrics.get("grad_norm", float("nan"))
            if args.skip_nonfinite and consec_skips <= args.max_consec_skips and not any(
                    not torch.isfinite(pp).all() for pp in list(model.parameters())[:1]):
                pass  # tolerated skip (or finite step)
            elif (not math.isfinite(metrics["L_total"])) or (not math.isfinite(gn)) or consec_skips > args.max_consec_skips:
                saw_nan = True
                stop_reason = "nan" if consec_skips <= args.max_consec_skips else "too_many_consecutive_skips"
                print(f"  NaN/Inf at step={step} metrics={metrics}")
                break

            wall_elapsed = time.perf_counter() - t_wall0
            rec = {
                "step": step,
                "wall_s": wall_elapsed,
                "update_s": update_seconds,
                "lr": cur_lr,
                "L_total": metrics["L_total"],
                "grad_norm": gn,
                "step_s": dt,
            }
            for lk in ("logit_pre", "logit_core0", "logit_core1", "logit_post"):
                if lk in metrics:
                    rec[lk] = metrics[lk]
            if args.ckpt_every > 0 and step % args.ckpt_every == 0:
                cpath = ckpt_dir / f"ckpt_step{step:06d}.pt"
                save_ckpt(cpath, model, ema, opt, step, vars(args), cfg,
                          extra={"wall_s": wall_elapsed, "update_s": update_seconds})
                print(f"  saved periodic ckpt {cpath}", flush=True)
                if args.ckpt_eval:
                    import subprocess
                    eo = out_dir / "ckpt_evals" / f"step{step:06d}"
                    eo.mkdir(parents=True, exist_ok=True)
                    subprocess.Popen([sys.executable, "-u", str(repo / "scripts" / "stage_e_eval.py"), "--ckpt", str(cpath),
                                      "--out", str(eo), "--Rs", "4", "--steps", "5", "--n-scenes", "8", "--no-video"],
                                     stdout=open(eo / "eval.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
                    print(f"  spawned ckpt eval -> {eo}", flush=True)
            if args.heldout_every > 0 and step % args.heldout_every == 0:
                log_heldout(step)
            if step % args.log_every == 0 or step == 1:
                el = eval_loss(model)
                rec["eval_loss"] = el
                rec["loss_ema100"] = sum(losses_window) / len(losses_window)
                logit_bits = ""
                for lk in ("logit_pre", "logit_core0", "logit_core1", "logit_post"):
                    if lk in metrics:
                        rec[lk] = metrics[lk]
                        logit_bits += f" {lk[6:]}={metrics[lk]:.1f}"
                print(
                    f"  step={step} wall={wall_elapsed:.0f}s upd={update_seconds:.0f}s "
                    f"loss={metrics['L_total']:.4f} eval={el:.4f} "
                    f"grad={gn:.3f} lr={cur_lr:.2e} step_ms={dt*1000:.1f}{logit_bits}"
                )
            curve_f.write(json.dumps(rec) + "\n")
            if step % args.log_every == 0:
                curve_f.flush()

            # Mid checkpoints by wall minutes
            for m_mark in ckpt_minute_marks:
                if m_mark in ckpt_saved_marks:
                    continue
                if wall_elapsed >= m_mark * 60.0:
                    path = ckpt_dir / f"ckpt_step{step:06d}_t{int(m_mark)}min.pt"
                    save_ckpt(
                        path,
                        model,
                        ema,
                        opt,
                        step,
                        vars(args),
                        cfg,
                        extra={"wall_s": wall_elapsed, "update_s": update_seconds},
                    )
                    ckpt_saved_marks.add(m_mark)
                    print(f"  saved mid ckpt {path}")

    wall_total = time.perf_counter() - t_wall0
    peak_alloc = (
        int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else 0
    )

    L_final = eval_loss(model)
    ema_model = LoopWorldModel(
        image_size=image_size,
        patch_size=int(cfg.get("patch_size", 4)),
        width=width,
        heads=heads,
    ).to(device)
    ema.copy_to(ema_model)
    L_ema = eval_loss(ema_model)

    final_ckpt = ckpt_dir / f"ckpt_final_step{step:06d}.pt"
    save_ckpt(
        final_ckpt,
        model,
        ema,
        opt,
        step,
        vars(args),
        cfg,
        extra={
            "wall_s": wall_total,
            "update_s": update_seconds,
            "stop_reason": stop_reason,
            "eval_loss": L_final,
            "eval_loss_ema": L_ema,
        },
    )
    # Convenience alias
    _alias = ckpt_dir / "ckpt_final.pt"
    if final_ckpt.exists():
        if _alias.exists():
            _alias.unlink()
        os.link(final_ckpt, _alias)  # hardlink alias: no extra disk
    else:
        print("  [diskguard] final ckpt was NOT saved (low disk)", flush=True)

    # Demo previews from EMA (fixed noise) — not Stage E matrix
    try:
        import numpy as np
        from PIL import Image

        torch.manual_seed(7)
        noise = torch.randn(4, image_size, image_size, 4, device=device)
        with torch.no_grad():
            x = sample_view(
                ema_model,
                eval_bd["context_rgbd"],
                eval_bd["context_plucker"],
                eval_bd["context_valid"],
                eval_bd["target_plucker"],
                num_loops=train_loops,
                steps=int(cfg.get("default_sampling_steps", 20)),
                noise=noise,
            )
            rgb, dinfo, oob = decode_rgbd(x)
        tiles = []
        for i in range(4):
            gt = (eval_batch_raw.target_rgb[i].cpu().clamp(0, 1).numpy() * 255).astype(
                "uint8"
            )
            pr = (rgb[i].cpu().clamp(0, 1).numpy() * 255).astype("uint8")
            tiles.append(np.concatenate([gt, pr], axis=0))
        grid = np.concatenate(tiles, axis=1)
        Image.fromarray(grid).resize(
            (grid.shape[1] * 4, grid.shape[0] * 4), Image.NEAREST
        ).save(preview_dir / "ema_demo_grid.png")
        print(f"  saved demo preview {preview_dir / 'ema_demo_grid.png'} oob={oob:.4f}")
        preview_ok = True
    except Exception as e:
        print(f"  preview failed: {e}")
        preview_ok = False
        oob = None

    mean_step = update_seconds / max(step - start_step, 1)
    summary = {
        "stage": "D",
        "role": "main",
        "passed": (not saw_nan) and step > 100,
        "stop_reason": stop_reason,
        "config_path": str(cfg_path),
        "image_size": image_size,
        "width": width,
        "heads": heads,
        "batch_size": batch_size,
        "train_loops": train_loops,
        "params": nparams,
        "steps": step,
        "wall_seconds": wall_total,
        "update_seconds": update_seconds,
        "wall_budget_seconds": wall_budget,
        "update_budget_seconds": update_budget,
        "mean_step_seconds": mean_step,
        "samples_seen": step * batch_size,
        "initial_eval_loss": L0,
        "final_eval_loss": L_final,
        "final_ema_eval_loss": L_ema,
        "drop_frac": (L0 - L_final) / max(L0, 1e-8),
        "drop_ema_frac": (L0 - L_ema) / max(L0, 1e-8),
        "saw_nan": saw_nan,
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_gib": peak_alloc / (1024**3),
        "peak_reserved_gib": peak_reserved / (1024**3),
        "last_train_metrics": {k: float(v) for k, v in last_metrics.items()},
        "checkpoints": {
            "final": str(final_ckpt),
            "final_alias": str(ckpt_dir / "ckpt_final.pt"),
            "mid_marks_min": sorted(ckpt_saved_marks),
        },
        "curve_path": str(curve_path),
        "preview_ok": preview_ok,
        "preview_rgb_oob": oob,
        "environment": env,
        "note": "Main training run (simple setting). Evaluate separately with stage_e_eval.py.",
        "resumed_from": args.resume or None,
        "start_step": start_step,
        "session_steps": step - start_step,
        "n_skipped_updates": n_skips,
        "lr_peak": lr,
        "grad_clip": grad_clip_v,
        "heldout_curve": heldout_log,
        "heldout_probe": "val seeds 200000-200003, mode S, j=4..24 step4, R=train_loops, 5 Euler steps, fixed noise",
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Stage D summary ===")
    print(f"  stop_reason={stop_reason} steps={step} wall={wall_total:.1f}s "
          f"update={update_seconds:.1f}s mean_step={mean_step*1000:.1f}ms")
    print(
        f"  eval L0={L0:.4f} L_final={L_final:.4f} L_ema={L_ema:.4f} "
        f"drop={(summary['drop_frac']*100):.1f}% drop_ema={(summary['drop_ema_frac']*100):.1f}%"
    )
    print(
        f"  peak_alloc={summary['peak_allocated_gib']:.2f}GiB "
        f"nan={saw_nan} ckpt={final_ckpt}"
    )
    print(f"  summary={out_dir / 'summary.json'}")
    print("=== Stage D: {} ===".format("DONE" if summary["passed"] else "FAILED"))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
