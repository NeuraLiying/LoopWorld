#!/usr/bin/env python3
"""Maze v2 action-conditioned training (new script; Stage D script untouched).
Init from v1 ckpt (EMA weights) with zero-init action embedding. QK-Norm + fp32 SDPA (model.py),
bf16 autocast, EMA, diskguard, atomic ckpts, periodic ckpt, held-out mini AR eval.
--profile B1,B2,..: measure peak VRAM + samples/s per batch size and exit."""
import argparse, json, math, os, shutil, sys, time
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch
from loopworld.model_action import ActionLoopWorldModel
from loopworld.maze_v2_data import make_maze_batch
from loopworld.train import EMA, build_optimizer, train_step
sys.path.insert(0, os.path.dirname(__file__))
from maze_v2_eval import run_ar_eval, write_video

MIN_FREE = 5 * 1024**3; CKPT_EST = 800 * 1024**2


def ensure_space(d: Path, keep=None):
    while shutil.disk_usage(d).free < MIN_FREE + CKPT_EST:
        c = sorted(d.glob("ckpt_step*.pt"), key=lambda q: q.stat().st_mtime)[:-1]
        c = [x for x in c if keep is None or x.resolve() != Path(keep).resolve()]
        if not c:
            print("  [diskguard] low disk, nothing deletable -> SKIP save", flush=True); return False
        print(f"  [diskguard] deleting {c[0]}", flush=True); c[0].unlink()
    return True


def save_ckpt(path, model, ema, opt, step, args, cfg, keep=None, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not ensure_space(path.parent, keep): return False
    pl = {"model": model.state_dict(), "ema": ema.shadow, "optimizer": opt.state_dict(), "step": step, "ema_n": ema.n,
          "args": vars(args), "config": cfg, "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()}
    if extra: pl.update(extra)
    tmp = path.with_suffix(".pt.tmp"); torch.save(pl, tmp); os.replace(tmp, path); return True


def lr_at(step, total, base, warm, floor=0.1):
    if step < warm: return base * (step + 1) / warm
    t = min(max((step - warm) / max(total - warm, 1), 0), 1)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))


def _init_worker():
    torch.set_num_threads(1)


def _mk_ep(sd):
    from loopworld.maze_v2_data import MazeEpisode
    ep = MazeEpisode(sd, 128, "cpu"); ep.c2w = ep.c2w.numpy()  # numpy: avoid torch fd sharing
    return ep


def build(init, device):
    ck = torch.load(init, map_location="cpu", weights_only=False); cfg = dict(ck["config"])
    m = ActionLoopWorldModel(image_size=int(cfg["image_size"]), patch_size=int(cfg.get("patch_size", 4)),
                             width=int(cfg["width"]), heads=int(cfg["heads"])).to(device)
    missing, unexpected = m.load_state_dict(ck["ema"], strict=False)
    assert all(k.startswith("action_") for k in missing) and not unexpected, (missing, unexpected)
    return m, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="exp_records/stage_d_qk_norm_cont4h2_w768_s96_b16_20261014_003006/checkpoints/ckpt_final_step083864.pt")
    p.add_argument("--resume", default="")
    p.add_argument("--out-dir", default=""); p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr-base", type=float, default=1e-4, help="peak LR at batch 16; scaled by sqrt(B/16)")
    p.add_argument("--warmup", type=int, default=500); p.add_argument("--max-steps", type=int, default=60000)
    p.add_argument("--wall-hours", type=float, default=4.0); p.add_argument("--ckpt-every", type=int, default=4000)
    p.add_argument("--heldout-every", type=int, default=1000); p.add_argument("--heldout-scenes", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50); p.add_argument("--no-pose", action="store_true")
    p.add_argument("--p-revisit", type=float, default=0.3); p.add_argument("--n-train-mazes", type=int, default=8192)
    p.add_argument("--profile", default=""); p.add_argument("--seed", type=int, default=20261014)
    a = p.parse_args()
    dev = torch.device("cuda"); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    model, cfg = build(a.init, dev); H = model.image_size
    exit_w = list(cfg["exit_loss_weights"]); R = int(cfg["train_loops"])
    train_seeds = np.arange(100000, 100000 + a.n_train_mazes)
    t0 = time.time()
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(16, initializer=_init_worker) as pool:
        eps_ = pool.map(_mk_ep, [int(s) for s in train_seeds], chunksize=64)
    cache = {}
    for sd, ep in zip(train_seeds, eps_):
        ep.c2w = torch.from_numpy(ep.c2w).to(dev); cache[sd] = ep
    print(f"  prebuilt {len(cache)} maze episodes in {time.time()-t0:.0f}s", flush=True)
    mk = lambda B: make_maze_batch(rng.choice(train_seeds, B), H, dev, rng, p_revisit=a.p_revisit, use_pose=not a.no_pose, ep_cache=cache)

    if a.profile:
        res = []
        for B in map(int, a.profile.split(",")):
            opt = build_optimizer(model, lr=1e-5); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                ts = []
                for i in range(25):
                    t0 = time.perf_counter(); b = mk(B); t1 = time.perf_counter(); model.cur_action = b["action"]
                    train_step(model, b, opt, train_loops=R, exit_weights=exit_w, grad_clip=1.0, use_bf16=True)
                    torch.cuda.synchronize(); ts.append((time.perf_counter() - t0, t1 - t0))
                tt = np.array(ts[5:]); pk = torch.cuda.max_memory_reserved() / 2**30
                r = dict(B=B, peak_reserved_GiB=round(pk, 2), peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                         step_ms=round(tt[:, 0].mean() * 1e3, 1), data_ms=round(tt[:, 1].mean() * 1e3, 1),
                         samples_per_s=round(B / tt[:, 0].mean(), 1))
            except torch.cuda.OutOfMemoryError:
                r = dict(B=B, oom=True)
            del opt; model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            print("PROFILE", json.dumps(r), flush=True); res.append(r)
        return

    B = a.batch_size; lr = a.lr_base * math.sqrt(B / 16)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(a.out_dir or f"exp_records/maze_v2_action_b{B}_{stamp}"); (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg.update(dict(maze_v2=True, action_conditioning="adaln_add_to_t", use_pose=not a.no_pose, batch_size=B, lr_peak=lr,
                    plucker_moment_scale=0.6, context="[earlier(j-k,k~U[2,32]; p_revisit first-visit), prev]"))
    json.dump({"cfg": cfg, "cli": vars(a), "gpu": torch.cuda.get_device_name(0), "cvd": os.environ.get("CUDA_VISIBLE_DEVICES")},
              open(out / "config_resolved.json", "w"), indent=2)
    opt = build_optimizer(model, lr=lr, weight_decay=float(cfg.get("weight_decay", 0.01))); ema = EMA(model, 0.999)
    step = 0
    if a.resume:
        rk = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(rk["model"]); opt.load_state_dict(rk["optimizer"]); ema.shadow = {k: v.to(dev) for k, v in rk["ema"].items()}
        ema.n = rk["ema_n"]; step = rk["step"]
    probe = ActionLoopWorldModel(image_size=H, patch_size=int(cfg.get("patch_size", 4)), width=int(cfg["width"]), heads=int(cfg["heads"])).to(dev)
    curve = open(out / "train_curve.jsonl", "a"); hcurve = open(out / "heldout_curve.jsonl", "a")
    best = (-1e9, None); t_start = time.time(); win = []; tlast = time.time()
    print(f"=== maze v2 action train === out={out} B={B} lr={lr:.2e} R={R} use_pose={not a.no_pose} steps={a.max_steps} wall={a.wall_hours}h", flush=True)

    def heldout(s):
        nonlocal best
        ema.copy_to(probe); t0 = time.time()
        r, _ = run_ar_eval(probe, a.heldout_scenes, R=4, steps=5, use_pose=not a.no_pose)
        rec = {"step": s, "sec": time.time() - t0, **{k: r[k] for k in ("psnr_all", "psnr_seg", "absrel_all", "revisit")}}
        hcurve.write(json.dumps(rec) + "\n"); hcurve.flush()
        rv = r["revisit"]
        print(f"  [heldout] step={s} AR psnr={r['psnr_all']:.2f} seg={ {k: round(v, 2) for k, v in r['psnr_seg'].items()} } "
              f"absrel={r['absrel_all']:.3f} revisit self={rv['psnr_pred_vs_own_first']:.2f} gt={rv['psnr_pred_vs_gt_at_revisit']:.2f} ({rec['sec']:.0f}s)", flush=True)
        if r["psnr_all"] > best[0]:
            path = out / "checkpoints" / "ckpt_best.pt"
            if save_ckpt(path, probe, ema, opt, s, a, cfg): best = (r["psnr_all"], path)
    if step == 0: heldout(0)
    stop = "max_steps"
    while step < a.max_steps:
        if time.time() - t_start > a.wall_hours * 3600: stop = "wall"; break
        cur = lr_at(step, a.max_steps, lr, a.warmup)
        for g in opt.param_groups: g["lr"] = cur
        b = mk(B); model.cur_action = b["action"]
        mt = train_step(model, b, opt, train_loops=R, exit_weights=exit_w, grad_clip=float(cfg.get("grad_clip_norm", 1.0)),
                        use_bf16=True, skip_nonfinite=True)
        step += 1
        if mt["skipped"]:
            print(f"  SKIP nonfinite step {step}", flush=True)
            if not all(torch.isfinite(q).all() for q in model.parameters()): stop = "nan"; break
        else:
            ema.update(model)
        win.append(mt["L_total"]); win = win[-100:]
        if step % a.log_every == 0:
            now = time.time(); sps = a.log_every / (now - tlast); tlast = now
            rec = dict(step=step, wall_s=now - t_start, lr=cur, L=mt["L_total"], L_ema100=float(np.mean(win)), grad=mt["grad_norm"],
                       it_s=sps, mem_GiB=torch.cuda.max_memory_reserved() / 2**30)
            curve.write(json.dumps(rec) + "\n"); curve.flush()
            print(f"step {step} L={mt['L_total']:.4f} L100={rec['L_ema100']:.4f} grad={mt['grad_norm']:.3f} lr={cur:.2e} it/s={sps:.2f} mem={rec['mem_GiB']:.1f}G", flush=True)
        if step % a.ckpt_every == 0:
            save_ckpt(out / "checkpoints" / f"ckpt_step{step:06d}.pt", model, ema, opt, step, a, cfg, keep=best[1]); print(f"  saved ckpt step {step}", flush=True)
        if step % a.heldout_every == 0: heldout(step)
    save_ckpt(out / "checkpoints" / f"ckpt_final_step{step:06d}.pt", model, ema, opt, step, a, cfg, keep=best[1])
    heldout(step)
    try:
        ema.copy_to(probe); _, vid = run_ar_eval(probe, 3, R=4, steps=5, use_pose=not a.no_pose, keep_video=3)
        write_video(vid, out / "final_ar_rollout_R4_S5_h264.mp4")
    except Exception as e:
        print("video failed", e, flush=True)
    json.dump(dict(stop=stop, step=step, wall_s=time.time() - t_start, best_psnr=best[0]), open(out / "summary.json", "w"), indent=1)
    print(f"=== DONE stop={stop} step={step} ===", flush=True)

if __name__ == "__main__":
    main()
