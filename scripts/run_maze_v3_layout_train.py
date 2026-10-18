#!/usr/bin/env python3
"""Layout-error round (v3): SF + hard-frame oversampling/weighting + confidence-masked memory + slower curriculum. New script.
Based on run_maze_v2_sf_train.py (unchanged).

Streaming rollout: B persistent episode slots. Each slot holds an episode, cursor c, a PointMemory2 built
from frames the model itself generated (mixed with GT per curriculum), and its last own prediction.
Every train step, for each slot, target t=c+1:
  context = [memory reprojected to camera t, prev = own pred of c (prob p_own) else GT c]
  1) denoising loss on GT target t given this self-generated context (grad)
  2) no-grad sample of frame t with the SAME context (current weights, S_roll steps, R_roll loops)
  3) prev <- own pred t (or GT per p_own); integrate own pred (or GT per p_own) into memory every `stride`
  4) c <- t; on episode end, reset slot to new maze at c=0 (GT frame 0).
Slots start at random cursors (memory warm-started from GT frames) so targets are de-correlated."""
import argparse, json, math, os, shutil, sys, time
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch
from loopworld.cameras import make_K
from loopworld.encoding import encode_rgbd, decode_rgbd
from loopworld.flow import euler_sample
from loopworld.maze_v2 import render_maze_multi
from loopworld.maze_v2_data import maze_plucker
from loopworld.geom_memory2 import PointMemory2
from loopworld.model_action import ActionLoopWorldModel
from loopworld.train import EMA, build_optimizer, train_step
sys.path.insert(0, os.path.dirname(__file__))
from run_maze_v2_mem_train import save_ckpt, lr_at, _init_worker, _mk_ep
from maze_v3_eval import run_v3_eval
from loopworld.layout_utils import hard_frames, confident_mask, weighted_train_step

T = 128


class Slot:
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="checkpoints/maze_sf.pt")
    p.add_argument("--out-dir", default=""); p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr-base", type=float, default=3e-5); p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--max-steps", type=int, default=30000); p.add_argument("--wall-hours", type=float, default=4.0)
    p.add_argument("--ckpt-every", type=int, default=3000); p.add_argument("--heldout-every", type=int, default=1000)
    p.add_argument("--heldout-scenes", type=int, default=8); p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--roll-steps", type=int, default=3); p.add_argument("--roll-loops", type=int, default=4)
    p.add_argument("--p-own-start", type=float, default=0.5); p.add_argument("--p-own-end", type=float, default=0.85)
    p.add_argument("--hard-weight", type=float, default=3.0); p.add_argument("--hard-start-prob", type=float, default=0.5)
    p.add_argument("--rgb-tol", type=float, default=0.08); p.add_argument("--depth-tol", type=float, default=0.05)
    p.add_argument("--curriculum-steps", type=int, default=12000)
    p.add_argument("--stride", type=int, default=4); p.add_argument("--splat", type=int, default=2)
    p.add_argument("--hole-fill", action="store_true"); p.add_argument("--max-points", type=int, default=300000)
    p.add_argument("--n-train-mazes", type=int, default=8192); p.add_argument("--profile", default="")
    p.add_argument("--seed", type=int, default=20261015)
    a = p.parse_args()
    dev = torch.device("cuda"); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    ck = torch.load(a.init, map_location="cpu", weights_only=False); cfg = dict(ck["config"])
    model = ActionLoopWorldModel(image_size=int(cfg["image_size"]), patch_size=int(cfg.get("patch_size", 4)),
                                 width=int(cfg["width"]), heads=int(cfg["heads"])).to(dev)
    model.load_state_dict(ck["ema"]); H = model.image_size; K = make_K(H, device=dev)
    exit_w = list(cfg["exit_loss_weights"]); R = int(cfg["train_loops"])
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("spawn").Pool(16, initializer=_init_worker) as pool:
        eps_ = pool.map(_mk_ep, list(range(100000, 100000 + a.n_train_mazes)), chunksize=64)
    for ep in eps_: ep.c2w = torch.from_numpy(ep.c2w).to(dev)
    print(f"  prebuilt {len(eps_)} episodes in {time.time()-t0:.0f}s", flush=True)
    mem_kw = dict(max_points=a.max_points, splat=a.splat, hole_fill=a.hole_fill)

    def new_slot(random_start):
        s = Slot(); s.ep = eps_[int(rng.integers(len(eps_)))]
        rgb, z, m = render_maze_multi([s.ep.scene] * T, s.ep.c2w, H)
        s.gr, s.gz, s.gm = rgb, z, m; s.genc = encode_rgbd(rgb, z, m); s.pl = maze_plucker(s.ep.c2w, K, H, True)
        s.hard = hard_frames(s.ep, z, m)
        hidx = np.nonzero(s.hard.numpy()[1:T - 8])[0]
        if random_start and len(hidx) and rng.random() < a.hard_start_prob:
            s.c = max(0, int(hidx[int(rng.integers(len(hidx)))]) - int(rng.integers(1, 6)))  # start just before a hard frame
        else:
            s.c = int(rng.integers(0, T - 8)) if random_start else 0
        s.mem = PointMemory2(**mem_kw)
        for i in sorted(set(list(range(0, s.c + 1, a.stride)) + [0, s.c])):
            s.mem.integrate(rgb[i], z[i], m[i], s.ep.c2w[i], K)
        s.prev = s.genc[s.c].clone()
        return s

    def p_own_at(step):
        f = min(step / max(a.curriculum_steps, 1), 1.0)
        return a.p_own_start + (a.p_own_end - a.p_own_start) * f

    tm = {"ctx": 0.0, "train": 0.0, "roll": 0.0, "integ": 0.0}

    def do_step(slots, opt, step, train=True):
        B = len(slots); pown = p_own_at(step)
        t_a = time.perf_counter()
        tg = [s.c + 1 for s in slots]
        reps = [s.mem.reproject(s.ep.c2w[t], K, H) for s, t in zip(slots, tg)]
        enc_m = torch.stack([encode_rgbd(*r) for r in reps]); mv = torch.stack([r[2].any() for r in reps])
        prev = torch.stack([s.prev for s in slots])
        batch = {"target_rgbd": torch.stack([s.genc[t] for s, t in zip(slots, tg)]),
                 "target_plucker": torch.stack([s.pl[t] for s, t in zip(slots, tg)]),
                 "context_rgbd": torch.stack([enc_m, prev], 1),
                 "context_plucker": torch.stack([torch.stack([s.pl[t], s.pl[t - 1]]) for s, t in zip(slots, tg)]),
                 "context_valid": torch.stack([mv, torch.ones_like(mv)], 1),
                 "action": torch.tensor([int(s.ep.actions[t]) for s, t in zip(slots, tg)], device=dev)}
        torch.cuda.synchronize(); t_b = time.perf_counter()
        model.cur_action = batch["action"]
        wts = torch.tensor([a.hard_weight if bool(s.hard[t]) else 1.0 for s, t in zip(slots, tg)], device=dev)
        mt = weighted_train_step(model, batch, opt, wts, R, exit_w, float(cfg.get("grad_clip_norm", 1.0))) if train else {}
        if train: mt["hard_frac"] = float((wts > 1).float().mean())
        torch.cuda.synchronize(); t_c = time.perf_counter()
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            fn = lambda x, t: model.forward_velocity(x, t, batch["target_plucker"], batch["context_rgbd"], batch["context_plucker"],
                                                     batch["context_valid"], num_loops=a.roll_loops)
            two = lambda key: batch[key].repeat(2, *([1] * (batch[key].ndim - 1)))
            model.cur_action = batch["action"].repeat(2)
            fn = lambda x, t: model.forward_velocity(x, t, two("target_plucker"), two("context_rgbd"), two("context_plucker"),
                                                     two("context_valid"), num_loops=a.roll_loops)
            xx = euler_sample(fn, torch.randn(2 * B, *batch["target_rgbd"].shape[1:], device=dev), steps=a.roll_steps).float().clamp(-1, 1)
            own, own2 = xx[:B], xx[B:]
        conf = confident_mask(own, own2, a.rgb_tol, a.depth_tol)
        model.train(); model.cur_action = None
        torch.cuda.synchronize(); t_d = time.perf_counter()
        pr, di, _ = decode_rgbd(own)
        for k, (s, t) in enumerate(zip(slots, tg)):
            use_own = rng.random() < pown
            s.prev = own[k] if use_own else s.genc[t]
            if t % a.stride == 0:
                if rng.random() < pown:
                    s.mem.integrate(pr[k], di["z_metric"][k], conf[k], s.ep.c2w[t], K)
                else:
                    s.mem.integrate(s.gr[t], s.gz[t], s.gm[t], s.ep.c2w[t], K)
            s.c = t
            if s.c >= T - 1: slots[k] = new_slot(False)
        torch.cuda.synchronize(); t_e = time.perf_counter()
        for kk, dt in zip(("ctx", "train", "roll", "integ"), (t_b - t_a, t_c - t_b, t_d - t_c, t_e - t_d)): tm[kk] += dt
        if train: mt["conf_frac"] = float(conf.float().mean())
        return mt, pown

    if a.profile:
        for B in map(int, a.profile.split(",")):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); opt = build_optimizer(model, lr=1e-6)
            for k in tm: tm[k] = 0.0
            try:
                slots = [new_slot(True) for _ in range(B)]
                for i in range(25):
                    if i == 5:
                        for k in tm: tm[k] = 0.0
                        t0 = time.perf_counter()
                    do_step(slots, opt, 0)
                torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 20
                r = dict(B=B, peak_reserved_GiB=round(torch.cuda.max_memory_reserved() / 2**30, 2), step_ms=round(dt * 1e3, 1),
                         samples_per_s=round(B / dt, 1), share={k: round(v / 20 / dt, 3) for k, v in tm.items()})
            except torch.cuda.OutOfMemoryError:
                r = dict(B=B, oom=True)
            del opt; slots = None; model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            print("PROFILE", json.dumps(r), flush=True)
        return

    B = a.batch_size; lr = a.lr_base * math.sqrt(B / 16)
    out = Path(a.out_dir or f"exp_records/maze_v3_layout_b{B}_{time.strftime('%Y%m%d_%H%M%S')}"); (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg.update(dict(layout_round_v3=True, hard_weight=a.hard_weight, hard_start_prob=a.hard_start_prob, conf_tol=[a.rgb_tol, a.depth_tol], self_forcing=True, roll_steps=a.roll_steps, roll_loops=a.roll_loops, p_own=[a.p_own_start, a.p_own_end, a.curriculum_steps],
                    stride=a.stride, memory="PointMemory2", mem_kw=mem_kw, batch_size=B, lr_peak=lr, init=a.init))
    json.dump({"cfg": cfg, "cli": vars(a), "cvd": os.environ.get("CUDA_VISIBLE_DEVICES")}, open(out / "config_resolved.json", "w"), indent=2)
    opt = build_optimizer(model, lr=lr, weight_decay=float(cfg.get("weight_decay", 0.01))); ema = EMA(model, 0.999)
    probe = ActionLoopWorldModel(image_size=H, patch_size=int(cfg.get("patch_size", 4)), width=int(cfg["width"]), heads=int(cfg["heads"])).to(dev)
    curve = open(out / "train_curve.jsonl", "a"); hcurve = open(out / "heldout_curve.jsonl", "a")
    best = [-1e9, None]; t_start = time.time(); win = []; tlast = time.time()
    print(f"=== maze v3 LAYOUT round === out={out} B={B} lr={lr:.2e} roll S{a.roll_steps} R{a.roll_loops} mem={mem_kw} stride={a.stride}", flush=True)

    def heldout(s):
        ema.copy_to(probe); t0 = time.time()
        r = run_v3_eval(probe, "own_conf", a.heldout_scenes, mem_kw=mem_kw, stride=a.stride, rgb_tol=a.rgb_tol, depth_tol=a.depth_tol)
        rec = {"step": s, "sec": time.time() - t0, **r}; hcurve.write(json.dumps(rec) + "\n"); hcurve.flush()
        rv = r["revisit"]
        print(f"  [heldout] step={s} AR psnr={r['psnr_all']:.2f} seg={ {k: round(v, 2) for k, v in r['psnr_seg'].items()} } absrel={r['absrel_all']:.3f} "
              f"revisit gt={rv['psnr_pred_vs_gt_at_revisit']:.2f} own={rv['psnr_pred_vs_own_first']:.2f} cov={r['mem_cov']:.2f} hint={r['hint_psnr_revisit']:.2f} absrel_med={r['depth']['absrel_median']:.4f} d1={r['depth']['delta1']:.3f} iou_wall={r['depth']['iou_wall']:.3f} ({rec['sec']:.0f}s)", flush=True)
        if r["psnr_all"] > best[0] and save_ckpt(out / "checkpoints" / "ckpt_best.pt", probe, ema, opt, s, a, cfg):
            best[0], best[1] = r["psnr_all"], out / "checkpoints" / "ckpt_best.pt"

    heldout(0)
    slots = [new_slot(True) for _ in range(B)]; step = 0; stop = "max_steps"
    while step < a.max_steps:
        if time.time() - t_start > a.wall_hours * 3600: stop = "wall"; break
        cur = lr_at(step, a.max_steps, lr, a.warmup)
        for g in opt.param_groups: g["lr"] = cur
        mt, pown = do_step(slots, opt, step); step += 1
        if mt["skipped"]:
            print(f"  SKIP nonfinite step {step}", flush=True)
            if not all(torch.isfinite(q).all() for q in model.parameters()): stop = "nan"; break
        else:
            ema.update(model)
        win = (win + [mt["L_total"]])[-100:]
        if step % a.log_every == 0:
            now = time.time(); sps = a.log_every / (now - tlast); tlast = now; tot = sum(tm.values())
            sh = {k: round(v / max(tot, 1e-9), 3) for k, v in tm.items()}
            for k in tm: tm[k] = 0.0
            rec = dict(step=step, wall_s=now - t_start, lr=cur, p_own=pown, L=mt["L_total"], L100=float(np.mean(win)), grad=mt["grad_norm"],
                       it_s=sps, mem_GiB=torch.cuda.max_memory_reserved() / 2**30, share=sh)
            curve.write(json.dumps(rec) + "\n"); curve.flush()
            print(f"step {step} L={mt['L_total']:.4f} L100={rec['L100']:.4f} grad={mt['grad_norm']:.3f} lr={cur:.2e} p_own={pown:.2f} hard={mt.get('hard_frac',0):.2f} conf={mt.get('conf_frac',0):.2f} it/s={sps:.2f} mem={rec['mem_GiB']:.1f}G share={sh}", flush=True)
        if step % a.ckpt_every == 0:
            save_ckpt(out / "checkpoints" / f"ckpt_step{step:06d}.pt", model, ema, opt, step, a, cfg, keep=best[1]); print(f"  saved ckpt {step}", flush=True)
        if step % a.heldout_every == 0: heldout(step)
    save_ckpt(out / "checkpoints" / f"ckpt_final_step{step:06d}.pt", model, ema, opt, step, a, cfg, keep=best[1]); heldout(step)
    json.dump(dict(stop=stop, step=step, wall_s=time.time() - t_start, best_psnr=best[0]), open(out / "summary.json", "w"), indent=1)
    print(f"=== DONE stop={stop} step={step} ===", flush=True)

if __name__ == "__main__":
    main()
