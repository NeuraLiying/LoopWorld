# QK-Norm + fp32 attention (2026-10-13)

Approved fix from `instability_diagnosis_20261013.md`: **F1 + F3 only** (no adaLN weight decay, no β2 change).

## Code changes

### `loopworld/model.py`
- Replaced `nn.MultiheadAttention` with explicit `Attention`:
  - Packed QKV via `in_proj_weight` / `in_proj_bias` (same Parameter names as MHA)
  - `out_proj` Linear (same names)
  - Per-head `RMSNorm` on q and k (`q_norm.weight`, `k_norm.weight`, init ones) — LightningDiT/SD3 style
  - `F.scaled_dot_product_attention` in **fp32** (q/k/v cast to float32, result cast back)
- Padding: float additive mask with `-inf` (bool SDPA mask on this torch 2.4 build did not match MHA and produced NaNs)
- Optional `model.record_attn_logit` + chunked max `|logit|` logging (pre/core0/core1/post)
- AdaLN-Zero, loop structure, exits, head unchanged

### `scripts/run_stage_d_main.py`
- `--logit-every N`: record/log max attention logits every N steps
- Resume with `strict=False`:
  - Expected missing keys: `{pre,core.0,core.1,post}.attn.{q,k}_norm.weight` (8 keys)
  - Unexpected keys → hard error
  - EMA: copy matching keys; init missing QK-norm gains from the (freshly inited) model weights (ones)
  - If any keys missing → **rebuild AdamW** (new params have no moments; prior Adam state dropped). Documented in train.log

Backups: `loopworld/model.py.bak_pre_qknorm_20261013`, `scripts/run_stage_d_main.py.bak_pre_qknorm_20261013`

## Checkpoint load notes (from `ckpt_final_step014000.pt`)
- Loaded: all prior weights including `attn.in_proj_*` and `out_proj.*`
- Missing (inited to ones): 8× `*.attn.q_norm.weight` / `*.attn.k_norm.weight`
- Smoke @14k+QK: max |logit| ≈ 7–8 on all blocks (was ~83 pre without QK-norm); train_step finite

## Validation run
- Tag: `stage_d_qk_norm_val3k_*`
- Resume: `ckpt_final_step014000.pt`, GPU5, batch 16
- Schedule: reset, warmup 200 → peak 1e-4, cosine floor 0.5× (=5e-5), max_steps 17000 (~3k steps)
- Stability: grad clip 0.5, skip-nonfinite, logit-every 100, heldout-every 1000
- Pass: max logit ≲ 200; p99 grad < 1; held-out EMA R4 ≳ 24.5 at 16k–17k

## Validation results (`stage_d_qk_norm_val3k_w768_s96_b16_20261013_160835`)

- PID 1134967; GPU5; stopped at step **16980** (`update_budget`); wall 1516s; mean step 503.5 ms; peak alloc 19.05 GiB; skips=0
- Schedule: warmup 200 → 1e-4, cosine floor 5e-5; clip 0.5; Adam rebuilt on resume (QK-norm keys new)

### Pass criteria

| criterion | result | value |
|---|---|---|
| max \|logit\| ≲ 200 | **PASS** | max 9.06 (med ~8.0); flat 7.5–9.1 over whole run |
| p99 grad < 1 | **PASS** | med 0.068, p99 0.221, max 0.476, frac>1 = 0 |
| held-out EMA R4 ≳ 24.5 at 16–17k | **PASS at ~17k** | 16k: 24.26; post-hoc 16980: **24.54** |

### Held-out EMA R4 trajectory (QK-norm insert drops quality until adapted)

| step | EMA R4 | model R4 |
|---|---|---|
| 14000 (just loaded) | 20.89 | 20.90 |
| 15000 | 23.28 | 23.17 |
| 16000 | 24.26 | 24.07 |
| 16980 | 24.54 | 24.36 |

Clean 14k baseline without QK-norm was EMA R4 **24.63**. Load-time drop is expected (RMSNorm changes Q/K geometry); recovered within ~3k steps.

### vs baseline unstable 2h run (same 14k→17k window, no QK-norm)

| | QK-norm val | baseline 2h (diverged later) |
|---|---|---|
| med grad | 0.068 | 0.077 |
| p99 grad | 0.22 | 0.40 |
| max grad | 0.48 | 4.47 |
| frac grad>1 | 0% | 0.2% |
| pre logit @16k | ~8 | ~11,136 (probe) |
| held-out @16k | 24.26 | 24.49 then collapse after 20k |

### Notes / follow-ups
- fp32 SDPA + logit logging raised wall step ~0.50s (was ~0.44); VRAM ~19 GiB (was ~15.7). Logit-every can be raised to 500 for long runs.
- Bool SDPA mask was wrong on torch 2.4 here; float `-inf` mask is required (documented in model.py).
- Recommended next (needs approval): longer continue from `ckpt_final_step016980.pt` or from 14k with QK-norm, e.g. 2h+, to confirm no late collapse past 20k.
