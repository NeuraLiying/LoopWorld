# LoopWorld R1 — Stage D-scaled instability diagnosis (2026-10-13)

Scope: read-only analysis and offline probes on GPU5. No training code was changed and no training was launched.
Runs analysed:
- `stage_d_scaled_2h_lr1e-4_w768_s96_b16_20261013_110523` (the "2h rerun"). Stopped with SIGINT at about 12:29 Asia/Shanghai, after step ~25000. Checkpoints kept: 16000, 20000, 24000. The 14000 source checkpoint is untouched. Because it was interrupted, no `ckpt_final` was written.
- `stage_d_scaled_long8h_w768_s96_b16_20261013_005651` (the 8h run at LR 3e-4; inf at step 22359).

Probe scripts: `/tmp/lw_probe_instability.py` and `/tmp/lw_probe2.py` on mars. Raw output: `/tmp/lw/probe_out.json`.
Each probe uses 3 fixed online train batches (B=8) and runs forward/backward in both bf16-autocast and fp32. It records per-block statistics for every application (pre, 4 loops × 2 shared core blocks, post): residual RMS and absmax, attention/MLP update size, adaLN scale/gate magnitudes, max |q| and |k| per head, max attention logit, and dL/dh. It also records weight norms, top singular values of Wq/Wk, and AdamW state.

## 1. Timeline from the logs (2h rerun; peak LR 1e-4 at step 15000)

Each row is a 1000-step window.

| window | median grad | p99 grad | share of steps with grad > 1 | median loss | held-out EMA R4 |
|---|---|---|---|---|---|
| 14k | 0.065 | 0.23 | 0% | 0.0076 | 24.63 |
| 15k | 0.082 | 0.25 | 0% | 0.0089 | 24.68 |
| 16k | 0.082 | 0.72 | 0.5% | 0.0095 | 24.49 |
| 17k | 0.167 | 6.7 | 11% | 0.0160 | 24.52 |
| 18k | 0.150 | 58 | 17% | 0.0152 | 24.38 |
| 19k | 1.17 | 188 | 52% | 0.039 | 24.60 |
| 20k | 53 | 1.6e6 | 87% | 0.126 | 24.47 → 23.67 (21k) |
| 21k–25k | 90–560 | up to 8e6 | ~90–95% | 0.11–0.17 | 23.4 / 23.1 / 22.87 / 22.98 |

- Thresholds were first crossed as follows (lr is the learning rate at that step):
  - grad > 0.5 at step 15786 (lr 9.9e-5)
  - grad > 1 at step 16666
  - grad > 10 at step 17297
  - grad > 100 at step 18455
  - grad > 1e4 at step 20254 (lr 7.6e-5)
- **The growth is gradual and self-reinforcing.** It started about 800 steps after reaching peak LR and continued while the LR was already decaying. It is not one bad batch. Batches are generated online, so data corruption can be ruled out.
- 0 SKIP events: the gradients were huge but always finite, so the skip logic never triggered.
- The 8h run shows the same pattern, earlier and harder: grad 2.9 at step 14312 during the warmup to 3e-4, intermittent >10 spikes from step 19169, then 1.6e6 and inf at 22348–22359.
- Per-term losses: the logs only record L_total. The probes show that once the model breaks, all 4 exits and both RGB and depth degrade together (L_k ≈ 0.28–0.30 at 24k vs ≈0.009 at 14k). Neither the depth term nor the exit weighting is the trigger.

## 2. Checkpoint probe evidence

### 2.1 The first thing to blow up is attention logits in the non-shared pre block

Max |pre-softmax attention logit|, model weights, bf16 forward, batch 0:

| ckpt | pre | L1.core0 | L1.core1 | post | loss | grad norm |
|---|---|---|---|---|---|---|
| 14000 | 83 | 48 | 121 | 78 | 0.0099 | 0.19 |
| 16000 | **11,136** | 50 | 120 | 75 | 0.0123 | 0.13 |
| 20000 | **1.7e7** | 12,096 | 180,224 | 169 | 0.207 | 115 |
| 24000 | **5.4e7** | 20,864 | 3.0e6 | 1,016 | 0.288 | 544 / 4,722 |

- At 16000 the loss and gradient still look healthy, but the pre-block logits are already ~130× larger. The softmax in that block is effectively one-hot (attention entropy collapse). This is the leading indicator. It sits about 4k steps ahead of the grad explosion and about 5k steps ahead of the held-out PSNR drop.
- Next, the collapse spreads to the shared core block 1 (logits 1.8e5 at 20k, 3e6 at 24k), then to core 0.
- Max |q| per head in pre: 27 → 302 → 12,119 → 21,507.

### 2.2 Mechanism: Wq/Wk spectral growth plus adaLN scale growth

Top singular values of Wq/Wk, from `in_proj_weight`:

| ckpt | pre σ(Wq)/σ(Wk) | core.0 | core.1 | post |
|---|---|---|---|---|
| 14000 | 4.6 / 5.7 | 4.7 / 4.9 | 5.3 / 5.2 | 4.3 / 4.4 |
| 16000 | **11.3 / 11.3** | 4.8 / 5.1 | 5.3 / 5.2 | 4.3 / 4.5 |
| 20000 | **21.5 / 21.1** | 5.8 / 6.6 | 8.2 / 9.3 | 4.5 / 4.9 |
| 24000 | 21.7 / 21.2 | 6.0 / 6.8 | 9.9 / 10.9 | 4.5 / 5.1 |

- The Frobenius norm of in_proj barely moved (pre 36 → 40 → 47). The growth is **rank-1 / low-rank in the q and k directions**, which is the classic attention-logit-growth failure. The q/k biases stay small (|bq| 0.5–0.9), so they are not the cause.
- Logits scale roughly as σq·σk·(1+scale)². The adaLN modulation `scale` that multiplies norm1(x) before the QKV projection also grew:

  | ckpt | pre | core.1 | post |
  |---|---|---|---|
  | 14000 | 1.9 | 2.6 | 3.1 |
  | 16000 | 2.3 | | |
  | 20000 | 50 | 20 | 20 |
  | 24000 | 91 | 46 | 47 |

  The adaLN biases stay ~0.05, so the growth comes from adaLN.weight × t_emb. In `build_optimizer`, every parameter whose name contains "adaLN" is in the **no-weight-decay** group, so nothing opposes this growth. pre.adaLN Frobenius norm: 21.4 → 24.9 → 35 → 36.5.
- The LayerNorm before attention has `elementwise_affine=False`, and there is **no normalisation of q/k**. LayerNorm bounds the input to attention, but nothing bounds Wq/Wk or (1+scale).

### 2.3 Residual growth across loops: a consequence, not the trigger

- At 14000 the residual RMS is about 0.76–0.97 and grows slowly and monotonically across the 8 shared core applications. dL/dh shrinks backward through the loops (0.0034 at pre, 0.0007 at L4); there is no gradient amplification through the recurrence.
- At 24000 the residual RMS is 90–180, the post block MLP update is about 960 RMS, and absmax reaches 41k. This happens only after the attention collapse, so loop residual growth is secondary amplification.

### 2.4 bf16 vs fp32

- Up to 16000, bf16 and fp32 losses and grads match to within about 2%.
- At 20000 and 24000, bf16 grad norms are 5–100× the fp32 values on the same batch (for example 114.8 vs 17.2, and 4,722 vs 3.6). The forward losses are identical.
- So bf16 precision in the softmax and attention backward makes the spikes much larger once logits are around 1e4 or more. It does not cause the collapse, because fp32 logits are equally saturated. Fp32 attention would reduce the symptom, not the cause.

### 2.5 Gradient location

- Healthy (14000): the largest grads are in patch_embed (0.18) and head.proj.
- Broken (24000, bf16): the largest grads are in t_embed.mlp.0 (406), core.1.attn.in_proj (208), pre.adaLN (169) and pre.attn.in_proj (129). These are exactly the time-conditioning → modulation → QK path.

### 2.6 Optimizer, clipping, skip, EMA checks

- **Clipping:** bf16 autocast has no GradScaler, so there is no unscale ordering issue. `clip_grad_norm_` runs after backward and before `optimizer.step()`, which is correct.
  - But with Adam, clipping a 1e4 grad to 0.5 does not bound the step size per parameter. The update is about lr·m/√v.
  - √v is tiny: median about 1e-6 to 1e-5 because the healthy grads are about 0.05.
  - The maximum |m|/√v ratio grows from 2.5 at 14k to about 3.6–4 at 20k/24k.
  - So each spike still moves the q/k and adaLN weights by a full LR-sized step in a consistent direction.
- **AdamW settings:** betas (0.9, 0.999), eps 1e-8, wd 0.01. No weight decay on norm, 1-D, bias and **all adaLN** parameters. Decay does apply to pos/slot embeddings, which is harmless.
  - With beta2 = 0.999, √v takes about 1000 steps to adapt to the new gradient scale, so the step-to-step response lags the spikes.
- **Skip logic:** it works as designed. It never fired because nothing went non-finite in this run; inf only appeared in the 8h run.
- **EMA (cap 0.999, about 1000-step horizon):** it delayed but did not prevent contamination. EMA R4 held up to 20k (24.47), then followed the model down (22.87 at 24k). The EMA of the 20k checkpoint already has pre logits of 4e6. There is no clean EMA after about 19k.
- **Learning rate:** both runs diverged, at peak LR 3e-4 and 1e-4. Lower LR delayed onset but did not prevent it. The original 0→14k training at 3e-4 was stable, but the instability is cumulative: the q/k spectra keep growing as training continues.

## 3. Root cause ranking

1. **Most likely: attention-logit growth and entropy collapse with no QK normalisation, starting in the pre block and spreading to the shared core.** Evidence: logits rise 83 → 11k while loss is still normal, Wq/Wk spectral norm rises 4.6 → 21, the logit growth comes first in time, and the largest grads sit on the attention path.
2. **Strong contributing factor: unbounded adaLN modulation scale, with no weight decay on adaLN and adaLN driven by t_emb.** Scale grows from ~2 to ~90, and (1+scale)² multiplies the QK logits directly. The broken-state gradients concentrate in t_embed and adaLN.
3. **Amplifier: bf16 softmax/attention backward.** Grad norms are 5–100× the fp32 values once logits saturate.
4. **Amplifier: Adam with beta2 0.999 and tiny √v.** Clipping does not limit per-parameter step size, so each spike pushes weights further in the same direction.
5. **Minor or secondary: residual growth across loops and the exit weighting.** These appear only after the collapse, and all exits and both RGB and depth degrade together.

Not causes: data (generated online), skip/clip ordering, the LR schedule mechanics, NaN/inf in the inputs.

## 4. Proposed fix plan (for approval; nothing implemented)

Minimal and targeted. No width or resolution change and no new modules beyond normalisation.

**F1. QK-norm (primary).**
- Replace `nn.MultiheadAttention` with an equivalent manual attention: the same `in_proj` and `out_proj` weights, plus a per-head RMSNorm on q and k with a learnable gain (init 1), then SDPA.
- Apply it in pre, core and post.
- The checkpoint loads with q/k gains initialised to 1 (new params). The logit scale changes, so a short re-adaptation is expected. Alternatively, initialise the gain to match the current per-head |q| at 14k.

**F2. Bound the adaLN modulation.**
- Put weight decay back on adaLN.weight (keep the biases excluded).
- Optionally cap the scale, e.g. `scale = s_max·tanh(scale/s_max)` with s_max around 4. This is an optional sub-item.

**F3. Fp32 attention softmax.** Compute attention logits and softmax in fp32 (upcast q/k inside the attention). This is cheap at 2,304 tokens.

**F4. Optimizer.**
- Set AdamW beta2 = 0.95 so the second moment adapts quickly to spikes.
- Optionally use eps 1e-6.
- Keep the clip at 0.5 and the skip logic.

**F5. (Optional) z-loss-style penalty.** Add a small penalty on the max attention logit, or log the logits only. F1 should make this unnecessary. Recommend logging only, not a penalty.

**Monitoring (no behaviour change):**
- Every 100 steps, log per-block max attention logit, max |adaLN scale|, σ(Wq) and σ(Wk), and per-exit and RGB/depth losses.
- Alarm when any logit exceeds 1e3.

## 5. How to validate (short repro, about 25–35 min on GPU5)

**Baseline repro, current code.** Resume from 14000 with LR 1e-4 and warmup 1000.
- Expected: pre logits pass ~1e4 by step ~16000 and grads exceed 1 by ~16.7k.
- This can be a short ~2.5k-step diagnostic run with the new monitoring, or the existing 16000 checkpoint can serve as the reference. Reusing the existing data means no rerun.

**Candidate A: F1 + F3 (+ F4).** Same resume and schedule, 3000 steps (about 22 min).
- Pass criteria:
  - every block's max logit stays below ~200
  - p99 grad < 1 and no step with grad > 10
  - held-out EMA R4 ≥ 24.5 at 16k and 17k
- Optional stress test: rerun at LR 3e-4 for 2000 steps. That LR broke at warmup in the 8h run.

**Candidate B: A + F2,** if A still shows the scale or logit trend growing.

**Ablation (cheap, optional):** F4 alone, to confirm the optimizer change only delays the collapse and does not prevent it.

After a candidate passes, an approved longer continue run (for example 2h) from 14000 with the same S/T/AR checkpoint evals.

Note: F1 changes the attention module, so it is a model-code change. It needs approval before implementation. Checkpoints from 14k can be loaded by mapping the `attn.in_proj_*` and `out_proj` keys unchanged.
