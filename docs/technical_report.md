# LoopWorld · technical report (portfolio)

Companion to the README. Internal stage names (Stage E, Layout round, cont4h2, …) live here so the homepage can stay a capability showcase.

## Scope

- Synthetic static scenes, **96²**, ~**46M** params, **one RTX 4090**
- Camera-conditioned **RGB-D** prediction (first frame = RGB-D + Plücker pose)
- Maze: discrete actions + geometric memory; paths planned on GT map (DFS / episode planners) — **not** autonomous planning

## Orbit chronology (headline = step 83864)

1. Capacity / smoke runs at 96²
2. Scaled Stage D training with QK-Norm + fp32 attention logits (stability)
3. Multi-continuation training ending at **`ckpt_final_step083864`** (EMA)
4. **Stage E** protocol: 24 held-out seeds, modes **S / T / AR**, loops **R∈{1,2,4}**, 5 Euler steps  
   → published JSON: `results/orbit_stage_e_step083864.json`  
   → AR loop gain R4−R1 = **+1.68 dB** (raw subtraction)
5. R×steps extrapolation grid (incl. R6/R8): `results/orbit_loop_rxsteps_grid.json`  
   → peak near **R4 + 2 steps**; deeper R hurts AR

The README “~1 hour starter” recipes are for onboarding. They are **not** the orbit headline run.

## Maze chronology

| Stage | Idea | Notes |
|---|---|---|
| Action | AdaLN discrete actions, 128-frame episodes, revisit protocol | No geometric memory |
| Geometric memory | Lift RGB-D → points → reproject | Own vs oracle diagnosis |
| Self-forcing | Train memory on EMA rollouts | Closes train/test memory gap |
| Layout round | Hard-frame / turn oversampling + confidence-weighted memory | Final closed-loop recipe |

**Published test suite (README):** seeds **700000–700031**, layout EMA, `own_conf` AR **20.22**, revisit vs GT **21.10**, oracle AR **33.24**.  
JSON: `results/maze_layout_test_seeds700000_700031.json`.

Developmental curves on an earlier seed band (used while iterating checkpoints) are **not** the public test headline. Figures below are kept for the progression story:

- `docs/media/figures/fig_maze_progression.png`
- `docs/media/figures/fig_maze_oracle_memory.png`
- `docs/media/maze/sf_ar_own_memory.gif`

## Demo path semantics

| Asset | Planner | Use |
|---|---|---|
| `layout_explore_clean` / `_diag` | **DFS** on GT map | Portfolio teaser + diagnostic (seed 700004) |
| MazeEpisode eval trajectories | Rectangular revisit schedule | Aggregate PSNR table (32 seeds) |

DFS long explorations typically land ~15 dB PSNR; short MazeEpisode eval averages ~20 dB. Different path design — not a contradiction.

## Eval protocols (short)

- **S:** condition on GT first RGB-D frame only  
- **T:** first frame + GT previous frame (perfect history)  
- **AR:** first frame + model’s own previous frame  
- **own_conf:** geometric memory from model predictions, confidence-gated  
- **oracle_mem:** memory integrated from GT RGB-D (diagnostic upper bound)

## Reproduce tables

```bash
python scripts/print_result_tables.py
```

GPU re-eval commands are in the README Quick start.
