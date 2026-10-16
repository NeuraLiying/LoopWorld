# LoopWorld release media manifest

Curated assets under `docs/media/` for the public README. GIFs are embedded in GitHub; paired H.264 `.mp4` files are for smoother playback and download.

## `figures/`

| File | Use |
|---|---|
| `fig_simple_stage_e.png` | Simple setting — Stage E S/T/AR R1 vs R4 |
| `fig_maze_progression.png` | Complex setting — action → memory → SF → layout progression |
| `fig_maze_oracle_memory.png` | Own vs oracle memory read upper bound |

## `simple_orbit/`

| File | Use |
|---|---|
| `D1_loop_R1_R2_R4_split.gif` | README hero — same weights, R1/R2/R4 side by side |
| `D1_loop_R1_R2_R4_split_h264.mp4` | H.264 pair for D1 loop split |
| `D3_psnr_curve.png` | AR PSNR vs view index |
| `orbit_demo_step083864_h264.mp4` | Four-panel orbit RGB-D demo |

## `maze/`

| File | Use |
|---|---|
| `layout_explore.gif` | Maze exploration (first-person + path) |
| `layout_explore_h264.mp4` | H.264 pair for explore demo |
| `layout_best.gif` | Layout-best closed-loop AR |
| `layout_best_h264.mp4` | H.264 pair for layout best |
| `sf_ar_own_memory.gif` | Self-forcing own-memory AR |
| `sf_ar_own_memory_h264.mp4` | H.264 pair for SF AR |

## Excluded from this tree

Point-cloud exports, return-to-start-only clips, duplicate layout previews, and other low-signal or misleading demos from internal `exp_records/` were not shipped. Checkpoints and raw run logs are never in the repo (see root `.gitignore`).
