# Release media · curated · 2026-10-16

相对路径：`docs/media/`。  
原则：每条素材一个清晰展示目的；首页优先干净 teaser，诊断面板单独标注。

## Figures（`figures/`）

| 文件 | 展示目的 |
|---|---|
| `figures/fig_simple_stage_e.png` | Orbit Stage E：S/T/AR 下 R1 vs R4 |
| `figures/fig_maze_progression.png` | 迷宫开发递进（技术报告；非公开 test headline） |
| `figures/fig_maze_oracle_memory.png` | own vs oracle 诊断图（技术报告） |

## Simple setting · orbit（`simple_orbit/`）

| 文件 | 展示目的 |
|---|---|
| `simple_orbit/D1_loop_R1_R2_R4_split.gif`（+ `_h264.mp4`） | 同权多 loop 分屏 — 首页 teaser |
| `simple_orbit/D3_psnr_curve.png` | AR 随视角 PSNR |
| `simple_orbit/orbit_demo_step083864_h264.mp4` | 四栏 RGB-D 轨道 demo |

## Complex setting · maze（`maze/`）

| 文件 | 展示目的 |
|---|---|
| `maze/layout_explore_clean.gif`（+ `_h264.mp4`） | **DFS** 探索干净 teaser（seed 700004，own_conf）；地图 = GT reference |
| `maze/layout_explore_diag.gif`（+ `_h264.mp4`） | 同轨迹六面板诊断（GT / pred / memory / depth） |
| `maze/sf_ar_own_memory.gif`（+ `_h264.mp4`） | 较早 self-forcing own-memory AR（对照用） |

未纳入：旧 layout_explore / layout_best（矩形 MazeEpisode 路径，已用 DFS 新 seed 替换）、弱早期 rollout、失败特写。
