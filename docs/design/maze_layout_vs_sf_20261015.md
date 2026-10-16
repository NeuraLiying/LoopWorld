# Layout-fix vs Self-forcing 全量对比 · 2026-10-15

- 协议：32 个测试迷宫（seed 600000–600031），每个 128 帧，R4，S5，与 `maze_mem_diagnosis_20261014` / `maze_sf_eval_20261015` 同一套噪声种子，1712 个回访对。
- Layout run：`maze_v3_layout_b17_4h_20261015_094529`（墙钟停在 step 13268）。ckpt_best = **step 10000**（mini AR 20.15）。final = step 13268（mini AR 19.61）。
- SF：`maze_v2_sf_b16_4h_20261015_012547` final = step 19351。
- 主评测模式：`own_conf`（记忆只积分两次采样一致的像素）。SF 额外报 `own`（无置信过滤，与 `maze_sf_eval_20261015` 一致）。Layout best 另报 `oracle_mem`。
- 原始结果：`exp_records/maze_layout_vs_sf_20261015/{layout_best,layout_final,sf_final}/v3_eval.json`。

## 主表（32 迷宫，own_conf，除非另标）
| 模型 | 模式 | AR 全段 | 1-8 | 9-32 | 33-64 | 65-127 | 回访 vs GT | 回访 vs 自身 | AbsRel 均值 | AbsRel 中位 | δ&lt;1.25 | 墙 IoU | 记忆覆盖 | 提示 PSNR | conf 占比 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Layout best (10k) | own_conf | **19.43** | 31.46 | **17.35** | **18.07** | **19.40** | **19.67** | 24.31 | **0.450** | **0.0058** | **0.761** | **0.934** | 0.581 | 18.96 | 0.707 |
| Layout final (13.3k) | own_conf | 19.35 | 31.24 | 17.31 | 18.07 | 19.27 | 19.65 | 22.95 | 0.488 | 0.0061 | 0.749 | 0.931 | 0.538 | 19.32 | 0.643 |
| SF final (19.4k) | own_conf | 19.10 | 30.05 | 17.03 | 17.93 | 19.09 | 19.36 | 23.64 | 0.632 | 0.0060 | 0.742 | 0.926 | 0.597 | 18.98 | 0.699 |
| SF final | own（无 conf） | 18.98 | 29.85 | 17.16 | 17.90 | 18.84 | 19.05 | 24.08 | 0.499 | 0.0059 | 0.759 | 0.933 | 0.701 | 18.29 | – |
| Layout best | oracle_mem | 33.24 | 32.34 | 30.49 | 33.21 | 34.42 | 35.60 | 31.43 | 0.022 | 0.0031 | 0.987 | 0.996 | 0.704 | 27.83 | – |

对照：SF final 在 `maze_sf_eval` 里的 own 也是 18.98（一致）。SF 的 oracle_mem 当时是 31.67。

## 解读
1. **Layout best 略胜 SF**，在相同 `own_conf` 协议下：AR +0.33 dB（19.43 vs 19.10），回访 vs GT +0.31 dB，远段（65-127）+0.31 dB，AbsRel 均值从 0.63 降到 0.45。幅度小，不是质变。
2. **ckpt_best（10k）优于 final（13.3k）**：与 mini 曲线一致（10k=20.15，13.3k=19.61）。课程到 12k 才结束，后半段 lr 已衰减，继续训没有收益。
3. **置信过滤对 SF 也略有帮助**（own 18.98 → own_conf 19.10），但 AbsRel 均值变差（0.50 → 0.63）：过滤写得更少（覆盖 70%→60%），空洞处更容易出近墙/远廊的大误差。Layout 训练过 conf 记忆，所以 own_conf 下 AbsRel 均值更好（0.45）。
4. **oracle 仍远高于 own**：Layout best oracle 33.24 vs own_conf 19.43（差约 14 dB）。模型用好记忆的能力又涨了一点（相对 SF oracle 的 31.7），但错误首访几何写进记忆的问题没有解决。
5. **墙 IoU / δ1 / AbsRel 中位** 三个 run 几乎持平（IoU≈0.93，δ1≈0.75，中位≈0.006）。布局专项没有显著改善"墙在不在"的 mask 指标；提升主要来自像素级 PSNR 和均值 AbsRel（后者对重尾敏感）。

## Mini held-out 曲线回顾（layout，8 迷宫）
峰值在 **step 10000（AR 20.15，回访 vs GT 20.77）**；之后 11–13k 缓慢回落。与 32 迷宫结论一致：best 在中段，不是末段。

## 建议：**先停迷宫这条线**
理由：
- 从 action（16.2）→ mem（17.5）→ SF（19.0）→ layout（19.4），每次 +1–2 dB，但 own↔oracle 差距仍约 12–14 dB，说明核心瓶颈（首访拓扑错误 → 脏记忆）没被这轮过采样/置信过滤撬开。
- Layout 相对 SF 的增益 <0.4 dB，投入产出已经很低；再加长课程或加权难帧，预期边际更小。
- 若还要继续，应换机制，而不是再 fine-tune：例如可纠正的记忆（检测冲突、用新观察覆盖旧错误墙）、或显式"墙/空"分类头、或更强的几何先验。否则建议转向 GitHub/Arena 故事和其他场景，把当前 best（layout step 10k 或 SF）冻成基线。

## 产出
- 本文：`exp_records/maze_layout_vs_sf_20261015.md`
- 视频（layout best，2 迷宫，own 记忆，无 conf）：`layout_best_video/sf_ar_own_memory_h264.mp4`、`layout_best.gif`
- Mac：`~/loopworld_demos_20261014/layout_best_h264.mp4`、`layout_best.gif`
- Box：`/workspace/maze_layout/`
