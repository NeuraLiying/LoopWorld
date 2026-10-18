# LoopWorld · GitHub 发布故事稿

> 中文发布叙事。首页能力展示见仓库 README；阶段年表与内部代号见 [`technical_report.md`](technical_report.md)；媒体见 [`media/MEDIA_MANIFEST.md`](media/MEDIA_MANIFEST.md)。

---

## Hook

**在一张 RTX 4090 上，从零完整搭好一个世界模型。**

共享权重 loop DiT：推理多跑几圈（R1→R4），自回归质量可测上升。同一套代码，先在合成环绕场景上把 **RGB-D + 位姿** 条件与 S/T/AR 评测跑通，再接到带离散动作与几何记忆的迷宫 rollout（路径在 GT 地图上 DFS 规划，**不是**模型自主规划）。

---

## 三个发现

1. **同权 R1→R4，24 个 held-out 环绕场景，AR +1.68 dB**（Stage E，5 Euler steps；以 JSON 原始相减为准）。
2. **循环深度有有效范围**；超出训练深度（R6/R8）会掉点；在本 flow 设定下，加 loop 往往比盲目加 denoising 步更划算。
3. **回访自洽 ≠ 与真场景一致**；oracle-history 说明系统对 rollout / 记忆保真度敏感，用作诊断，而不是断言「瓶颈只在写入」。

---

## 复现入口

本地训练后用 `--ckpt` / `--init` 指向 EMA。`results/` 下的 JSON 可用 `python scripts/print_result_tables.py` 核对表数字；训练与 demo 脚本见根目录 README。

---

## 简单设定 · 合成环绕

| 模式 | 条件 | R4 PSNR | 循环增益 R4−R1 | AbsRel |
|---|---|---|---|---|
| **S** | GT 首帧 RGB-D | **28.36** | **+1.93 dB** | 0.026 |
| **T** | 首帧 + GT 前帧 | **31.14** | +2.01 dB | 0.012 |
| **AR** | 首帧 + 自身前帧 | **27.40** | **+1.68 dB** | 0.032 |

ckpt：EMA step **83864**（多次续训后的 headline，不是一小时入门配方）。

媒体：`media/simple_orbit/D1_loop_R1_R2_R4_split.gif` 等。

---

## 复杂设定 · 迷宫

**公开测试集：seeds 700000–700031**（32 场景），layout EMA，`own_conf`：

| 记忆 | AR 全段 | 回访 vs GT |
|---|---|---|
| Own（conf） | **20.22** | **21.10** |
| Oracle | 33.24 | 35.23 |

演示（同一 DFS 轨迹，seed 700004）：

- 干净 teaser：`media/maze/layout_explore_clean.gif`
- 六面板诊断：`media/maze/layout_explore_diag.gif`（地图为 GT reference）

阶段递进图与旧协议数字见 [`technical_report.md`](technical_report.md)。

---

## 归因

受 **Atlas / Genie** 启发。LoopWorld 为独立从零实现与实验。

---

## 快速开始

见仓库 README（Download → demo → table；训练命令在后）。
