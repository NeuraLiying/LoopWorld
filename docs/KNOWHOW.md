# LoopWorld · Know-how（配方笔记）

> 配套 [`STORY.zh.md`](STORY.zh.md)。可复用的训练、评测与诊断做法。

---

## 1. 长训稳定：QK-Norm + fp32 softmax

**现象**：高 LR 或长续训时 loss/grad 非有限。  
**测量**：记录 attention **max logit**（以及可选熵）。  
**做法**：per-head RMSNorm on Q/K + **fp32** SDPA/softmax。max logit 回到 ~18 后可稳定长训。

相关：`instability_diagnosis_20261013.md`、`qk_norm_fp32_change_20261013.md`。  
配套：原子 ckpt、diskguard、`--resume` 时按需 `--reset-schedule`。

---

## 2. 评测拆开 S / T / AR

| 模式 | 条件 | 用途 |
|---|---|---|
| **S** | 仅 GT 首帧 | 单条件生成 |
| **T** | + GT 前帧 | 完美历史参考 |
| **AR** | + 自身前帧 | 闭环主指标 |

环绕设定若首帧始终在条件中，「回到起点」帧分数易偏高；读曲线时看中段更有信息。迷宫协议成对报 **回访 vs GT** 与 **回访 vs 自身首访**。

简单设定备忘（ckpt 83864，R4）：S/T/AR ≈ **28.36 / 31.14 / 27.40**；循环增益约 **+1.9 dB**。

---

## 3. 记忆：own 与 oracle 成对

迷宫几何记忆上固定 ckpt 与噪声，切换写入来源：

1. **own** — 预测积分  
2. **oracle_mem** — GT 积分（读通路上界）  
3. **oracle_mem_gtprev** — 再加 GT 前帧  
4. **nomem** — 关记忆槽  

典型：own **17.5**、oracle **28.6**、nomem **15.1**。干净记忆可用且收益大。  
脚本：`scripts/maze_mem_diag.py`。

---

## 4. 深度指标

AbsRel **均值**对少数大误差敏感；同时报 **中位数**、δ&lt;1.25、墙 IoU。

---

## 5. 单卡利用率

96² / width 768 / 4090：batch **16** 可接近满 util；更大易 OOM 且未必更快。先拆步进里的 data vs compute。

---

## 6. 媒体包装

每条素材一个展示目的（见 [`media/MEDIA_MANIFEST.md`](media/MEDIA_MANIFEST.md)）。简单设定突出 loop 与 S/T/AR；复杂设定突出探索与 own/oracle 图。
