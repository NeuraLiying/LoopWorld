# 几何记忆（长程）设计短注 · 2026-10-14

## 问题（已由 maze action 基线定量确认）
`maze_v2_action_b16_4h` 最终 held-out：AR PSNR 全段 16.4，近端(1-8) 29.5，远端(33+) ~15；回访自洽 36.5 vs 对 GT 16.2。模型"记住了自己的错误预测"，但没记住几何。这正是显式几何记忆要填的缺口。

## 方案（简单、无新参数）
**点云缓冲 + z-buffer 重投影**，作为现有 2 个 context slot 之一的内容：
- **slot0 = 记忆重投影**（把 0..j−1 的 RGB-D 累积到世界坐标，再投到目标相机）
- **slot1 = 上一帧**（局部时序）
- **动作条件保留**（AdaLN 加到 t embedding）
- **Plücker 仍输入**（矩×0.6），消融可 `--no-pose`

不改模型权重结构：`ActionLoopWorldModel` 参数与 maze action ckpt 完全对齐，直接 fine-tune。

### 记忆细节
- 积分：反投影 hit 像素 → 世界点 (xyz, rgb)；像素 stride=2；总点数 cap 12 万，超出则保留一半最近 + 随机旧点。
- 训练：用 **GT** 帧积分（teacher forcing），时序 stride=4，并强制纳入 j−1。
- 评测 AR：用 **模型自己的预测** 积分（闭环），同样 stride。
- 重投影：相机系投影 + 最近深度 z-buffer，输出 RGB / z / mask，再走现有 `encode_rgbd`。
- 空记忆（j=1 且无历史）→ mask=False，与缺席 context 一致。

### 为何不用体素 / 不加第 3 slot
- 点云 splat 实现短、与连续几何一致；迷宫尺度小，12 万点足够。
- 第 3 slot 会改 `n_tokens` / `pos_embed`，无法从现有 ckpt 无缝加载。用 slot0 替换"随机 earlier 帧"在语义上更干净：记忆已覆盖回访首访几何。

### 训练
- 从 `ckpt_best.pt`（maze action，best_psnr≈16.61）fine-tune ~4h，B 以 profile 为准，lr 约 5e-5×sqrt(B/16)。
- 评测协议不变：分段漂移 + 回访自洽/对 GT。

### 文件（全部新增或仅追加，旧训练脚本不改行为）
- `loopworld/geom_memory.py`
- `loopworld/maze_v2_mem_data.py`
- `scripts/run_maze_v2_mem_train.py` / `scripts/maze_v2_mem_eval.py`
