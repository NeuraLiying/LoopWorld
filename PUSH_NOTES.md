# LoopWorld GitHub push notes

Prepared: 2026-10-16 (Asia/Singapore).  
Target: https://github.com/NeuraLiying/LoopWorld (private, empty).  
Tree: `/workspace/loopworld_github_push/` · tarball: `/workspace/loopworld_github_push.tar.gz`

## Included

| Path | Why |
|---|---|
| `loopworld/` | Full publishable package from mars: orbit model + maze + geom memory + layout utils (no `.bak`) |
| `configs/` | `r1_smoke_256.yaml`, `r1_main_640.yaml`, `r1_main_768_96.yaml` |
| `scripts/` | Published train/eval/demo entrypoints only (simple Stage A/D/E + maze action/mem/SF/layout + demos) |
| `docs/media/` | Curated release media (figures + simple_orbit + maze GIFs/mp4s) from `/workspace/loopworld_release/` |
| `docs/KNOWHOW.md` | Training/eval recipes |
| `docs/STORY.zh.md` | Chinese narrative (简单设定 / 复杂设定; no V1/V2 framing) |
| `docs/design/` | Selected design & diagnosis notes (QK-Norm, memory, SF, layout, maze protocol) |
| `README.md` | English landing page — capability-first, embeds figures/GIFs, tables |
| `LICENSE` | MIT |
| `requirements.txt` | Public-facing deps |
| `.gitignore` | Excludes ckpts, wandb, datasets, bulk exp dumps |

## Excluded (on purpose)

| Source on mars | Why dropped |
|---|---|
| `external_baselines/` (~12G) | Unrelated third-party baselines / explorations |
| `reports/`, `drafts/` | CVPR review notes / internal drafts — not the LoopWorld product story |
| `docs/` old capacity plans (mars) | Superseded; replaced by KNOWHOW + design notes |
| Most of `exp_records/` (~54G) | Raw runs, logs, checkpoints, smoke dirs, aborted jobs |
| `scripts/round2/` | Round-2 evidence / DL3DV / LagerNVS engineering smoke — other-direction exploration |
| `scripts/profile_scaled_*.py`, `run_stage_c_profile.py`, `wait_and_launch_sf.sh` | Profiling / ops helpers, not published recipes |
| `*.bak*`, `__pycache__`, `*.pt` | Backups / caches / weights |
| Non-manifest demos (pointcloud, return-to-start, scene wall, maze preview, layout_complex) | Already curated out of MEDIA_MANIFEST (low info / easy to misread / duplicate) |

## Media size policy

All curated GIF + H.264 pairs kept (README embeds GIFs; mp4s for download/playback). Uncompressed tree ≈ 31 MB; tarball should stay under ~40 MB because media is already compressed.

## SSH note

Mars host from Ying's Mac SSH config: **`10.0.3.233`** (hostname `mars`, user `liying`, port 22022). Project: `/home/liying/projects/loopworld`. Code curated there then copied via Mac → box.

## How to push (for cloud agent / Ying)

```bash
cd /workspace/loopworld_github_push   # or extract the tarball
git init
git add .
git commit -m "Initial LoopWorld release: loop DiT baseline (orbit + maze memory)"
git branch -M main
git remote add origin git@github.com:NeuraLiying/LoopWorld.git
git push -u origin main
```

Do **not** commit checkpoints. Point `--ckpt` / `--init` at local weights after training.

## Framing checklist (Ying)

- [x] No V1/V2 naming in README
- [x] Simple vs complex setting, same codebase
- [x] Showcase value (numbers + figures + GIFs); no apology / "honest diagnosis" landing tone
- [x] No tiny-atlas comparisons
- [x] Unrelated explorations excluded

## Script default `--init` / `--ckpt` paths

Several maze scripts still default to mars-local `exp_records/.../ckpt_*.pt` paths. Those weights are **not** in the repo. Callers must pass `--init` / `--ckpt` explicitly. README already documents this; no code change required for the first push.
