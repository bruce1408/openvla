# Official LIBERO Evaluation User Guide

> OpenVLA 官方 LIBERO 评测使用指南 · 路径：`experiments/robot/libero/official_repro/OFFICIAL_EVAL_GUIDE.md`

本目录用于**严格复现 OpenVLA 论文**中的 LIBERO 成功率（例如 LIBERO-Spatial **84.7 ± 0.9%**）。

与 `experiments/robot/libero/run_expanded_eval_multigpu.sh`（deploy 路径、`portable` 预处理）不同，这里使用 **TensorFlow 官方图像预处理**，数字才能和论文对齐。

---

## 0. 每次开跑前（必做）

在 **tmux** 里执行（评测要跑数小时，防止 SSH 断开）：

```bash
tmux new -s libero-official

cd /share_data/bruce/workspace/ai/openvla
source env_gpu.sh
```

一次性环境检查（首次或换机器时）：

```bash
conda activate torch270_128
pip install tensorflow==2.15.0    # 官方预处理必需
bash experiments/robot/libero/official_repro/check_env.sh
```

---

## 1. 我要跑哪种评测？（选一条路径）

| 目标 | 需要几步 | 推荐脚本 |
| --- | --- | --- |
| **A. 快速验证环境** | 1 步 | `run_smoke_test_suite.sh` |
| **B. 单 suite BF16 官方成功率** | 1 步 | `run_official_bf16_multigpu_suite.sh` |
| **C. 四个 suite 全部 BF16** | 1 步 | `run_all_suites_official.sh` |
| **D. 单 suite FP8 官方成功率** | 2 步（先构建产物） | `build_fp8_artifacts_for_suite.sh` → `run_official_eval_multigpu_balanced.sh` |
| **E. BF16 vs FP8 配对对比** | 2～3 步 | 先跑 B + D，再 `run_suite_bf16_fp8_compare.sh` |
| **F. 论文 3-seed 复现** | 1 步 | `run_paper_3seeds_suite.sh` |

**公共参数**：所有脚本用 `SUITE=` 切换 benchmark：

| `SUITE=` | 含义 | Checkpoint |
| --- | --- | --- |
| `spatial` | LIBERO-Spatial | `openvla-7b-finetuned-libero-spatial` |
| `object` | LIBERO-Object | `openvla-7b-finetuned-libero-object` |
| `goal` | LIBERO-Goal | `openvla-7b-finetuned-libero-goal` |
| `10` | LIBERO-Long-10 | `openvla-7b-finetuned-libero-10` |

每个 suite：**10 tasks × 50 trials = 500 episodes**（默认 `SEED=7`）。

---

## 2. 具体命令（逐步复制执行）

下面所有命令默认已在仓库根目录，且已 `source env_gpu.sh`。

### A. 冒烟测试（约 5 分钟，1 task × 2 trials）

**脚本**：`run_smoke_test_suite.sh`

```bash
# BF16 冒烟（Spatial）
SUITE=spatial bash experiments/robot/libero/official_repro/run_smoke_test_suite.sh

# 其他 suite 把 spatial 换成 object / goal / 10
SUITE=object bash experiments/robot/libero/official_repro/run_smoke_test_suite.sh
```

**输出**：`experiments/logs/libero_official/official-{suite}-smoke-bf16.summary.json`

---

### B. BF16 官方评测 — 单 suite，8 卡（推荐）

**脚本**：`run_official_bf16_multigpu_suite.sh`  
**入口**：`run_official_shard.py`（论文同款 TensorFlow 预处理）

```bash
# Spatial，500 episodes，seed=7
SUITE=spatial SEED=7 bash experiments/robot/libero/official_repro/run_official_bf16_multigpu_suite.sh

# Object
SUITE=object SEED=7 bash experiments/robot/libero/official_repro/run_official_bf16_multigpu_suite.sh

# Goal
SUITE=goal SEED=7 bash experiments/robot/libero/official_repro/run_official_bf16_multigpu_suite.sh

# Long-10（单 episode 最多 520 步，更慢）
SUITE=10 SEED=7 bash experiments/robot/libero/official_repro/run_official_bf16_multigpu_suite.sh
```

**输出**：

- 合并结果：`experiments/logs/libero_official/official-{suite}-seed7-8gpu.summary.json`
- 各卡 shard：`official-{suite}-seed7-8gpu-s{0..7}.jsonl`

> 续跑 / 重跑 / 清旧结果：见 **[§2.1 续跑、重新跑与清掉旧结果](#21-续跑重新跑与清掉旧结果)**。

**单卡版**（无 8 卡时用，较慢）：

```bash
SUITE=spatial bash experiments/robot/libero/official_repro/run_official_bf16.sh
```

> **GPU 占用**：8 卡脚本默认使用 GPU 0–7。若 `nvidia-smi` 显示部分 GPU 已被其他任务占用，请先释放或等占用任务结束后再跑。

---

### C. 四个 suite 批量 BF16

**脚本**：`run_all_suites_official.sh`

```bash
MODE=bf16 SEED=7 bash experiments/robot/libero/official_repro/run_all_suites_official.sh
```

只跑部分 suite：

```bash
MODE=bf16 SUITES="object goal" SEED=7 bash experiments/robot/libero/official_repro/run_all_suites_official.sh
```

跑完后自动打印汇总表；也可手动：

```bash
python experiments/robot/libero/official_repro/summarize_all_suites.py --seed 7
```

---

### D. FP8 官方评测 — 单 suite，8 卡

FP8 **不能共用** Spatial 的 TensorRT engine，每个 suite 要先构建专属产物。

#### 第 1 步：构建 FP8 产物（每个 suite 一次，耗时较长）

**脚本**：`build_fp8_artifacts_for_suite.sh`

```bash
# Spatial（若已有 legacy 产物可跳过，见下方说明）
SUITE=spatial bash experiments/robot/libero/official_repro/build_fp8_artifacts_for_suite.sh

# Object / Goal / 10 必须各自构建
SUITE=object bash experiments/robot/libero/official_repro/build_fp8_artifacts_for_suite.sh
SUITE=goal   bash experiments/robot/libero/official_repro/build_fp8_artifacts_for_suite.sh
SUITE=10     bash experiments/robot/libero/official_repro/build_fp8_artifacts_for_suite.sh
```

产物位置：

```text
deploy/tensorrt/artifacts/suites/{spatial,object,goal,10}/
  engines/vision_projector_fp8.plan
  engines/openvla_llama_fp8/llm.engine
  action_meta/action_meta.json
```

> Spatial 若已有 `deploy/tensorrt/artifacts/engines/` 下的 legacy FP8 产物，评测脚本会自动使用，可跳过构建。

#### 第 2 步：FP8 8 卡评测（episode 均衡分片）

**脚本**：`run_official_eval_multigpu_balanced.sh`

```bash
SUITE=spatial SEED=7 BACKEND=fp8 bash experiments/robot/libero/official_repro/run_official_eval_multigpu_balanced.sh

SUITE=object SEED=7 BACKEND=fp8 bash experiments/robot/libero/official_repro/run_official_eval_multigpu_balanced.sh
```

**输出**：`experiments/logs/libero_official/official-{suite}-seed7-8gpu-fp8-balanced.summary.json`

> 续跑 / 重跑 / 清旧结果：见 **[§2.1](#21-续跑重新跑与清掉旧结果)**（FP8 同样支持 `RESUME` / `FORCE_RESTART`）。

---

### 2.1 续跑、重新跑与清掉旧结果

三种常见场景：

| 场景 | 你想做什么 | 用什么方式 |
| --- | --- | --- |
| **续跑** | 上次跑到一半断了，接着跑 | 直接再执行同一命令（默认 `RESUME=1`） |
| **已完成再执行** | 500 ep 都跑完了，再敲一次命令 | 正常退出，只刷新 summary，**不会重跑仿真** |
| **强制重跑** | 结果不要了，从零再跑 500 ep | `FORCE_RESTART=1`（自动删旧 shard 文件） |

#### BF16 8 卡（`run_official_bf16_multigpu_suite.sh`）

```bash
# 【续跑】中断后继续（默认 RESUME=1，可省略）
SUITE=spatial SEED=7 bash experiments/robot/libero/official_repro/run_official_bf16_multigpu_suite.sh

# 【已完成】再执行一次：检测到 jsonl 已满 500 条 → 刷新 summary 后退出
# （与上面命令相同，不会重复跑仿真）

# 【强制重跑】清掉该 RUN_TAG 下所有 shard 输出，从头跑 500 ep
SUITE=spatial SEED=7 FORCE_RESTART=1 bash experiments/robot/libero/official_repro/run_official_bf16_multigpu_suite.sh

# 【换 seed 新跑】不删旧结果，用新 seed 产生新的一套日志
SUITE=spatial SEED=42 bash experiments/robot/libero/official_repro/run_official_bf16_multigpu_suite.sh
# → 输出 tag 变为 official-spatial-seed42-8gpu
```

`FORCE_RESTART=1` 会删除（以 `official-spatial-seed7-8gpu` 为例）：

```text
experiments/logs/libero_official/
  official-spatial-seed7-8gpu.summary.json      # 合并 summary
  official-spatial-seed7-8gpu-s{0..7}.jsonl     # 各卡逐 episode 记录
  official-spatial-seed7-8gpu-s{0..7}.summary.json
  official-spatial-seed7-8gpu-s{0..7}.worker.log
```

关闭续跑（极少数调试场景，会重复写同一 jsonl）：

```bash
SUITE=spatial SEED=7 RESUME=0 FORCE_RESTART=1 bash experiments/robot/libero/official_repro/run_official_bf16_multigpu_suite.sh
```

#### BF16 单卡（`run_official_bf16.sh`）

单卡脚本**没有** `FORCE_RESTART`，需手动删文件后重跑：

```bash
# 续跑（默认 RESUME=1）
SUITE=spatial SEED=7 bash experiments/robot/libero/official_repro/run_official_bf16.sh

# 强制重跑：先删旧输出
rm -f experiments/logs/libero_official/official-spatial-seed7.jsonl
rm -f experiments/logs/libero_official/official-spatial-seed7.summary.json
SUITE=spatial SEED=7 bash experiments/robot/libero/official_repro/run_official_bf16.sh
```

#### FP8 8 卡均衡（`run_official_eval_multigpu_balanced.sh`）

```bash
# 续跑
SUITE=spatial SEED=7 BACKEND=fp8 bash experiments/robot/libero/official_repro/run_official_eval_multigpu_balanced.sh

# 强制重跑（清 shard + 合并 jsonl + shard_plan）
SUITE=spatial SEED=7 BACKEND=fp8 FORCE_RESTART=1 bash experiments/robot/libero/official_repro/run_official_eval_multigpu_balanced.sh
```

默认 `RUN_TAG=official-{suite}-seed{N}-8gpu-fp8-balanced`。自定义 tag 时：

```bash
SUITE=spatial SEED=7 BACKEND=fp8 RUN_TAG=my-fp8-run FORCE_RESTART=1 \
  bash experiments/robot/libero/official_repro/run_official_eval_multigpu_balanced.sh
```

#### 手动清理（不按脚本变量时）

```bash
LOGDIR=/share_data/bruce/workspace/ai/openvla/experiments/logs/libero_official
TAG=official-spatial-seed7-8gpu          # 或 official-spatial-seed7-8gpu-fp8-balanced

rm -f "$LOGDIR/${TAG}"*.jsonl "$LOGDIR/${TAG}"*.summary.json "$LOGDIR/${TAG}.log"
rm -f "$LOGDIR/${TAG}"-s*.worker.log "$LOGDIR/${TAG}.shard_plan.json" 2>/dev/null
```

清理后重新执行对应的评测命令即可。

---

### E. BF16 vs FP8 配对对比

**前提**：同一 `SUITE` + `SEED` 下，BF16 和 FP8 的 500 episodes 都已跑完。

**脚本**：`run_suite_bf16_fp8_compare.sh`

```bash
# 若 FP8 已跑完，只生成对比 JSON：
SUITE=spatial SEED=7 SKIP_FP8=1 bash experiments/robot/libero/official_repro/run_suite_bf16_fp8_compare.sh

# 若 FP8 还没跑，会自动先跑 FP8 再对比：
SUITE=spatial SEED=7 bash experiments/robot/libero/official_repro/run_suite_bf16_fp8_compare.sh
```

**输出**：`experiments/logs/libero_official/official-spatial-seed7-8gpu-bf16-vs-fp8-official.json`

**完整手动流程（三步）**：

```bash
# 1) BF16
SUITE=spatial SEED=7 bash experiments/robot/libero/official_repro/run_official_bf16_multigpu_suite.sh

# 2) FP8（产物已就绪时）
SUITE=spatial SEED=7 BACKEND=fp8 bash experiments/robot/libero/official_repro/run_official_eval_multigpu_balanced.sh

# 3) 对比
SUITE=spatial SEED=7 SKIP_FP8=1 bash experiments/robot/libero/official_repro/run_suite_bf16_fp8_compare.sh
```

---

### F. 论文 3-seed 复现（BF16）

**脚本**：`run_paper_3seeds_suite.sh`  
默认 seeds：`7, 42, 123`；每个 seed 500 episodes。

```bash
SUITE=spatial bash experiments/robot/libero/official_repro/run_paper_3seeds_suite.sh

# 其他 suite
SUITE=object bash experiments/robot/libero/official_repro/run_paper_3seeds_suite.sh
```

---

## 3. 辅助命令

### 查看正在跑的任务进度

```bash
bash experiments/robot/libero/official_repro/process_info.sh
```

### 预览 FP8 episode 均衡分片

```bash
python experiments/robot/libero/official_repro/episode_balanced_sharding.py --format table
```

### 四 suite 一键编排（高级）

**脚本**：`run_all_suites_official.sh`

```bash
# 仅为缺失 suite 构建 FP8
MODE=build_fp8 SUITES=object bash experiments/robot/libero/official_repro/run_all_suites_official.sh

# 四 suite 全部 BF16
MODE=bf16 bash experiments/robot/libero/official_repro/run_all_suites_official.sh

# 四 suite 全部 FP8（需产物就绪）
MODE=fp8 bash experiments/robot/libero/official_repro/run_all_suites_official.sh

# 四 suite 全部对比（需 BF16+FP8 都已跑完）
MODE=compare bash experiments/robot/libero/official_repro/run_all_suites_official.sh

# 全流程：build → bf16 → fp8 → compare
MODE=all bash experiments/robot/libero/official_repro/run_all_suites_official.sh
```

---

## 4. 输出文件说明

日志目录：`experiments/logs/libero_official/`

| 文件 | 说明 |
| --- | --- |
| `official-{suite}-seed{N}-8gpu.summary.json` | BF16 合并成功率 |
| `official-{suite}-seed{N}-8gpu-fp8-balanced.summary.json` | FP8 合并成功率 |
| `official-{suite}-seed{N}-8gpu-bf16-vs-fp8-official.json` | BF16 vs FP8 配对对比 |
| `official-{suite}-seed{N}-8gpu-s{0..7}.jsonl` | 各 GPU 逐 episode 记录（可 `--resume`） |
| `official-{suite}-seed{N}-8gpu.log` | 主脚本日志 |

---

## 5. 本机已有结果（seed=7, Spatial）

| Backend | 成功率 | Summary 文件 |
| --- | --- | --- |
| BF16 official | 82.8% (414/500) | `official-spatial-seed7-8gpu.summary.json` |
| FP8 balanced | 83.8% (419/500) | `official-spatial-seed7-8gpu-fp8-balanced.summary.json` |

补跑 Spatial 对比：

```bash
SUITE=spatial SEED=7 SKIP_FP8=1 bash experiments/robot/libero/official_repro/run_suite_bf16_fp8_compare.sh
```

---

## 6. 脚本与 Python 文件对照

| 文件 | 作用 |
| --- | --- |
| **`OFFICIAL_EVAL_GUIDE.md`** | **本使用指南** |
| `libero_suites.sh` | 四 suite 配置（checkpoint、FP8 路径），被其他脚本 source |
| `check_env.sh` | 依赖检查 |
| `install_deps.sh` | 安装依赖 |
| `process_info.sh` | 查看运行进度 |
| `run_smoke_test_suite.sh` | 冒烟 |
| `run_official_bf16.sh` | BF16 单卡 |
| `run_official_bf16_multigpu_suite.sh` | **BF16 8 卡（论文复现主入口）** |
| `build_fp8_artifacts_for_suite.sh` | FP8 TensorRT 产物构建 |
| `run_official_eval_multigpu_balanced.sh` | **FP8 8 卡（episode 均衡）** |
| `run_suite_bf16_fp8_compare.sh` | BF16 vs FP8 对比 |
| `run_all_suites_official.sh` | 四 suite 批量编排 |
| `run_paper_3seeds_suite.sh` | 论文 3-seed |
| `run_official_shard.py` | BF16 核心 Python 入口 |
| `merge_official_summaries.py` | 合并 BF16 shard |
| `compare_official_bf16_fp8.py` | 对比逻辑 |
| `summarize_all_suites.py` | 跨 suite 汇总表 |

---

## 7. 与 deploy 评测的区别

| 项目 | 本目录（官方） | deploy 路径 |
| --- | --- | --- |
| 使用指南 | `official_repro/OFFICIAL_EVAL_GUIDE.md` | `libero/DEPLOY_EVAL.md` |
| Shell 入口 | `official_repro/run_*.sh` | `run_expanded_eval_multigpu.sh` |
| Python 入口 | `run_official_shard.py` | `run_libero_deploy_eval.py` |
| 图像预处理 | TensorFlow（official） | 默认 portable（Pillow） |
| 论文数字 | ✅ 可对齐 | ❌ 不能直接对比（Spatial 约 35% vs 84%） |

---

## 8. 论文 84.7% 的含义

- **Suite**：LIBERO-Spatial（10 tasks）
- **Rollouts**：每 task 50 次 → 每 seed 500 episodes
- **Seeds**：论文报告 **3 个 seed 的均值 ± 标准差**
- **关键参数**：`center_crop=True`（与微调时 90% random crop 对齐）

---

## 9. 目录结构与冗余说明（复习用）

本节说明 `official_repro/` 里各文件的职责、哪些看起来重复其实不能删，以及推荐怎么用。便于日后复习，避免误删脚本或跑错入口。

### 9.1 总体结论

| 状态 | 说明 |
| --- | --- |
| ✅ 无冗余文件 | 历史上 spatial 专用旧脚本、薄包装脚本已清理；目前没有两个文件做同一件事 |
| ✅ 职责清晰 | 每个文件对应一个明确用途 |
| ⚠️ 有逻辑重叠 | 存在 2 套 BF16 8 卡路径、2 个 compare 脚本、multigpu shell 模板代码相似 |
| ❌ 不建议再删 | 重叠部分服务于不同评测后端，强行合并容易引入 bug |

当前目录共 **18 个文件**（1 文档 + 11 shell + 6 Python）。

### 9.2 文件清单（按层级）

```text
official_repro/
├── OFFICIAL_EVAL_GUIDE.md          # 本使用指南
├── libero_suites.sh                # 配置中心（被其他脚本 source）
│
├── 环境
│   ├── check_env.sh                # 检查依赖
│   └── install_deps.sh             # 安装 TensorFlow
│
├── 评测入口（8 个 shell）
│   ├── run_smoke_test_suite.sh     # 冒烟（1 task × 2 trials）
│   ├── run_official_bf16.sh        # BF16 单卡
│   ├── run_official_bf16_multigpu_suite.sh   # BF16 8 卡 ★ 论文主入口
│   ├── run_official_eval_multigpu_balanced.sh # FP8 8 卡（及 deploy 路径 BF16）
│   ├── build_fp8_artifacts_for_suite.sh
│   ├── run_suite_bf16_fp8_compare.sh
│   ├── run_all_suites_official.sh  # 四 suite 批量编排
│   └── run_paper_3seeds_suite.sh   # 3-seed 循环
│
├── 工具
│   └── process_info.sh             # 查看运行进度
│
└── Python（6 个）
    ├── run_official_shard.py       # BF16 核心评测
    ├── merge_official_summaries.py # 合并 BF16 shard summary
    ├── compare_official_bf16_fp8.py
    ├── episode_balanced_sharding.py
    └── summarize_all_suites.py
```

### 9.3 容易误以为「重复」的地方（其实不能删）

#### 两套 BF16 8 卡路径

这是最容易混淆的一点：

| 脚本 | Python 入口 | 分片方式 | 典型 RUN_TAG | 用途 |
| --- | --- | --- | --- | --- |
| `run_official_bf16_multigpu_suite.sh` | `run_official_shard.py` | 按 **task** 分片 | `official-{suite}-seed{N}-8gpu` | **论文复现主入口** |
| `run_official_eval_multigpu_balanced.sh` + `BACKEND=bf16` | `run_libero_deploy_eval.py` | 按 **episode** 均衡 | `official-{suite}-seed{N}-8gpu-bf16-balanced` | deploy 路径 / 与 FP8 配对 |

两者都能跑 BF16 + official 预处理，但代码路径、分片策略、合并脚本、日志 tag 都不同。

**推荐**：日常与论文复现 **只用** `run_official_bf16_multigpu_suite.sh`。  
`BACKEND=bf16` 的 balanced 脚本主要服务于 FP8 对比链路，一般不必单独跑。

#### 两个 compare 脚本（本目录 vs 父目录）

| 文件 | 位置 | 用途 |
| --- | --- | --- |
| `compare_official_bf16_fp8.py` | `official_repro/` | 读官方 jsonl shard，做 BF16 vs FP8 配对对比 |
| `compare_libero_results.py` | `libero/` | deploy 路径的 BF16 vs FP8 对比 |

结构相似，但输入格式与校验字段不同，**不能合并**。

#### 两个 merge 脚本

| 文件 | 合并对象 |
| --- | --- |
| `merge_official_summaries.py` | BF16 shard 的 `.summary.json` |
| `merge_libero_deploy_shards.py`（父目录） | FP8 / deploy 路径的 `.jsonl` |

schema 不同，**不重复**。

### 9.4 轻微重叠（可优化，但不必须删）

| 重叠点 | 说明 | 建议 |
| --- | --- | --- |
| `run_smoke_test_suite.sh` vs `run_official_bf16.sh` | 冒烟 = 1 task × 2 trials；单卡 = 10 task × 50 trials；都调用 `run_official_shard.py` | 保留；冒烟命令更短 |
| `run_paper_3seeds_suite.sh` | 循环 3 次调用 multigpu / 单卡 | 保留；论文 3-seed 常用 |
| `run_all_suites_official.sh` | 批量调用 build / bf16 / fp8 / compare | 保留；编排层 |
| `install_deps.sh` + `check_env.sh` | 前者装 TF 后调用后者 | 可合并，但文件很小，现状可接受 |
| 两个 multigpu shell 的 launch/wait/print | 结构约 80% 相似 | 可抽公共 `multigpu_common.sh`，目前非必须 |

### 9.5 调用关系（谁调谁）

```text
run_all_suites_official.sh
  ├── build_fp8_artifacts_for_suite.sh
  ├── run_official_bf16_multigpu_suite.sh
  │     └── run_official_shard.py → merge_official_summaries.py
  ├── run_official_eval_multigpu_balanced.sh
  │     └── run_libero_deploy_eval.py → merge_libero_deploy_shards.py（父目录）
  └── run_suite_bf16_fp8_compare.sh
        └── compare_official_bf16_fp8.py

run_paper_3seeds_suite.sh
  └── run_official_bf16_multigpu_suite.sh（或 run_official_bf16.sh）

run_smoke_test_suite.sh
  └── run_official_shard.py（BF16）/ run_libero_deploy_eval.py（FP8）
```

无循环依赖，也没有「A 与 B 完全等价」的脚本对。

### 9.6 快速选型（复习卡片）

| 我想… | 用哪个 |
| --- | --- |
| 论文 BF16 500 ep | `run_official_bf16_multigpu_suite.sh` |
| FP8 500 ep | `build_fp8_artifacts_for_suite.sh` → `run_official_eval_multigpu_balanced.sh` |
| BF16 vs FP8 对比 | `run_suite_bf16_fp8_compare.sh` |
| 四 suite 批量 | `run_all_suites_official.sh` |
| 快速验环境 | `run_smoke_test_suite.sh` |
| 看进度 | `process_info.sh` |

### 9.7 已清理的历史冗余（供参考）

以下脚本曾在早期版本存在，**已删除**，勿再寻找：

- `run_official_bf16_multigpu.sh`（spatial 专用）→ 用 `run_official_bf16_multigpu_suite.sh`
- `run_official_fp8_multigpu.sh`（task 分片 FP8）→ 用 `run_official_eval_multigpu_balanced.sh`
- `run_official_fp8_multigpu_balanced.sh` / `run_official_bf16_fp8_compare.sh`（薄包装）→ 已合并到通用脚本
- `run_smoke_test.sh` / `run_paper_3seeds.sh`（spatial 专用）→ 用 `*_suite.sh` 版本
