# TensorRT Pipeline 脚本索引

按精度分类的部署与评测脚本。典型执行顺序：**fp16（基准）→ fp8 / nvfp4（量化路径）**。

## fp16/ — 基准导出与 Vision Engine

| 脚本 | 说明 |
|------|------|
| `00_dump_golden.py` | 导出 PyTorch BF16/FP16 golden 张量（精度对照基准） |
| `01_export_vision_projector_onnx.py` | 导出 FP16 vision ONNX |
| `02_extract_llama_checkpoint.py` | 提取 Llama 语言模型 checkpoint |
| `03_convert_llm_x86.sh` | x86 上 LLM 量化 + ONNX 导出（bf16 / fp8 / nvfp4） |
| `04_export_action_params.py` | 导出动作解码 sidecar 参数 |
| `05_build_vision_engine.sh` | 构建 vision TensorRT engine（默认 FP16，可选 FP8） |

## fp8/ — FP8 量化与评测

| 脚本 | 说明 |
|------|------|
| `01b_quantize_vision_fp8.py` | Vision ONNX FP16 → FP8 PTQ 量化 |
| `06_measure_llm_latency.sh` | LLM prefill/decode 时延（参数：`fp8` 或 `nvfp4`） |
| `09_prof_trace_e2e.py` | E2E trace + 逐层精度验证（`--precision fp8\|nvfp4`） |
| `10_operator_categories.py` | 算子类别聚合（GEMM / Attention / Memory 等） |
| `11_measure_fp8_trt.py` | FP8 TRT 端到端 + 模块级时延采集 |
| `12_fp8_action_accuracy.py` | FP8 7-DoF 动作精度（RMSE vs BF16 golden） |
| `13_nsys_cpu_overhead.py` | FP8 Edge-LLM host 侧 CPU 开销（nsys） |

## nvfp4/ — NVFP4 量化与评测

| 脚本 | 说明 |
|------|------|
| `07_measure_e2e_latency.py` | 真端到端时延（`--precision nvfp4\|fp8`） |
| `08_measure_component_latency.sh` | 三组件时延基准（NVFP4 LLM + FP16 Vision） |
| `14_nvfp4_action_accuracy.py` | NVFP4 7-DoF 动作精度（RMSE vs BF16 golden） |
| `14b_nvfp4_token_accuracy.py` | NVFP4 token 级精度（无需加载 BF16 model） |
| `14c_nvfp4_nsys_cpu_overhead.py` | NVFP4 Edge-LLM host 侧 CPU 开销（nsys） |
| `14_nvfp4_collect_all.sh` | NVFP4 采集编排（trace + 精度 + nsys 一键跑） |
| `15_nvfp4_multi_sample_accuracy.py` | 多样本 golden dump + token/RMSE 聚合 |

## 常用命令示例

```bash
# FP16 基准
python deploy/tensorrt/pipeline/fp16/00_dump_golden.py --dtype bf16

# FP8 vision 量化 + engine
python deploy/tensorrt/pipeline/fp8/01b_quantize_vision_fp8.py
bash deploy/tensorrt/pipeline/fp16/05_build_vision_engine.sh \
  deploy/tensorrt/artifacts/onnx/vision_projector_fp8.onnx \
  deploy/tensorrt/artifacts/engines/vision_projector_fp8.plan \
  fp8

# FP8 动作精度
python deploy/tensorrt/pipeline/fp8/12_fp8_action_accuracy.py --mode action

# NVFP4 一键采集
bash deploy/tensorrt/pipeline/nvfp4/14_nvfp4_collect_all.sh
```
