"""OpenVLA Thor 端侧推理性能评测脚本。

用途
====
这个脚本用于在 NVIDIA Thor / T5000 设备上评测 OpenVLA 的端侧单步动作推理性能。
它会加载本地 Hugging Face 缓存中的 OpenVLA 模型，构造一组「图像 + 语言指令」输入，
反复调用 `model.predict_action()`，并把每次迭代的耗时记录到 JSONL 文件，最后生成一个
summary JSON，方便写部署评测报告。

它测的不是训练速度，也不是服务器吞吐，而是机器人控制链路里最关心的单次 action latency：
    当前相机图像 + 当前任务指令 -> OpenVLA 输出下一步 7-DoF 动作

输出文件
========
1. openvla_thor_benchmark_*.jsonl
   一行一个 JSON，每一行对应一次正式测试迭代。适合做后处理、画图、排查异常 spike。

2. openvla_thor_benchmark_*.summary.json
   对 JSONL 中所有 *_ms 延迟字段做统计汇总，包括 mean、p50、p90、p95、p99、min、max。
   写评测报告时通常引用这个文件。

核心指标
========
- processor_time_ms：图像预处理 + prompt tokenization 的耗时。
- h2d_time_ms：输入 tensor 从 host/CPU 内存搬到 device/GPU 的耗时。
- predict_action_total_time_ms：OpenVLA 模型执行 predict_action() 的耗时。
- model_e2e_time_ms：从 processor 开始到拿到 action 的端到端耗时。
- generate_*：可选指标，只在 --measure-generate 时出现，用于分析 LLM 生成阶段。

控制频率参考
============
如果目标是 10Hz 控制闭环，一次模型侧端到端耗时最好低于 100 ms。
如果目标是 20Hz 控制闭环，一次模型侧端到端耗时最好低于 50 ms。
实际报告里建议重点看 model_e2e_time_ms 的 mean、p95、p99，而不是只看平均值。
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 这里把 app 加到 Python import 路径最前面。
# 原因：这个 benchmark 文件现在从 tools/ 目录运行；
# 加这行后，无论当前工作目录在哪里，都能稳定导入 runtime_env.py。
# runtime_env.py 的作用是：
#   1. source env.sh；
#   2. 设置 HF_HOME、TRANSFORMERS_OFFLINE 等 Hugging Face 离线缓存变量；
#   3. 导出 MODEL_PATH，供下面 from_pretrained() 使用。
RUNTIME_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUNTIME_DIR))
from runtime_env import MODEL_PATH, MODEL_REVISION


import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


# MODEL_PATH 可能有两种形态：
#   1. Hugging Face 模型 ID，例如 "openvla/openvla-7b"；
#   2. 本地模型目录，例如 "/data/models/openvla-7b"。
# 具体取决于 env.sh 里的 OPENVLA_MODEL_ID，以及 runtime_env.py 的处理逻辑。
MODEL_ID = str(MODEL_PATH)

# 推理设备。
# Thor/T5000 正常应使用 cuda:0；如果 CUDA 不可用，则回退到 CPU。
# CPU 路径主要用于调试脚本逻辑，不适合真实性能评测。
DEVICE = os.getenv("OPENVLA_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")

# Attention 后端。
# sdpa 是 PyTorch 内置 scaled dot product attention，兼容性最好。
# 如果环境安装并支持 flash_attention_2，可以通过环境变量切换：
#   export OPENVLA_ATTN_IMPLEMENTATION=flash_attention_2
ATTN_IMPLEMENTATION = os.getenv("OPENVLA_ATTN_IMPLEMENTATION", "sdpa")

# 动作反归一化 key。
# OpenVLA 先生成归一化动作 token，再根据训练数据集统计量还原成真实动作量。
# bridge_orig 对应 BridgeData/WidowX 常用统计；如果换机器人或微调模型，需要确认这个 key 是否匹配。
UNNORM_KEY = os.getenv("OPENVLA_UNNORM_KEY", "bridge_orig")

# 记录测试时的功耗模式，例如 MAX-N。
# 这个脚本只记录，不主动切换功耗模式；切换功耗通常要在脚本外通过 nvpmodel/jetson_clocks 完成。
POWER_MODE = os.getenv("THOR_POWER_MODE", "unknown")


class TokenTimingStreamer:
    """用于测量 model.generate() 的 token 级时间。

    Hugging Face generate() 支持 streamer 回调。模型生成 token 时会调用 streamer.put()。
    这里利用这个机制记录：
      - first_token_time：第一个新 token 出来的时间；
      - last_token_time：最后一个新 token 出来的时间；
      - token_count：生成的新 token 数量。

    注意：这不是 OpenVLA 控制链路的主指标。
    机器人控制主要走 predict_action()，这个类只在 --measure-generate 打开时用于补充分析。
    """

    def __init__(self) -> None:
        # 第一个新 token 产生的时间，用于计算 TTFT/time-to-first-token。
        self.first_token_time: float | None = None
        # 最后一个新 token 产生的时间，当前代码主要保留这个字段用于后续扩展。
        self.last_token_time: float | None = None
        # generate() 实际生成的新 token 数量。
        self.token_count = 0
        # generate() 的 streamer 第一次 put() 往往会收到 prompt/input 本身，不是新生成 token。
        # 这里用 _seen_prompt 跳过第一次回调，避免把 prompt 当成 decode token 统计。
        self._seen_prompt = False

    def put(self, value: Any) -> None:
        # perf_counter() 适合做短时间间隔测量，比 time.time() 更适合 benchmark。
        now = time.perf_counter()

        # 第一次回调跳过：通常是 prompt/input，不代表模型 decode 出来的新 token。
        if not self._seen_prompt:
            self._seen_prompt = True
            return

        # value 可能是一个 tensor，也可能是单个 token 值；这里尽量兼容两种形式。
        token_count = int(value.numel()) if hasattr(value, "numel") else 1

        # 第一次看到新 token 时记录 first_token_time，后面不再覆盖。
        if self.first_token_time is None:
            self.first_token_time = now

        self.last_token_time = now
        self.token_count += token_count

    def end(self) -> None:
        # Hugging Face streamer 需要提供 end() 方法。
        # 当前 benchmark 不需要在结束时做额外处理，所以留空。
        return


def dtype() -> torch.dtype:
    """返回模型推理使用的数据类型。"""

    # GPU 上使用 bfloat16：相比 float32 更省显存、通常更快；相比 float16 数值范围更宽。
    # OpenVLA 官方示例也倾向在 CUDA 上使用 bfloat16。
    if DEVICE.startswith("cuda"):
        return torch.bfloat16

    # CPU 上很多算子对 bfloat16 支持不如 GPU 完整，调试时用 float32 更稳。
    return torch.float32


def sync() -> None:
    """在 CUDA 设备上同步，保证计时覆盖真实 GPU 执行时间。"""

    # PyTorch CUDA kernel 默认异步提交：Python 代码返回时 GPU 可能还没真正算完。
    # 如果不 synchronize()，测到的时间会偏小，尤其是推理阶段会非常不准。
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


def now_iso() -> str:
    """返回 UTC ISO 时间戳，用于 JSONL 每条记录。"""

    # 使用 UTC 方便和系统日志、tegrastats、nvidia-smi 采样做时间对齐。
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], pct: float) -> float:
    """计算简单百分位数，用于 summary 里的 p50/p90/p95/p99。"""

    if not values:
        return 0.0

    # 这里使用最简单的排序+取近似下标方式，不做插值。
    # 对 benchmark 报告来说，重点是快速判断尾延迟水平，足够使用。
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """把所有迭代的 JSONL row 汇总成 summary JSON。"""

    # 先写入本次测试的固定元数据。
    # 这些字段不是性能指标，但写报告时需要说明测试配置。
    metrics: dict[str, Any] = {
        "count": len(rows),
        "model_id": MODEL_ID,
        "device": DEVICE,
        "attention": ATTN_IMPLEMENTATION,
        "power_mode": POWER_MODE,
    }

    # 自动收集所有以 _ms 结尾的数值字段。
    # 好处是：如果后面新增 latency 字段，只要字段名以 _ms 结尾，就会自动进入 summary。
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key.endswith("_ms")
        }
    )

    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row]
        metrics[key] = {
            # mean：平均值，适合描述整体性能，但容易被少数 spike 拉高或拉低。
            "mean": statistics.mean(values),
            # p50：中位数，一半请求低于这个耗时，代表常态水平。
            "p50": percentile(values, 50),
            # p90/p95/p99：尾延迟。实时机器人控制更应关注这些值。
            "p90": percentile(values, 90),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            # min/max：最快和最慢一次。max 能帮助发现偶发卡顿。
            "min": min(values),
            "max": max(values),
        }

    if DEVICE.startswith("cuda"):
        # allocated：PyTorch 当前/历史实际分配给 tensor 的显存峰值。
        metrics["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        # reserved：PyTorch CUDA caching allocator 向系统保留的显存峰值。
        # reserved 通常大于 allocated，不代表模型真的一直用掉这么多 tensor 内存。
        metrics["max_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024 / 1024

    return metrics


def load_image(path: str | None) -> Image.Image:
    """加载输入图片；如果没给图片，就创建一张 224x224 灰色图。"""

    if path:
        # convert("RGB") 保证输入是三通道 RGB，避免 PNG alpha、灰度图等格式影响 processor。
        return Image.open(path).convert("RGB")

    # 灰色图用于冒烟测试：可以验证模型加载、processor、predict_action 链路是否通。
    # 但正式报告建议使用真实相机帧或 test_data 中的 BridgeData 样本图。
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


def prompt_for(instruction: str) -> str:
    """把自然语言任务指令包装成 OpenVLA 官方推理 prompt。"""

    # OpenVLA 训练/推理约定的格式大致是：
    #   In: What action should the robot take to <instruction>?
    #   Out:
    # 模型看到图像和这个 prompt 后，会在 Out: 后生成动作 token。
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def timed_predict_action(model: Any, processor: Any, image: Image.Image, instruction: str) -> dict[str, Any]:
    """执行一次 OpenVLA 动作预测，并返回分阶段耗时。"""

    # total_start 到 total_end 覆盖模型侧完整链路：processor -> H2D -> predict_action。
    # 它不包含相机采集、网络传输、机器人控制器执行等外部系统耗时。
    total_start = time.perf_counter()

    # 1) Processor 阶段。
    # processor 会做两类事情：
    #   - 文本：prompt tokenization，生成 input_ids/attention_mask；
    #   - 图像：resize/crop/normalize，生成 pixel_values。
    # 这部分通常在 CPU 上执行，可能受 PIL、tokenizer、CPU 性能影响。
    processor_start = time.perf_counter()
    inputs = processor(prompt_for(instruction), image)
    processor_end = time.perf_counter()

    # 2) Host-to-device 阶段。
    # 这里把 processor 产生的 tensor 移到 GPU，并转换成 dtype() 指定的精度。
    # 如果这项很高，可能说明 CPU/GPU 数据搬运、pin memory 或输入尺寸处理有瓶颈。
    h2d_start = time.perf_counter()
    inputs = inputs.to(DEVICE, dtype=dtype())
    sync()
    h2d_end = time.perf_counter()

    # 3) 模型动作预测阶段。
    # predict_action 内部会：
    #   - 使用视觉编码器处理图像；
    #   - 把视觉特征投影到语言模型空间；
    #   - 自回归生成 action token；
    #   - 将 action token 解码为归一化动作；
    #   - 根据 UNNORM_KEY 做反归一化，输出连续动作向量。
    infer_start = time.perf_counter()
    with torch.inference_mode():
        action = model.predict_action(
            **inputs,
            unnorm_key=UNNORM_KEY,
            do_sample=False,
        )
    sync()
    infer_end = time.perf_counter()

    total_end = time.perf_counter()
    return {
        # 图像+文本预处理耗时，单位 ms。
        # 报告里可作为 I/O preprocessing latency。
        "processor_time_ms": (processor_end - processor_start) * 1000.0,
        # CPU/host 到 GPU/device 的输入搬运耗时，单位 ms。
        "h2d_time_ms": (h2d_end - h2d_start) * 1000.0,
        # predict_action 本身耗时，单位 ms。
        # 这是评估 OpenVLA 模型推理速度的核心指标。
        "predict_action_total_time_ms": (infer_end - infer_start) * 1000.0,
        # 端到端模型侧耗时，单位 ms。
        # 近似等于 processor_time_ms + h2d_time_ms + predict_action_total_time_ms，
        # 但由于 Python 调度和计时代码开销，可能有极小差异。
        "model_e2e_time_ms": (total_end - total_start) * 1000.0,
        # 动作输出预览。
        # OpenVLA 常见输出是 7 维连续动作：[x, y, z, roll, pitch, yaw, gripper]。
        # 这里只截取字符串前 160 个字符，避免日志过长。
        "action_preview": str(action)[:160],
    }


def timed_generate_tokens(
    model: Any,
    processor: Any,
    image: Image.Image,
    instruction: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """可选的 generate() token 级性能测试。"""

    # 这个函数不走 predict_action 的反归一化逻辑，而是直接调用底层 generate()。
    # 它适合分析 LLM/VLM 的 prefill/decode 性能，但不能直接代表机器人 action 输出质量。
    inputs = processor(prompt_for(instruction), image).to(DEVICE, dtype=dtype())
    streamer = TokenTimingStreamer()

    sync()
    start = time.perf_counter()
    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            streamer=streamer,
        )
    sync()
    end = time.perf_counter()

    total_ms = (end - start) * 1000.0

    if streamer.first_token_time is None:
        # 如果没有捕获到新 token，只返回总耗时和 token 数。
        # 这种情况可能由模型 generate 行为、streamer 兼容性或 max_new_tokens 设置导致。
        return {
            "generate_total_time_ms": total_ms,
            "generate_token_count": streamer.token_count,
        }

    # TTFT/time-to-first-token：从 generate 开始到第一个新 token 出来的时间。
    # 在 LLM 性能分析里，它通常近似 prefill 阶段开销。
    ttft_ms = (streamer.first_token_time - start) * 1000.0
    decode_total_ms = total_ms - ttft_ms
    # TTFT already includes the first generated token. Only subsequent tokens
    # belong in the average decode-token denominator.
    decode_token_count = max(streamer.token_count - 1, 1)

    return {
        # generate() 整体耗时。
        "generate_total_time_ms": total_ms,
        # 首 token 时间，也写成 prefill_total，便于报告按 prefill/decode 拆分。
        "generate_ttft_ms": ttft_ms,
        "generate_prefill_total_time_ms": ttft_ms,
        # 除首 token 前等待外，剩余 decode 阶段总耗时。
        "generate_decode_total_time_ms": decode_total_ms,
        # streamer 捕获的新 token 数。
        "generate_token_count": streamer.token_count,
        # decode 阶段平均每 token 耗时。
        "generate_decode_token_count": max(streamer.token_count - 1, 0),
        "generate_time_per_decode_token_ms": decode_total_ms / decode_token_count,
    }


def main() -> None:
    """命令行入口：加载模型、执行 warmup、跑正式 benchmark、写结果文件。"""

    parser = argparse.ArgumentParser(description="Benchmark OpenVLA predict_action latency on NVIDIA Thor/T5000.")
    parser.add_argument(
        "--image",
        default="/workspace/openvla/test_data/bridge_sample_0001.jpg",
        help=(
            "评测图片路径。app 入口默认 None，会生成灰色图；"
            "tools 入口默认使用 test_data/bridge_sample_0001.jpg。"
        ),
    )
    parser.add_argument(
        "--instruction",
        default="pick up the blue object",
        help="输入给 OpenVLA 的自然语言任务指令。",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="正式统计前预热次数。预热结果不会写入 JSONL。",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="正式统计迭代次数。每次迭代写入 JSONL 一行。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="只在 --measure-generate 时使用；默认取 model.get_action_dim(unnorm_key)。",
    )
    parser.add_argument(
        "--measure-generate",
        action="store_true",
        help="额外测量 model.generate() 的 token 级耗时。默认关闭。",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="自定义 JSONL 输出路径。默认写到 $OPENVLA_PREFIX/logs。",
    )
    args = parser.parse_args()

    # 日志目录默认是 $OPENVLA_PREFIX/logs。
    # OPENVLA_PREFIX 来自 env.sh，当前通常是 /workspace/openvla。
    # 默认输出到 /workspace/outputs/openvla,可用 OPENVLA_LOGS_DIR 覆盖
    output_dir = Path(os.getenv("OPENVLA_LOGS_DIR", "/workspace/outputs/openvla"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSONL 文件保存每轮明细；summary JSON 保存统计汇总。
    output_path = Path(args.output) if args.output else output_dir / f"openvla_thor_benchmark_{time.strftime('%Y_%m%d_%H%M%S')}.jsonl"
    summary_path = output_path.with_suffix(".summary.json")

    # 打印测试环境信息，方便终端记录和报告截图。
    # 注意：这里打印的 power_mode 只来自环境变量 THOR_POWER_MODE，脚本不会自动探测 nvpmodel。
    print("model:", MODEL_ID)
    print("device:", DEVICE)
    print("attention:", ATTN_IMPLEMENTATION)
    print("power_mode:", POWER_MODE)
    print("output:", output_path)
    print("torch:", torch.__version__)
    print("cuda runtime:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
        # 重置 PyTorch 显存峰值统计，确保 summary 中的峰值主要来自本次 benchmark。
        torch.cuda.reset_peak_memory_stats()

    # 加载输入图片和模型。
    # trust_remote_code=True 是 OpenVLA Hugging Face 导出模型需要的，因为它依赖自定义 Prismatic 类。
    # local_files_only=True 强制只用本地缓存，避免 benchmark 过程中联网下载导致耗时污染。
    image = load_image(args.image)
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        revision=MODEL_REVISION,
        attn_implementation=ATTN_IMPLEMENTATION,
        torch_dtype=dtype(),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).to(DEVICE)

    # 切换到 eval 模式，关闭 dropout 等训练行为。
    model.eval()
    generate_tokens = args.max_new_tokens or int(model.get_action_dim(UNNORM_KEY))

    # Warmup 阶段。
    # 第一次运行通常包含 CUDA kernel 初始化、内存分配、算子 cache 等冷启动开销。
    # 如果把这些统计进正式结果，会让 p95/p99 和 mean 被冷启动污染。
    for index in range(args.warmup):
        timed_predict_action(model, processor, image, args.instruction)
        print(f"warmup {index + 1}/{args.warmup}")

    # 正式测试阶段。
    # rows 保存在内存里用于最后 summary；同时每轮立刻 flush 到 JSONL，避免长时间测试中断后数据丢失。
    rows: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as log_file:
        for index in range(args.iters):
            # 每一轮 row 都是一条完整 JSONL 记录。
            # 先写元数据，再用 timed_predict_action() 追加性能指标。
            row = {
                "ts": now_iso(),
                "iter": index,
                "model_id": MODEL_ID,
                "device": DEVICE,
                "attention": ATTN_IMPLEMENTATION,
                "power_mode": POWER_MODE,
                "instruction": args.instruction,
            }

            # 主指标：一次 OpenVLA 动作预测的分阶段耗时。
            row.update(timed_predict_action(model, processor, image, args.instruction))

            # 可选指标：底层 generate() token 级耗时。
            # 如果 generate() 路径不兼容或报错，记录错误字符串，但不影响 predict_action 主结果。
            if args.measure_generate:
                try:
                    row.update(
                        timed_generate_tokens(
                            model,
                            processor,
                            image,
                            args.instruction,
                            generate_tokens,
                        )
                    )
                except Exception as exc:
                    row["generate_timing_error"] = repr(exc)

            rows.append(row)

            # JSONL 是 line-delimited JSON：每行一个独立 JSON object。
            # ensure_ascii=True 会把中文等字符转义，保证日志在各种终端/工具里稳定可读。
            log_file.write(json.dumps(row, ensure_ascii=True) + "\n")
            log_file.flush()

            # 终端进度只打印两个最关键指标，避免输出太吵。
            print(
                f"iter {index + 1}/{args.iters}: "
                f"predict_action={row['predict_action_total_time_ms']:.2f} ms, "
                f"e2e={row['model_e2e_time_ms']:.2f} ms"
            )

    # 生成 summary JSON。
    # 报告建议重点引用：
    #   model_e2e_time_ms.mean / p95 / p99
    #   predict_action_total_time_ms.mean / p95 / p99
    #   max_memory_allocated_mb / max_memory_reserved_mb
    summary = summarize(rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
