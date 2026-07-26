#!/usr/bin/env python3
# =============================================================================
# 01b_quantize_vision_fp8.py - 把 vision ONNX 从 FP16 量化到 FP8 (PTQ + 真实图片校准)
# =============================================================================
#
# 目标:
#   对 01_export_vision_projector_onnx.py 导出的 FP16 vision ONNX
#   (DINOv2 + SigLIP + projector) 做训练后量化 (PTQ) 到 FP8。
#   用真实图片经 OpenVLA processor 预处理后的 pixel_values 做校准,
#   在图中插入 FP8 的 Q/DQ 节点,产出可被 TensorRT --fp8 编译的量化 ONNX。
#
# 前置依赖 (已安装):
#   nvidia-modelopt[onnx], onnx_graphsurgeon, onnxruntime (CPU 校准即可)
#
# 流程位置:
#   01 (导出 FP16 ONNX) -> [本脚本: 量化成 FP8 ONNX] -> 05 (trtexec --fp8 构建 engine)
#
# 用法:
#   python 01b_quantize_vision_fp8.py                       # 用 test_data 全部图片校准
#   python 01b_quantize_vision_fp8.py --num-calib 32        # 只用前 32 张
#   python 01b_quantize_vision_fp8.py --calibration-method max
#
# 产物:
#   artifacts/onnx/vision_projector_fp8.onnx(+ .onnx.data)   量化后的 QDQ ONNX
#   artifacts/onnx/vision_projector_fp8.manifest.json         量化元数据
#
# 之后构建 FP8 engine:
#   bash 05_build_vision_engine.sh \
#       artifacts/onnx/vision_projector_fp8.onnx \
#       artifacts/engines/vision_projector_fp8.plan
#   (05 需带 --fp8;见脚本末尾提示)
# =============================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# 必须在 transformers 之前(runtime_env 会设好 HF 缓存路径)
from deploy.tensorrt.common import (  # noqa: E402
    load_openvla,
    move_batch_to_device,
    prompt_for,
    tensor_to_numpy,
    torch_dtype,
    write_json,
)

ARTIFACTS = REPO_ROOT / "deploy/tensorrt/artifacts"


def build_calibration_data(
    image_paths: list[Path],
    instruction: str,
    device: str,
    num_calib: int,
) -> np.ndarray:
    """把真实图片经 OpenVLA processor 预处理成 pixel_values,堆叠成校准数据。

    返回 shape [N, 6, 224, 224] 的 float16 numpy 数组,N=校准样本数。
    这是 vision ONNX 的唯一输入 (pixel_values),ModelOpt 用它统计激活分布来定标 FP8。
    """
    from PIL import Image

    processor, _model = load_openvla(device, "fp16")
    paths = image_paths[:num_calib]
    if not paths:
        raise SystemExit("没有可用的校准图片")

    samples = []
    for p in paths:
        image = Image.open(p).convert("RGB")
        inputs = processor(prompt_for(instruction), image, return_tensors="pt")
        inputs = move_batch_to_device(inputs, device, torch_dtype("fp16"))
        pv = tensor_to_numpy(inputs["pixel_values"])  # [1, 6, 224, 224]
        samples.append(pv.astype(np.float16))
    data = np.concatenate(samples, axis=0)  # [N, 6, 224, 224]
    print(f"校准数据: {data.shape} (dtype={data.dtype}, 来自 {len(paths)} 张图片)")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision ONNX FP16 -> FP8 量化 (PTQ)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--instruction", default="pick up the blue object")
    parser.add_argument("--num-calib", type=int, default=64, help="校准样本数 (32-100 通常足够)")
    parser.add_argument(
        "--calibration-method",
        choices=("entropy", "max"),
        default="max",
        help="FP8 校准方法; vision 特征建议 max (对离群值更稳),entropy 亦可试。",
    )
    parser.add_argument(
        "--calibration-eps",
        nargs="+",
        default=["cpu"],
        help="校准用的 onnxruntime EP。本机仅 CPU 版 onnxruntime,故默认 ['cpu']。",
    )
    parser.add_argument("--image-dir", type=Path, default=REPO_ROOT / "test_data")
    parser.add_argument(
        "--input-onnx",
        type=Path,
        default=ARTIFACTS / "onnx/vision_projector_fp16.onnx",
    )
    parser.add_argument(
        "--output-onnx",
        type=Path,
        default=ARTIFACTS / "onnx/vision_projector_fp8.onnx",
    )
    args = parser.parse_args()

    if not args.input_onnx.exists():
        raise SystemExit(
            f"找不到输入 ONNX: {args.input_onnx}\n请先运行 01_export_vision_projector_onnx.py"
        )

    # 收集校准图片
    image_paths = sorted(args.image_dir.glob("*.jpg")) + sorted(args.image_dir.glob("*.png"))
    calib_data = build_calibration_data(image_paths, args.instruction, args.device, args.num_calib)

    # ModelOpt 的输入名要与 ONNX 里的输入名一致 (01 导出时为 "pixel_values")
    calibration_dict = {"pixel_values": calib_data}

    # 延后 import,确保 runtime_env 已设好环境
    import modelopt.onnx.quantization as moq

    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)
    print(f"开始 FP8 量化: {args.input_onnx.name} -> {args.output_onnx.name}")
    print(f"  校准方法: {args.calibration_method} | 样本数: {calib_data.shape[0]}")

    moq.quantize(
        onnx_path=str(args.input_onnx),
        quantize_mode="fp8",
        calibration_data=calibration_dict,
        calibration_method=args.calibration_method,
        calibration_eps=args.calibration_eps,  # 本机 onnxruntime 仅 CPU EP;不指定会默认尝试 cuda/trt 而失败
        output_path=str(args.output_onnx),
        use_external_data_format=True,  # vision 权重大,需外部数据格式
    )

    manifest = {
        "precision": "fp8",
        "source_onnx": str(args.input_onnx),
        "quantize_mode": "fp8",
        "calibration_method": args.calibration_method,
        "num_calibration_samples": int(calib_data.shape[0]),
        "input_name": "pixel_values",
        "input_shape": list(calib_data.shape[1:]),
        "note": "PTQ FP8 QDQ ONNX; build engine with trtexec --fp8.",
    }
    write_json(args.output_onnx.with_suffix(".manifest.json"), manifest)
    print(f"\nFP8 量化 ONNX 已保存: {args.output_onnx}")
    print(manifest)
    print("\n下一步 (构建 FP8 engine):")
    print(
        "  bash 05_build_vision_engine.sh \\\n"
        f"      {args.output_onnx} \\\n"
        f"      {ARTIFACTS}/engines/vision_projector_fp8.plan"
    )
    print("  注意: 05 脚本需带 --fp8 (见脚本 FP8 提示)。")


if __name__ == "__main__":
    main()
