# OpenVLA TensorRT Phase 1

This package implements the first deployment gate for the `cdd_dev` branch:

1. Capture a golden trace from the real `predict_action()` path.
2. Export DINOv2 + SigLIP + Prismatic projector to fixed-shape ONNX.
3. Build the vision TensorRT engine on Jetson AGX Thor.
4. Replace Hugging Face `generate()` with explicit prefill + KV-cache decode.
5. Compare all generated action token IDs against the original runtime.

## 0. Copy into the repository

Copy the `deploy/` directory to the root of the OpenVLA repository.

```bash
cd /workspace/openvla
source /opt/openvla-venv/bin/activate
source env.sh
python -m pip install onnx
```

Add generated artifacts to `.gitignore`:

```gitignore
deploy/tensorrt/artifacts/
```

## 1. Golden trace

```bash
python deploy/tensorrt/export/00_dump_golden.py \
  --image test_data/YOUR_IMAGE.jpg \
  --instruction "pick up the object" \
  --unnorm-key bridge_orig
```

The trace includes:

- processor input IDs and image tensor;
- actual input IDs intercepted inside `predict_action()`;
- DINOv2 + SigLIP patch features;
- projected visual embeddings;
- actual multimodal embeddings passed to Llama;
- one last-token logits vector for every LLM call;
- generated action token IDs;
- normalized and unnormalized actions.

## 2. Validate manual decoding before TensorRT

This is the most important initial gate. It keeps the vision path in PyTorch but
removes Hugging Face `generate()`.

```bash
python deploy/tensorrt/runtime/hybrid_runtime.py \
  --vision-backend pytorch \
  --image test_data/YOUR_IMAGE.jpg \
  --instruction "pick up the object" \
  --unnorm-key bridge_orig \
  --warmup 3 \
  --iters 10
```

Required result:

```text
"tokens_exact_match": true
"action_max_abs_error": 0.0
```

Do not continue to TensorRT until this passes.

## 3. Export vision + projector

```bash
python deploy/tensorrt/export/01_export_vision_projector_onnx.py \
  --output deploy/tensorrt/artifacts/onnx/vision_projector_fp16.onnx \
  --dtype fp16 \
  --opset 17 \
  --check
```

Expected interface:

```text
pixel_values                 [1, 6, 224, 224] FP16
projected_patch_embeddings   [1, 256, 4096]   FP16
```

The script derives the actual input shape through the repository processor.
Treat the dimensions above as expected values, not assumptions.

If the combined export fails, retain the complete exception. The fallback is
to export the two timm featurizers and projector as three graphs; the likely
compatibility boundary is the monkey-patched `get_intermediate_layers()` call.

## 4. Build on Jetson AGX Thor

TensorRT engines must be built on the target Thor software/hardware stack.

```bash
chmod +x deploy/tensorrt/build/build_vision_engine.sh

deploy/tensorrt/build/build_vision_engine.sh \
  deploy/tensorrt/artifacts/onnx/vision_projector_fp16.onnx \
  deploy/tensorrt/artifacts/engines/vision_projector_fp16.plan
```

## 5. TensorRT vision + PyTorch Llama

```bash
python deploy/tensorrt/runtime/hybrid_runtime.py \
  --vision-backend tensorrt \
  --engine deploy/tensorrt/artifacts/engines/vision_projector_fp16.plan \
  --image test_data/YOUR_IMAGE.jpg \
  --instruction "pick up the object" \
  --unnorm-key bridge_orig \
  --warmup 5 \
  --iters 50
```

The first quality target is exact action-token equality. If tokens differ,
compare in this order:

1. `pixel_values.npy`
2. `projected_patch_embeddings.npy`
3. first row of `step_last_logits.npy`
4. first generated action token
5. later decode tokens

A first-token mismatch points to preprocessing, vision/projector output,
multimodal insertion, mask construction, or prefill. A later-token mismatch
points to KV-cache or decode handling.

## Notes

- Keep the original `modeling_prismatic.py` unchanged as the golden reference.
- Use fixed batch 1 and fixed image shape for the first TensorRT engine.
- The TensorRT runner uses PyTorch CUDA tensor pointers and
  `execute_async_v3()`, so it does not require PyCUDA.
- The engine output is cast to the LLM dtype before multimodal concatenation.
- Benchmarking synchronizes CUDA around the complete manual action prediction;
  it does not insert synchronizations between individual kernels.
