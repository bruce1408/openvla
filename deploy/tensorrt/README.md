# OpenVLA on Jetson AGX Thor: staged TensorRT deployment

This directory turns the deployment proposal into explicit correctness gates. It intentionally keeps
`prismatic/extern/hf/modeling_prismatic.py` unchanged as the PyTorch oracle.

## Review outcome

The high-level split is sound:

1. GPU-native preprocessing.
2. TensorRT for DINOv2 + SigLIP + Prismatic projector.
3. A KV-cache-aware LLM runtime.
4. A small deterministic action detokenizer.

Three constraints must be treated as gates, not assumptions:

- TensorRT Edge-LLM's current public matrix explicitly lists Llama 3.x, not Llama 2. The extracted OpenVLA
  Llama 2 checkpoint must first pass an exporter/build probe. A custom model registration may be required.
- A 256-row action-only LM head is equivalent only when decoding is constrained to OpenVLA's action-token
  interval and the reference model also selects a token in that interval. The golden dump records this fact.
- `n_action_bins=256` produces 256 action token symbols but only 255 bin centers in the current OpenVLA code.
  Do not generate the C++ lookup table from an assumed length.

## Produced code

```text
deploy/tensorrt/
├── common.py
├── export/
│   ├── 00_dump_golden.py
│   ├── 01_export_vision_projector_onnx.py
│   ├── 02_extract_llama_checkpoint.py
│   └── 03_export_action_meta.py
├── build/build_vision_engine.sh
├── runtime/
│   ├── trt_runner.py
│   ├── hybrid_runtime.py
│   └── cpp/action_decoder.{h,cpp}
├── validate/validate_vision_engine.py
└── tests/test_common.py
```

Generated models and logs live below `deploy/tensorrt/artifacts/` and are ignored by git.

## 0. Freeze the environment

Run on Thor from the repository root. Keep the JetPack-provided CUDA and TensorRT packages together.

```bash
source env.sh

cat /etc/nv_tegra_release
nvcc --version
trtexec --version || /usr/src/tensorrt/bin/trtexec --version

python - <<'PY'
import torch
import tensorrt as trt
print("torch", torch.__version__)
print("torch CUDA", torch.version.cuda)
print("TensorRT", trt.__version__)
print("GPU", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
PY
```

Record the power mode separately. Use the mode approved for the robot's thermal and power envelope; do not
compare benchmark runs made under different modes.

## 1. PyTorch baseline

The formal end-to-end number comes from `benchmark_openvla_thor.py`. The hook-heavy `eval_latency.py` is for
diagnosis because its synchronizations perturb natural asynchronous execution.

```bash
python tools/benchmark_openvla_thor.py \
  --image test_data/bridge_sample_0001.jpg \
  --instruction "pick up the blue object" \
  --warmup 10 \
  --iters 100 \
  --measure-generate

python tools/eval_latency.py \
  --image-dir test_data \
  --glob '*.jpg' \
  --limit 20 \
  --warmup 5
```

`--max-new-tokens` now defaults to `model.get_action_dim(OPENVLA_UNNORM_KEY)`. The per-decode-token metric also
excludes the first token, which is produced by prefill.

## 2. Gate 1: dump the golden tensors

```bash
python deploy/tensorrt/export/00_dump_golden.py \
  --image test_data/bridge_sample_0001.jpg \
  --instruction "pick up the blue object" \
  --unnorm-key bridge_orig \
  --dtype bf16
```

Inspect `deploy/tensorrt/artifacts/golden/sample_0001/metadata.json` before proceeding. For the standard model,
verify:

- `pixel_values` is `[1, 6, 224, 224]`;
- projected embeddings are `[1, 256, 4096]`;
- `empty_token_appended` is normally `false` for the standard prompt;
- `tokens_in_action_range` is `true`;
- the action dimension matches the selected normalization statistics.

Repeat the dump on a representative fixed validation set, not only one image, before quantization.

## 3. Export the fixed-shape vision graph

Export FP16 first:

```bash
python deploy/tensorrt/export/01_export_vision_projector_onnx.py \
  --mode combined \
  --opset 17
```

If export fails, isolate the failing branch:

```bash
python deploy/tensorrt/export/01_export_vision_projector_onnx.py \
  --mode split \
  --opset 17
```

Split mode writes separate DINOv2, SigLIP, and projector graphs under
`deploy/tensorrt/artifacts/onnx/split/`. This is especially useful because this branch monkey-patches TIMM
`forward()` with `get_intermediate_layers(..., n={second_to_last_layer})`.

Optional ONNX structural check:

```bash
python - <<'PY'
import onnx
p = 'deploy/tensorrt/artifacts/onnx/vision_projector_fp16.onnx'
m = onnx.load(p, load_external_data=True)
onnx.checker.check_model(m)
print('ONNX check passed')
PY
```

## 4. Build the vision engine on Thor

TensorRT plans are hardware/software-specific. Build the plan on the target Thor image rather than copying a
plan built on a different GPU or TensorRT release.

```bash
bash deploy/tensorrt/build/build_vision_engine.sh
```

The default command fixes the only supported input to `1x6x224x224`, enables FP16, and writes:

```text
deploy/tensorrt/artifacts/engines/vision_projector_fp16.plan
```

## 5. Gate 2: validate projected embeddings

```bash
python deploy/tensorrt/validate/validate_vision_engine.py \
  --engine deploy/tensorrt/artifacts/engines/vision_projector_fp16.plan \
  --golden-dir deploy/tensorrt/artifacts/golden/sample_0001 \
  --min-cosine 0.999
```

Cosine similarity is only an intermediate gate. Record mean/max absolute error as well, then continue to the
token gate.

## 6. Gates 3-5: hybrid runtime and exact token comparison

```bash
python deploy/tensorrt/runtime/hybrid_runtime.py \
  --engine deploy/tensorrt/artifacts/engines/vision_projector_fp16.plan \
  --image test_data/bridge_sample_0001.jpg \
  --instruction "pick up the blue object" \
  --unnorm-key bridge_orig \
  --llm-dtype bf16 \
  --compare-reference
```

This path does not call Hugging Face `generate()` for its candidate result. It explicitly performs:

```text
TRT projected embeddings
  -> BOS + 256 vision embeddings + remaining text embeddings
  -> PyTorch Llama prefill (first action token)
  -> KV-cache decode (remaining action tokens)
  -> deterministic q01/q99 unnormalization
```

Required FP16 acceptance criteria on the full validation set:

1. `tokens_in_action_range == true` for both paths.
2. First action token exact match.
3. All action token IDs exact match (target: 100%).
4. Continuous action max error recorded; exact token match should make detokenization identical.
5. No rollout-success regression on the real robot before replacing the reference runtime.

When token 1 differs, inspect preprocessing, vision/projector error, BOS insertion, attention mask, and prefill.
When token 1 matches and a later token differs, inspect the cache length, position offset, and decode input token.

## 7. Export the action sidecar

```bash
python deploy/tensorrt/export/03_export_action_meta.py \
  --unnorm-key bridge_orig
```

The JSON is the source of truth for a later FP32 C++ detokenizer. Do not hard-code `32000`, 256 bin centers, or
the action dimension in C++.

The C++17 decoder can be tested independently:

```bash
cmake -S deploy/tensorrt/runtime/cpp \
  -B deploy/tensorrt/runtime/cpp/build
cmake --build deploy/tensorrt/runtime/cpp/build -j
ctest --test-dir deploy/tensorrt/runtime/cpp/build --output-on-failure
```

## 8. Edge-LLM compatibility probe

Pin an Edge-LLM release/commit and keep its ONNX, builder, and runtime from that same revision. The current
public workflow is checkpoint -> `tensorrt-edgellm-export` -> `llm_build` -> C++ runtime.

Extract the fine-tuned language model, not base Llama 2:

```bash
python deploy/tensorrt/export/02_extract_llama_checkpoint.py \
  --device cpu
```

On the machine where TensorRT Edge-LLM's exporter is installed:

```bash
tensorrt-edgellm-export \
  deploy/tensorrt/artifacts/hf_llama \
  deploy/tensorrt/artifacts/edgellm_onnx
```

Treat this as a probe. If the default `CausalLM` cannot load/export the Llama 2 checkpoint, add a registered
model implementation and checkpoint mapping in Edge-LLM; do not silently substitute Llama 3 or base Llama 2.

After a successful export, build on Thor:

```bash
/path/to/TensorRT-Edge-LLM/build/examples/llm/llm_build \
  --onnxDir deploy/tensorrt/artifacts/edgellm_onnx/llm \
  --engineDir deploy/tensorrt/artifacts/engines/openvla_llama \
  --maxBatchSize 1 \
  --maxInputLen 512 \
  --maxKVCacheCapacity 512
```

`512` is sufficient only after measuring the actual text length plus 256 visual tokens and generation length.
Use `1024` while bringing the system up if there is any uncertainty.

The final Edge-LLM integration contract is not ordinary chat input. It must accept externally generated visual
embeddings at exactly the same sequence position as the hybrid runtime. Stabilize that contract before writing
the C++ multimodal adapter.

## 9. Optimization order

Apply one change per validation run:

1. Fixed-shape FP16 vision + BF16/FP16 LLM.
2. Manual prefill/decode and CUDA Graphs.
3. GPU camera preprocessing and device-resident buffers.
4. LLM FP8, followed by full token/rollout validation.
5. Vision FP8, followed by the same validation.
6. Action-only LM head only after proving constrained decode equivalence.
7. NVFP4 last; token agreement and rollout success are the release gate, not cosine similarity alone.

For production control, remove PIL, NumPy image copies, JSON transport, and Python from the camera-to-action
hot path only after the staged runtime is correct.
