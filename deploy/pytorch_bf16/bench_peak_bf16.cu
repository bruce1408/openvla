// BF16 Tensor Core peak via cuBLASLt heuristic search (Thor sm_110).
//
// Why not bench_peak_tensorcore.py?
//   That file times torch.mm, which picks ONE cuBLAS kernel. On this box that
//   is ~110-160 TFLOPS. Datasheet dense BF16 is ~228 (120W) / ~259 (MAXN).
//   This program asks cuBLASLt for many algorithms, times each, keeps the best.
//
// FLOPs = 2*M*N*K.  Inputs/outputs are CUDA_R_16BF, accumulate FP32
// (CUBLAS_COMPUTE_32F) — the usual BF16 Tensor Core path.
//
//   nvcc -O3 -std=c++17 -arch=sm_110 bench_peak_bf16.cu -lcublasLt -o bench_peak_bf16
//   ./bench_peak_bf16
//
// Or: python bench_peak_bf16.py   (compiles if needed, then runs)

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK_CUDA(x)                                                          \
  do {                                                                         \
    cudaError_t err = (x);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__,                   \
              cudaGetErrorString(err));                                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CHECK_LT(x)                                                            \
  do {                                                                         \
    cublasStatus_t st = (x);                                                   \
    if (st != CUBLAS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "cuBLASLt %s:%d status=%d\n", __FILE__, __LINE__,        \
              (int)st);                                                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct Shape {
  int m, n, k;
};

static double tflops(int m, int n, int k, double ms) {
  return (2.0 * (double)m * n * k) / (ms * 1e-3) / 1e12;
}

static const char *compute_name(cublasComputeType_t c) {
  switch (c) {
  case CUBLAS_COMPUTE_32F:
    return "COMPUTE_32F (BF16 in, FP32 acc)";
  case CUBLAS_COMPUTE_16F:
    return "COMPUTE_16F (FP16 acc)";
  default:
    return "other";
  }
}

// Time one already-configured matmul. Returns ms/iter.
static float time_algo(cublasLtHandle_t lt, cublasLtMatmulDesc_t op,
                       const void *A, cublasLtMatrixLayout_t Adesc,
                       const void *B, cublasLtMatrixLayout_t Bdesc, void *C,
                       cublasLtMatrixLayout_t Cdesc, const float *alpha,
                       const float *beta, const cublasLtMatmulAlgo_t *algo,
                       void *ws, size_t ws_bytes, cudaStream_t stream,
                       int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    CHECK_LT(cublasLtMatmul(lt, op, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc,
                            C, Cdesc, algo, ws, ws_bytes, stream));
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) {
    CHECK_LT(cublasLtMatmul(lt, op, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc,
                            C, Cdesc, algo, ws, ws_bytes, stream));
  }
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return ms / (float)iters;
}

struct Best {
  double tf = 0.0;
  float ms = 0.f;
  int m = 0, n = 0, k = 0;
  const char *compute = "";
  int algo_index = -1;
};

static Best bench_one(cublasLtHandle_t lt, cudaStream_t stream, int m, int n,
                      int k, cublasComputeType_t compute, int warmup,
                      int iters, size_t ws_bytes, void *ws) {
  Best best;
  best.m = m;
  best.n = n;
  best.k = k;
  best.compute = compute_name(compute);

  nv_bfloat16 *A = nullptr, *B = nullptr, *C = nullptr;
  CHECK_CUDA(cudaMalloc(&A, (size_t)m * k * sizeof(nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&B, (size_t)k * n * sizeof(nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&C, (size_t)m * n * sizeof(nv_bfloat16)));
  CHECK_CUDA(cudaMemset(A, 1, (size_t)m * k * sizeof(nv_bfloat16)));
  CHECK_CUDA(cudaMemset(B, 1, (size_t)k * n * sizeof(nv_bfloat16)));
  CHECK_CUDA(cudaMemset(C, 0, (size_t)m * n * sizeof(nv_bfloat16)));

  cublasLtMatmulDesc_t op = nullptr;
  CHECK_LT(cublasLtMatmulDescCreate(&op, compute, CUDA_R_32F));
  cublasOperation_t transN = CUBLAS_OP_N;
  CHECK_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA,
                                          &transN, sizeof(transN)));
  CHECK_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB,
                                          &transN, sizeof(transN)));

  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
  // Column-major: A is m x k lda=m, B is k x n ldb=k, C is m x n ldc=m.
  CHECK_LT(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16BF, m, k, m));
  CHECK_LT(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16BF, k, n, k));
  CHECK_LT(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m, n, m));

  cublasLtMatmulPreference_t pref = nullptr;
  CHECK_LT(cublasLtMatmulPreferenceCreate(&pref));
  CHECK_LT(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes,
      sizeof(ws_bytes)));

  const int kMaxAlgo = 32;
  cublasLtMatmulHeuristicResult_t heur[kMaxAlgo];
  int n_algo = 0;
  cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
      lt, op, Adesc, Bdesc, Cdesc, Cdesc, pref, kMaxAlgo, heur, &n_algo);
  if (hs != CUBLAS_STATUS_SUCCESS || n_algo == 0) {
    fprintf(stderr, "  heuristic failed m=%d n=%d k=%d status=%d n=%d\n", m, n,
            k, (int)hs, n_algo);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(op);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return best;
  }

  float alpha = 1.f, beta = 0.f;
  for (int i = 0; i < n_algo; ++i) {
    if (heur[i].state != CUBLAS_STATUS_SUCCESS)
      continue;
    if (heur[i].workspaceSize > ws_bytes)
      continue;
    float ms = time_algo(lt, op, A, Adesc, B, Bdesc, C, Cdesc, &alpha, &beta,
                         &heur[i].algo, ws, heur[i].workspaceSize, stream,
                         warmup, iters);
    double tf = tflops(m, n, k, ms);
    if (tf > best.tf) {
      best.tf = tf;
      best.ms = ms;
      best.algo_index = i;
    }
  }

  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(Adesc);
  cublasLtMatrixLayoutDestroy(Bdesc);
  cublasLtMatrixLayoutDestroy(Cdesc);
  cublasLtMatmulDescDestroy(op);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  return best;
}

int main(int argc, char **argv) {
  int warmup = 10;
  int iters = 40;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--iters") && i + 1 < argc)
      iters = std::atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warmup") && i + 1 < argc)
      warmup = std::atoi(argv[++i]);
  }

  int dev = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  printf("gpu: %s  cc %d.%d  SMs %d\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);
  printf("BF16 peak search: cuBLASLt heuristic, CUDA_R_16BF, C = A * B\n");
  printf("datasheet T5000 120W dense BF16 ~228 TFLOPS | MAXN ~259 TFLOPS\n");
  printf("FLOPs = 2*M*N*K   warmup=%d iters=%d\n\n", warmup, iters);
  fflush(stdout);

  cublasLtHandle_t lt = nullptr;
  CHECK_LT(cublasLtCreate(&lt));
  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreate(&stream));

  const size_t ws_bytes = 64ull * 1024ull * 1024ull;
  void *ws = nullptr;
  CHECK_CUDA(cudaMalloc(&ws, ws_bytes));

  const std::vector<Shape> shapes = {
      {2048, 2048, 2048}, {2048, 4096, 4096}, {2560, 2560, 8192},
      {3072, 3072, 8192}, {4096, 4096, 4096}, {4096, 4096, 8192},
      {4096, 8192, 8192}, {5120, 5120, 5120}, {6144, 6144, 6144},
      {8192, 4096, 8192}, {8192, 8192, 4096}, {8192, 8192, 8192},
      {1024, 8192, 8192}, {2048, 2048, 8192}, {7168, 7168, 4096},
  };

  // BF16 Tensor Core with FP32 accumulate. COMPUTE_16F has no heuristic on Thor.
  const cublasComputeType_t computes[] = {CUBLAS_COMPUTE_32F};

  Best global, global_long;
  printf("%-36s %5s %5s %5s %8s %8s %s\n", "compute", "M", "N", "K", "ms",
         "TFLOPS", "algo#");
  for (cublasComputeType_t cmp : computes) {
    for (Shape s : shapes) {
      Best b = bench_one(lt, stream, s.m, s.n, s.k, cmp, warmup, iters,
                         ws_bytes, ws);
      if (b.tf <= 0.0)
        continue;
      printf("%-36s %5d %5d %5d %8.3f %8.1f %d\n", b.compute, b.m, b.n, b.k,
             b.ms, b.tf, b.algo_index);
      fflush(stdout);
      if (b.tf > global.tf)
        global = b;
      if (b.ms >= 0.20f && b.tf > global_long.tf)
        global_long = b;
      CHECK_CUDA(cudaDeviceSynchronize());
    }
  }

  printf("\n--> best dense BF16 (cuBLASLt): %.1f TFLOPS  shape=%dx%dx%d  %s  "
         "algo#%d  %.3f ms\n",
         global.tf, global.m, global.n, global.k, global.compute,
         global.algo_index, global.ms);
  if (global_long.tf > 0.0) {
    printf("--> best with ms>=0.20 (less L2-inflated): %.1f TFLOPS  "
           "shape=%dx%dx%d  %.3f ms\n",
           global_long.tf, global_long.m, global_long.n, global_long.k,
           global_long.ms);
  }
  printf("    vs 120W spec 228:  %.0f%%\n", 100.0 * global.tf / 228.0);
  printf("    vs MAXN spec 259:  %.0f%%\n", 100.0 * global.tf / 259.0);
  printf("    256 TFLOPS is the MAXN paper number; this is the highest this\n");
  printf("    library will give you for dense BF16 Tensor Core GEMM.\n");

  cudaFree(ws);
  cudaStreamDestroy(stream);
  cublasLtDestroy(lt);
  return 0;
}
