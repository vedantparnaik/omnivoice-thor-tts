// Native TensorRT (NvInfer.h) runtime for the OmniVoice vocoder engine.
//
// Demonstrates the low-level edge-inference path the assignment asks for:
//   * NvInfer.h C++ bindings (deserialize + IExecutionContext)
//   * page-locked host memory via cudaHostAlloc (fast, async H2D/D2H)
//   * asynchronous execution on a dedicated cudaStream_t
//
// Usage: vocoder_trt <engine.plan> [codes_length] [iters]

#include "NvInfer.h"
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <string>

using namespace nvinfer1;

class Logger : public ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", msg);
  }
} gLogger;

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
  std::exit(1); } } while (0)

static size_t volume(const Dims& d) {
  size_t v = 1; for (int i = 0; i < d.nbDims; ++i) v *= d.d[i]; return v;
}

int main(int argc, char** argv) {
  const char* plan = argc > 1 ? argv[1] : "artifacts/vocoder_fp32.plan";
  int L    = argc > 2 ? std::atoi(argv[2]) : 200;   // codes length
  int iters= argc > 3 ? std::atoi(argv[3]) : 30;
  const int Q = 8;                                   // num quantizers
  const int codebook = 1024;

  // ---- load serialized engine ----
  std::ifstream f(plan, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", plan); return 1; }
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  printf("loaded plan %s (%zu bytes)\n", plan, blob.size());

  IRuntime* runtime = createInferRuntime(gLogger);
  ICudaEngine* engine = runtime->deserializeCudaEngine(blob.data(), blob.size());
  if (!engine) { fprintf(stderr, "deserialize failed\n"); return 1; }
  IExecutionContext* ctx = engine->createExecutionContext();

  const char* inName  = "audio_codes";
  const char* outName = "audio_values";

  // ---- set dynamic input shape (1, Q, L) ----
  Dims3 inShape(1, Q, L);
  if (!ctx->setInputShape(inName, inShape)) { fprintf(stderr, "setInputShape failed\n"); return 1; }
  Dims outShape = ctx->getTensorShape(outName);
  size_t inElems  = volume(inShape);
  size_t outElems = volume(outShape);
  printf("input (1,%d,%d) int64  ->  output samples=%zu\n", Q, L, outElems);

  // ---- page-locked (pinned) host buffers via cudaHostAlloc ----
  int64_t* hIn = nullptr;
  float*   hOut = nullptr;
  CUDA_CHECK(cudaHostAlloc((void**)&hIn,  inElems  * sizeof(int64_t), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc((void**)&hOut, outElems * sizeof(float),   cudaHostAllocDefault));

  std::mt19937 rng(0);
  std::uniform_int_distribution<int> dist(0, codebook - 1);
  for (size_t i = 0; i < inElems; ++i) hIn[i] = dist(rng);

  // ---- device buffers + stream ----
  void *dIn = nullptr, *dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dIn,  inElems  * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&dOut, outElems * sizeof(float)));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  ctx->setTensorAddress(inName,  dIn);
  ctx->setTensorAddress(outName, dOut);

  auto run_once = [&]() {
    // async pinned H2D -> enqueue -> async D2H, all on one stream
    CUDA_CHECK(cudaMemcpyAsync(dIn, hIn, inElems * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    if (!ctx->enqueueV3(stream)) { fprintf(stderr, "enqueueV3 failed\n"); std::exit(1); }
    CUDA_CHECK(cudaMemcpyAsync(hOut, dOut, outElems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  };

  // warmup
  for (int i = 0; i < 3; ++i) run_once();

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) run_once();
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

  // simple sanity on output
  double s = 0; float mx = 0;
  for (size_t i = 0; i < outElems; ++i) { s += hOut[i]; if (std::abs(hOut[i]) > mx) mx = std::abs(hOut[i]); }
  double audio_s = (double)outElems / 24000.0;
  printf("C++ NvInfer: %.2f ms/infer  (audio %.2fs, RTF %.4f)  out_absmax=%.3f mean=%.5f\n",
         ms, audio_s, (ms / 1000.0) / audio_s, mx, s / outElems);
  printf("CPP_TRT_OK\n");

  cudaFreeHost(hIn); cudaFreeHost(hOut);
  cudaFree(dIn); cudaFree(dOut);
  cudaStreamDestroy(stream);
  delete ctx; delete engine; delete runtime;
  return 0;
}
