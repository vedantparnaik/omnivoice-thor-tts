#!/usr/bin/env bash
# Build the native C++ TensorRT vocoder runner.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA=/usr/local/cuda-13.0/targets/sbsa-linux

g++ -O2 -std=c++17 "$HERE/vocoder_trt.cpp" -o "$HERE/vocoder_trt" \
    -I/usr/include/aarch64-linux-gnu \
    -I"$CUDA/include" \
    -L/lib/aarch64-linux-gnu -L"$CUDA/lib" \
    -lnvinfer -lcudart

echo "built $HERE/vocoder_trt"
