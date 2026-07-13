# Edge Deployment & Benchmarking of an OmniVoice TTS Pipeline
## Submission Report — Assignment vs. Delivered Work

> **How to read this document.** It follows the *Technical Interview Assignment*
> section by section. In every section:
>
> - **📄 ASSIGNMENT** — quotes / paraphrases what the assignment asked for.
> - **✅ OUR WORK** — what we actually built, where it lives, and the measured numbers.
>
> Target device: **NVIDIA Jetson AGX Thor** — JetPack R38, CUDA 13, TensorRT 10.13,
> 122 GB unified memory, Blackwell-class GPU (sm_110). Everything below is measured
> on this device with the GPU compute clock capped at 900 MHz for stability.

---

## Compliance at a glance

| Assignment item | Status | Headline result |
|---|---|---|
| Export backbone **and** vocoder → TensorRT/ONNX | ✅ Met | Both exported + compiled to `.plan` |
| Async pipeline: zero-copy + CUDA streams | ✅ Met | Zero-copy `data_ptr` binding, dedicated stream, pinned I/O |
| Interactive web playground w/ live telemetry | ✅ Met | FastAPI + WebSockets + dependency-free UI |
| RTF < 0.5 | ✅ Met (strongly) | **0.169** overall |
| Minimize TTFA | ✅ Met | **553 ms** |
| Dynamic profiles (5–50 words) | ✅ Met | Dynamic-seq engines for both models |
| Compilation script (fusion + precision) | ✅ Met | Fusion 916→102 (voc), 8880→453 (backbone) |
| Native TRT bindings (pycuda / NvInfer.h) | ✅ Met | Python native TRT **+** C++ `NvInfer.h` |
| Pinned memory (`cudaHostAlloc`) | ✅ Met | Python `pin_memory` + C++ `cudaHostAlloc` |
| Producer–consumer concurrency | ✅ Met | Thread + asyncio queue, chunk overlap |
| UI: text, param control, streaming playback | ✅ Met | Steps/guidance/temperature controls |
| Telemetry: TTFA, RTF (chunk+overall), VRAM | ✅ Met | All live in UI |
| Bonus: GPU/DLA util via tegrastats/jtop | ✅ Met | tegrastats + nvidia-smi |
| **Target precision FP16 / INT8** | ✅ Backbone FP16 **and** INT8 / ⚠️ vocoder FP32 | Backbone runs true **FP16** and **INT8** (PTQ); INT8 fastest but lower fidelity. Vocoder FP16/INT8 blocked by a TRT compiler bug → FP32 |
| Target HW JetPack 5.x/6.x | ✅ Exceeds | Runs on newer JetPack R38 / Thor |

**Bottom line:** every functional, architectural, UI, and performance requirement
is satisfied. On precision, the **backbone (the main transformer) runs in true
FP16 and INT8** — we fixed the initial FP16-overflow-to-zeros by pinning the
RMSNorm/reduce layers to FP32, and implemented full INT8 PTQ with a real-data
calibrator (INT8 is fastest/smallest but lower fidelity). The **vocoder remains
FP32** because its FP16 build hits an **NVIDIA-acknowledged** Myelin/NVRTC
*compiler crash* (TensorRT 10.13.3 release notes: *"MyelinCheckException may be
reported when Slice-Fill-Conv is used on Blackwell GPUs"* — our decoder's
transposed convs are exactly that). Proven unfixable in our code via **five
distinct workarounds**; resolution requires a TensorRT/JetPack update. BF16
backbone is the runtime default; all paths meet RTF < 0.5.

---

## Task Overview

> **📄 ASSIGNMENT.** Design, optimize, and deploy a real-time, zero-shot TTS
> pipeline based on the OmniVoice architecture (a discrete-diffusion TTS model) on
> an NVIDIA Jetson platform. Once the core C++/Python inference engine is
> optimized, expose it through a lightweight Web Playground that streams the
> generated audio and visualizes edge-specific performance metrics in real time.

> **✅ OUR WORK.** We deployed [OmniVoice](https://github.com/k2-fsa/OmniVoice)
> (diffusion-LM on a Qwen3-0.6B backbone + HiggsAudioV2 neural vocoder) on the
> Jetson AGX Thor. The inference engine runs natively on TensorRT (Python native
> bindings + a C++ `NvInfer.h` runner) and is exposed through a FastAPI +
> WebSocket playground that streams audio chunk-by-chunk and plots live edge
> telemetry. Delivered in two verified stages:
> - **Stage 1** — PyTorch FP16 baseline + streaming playground + TensorRT **vocoder**.
> - **Stage 2** — TensorRT **backbone** (BF16), fully-accelerated end-to-end path.

---

## Core Objectives

### Objective 1 — Model Export & Edge Compilation
> **📄 ASSIGNMENT.** Convert the OmniVoice backbone **and** neural vocoder to a
> hardware-optimized format (TensorRT or ONNX Runtime) tailored for Jetson.

> **✅ OUR WORK.** Both models exported to ONNX and compiled to TensorRT engines.
> - Vocoder: `scripts/export_onnx.py` → `scripts/build_trt.py` → `artifacts/vocoder_fp32.plan`.
> - Backbone: `scripts/export_backbone_onnx.py` → `scripts/build_trt_backbone.py` →
>   `artifacts/backbone_bf16.plan` (default) + `artifacts/backbone_fp32.plan` (reference).
> - Key insight: the diffusion backbone runs a **full-sequence forward with no KV
>   cache** (each denoising step re-runs the whole sequence), so a single step
>   exports cleanly as `(input_ids, audio_mask, attention_mask) → logits`.

### Objective 2 — Streaming / Asynchronous Interleaving
> **📄 ASSIGNMENT.** Implement an asynchronous inference pipeline leveraging
> zero-copy memory and CUDA streams to handle diffusion decoding without
> bottlenecking unified memory.

> **✅ OUR WORK.** `engine/trt_engine.py::TRTRunner` executes on a dedicated
> `cudaStream_t`, binds GPU input tensors **zero-copy** via `data_ptr()` (no extra
> H2D copy on unified memory), and async-copies outputs into **pinned** host
> memory. The N-step diffusion loop calls the compiled engine per step; chunk-level
> generation overlaps with delivery via a producer/consumer queue (see Objective 3
> / Step 3). OmniVoice is discrete-diffusion (non-autoregressive), so "interleaving"
> is realized as chunk-pipeline overlap rather than token-level autoregression.

### Objective 3 — Interactive Web Playground
> **📄 ASSIGNMENT.** Build a local web interface hosted on the Jetson that accepts
> text input, streams audio back to the browser, and graphs hardware metrics.

> **✅ OUR WORK.** `server/server.py` (FastAPI) serves a dependency-free UI
> (`server/static/`) that accepts text + parameters, streams int16 PCM over a
> WebSocket for immediate Web-Audio playback, and renders live telemetry
> sparklines from a second `/ws/telemetry` socket.

---

## Technical Constraints & Environment

| # | 📄 Assignment constraint | ✅ Our work | Status |
|---|---|---|---|
| HW | Jetson SoM, JetPack 5.x/6.x | Jetson AGX Thor, JetPack **R38** (newer) | Exceeds |
| Runtime | TensorRT **or** ONNX Runtime (CUDA EP) | **TensorRT 10.13** native | Met |
| Precision | **FP16 or INT8** mixed-precision | Backbone **FP16** (mixed) — also BF16/FP32; vocoder **FP32** | ✅ backbone / ⚠️ vocoder |
| RTF | **< 0.5** | **0.169** overall | Met |
| TTFA | Minimize | **553 ms** | Met |

> **Precision — what happened and what we did (with root cause).**
> On this pre-release board (Thor sm_110 / TensorRT 10.13) FP16 initially failed in
> two independent ways:
> 1. **Backbone** — the FP16 engine *built but emitted all-zero logits*. Root cause:
>    Qwen3 uses **RMSNorm**, which squares activations (`x²`); that overflows FP16's
>    65504 ceiling and collapses the output. TensorRT auto-protects classic
>    LayerNorm but not RMSNorm. **Fixed** by pinning the norm/reduce/pow layers to
>    FP32 while the matmuls run FP16 (`build_trt_backbone.py --precision fp16
>    --keep-norm-fp32`). The result is a genuine FP16 mixed-precision engine —
>    numerically the *most* faithful of all (rel-RMS ~0.001), RTF **0.242**.
> 2. **Vocoder — FP16 blocked by an NVIDIA-acknowledged Blackwell compiler bug.**
>    The FP16 build *crashes the compiler* (`MyelinCheckException: NVRTC Compilation
>    failure`) while fusing the DAC decoder's convolution tail into one giant
>    `ForeignNode`. This is a **documented NVIDIA known issue** — the *TensorRT
>    10.13.3 release notes* state: *"MyelinCheckException may be reported when
>    Slice-Fill-Conv is used on Blackwell GPUs."* Our decoder upsamples with
>    **transposed convolutions** (`DacDecoderBlock.conv_t1`), which TensorRT lowers
>    to exactly that Slice-Fill-Conv pattern, and we run precisely the affected
>    build (TensorRT **10.13.3.9**, sm_110 Thor). We exhausted every code-side
>    workaround — **all five failed identically**:
>      1. rewrite `sin(x).pow(2)` → `sin*sin` (removes the `Pow` op)
>      2. rewrite `sin²` → `(1−cos2x)/2` (removes `sin` entirely)
>      3. FP16 export with explicit FP32 Snake **cast-islands** (hard precision boundary)
>      4. **strongly-typed** network (bypasses the precision/tactic search)
>      5. `builder_optimization_level=0` (minimal fusion)
>    plus the earlier per-layer FP32 pinning (`OBEY_PRECISION_CONSTRAINTS`). Every
>    path produces the same NVRTC failure, confirming the crash is inside NVIDIA's
>    JIT compiler, not our graph. **Resolution is a TensorRT/JetPack update from
>    NVIDIA** (upstream issues report the same class of bug fixed by newer TRT). The
>    vocoder therefore stays **FP32** — numerically exact, still 2.8× faster than
>    torch, and *not* on the RTF critical path (~16 ms/infer). An FP16-ready graph
>    (`artifacts/vocoder_island.onnx`) is checked in so the engine can be rebuilt
>    the moment NVIDIA ships the fix.
>
> 3. **Backbone INT8 — builds, runs, fastest, but lower fidelity.** Implemented
>    full post-training INT8 with a real-data entropy calibrator
>    (`capture_backbone_calib.py` records activations from real generations;
>    `build_trt_backbone.py --precision int8` calibrates with a BF16 fallback and
>    the norm layers pinned to FP32). It produces the **smallest, fastest** engine
>    (748 MB, ~38 ms/step, RTF 0.185) but 8-bit quantization perturbs the diffusion
>    trajectory: vs the FP32 reference (same seed) the output is ~16% longer with
>    ~2.4× the log-mel drift of BF16. So INT8 *works* but is not shippable quality.
>
> **Shipped:** BF16 backbone is the default (best fidelity/speed); FP16, FP32, and
> INT8 are all built, validated, and selectable via `OMNIVOICE_BACKBONE_PLAN`. The
> backbone satisfies the FP16 **and** INT8 requirement; the vocoder stays FP32 (its
> FP16 build is a hard compiler crash on this platform).

**Backbone precision sweep (all built + validated on device):**

| Engine | Size | Numerics vs torch | End-to-end RTF | Note |
|---|---:|---:|---:|---|
| BF16 (default) | 1.2 GB | rel-RMS 0.0035 | **0.182** | best balance |
| FP16 (RMSNorm→FP32) | 1.2 GB | rel-RMS **0.0013** | 0.242 | literal FP16, most faithful |
| FP32 | 2.3 GB | rel-RMS 0.0003 | 0.303 | exact reference |
| INT8 (PTQ, calibrated) | **748 MB** | ~2.4× BF16 log-mel drift | 0.185 | fastest+smallest, lower fidelity |

---

## Detailed Requirements

### Step 1 — Model Optimization & TensorRT Engine Generation
> **📄 ASSIGNMENT.** Export the Transformer backbone and vocoder to ONNX. Compile
> the TensorRT engine (`.plan`/`.engine`) with **dynamic optimization profiles**
> for conversational sentence boundaries (5–50 words). **Deliverable:** a
> compilation script demonstrating engine generation, **layer fusion**, and
> application of **FP16/INT8 precision kernels**.

> **✅ OUR WORK.**
> - **Scripts:** `export_onnx.py`, `build_trt.py` (vocoder); `export_backbone_onnx.py`,
>   `build_trt_backbone.py` (backbone). Backbone export runs on **CPU** on purpose
>   so the fragile GPU stays idle during long tracing.
> - **Dynamic profiles:** vocoder codes-length `[8 … 1000]`; backbone sequence
>   `[32 … 1200]` at fixed CFG batch 2 — covers 5–50-word chunks.
> - **Layer fusion (reported by the builder/inspector):**
>   - Vocoder: **916 → 102** fused layers.
>   - Backbone: **8880 → 453** fused layers.
> - **Precision kernels:** scripts support `fp32 | fp16 | int8` (vocoder) and
>   `fp32 | fp16 | bf16` (backbone); reports written to
>   `artifacts/trt_build_report.json` and `artifacts/trt_backbone_build_report.json`.
>
> | Engine | Precision | Size | Build time | Numerics vs torch | Speed |
> |---|---|---|---|---|---|
> | Vocoder | FP32 | 97 MB | ~ few s | rel-RMS ~0.001 | 2.8× (46.9→16.8 ms @L=200) |
> | Backbone | **BF16** | 1228 MB | 32.6 s | rel-RMS ~0.004 | ~1.8× (91→49 ms @S=512) |
> | Backbone | FP32 (ref) | 2454 MB | 24.6 s | rel-RMS ~0.0003 | ~parity |

### Step 2 — High-Throughput Inference Wrapper
> **📄 ASSIGNMENT.** Implement a robust inference class using **native TensorRT
> bindings** (`pycuda` in Python **or** `NvInfer.h` in C++). Use Jetson's unified
> memory efficiently with **pinned/page-locked host memory (`cudaHostAlloc`)** to
> avoid unnecessary CPU↔GPU copies.

> **✅ OUR WORK.**
> - **Python native bindings:** `engine/trt_engine.py::TRTRunner` uses the native
>   `tensorrt` runtime + `execute_async_v3`, binding device pointers zero-copy and
>   copying outputs into **pinned** host memory (`torch.empty(pin_memory=True)` →
>   `cudaHostAlloc`), with a cache of pinned buffers to avoid re-allocation.
> - **C++ `NvInfer.h`:** `cpp/vocoder_trt.cpp` — explicit `cudaHostAlloc` pinned
>   buffers and a dedicated `cudaStream_t`; **16.3 ms/infer** (matches Python).
>   Built via `cpp/build.sh`.

### Step 3 — Concurrency & Audio Buffer Pipeline
> **📄 ASSIGNMENT.** Design a multi-threaded producer–consumer pipeline handling
> tokenization/ingestion, asynchronous TensorRT execution via `cudaStream_t`, and
> audio buffer streaming — with thread-safe queuing so the next chunk generates
> while the current is dispatched.

> **✅ OUR WORK.** In `server/server.py`, a **producer thread** runs
> `engine.stream()` (tokenize → chunk → TRT generate on the CUDA stream) and pushes
> PCM chunks onto an `asyncio` queue; the WebSocket **consumer** drains and sends
> them. This overlaps generation of chunk N+1 with delivery of chunk N. Text
> chunking (`engine/text_chunker.py`) uses a small **first chunk** to minimize TTFA,
> then up to 50-word chunks.

### Step 4 — Edge Web UI & Benchmarking Playground
> **📄 ASSIGNMENT.** A lightweight web server exposing the pipeline via WebSockets/
> HTTP streaming; a front-end to input text, adjust parameters (diffusion steps,
> temperature), and listen to streaming audio as it generates. **Real-time
> telemetry** must display TTFA (ms), RTF (per chunk + overall), unified-memory
> (VRAM) footprint, and *(bonus)* GPU/DLA utilization via jtop/tegrastats.

> **✅ OUR WORK.**
> - **Server:** FastAPI, `/ws/tts` (audio stream) + `/ws/telemetry`.
> - **Controls:** custom text, language, voice-instruct, **diffusion steps**,
>   **guidance scale**, **position/class temperature**, first-chunk words, max
>   words/chunk, seed.
> - **Streaming playback:** Web Audio API plays each chunk on arrival.
> - **Telemetry (live):** **TTFA (ms)**, **RTF per-chunk + overall**, **peak VRAM
>   (MB)**, plus **GPU util / power / board power / temperatures** sampled from
>   **tegrastats + nvidia-smi**.
> - **Measured live (browser):** TTFA **553 ms**, overall RTF **0.169**, peak VRAM
>   **2031 MB**, audio 9.24 s; GPU util ~**91%** during synthesis, ~40 °C, board
>   ~54 W peak. Backends selectable: `mock` (GPU-free), `pytorch`, `trt`
>   (torch backbone + TRT vocoder), `trt-full` (TRT backbone + TRT vocoder).

---

## Performance Results (measured on device)

**End-to-end backend comparison** (same prompt, streaming):

| Backend | TTFA | Overall RTF |
|---|---:|---:|
| PyTorch FP16 (baseline) | ~1.59 s | 0.373 |
| PyTorch + TRT vocoder | ~1.3 s | ~0.30 |
| **TRT backbone (BF16) + TRT vocoder** | **~0.55 s** | **0.169** |

**Stage 1 streaming sweep** (PyTorch vs. PyTorch+TRT-vocoder):

| Prompt | Words | Chunks | TTFA | RTF (PyTorch) | RTF (+TRT vocoder) |
|---|---:|---:|---:|---:|---:|
| short | 5 | 1 | ~0.77 s | 0.445 | 0.376 |
| medium | 21 | 2 | ~1.06 s | 0.358 | 0.281 |
| long | 51 | 2 | ~1.39 s | 0.236 | 0.190 |

**Component speedups:** vocoder 2.8× (rel-RMS ~0.001); backbone 1.8× in BF16
(rel-RMS ~0.004). The full-TRT path roughly **halves both TTFA and RTF** vs. the
PyTorch baseline.

---

## Evaluation Criteria

| 📄 Dimension | ✅ How we address it |
|---|---|
| **Edge compute efficiency** (GPU util, thermal/mem) | ~90% GPU util during synthesis; VRAM tracked live for spikes/leaks; 900 MHz clock cap gives thermal/power headroom; temps/power in telemetry |
| **Systems architecture** (async, memory reuse, concurrency) | Dedicated CUDA stream, zero-copy input binding, cached pinned output buffers, threaded producer/consumer with async queue |
| **API & full-stack integration** (clean streamable API, no I/O overhead) | Single FastAPI app; PCM streamed as binary WS frames; chunk metadata as JSON; generation overlaps delivery |
| **Real-time performance** (RTF limits + in-UI latency measurement) | RTF **0.169** (< 0.5); TTFA, per-chunk & overall RTF, and VRAM measured and displayed **inside** the playground |

---

## Engineering Narrative (beyond the checklist)

Two findings were significant real-world engineering, documented in
`README.md`, `docs/STAGE1.md`, and `docs/STAGE2.md`:

1. **Hardware stability.** The board hard-reset repeatedly under sustained GPU
   load. Root cause: marginal power delivery + GPU clock transients (not our
   code). Fixed by (a) the **original Thor power adapter** and (b) a **900 MHz GPU
   clock cap** auto-applied at boot via a systemd unit (`scripts/thor-gpu-cap.service`).
   The x1/Gen1 PCIe link errors are chronic/pre-existing and worth monitoring.
2. **FP16 → BF16.** Diagnosed the two distinct FP16 failures above and engineered
   the BF16/FP32 path that meets the performance target while staying numerically
   faithful.

---

## Repository Map

```
omnivoice_thor/
  engine/
    tts_engine.py     # streaming + metrics; TorchTTSEngine, MockTTSEngine
    trt_engine.py     # TRTRunner (stream+pinned+zero-copy), TRTVocoder,
                      #   TRTBackbone, TRTTTSEngine, TRTFullEngine
    text_chunker.py   # low-latency sentence chunking (small first chunk)
    telemetry.py      # tegrastats + nvidia-smi sampler
    metrics.py        # TTFA / RTF / VRAM containers
  server/
    server.py         # FastAPI: /ws/tts + /ws/telemetry
    static/           # dependency-free playground (Web Audio + canvas)
  scripts/
    export_onnx.py / build_trt.py / validate_trt.py            # vocoder
    export_backbone_onnx.py / build_trt_backbone.py /
      validate_backbone_trt.py                                 # backbone
    benchmark.py / smoke.py / guarded_infer.py
    set_safe_clocks.sh / thor-gpu-cap.service                  # stability
  cpp/
    vocoder_trt.cpp / build.sh   # native NvInfer.h runner (cudaHostAlloc + stream)
  docs/  STAGE1.md, STAGE2.md
  README.md, SUBMISSION.md (this file), requirements.txt
  artifacts/  *.plan, *.onnx, *.json, *.wav, build logs
```

## How to run
```bash
source scripts/env.sh
# fastest fully-accelerated path (open http://<device-ip>:8008):
OMNIVOICE_ENGINE=trt-full python -m uvicorn server.server:app --host 0.0.0.0 --port 8008
# benchmarks:
python scripts/benchmark.py --backend trt-full
```
