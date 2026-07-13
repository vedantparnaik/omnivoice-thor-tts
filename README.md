# OmniVoice Real-Time TTS on NVIDIA Jetson AGX Thor

Edge deployment + benchmarking of a real-time, zero-shot Text-to-Speech pipeline
based on [OmniVoice](https://github.com/k2-fsa/OmniVoice) (a diffusion-LM TTS on a
Qwen3-0.6B backbone with a HiggsAudioV2 neural vocoder), exposed through a
streaming web playground with live edge telemetry.

Target device: **NVIDIA Jetson AGX Thor** (JetPack R38 / CUDA 13 / TensorRT 10.13,
122 GB unified memory, Blackwell-class GPU, sm_110).

---

## Results (measured on this device)

Streaming, GPU capped at 900 MHz (see "Hardware stability" below).

**End-to-end backend comparison** (same prompt, `~9 s` of audio):

| Backend | TTFA | Overall RTF | Notes |
|---------|-----:|------------:|-------|
| PyTorch FP16 | ~1.59 s | 0.373 | baseline (Stage 1) |
| PyTorch + TRT vocoder | ~1.3 s | ~0.30 | vocoder only on TRT |
| **TRT backbone (BF16) + TRT vocoder** | **~0.55 s** | **0.169** | fully accelerated (Stage 2) |

The fully-TRT path roughly **halves both TTFA and RTF** vs the PyTorch baseline,
verified live in the browser (TTFA 553 ms, RTF 0.169, peak VRAM 2031 MB).

Per-stage detail:
- **TRT vocoder** (Stage 1): rel-RMS ~0.001 vs torch, **2.8x** faster decode
  (46.9 ms -> 16.8 ms at L=200); C++ `NvInfer.h` runner 16.3 ms/infer.
- **TRT backbone** (Stage 2): the diffusion-LM step (run once per denoising step)
  is **~1.8x** faster than torch FP16 in BF16 (e.g. 91 ms -> 49 ms at S=512),
  rel-RMS ~0.004, with coherent audio matching the baseline.

Artifacts: `artifacts/baseline*.json`, `artifacts/*.plan`, `artifacts/*.wav`,
`artifacts/trt_build_report.json`, `artifacts/trt_backbone_build_report.json`.

---

## Architecture

```
Browser playground (static/)  ──ws /ws/tts──►  FastAPI (server/server.py)
    │  text + params                               │  producer thread: engine.stream()
    │  ◄── chunk_meta JSON + int16 PCM ──          │  asyncio queue (overlaps gen N+1 / send N)
    │  ◄── ws /ws/telemetry (tegrastats) ──         ▼
                                              TTS engine (engine/)
                                                ├─ TorchTTSEngine  (PyTorch FP16 baseline)
                                                ├─ TRTTTSEngine    (torch backbone + TRT vocoder)
                                                └─ TRTFullEngine   (TRT backbone + TRT vocoder)
                                                        └─ TRTRunner: NvInfer + CUDA stream + pinned I/O
```

The diffusion backbone has **no KV cache** — each of the N denoising steps
re-runs the whole sequence — so a single step exports cleanly to ONNX/TensorRT
`(input_ids, audio_mask, attention_mask) -> logits`. `TRTFullEngine` keeps the
N-step unmasking loop (sampling / confidence scatter) in Python/torch and runs
each forward on the compiled backbone engine.

Streaming splits text into sentence chunks (small first chunk for fast TTFA,
then up to 50-word chunks), synthesizes each, and streams PCM as it is produced.

---

## Layout

```
engine/
  tts_engine.py    # BaseTTSEngine (streaming+metrics), TorchTTSEngine, MockTTSEngine
  trt_engine.py    # TRTRunner (native TRT + pinned mem + stream), TRTVocoder,
                   #   TRTBackbone, TRTTTSEngine, TRTFullEngine
  text_chunker.py  # sentence chunking for low-latency streaming
  telemetry.py     # tegrastats + nvidia-smi sampler (unified mem, power, temp, GPU util)
  metrics.py       # TTFA / RTF / VRAM containers
server/
  server.py        # FastAPI: /ws/tts (audio stream) + /ws/telemetry, static UI
  static/          # dependency-free playground (Web Audio + canvas telemetry)
scripts/
  env.sh                    # activates venv + library paths (see below)
  smoke.py                  # minimal load+generate check
  benchmark.py              # baseline/TRT benchmark -> artifacts/*.json
  export_onnx.py            # export vocoder decode to ONNX (dynamic length)
  build_trt.py              # vocoder ONNX -> TRT engine (fp16/fp32/int8)
  validate_trt.py           # numeric + latency compare (torch vs TRT vocoder)
  export_backbone_onnx.py   # export diffusion backbone step to ONNX (CPU-safe)
  build_trt_backbone.py     # backbone ONNX -> TRT engine (fp32/fp16/bf16, dyn seq)
  validate_backbone_trt.py  # numeric + latency compare (torch vs TRT backbone)
  guarded_infer.py          # single guarded inference (stability testing)
  set_safe_clocks.sh / thor-gpu-cap.service  # GPU clock cap (stability)
cpp/
  vocoder_trt.cpp  # native NvInfer.h runner (cudaHostAlloc + cudaStream_t)
  build.sh
```

---

## Setup & run

Prerequisite: an NVIDIA Jetson running **JetPack R38** (CUDA 13, Python 3.12,
TensorRT 10.13 — ships with JetPack). See the platform note below.

```bash
# 0) clone
git clone https://github.com/vedantparnaik/omnivoice-thor-tts.git
cd omnivoice-thor-tts

# 1) create the environment (torch/torchaudio pull from the NVIDIA Jetson index;
#    tensorrt is provided by JetPack and wired in via scripts/env.sh)
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt      # exact lock in requirements-freeze.txt
source scripts/env.sh                # LD_LIBRARY_PATH + PYTHONPATH (TensorRT) + HF cache

# 2) sanity (downloads the OmniVoice weights on first run)
python scripts/smoke.py

# 3) baseline benchmark
python scripts/benchmark.py --backend pytorch          # PyTorch FP16
python scripts/benchmark.py --backend trt              # + TensorRT vocoder

# 4) web playground  (open http://<device>:8008)
OMNIVOICE_ENGINE=pytorch python -m uvicorn server.server:app --host 0.0.0.0 --port 8008
#   OMNIVOICE_ENGINE=mock      -> GPU-free UI/dev testing
#   OMNIVOICE_ENGINE=trt       -> torch backbone + TensorRT vocoder
#   OMNIVOICE_ENGINE=trt-full  -> TensorRT backbone (BF16) + TensorRT vocoder (fastest)

# 5) TensorRT engine (re)generation
#    vocoder:
python scripts/export_onnx.py
python scripts/build_trt.py --precision fp32           # see FP16 note below
python scripts/validate_trt.py
#    backbone (export on CPU is safe; build on GPU):
python scripts/export_backbone_onnx.py --device cpu
python scripts/build_trt_backbone.py --precision bf16 --out artifacts/backbone_bf16.plan
python scripts/validate_backbone_trt.py

# 6) native C++ runner
bash cpp/build.sh && ./cpp/vocoder_trt artifacts/vocoder_fp32.plan 200 30
```

### Environment notes (this is a bleeding-edge platform)
The Jetson/CUDA-13 `torch` wheel dynamically links NVIDIA math libs shipped as
pip packages; `scripts/env.sh` puts them on `LD_LIBRARY_PATH` and adds the
system TensorRT 10.13 python bindings to `PYTHONPATH`. cuDNN 9 for CUDA 13 was
installed via apt (`libcudnn9-cuda-13`).

---

## Hardware stability (important)

During bring-up the board hard-reset repeatedly under sustained GPU load. Root
causes identified from logs (`pstore`/`journalctl`, PCIe AER, `tegrastats`):

1. **Power delivery**: crashes were brownout-style hard resets (no panic trace)
   under GPU current transients. Replacing the power adapter with the **original
   Jetson Thor adapter** removed the resets on the TensorRT path.
2. **GPU clock transients**: capping the GPU compute clock (`gpu-gpc-0`) to
   **900 MHz** (vs ~1341 MHz MAXN) eliminated resets during PyTorch inference.
   Applied at boot via `scripts/thor-gpu-cap.service` (systemd).
3. **Chronic PCIe link state**: the GPU PCIe link reports x1/Gen1 with correctable
   `RxErr` (present in logs predating this work). Monitor if instability returns.

## Precision on sm_110 (TRT 10.13): FP16 findings

- **Backbone — FP16, BF16, INT8 all work.** A naive FP16 build compiles but emits
  **all-zero logits**: Qwen3's **RMSNorm** squares activations (`x²`), which
  overflows FP16's 65504 ceiling (TRT auto-protects LayerNorm, not RMSNorm).
  Pinning the norm/reduce/pow layers to FP32 fixes it:
  `build_trt_backbone.py --precision fp16 --keep-norm-fp32` → correct output
  (rel-RMS ~0.001), RTF 0.242. **BF16** is the default runtime (no per-layer
  casts, ~1.8x vs torch, RTF 0.182); **FP32** is the exact reference. **INT8**
  (post-training calibration via `capture_backbone_calib.py` +
  `build_trt_backbone.py --precision int8 --keep-norm-fp32`) is the
  fastest/smallest (748 MB, RTF 0.185) but quantization perturbs the diffusion
  path (~16% longer output, ~2.4x BF16's log-mel drift) so it trades fidelity.
  Pick with `OMNIVOICE_BACKBONE_PLAN=artifacts/backbone_{bf16,fp16,fp32,int8}.plan`.
- **Vocoder — FP16 blocked by an NVIDIA-acknowledged Blackwell bug.** The FP16
  build *crashes the compiler* (`MyelinCheckException: NVRTC Compilation failure`)
  while fusing the DAC decoder's conv tail. NVIDIA documents this exact issue in
  the **TensorRT 10.13.3 release notes**: *"MyelinCheckException may be reported
  when Slice-Fill-Conv is used on Blackwell GPUs."* Our decoder upsamples with
  transposed convs (`DacDecoderBlock.conv_t1`) → the Slice-Fill-Conv pattern, on
  the affected build (TensorRT 10.13.3.9, sm_110). We proved it's not our graph by
  trying **five** independent workarounds, all failing identically:
  (1) `sin.pow(2)`→`sin*sin`, (2) `sin²`→`(1−cos2x)/2`, (3) FP16 export with FP32
  Snake cast-islands, (4) strongly-typed network, (5) `--opt-level 0`; plus
  per-layer FP32 pinning (`--keep-snake-fp32`). All hit the same NVRTC failure, so
  the fix must come from a **TensorRT/JetPack update**. **FP32 builds/runs**
  correctly (2.8x vs torch, ~16 ms/infer — not on the RTF critical path). An
  FP16-ready graph is saved at `artifacts/vocoder_island.onnx` to rebuild once
  NVIDIA ships the fix. INT8 mixes in FP16 and inherits this crash; DLA unused.

All FP16/INT8 paths are retained for when the platform matures.
