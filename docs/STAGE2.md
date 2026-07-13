# Stage 2 — Full TensorRT pipeline (backbone + vocoder)

Builds on Stage 1 (see `docs/STAGE1.md`). Adds the diffusion-LM **backbone** to
the TensorRT path, so the whole inference pipeline — backbone denoising steps and
vocoder synthesis — runs on compiled TRT engines. The N-step unmasking loop
(sampling / confidence selection / scatter) stays in Python/torch.

## Status: DONE and verified (live in browser)
- Backbone single denoising step exported to ONNX with dynamic batch + seq.
- Backbone TensorRT engine built and validated (BF16 default, FP32 fallback).
- `TRTFullEngine` routes the model's forward through TRT inside the diffusion loop.
- Server backend `OMNIVOICE_ENGINE=trt-full`, verified end-to-end in the playground.

## Measured results (900 MHz GPU cap, streaming, ~9 s prompt)
| Backend | TTFA | Overall RTF |
|---------|-----:|------------:|
| PyTorch FP16 (Stage 1) | ~1.59 s | 0.373 |
| TRT backbone (BF16) + TRT vocoder | **~0.55 s** | **0.169** |

- Backbone step ~1.8x faster in BF16 vs torch FP16 (91 ms -> 49 ms at S=512), rel-RMS ~0.004.
- Peak VRAM ~2.0 GB. No crashes.

## Key engineering finding: FP16 -> BF16
FP16 backbone *builds* but emits **all-zero logits** (layernorm/attention overflow
on sm_110 / TRT 10.13; builder warns and clips weights to 65504). **BF16** fixes
it: FP32 exponent range (no overflow) + Blackwell tensor-core speed. FP32 is also
built as a correctness reference (numerically exact but only parity-speed with
torch FP16, since it doubles the FLOPs). See README "Known limitation".

## How the backbone is exportable
The diffusion backbone runs a **full-sequence forward with no KV cache** (each of
the N steps re-runs the whole sequence). So one step is a clean function:

    input_ids (2B,8,S) int64 + audio_mask (2B,S) bool + attention_mask (2B,1,S,S) bool
        -> logits (2B,8,S,1025)

2B = cond+uncond (classifier-free guidance), 8 codebooks, vocab 1025. Exported on
CPU (safe, GPU idle); engine built for fixed batch=2 and dynamic seq [32,1200].
Chunks outside that window transparently fall back to the torch forward.

## New files (relative to Stage 1)
- `scripts/export_backbone_onnx.py`, `scripts/build_trt_backbone.py`, `scripts/validate_backbone_trt.py`
- `engine/trt_engine.py`: `TRTBackbone` + `TRTFullEngine`
- artifacts: `backbone.onnx(.data)`, `backbone_bf16.plan` (default), `backbone_fp32.plan` (fallback)

## Run
```bash
source scripts/env.sh
OMNIVOICE_ENGINE=trt-full python -m uvicorn server.server:app --host 0.0.0.0 --port 8008
python scripts/benchmark.py --backend trt-full
```

## Rebuild backbone engines (on this device)
```bash
python scripts/export_backbone_onnx.py --device cpu          # ~70 s, CPU only
python scripts/build_trt_backbone.py --precision bf16 --out artifacts/backbone_bf16.plan
python scripts/build_trt_backbone.py --precision fp32 --out artifacts/backbone_fp32.plan
python scripts/validate_backbone_trt.py
```

## Remaining / Stage 3 ideas
- INT8/FP16 once the TRT-Thor platform bug is fixed (would add further speedup).
- Fuse the Python unmask/scatter into fewer GPU ops; CUDA-graph the per-step loop.
- Multi-request batching (engine currently fixed to CFG batch=2 for B=1 streaming).
