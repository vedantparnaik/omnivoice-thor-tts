# Stage 1 — Checkpoint (working edge TTS + streaming playground)

Frozen snapshot of a fully working stage. Everything here is verified on the
NVIDIA Jetson AGX Thor (JetPack R38 / CUDA 13 / TensorRT 10.13).

## Status: DONE and verified
- Environment: venv with Jetson/CUDA-13 torch, OmniVoice, FastAPI, TensorRT bindings.
- PyTorch FP16 streaming engine with TTFA / RTF / VRAM instrumentation.
- FastAPI + WebSocket streaming playground with live Jetson telemetry.
- Vocoder exported to ONNX + compiled to a TensorRT engine (FP32), validated.
- Native TRT execution in Python (pinned mem + CUDA stream) and C++ (`NvInfer.h`).

## Measured results (900 MHz GPU cap, streaming)
| Prompt | Words | TTFA | RTF (PyTorch) | RTF (PyTorch+TRT vocoder) |
|--------|------:|-----:|--------------:|--------------------------:|
| short  | 5  | ~0.77 s | 0.445 | 0.376 |
| medium | 21 | ~1.06 s | 0.358 | 0.281 |
| long   | 51 | ~1.39 s | 0.236 | 0.190 |

TRT vocoder: rel-RMS ~0.001 vs torch, 2.8x faster decode; C++ runner 16.3 ms/infer.

## What is NOT in Stage 1 (Stage 2 targets)
- Backbone (diffusion-LM) ONNX/TensorRT export — vocoder only so far.
- FP16 / INT8 TRT engine — blocked by a TensorRT-10.13/Thor Myelin nvrtc bug
  on the Snake activation (sm_110); FP32 works. See README "Known limitation".
- Deeper async interleaving of the diffusion decode itself.

## Hardware settings required for stability
- Original Jetson Thor power adapter (marginal adapters caused hard resets).
- GPU compute clock capped to 900 MHz (`scripts/thor-gpu-cap.service`, auto at boot).
- The GPU PCIe link reports x1/Gen1 + correctable RxErr (chronic/pre-existing).

## Restore from the zip
```bash
cd /home/idrivethor/OA
unzip omnivoice_thor_stage1.zip -d omnivoice_thor_stage1
cd omnivoice_thor_stage1

# recreate the venv (not included; ~3 GB)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt \
  --extra-index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple/
# (torch/torchaudio come from the jetson index; cuDNN 9 via: sudo apt install libcudnn9-cuda-13)

source scripts/env.sh
python scripts/smoke.py                 # re-downloads model weights (hf_cache, not zipped)
OMNIVOICE_ENGINE=pytorch python -m uvicorn server.server:app --host 0.0.0.0 --port 8008
```

## Included in the zip
- All source: `engine/`, `server/`, `scripts/`, `cpp/`, `docs/`, `README.md`, `requirements.txt`
- Built artifacts kept to avoid the fragile rebuild: `artifacts/vocoder.onnx(.data)`,
  `artifacts/vocoder_fp32.plan`, `artifacts/*.json`, sample `*.wav`.

## Excluded (regenerable)
- `.venv/` (~3 GB) — rebuild via requirements.txt
- `artifacts/hf_cache/` (~3 GB) — model weights re-download on first run
