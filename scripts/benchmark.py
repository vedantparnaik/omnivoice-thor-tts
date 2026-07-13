"""Benchmark the TTS engine and record baseline metrics to artifacts/.

Usage:
    python scripts/benchmark.py [--backend pytorch] [--num-step 32]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import GenParams, TorchTTSEngine  # noqa: E402


def make_engine(backend: str):
    if backend == "pytorch":
        return TorchTTSEngine()
    if backend == "trt":
        from engine.trt_engine import TRTTTSEngine
        return TRTTTSEngine()
    if backend == "trt-full":
        from engine.trt_engine import TRTFullEngine
        return TRTFullEngine()
    raise ValueError(backend)

PROMPTS = {
    "short": "Hello from the Jetson Thor.",
    "medium": "This is a real time text to speech test running on an NVIDIA edge device. "
    "It streams audio while measuring latency.",
    "long": "Edge deployment of neural text to speech requires careful attention to memory, "
    "concurrency, and thermal limits. On this platform we compile the model for the GPU, "
    "stream audio chunk by chunk, and report the real time factor for every segment so "
    "engineers can see exactly where the time goes during inference.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="pytorch", choices=["pytorch", "trt", "trt-full"])
    ap.add_argument("--num-step", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="artifacts/baseline.json")
    args = ap.parse_args()

    t0 = time.time()
    engine = make_engine(args.backend)
    load_s = time.time() - t0
    print(f"[load] backend={engine.backend} in {load_s:.1f}s sr={engine.sampling_rate}")

    # warmup so autotuning is not counted
    engine.synthesize("warm up sentence for the kernels.", GenParams(num_step=16, seed=args.seed))

    report = {"backend": engine.backend, "load_s": round(load_s, 2),
              "num_step": args.num_step, "sampling_rate": engine.sampling_rate,
              "cases": {}}

    os.makedirs("artifacts", exist_ok=True)
    for name, prompt in PROMPTS.items():
        params = GenParams(num_step=args.num_step, seed=args.seed)
        audio, sm = engine.synthesize(prompt, params)
        sf.write(f"artifacts/baseline_{name}.wav", audio, engine.sampling_rate)
        report["cases"][name] = {
            "words": len(prompt.split()),
            "ttfa_ms": round(sm.ttfa_ms, 1) if sm.ttfa_ms else None,
            "overall_rtf": round(sm.overall_rtf, 3) if sm.overall_rtf else None,
            "total_wall_s": round(sm.total_wall_s, 3),
            "total_audio_s": round(sm.total_audio_s, 3),
            "peak_vram_mb": sm.peak_vram_mb,
            "n_chunks": sm.n_chunks,
            "per_chunk_rtf": [round(c.rtf, 3) for c in sm.chunks],
        }
        c = report["cases"][name]
        print(f"[{name:6}] words={c['words']:3} chunks={c['n_chunks']} "
              f"TTFA={c['ttfa_ms']}ms RTF={c['overall_rtf']} "
              f"audio={c['total_audio_s']}s peakVRAM={c['peak_vram_mb']}MB")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
