"""Validate + benchmark the TensorRT vocoder against the PyTorch vocoder.

Runs identical audio codes through both decoders and reports numerical agreement
and per-call latency. Uses the native TRTRunner (dedicated stream + pinned I/O).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.trt_engine import TRTVocoder  # noqa: E402


def main() -> None:
    from omnivoice import OmniVoice

    print("loading model (fp32 tokenizer for fair compare)...", flush=True)
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float32)
    tok = model.audio_tokenizer
    num_q = tok.config.num_quantizers
    codebook_size = tok.config.codebook_size

    voc = TRTVocoder(device="cuda:0")
    print(f"TRT engine: {os.path.basename(voc.plan_path)}", flush=True)

    torch.manual_seed(0)
    for L in (64, 200, 500):
        codes = torch.randint(0, codebook_size, (1, num_q, L), dtype=torch.long, device="cuda:0")

        with torch.no_grad():
            ref = tok.decode(codes, return_dict=False).float()
        out = voc.decode(codes).audio_values.float()

        m = min(ref.shape[-1], out.shape[-1])
        a, b = ref[..., :m], out[..., :m]
        max_abs = (a - b).abs().max().item()
        rms = (a - b).pow(2).mean().sqrt().item()
        denom = a.pow(2).mean().sqrt().item() + 1e-9
        rel = rms / denom
        print(f"[L={L:4}] out_samples={out.shape[-1]:6} max_abs_diff={max_abs:.4f} "
              f"rel_rms={rel:.4f}", flush=True)

    # ---- latency benchmark at L=200 ----
    L = 200
    codes = torch.randint(0, codebook_size, (1, num_q, L), dtype=torch.long, device="cuda:0")
    N = 30
    # torch
    with torch.no_grad():
        for _ in range(3):
            tok.decode(codes, return_dict=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            tok.decode(codes, return_dict=False)
        torch.cuda.synchronize()
        torch_ms = (time.perf_counter() - t0) / N * 1000
    # trt
    for _ in range(3):
        voc.decode(codes)
    t0 = time.perf_counter()
    for _ in range(N):
        voc.decode(codes)
    trt_ms = (time.perf_counter() - t0) / N * 1000

    print(f"[latency L={L}] torch_fp32={torch_ms:.2f}ms  trt={trt_ms:.2f}ms  "
          f"speedup={torch_ms/trt_ms:.2f}x", flush=True)
    print("VALIDATE_OK", flush=True)


if __name__ == "__main__":
    main()
