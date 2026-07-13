"""Capture real backbone activation data for INT8 PTQ calibration.

Runs a few real generations, records every backbone forward's inputs
(input_ids, audio_mask, attention_mask) — one per diffusion step — and
pads/crops each to a fixed calibration length so TensorRT's entropy calibrator
can estimate activation ranges. Saved to artifacts/backbone_calib.pt.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.tts_engine import TorchTTSEngine, GenParams  # noqa: E402

SENTENCES = [
    "Hello from the Jetson Thor.",
    "This is a real-time, zero-shot text to speech pipeline running on the edge.",
    "The quick brown fox jumps over the lazy dog while the sun sets slowly.",
    "Edge inference demands careful management of memory, power, and thermals.",
    "She sells sea shells by the sea shore on a bright summer morning.",
    "TensorRT compiles the transformer backbone into fused kernels for the GPU.",
]


def pad_crop(ii, am, at, L, pad_id):
    B2, C, S = ii.shape
    if S == L:
        return ii, am, at
    if S > L:
        return ii[:, :, :L].contiguous(), am[:, :L].contiguous(), at[:, :, :L, :L].contiguous()
    # pad up to L
    ii2 = torch.full((B2, C, L), pad_id, dtype=ii.dtype)
    ii2[:, :, :S] = ii
    am2 = torch.zeros((B2, L), dtype=am.dtype)
    am2[:, :S] = am
    at2 = torch.zeros((B2, 1, L, L), dtype=at.dtype)
    at2[:, :, :S, :S] = at
    # let padded positions self-attend (mirrors the model's own uncond padding)
    idx = torch.arange(S, L)
    at2[:, :, idx, idx] = True
    return ii2, am2, at2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/backbone_calib.pt")
    ap.add_argument("--calib-len", type=int, default=384)
    ap.add_argument("--max-batches", type=int, default=64)
    args = ap.parse_args()

    eng = TorchTTSEngine()
    model = eng.model
    pad_id = model.config.audio_mask_id
    orig_forward = model.forward

    captured = []

    def recording_forward(input_ids=None, audio_mask=None, attention_mask=None, **kw):
        if attention_mask is not None and input_ids.shape[0] == 2:
            ii, am, at = pad_crop(
                input_ids.detach().cpu(),
                audio_mask.detach().cpu(),
                attention_mask.detach().cpu(),
                args.calib_len, pad_id,
            )
            captured.append((ii, am, at))
        return orig_forward(input_ids=input_ids, audio_mask=audio_mask,
                            attention_mask=attention_mask, **kw)

    model.forward = recording_forward
    gp = GenParams()
    for s in SENTENCES:
        if len(captured) >= args.max_batches:
            break
        for _ in eng.stream(s, gp):
            pass
        print(f"  after '{s[:32]}...': {len(captured)} batches", flush=True)

    model.forward = orig_forward
    captured = captured[:args.max_batches]

    input_ids = torch.stack([c[0] for c in captured])       # [N,2,8,L] int64
    audio_mask = torch.stack([c[1] for c in captured])       # [N,2,L]   bool
    attention_mask = torch.stack([c[2] for c in captured])   # [N,2,1,L,L] bool
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "input_ids": input_ids,
        "audio_mask": audio_mask,
        "attention_mask": attention_mask,
        "calib_len": args.calib_len,
    }, args.out)
    print(f"[done] {input_ids.shape[0]} calibration batches @ L={args.calib_len} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
