"""Export the OmniVoice neural vocoder (HiggsAudioV2 decoder) to ONNX.

The vocoder is the cleanly feed-forward part of the pipeline:

    audio_codes (B, Q, L)  --RVQ embed--> quantized (B, H, L)
                           --fc2 linear--> (B, H', L)
                           --DAC decoder--> waveform (B, 1, T)

This is exactly what TensorRT accelerates well (conv / conv-transpose stacks),
and it exports without the dynamic control flow of the diffusion backbone. We
export with dynamic ``codes_length`` so a single engine covers 5-50 word
sentences.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class VocoderDecode(nn.Module):
    """Thin ONNX-friendly wrapper around ``audio_tokenizer.decode``."""

    def __init__(self, audio_tokenizer):
        super().__init__()
        self.tok = audio_tokenizer

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        # returns (B, 1, T) float waveform
        return self.tok.decode(audio_codes, return_dict=False)


def _patch_snake(mode: str) -> None:
    """Rewrite DAC Snake1d.forward to a numerically-equivalent, Myelin-friendly form.

    The stock impl uses ``torch.sin(a*x).pow(2)``; TRT-10.13's Myelin JIT crashes
    (NVRTC compile failure) when fusing that Pow-of-sin subgraph in FP16 on sm_110.
      - "mul": replace pow(·,2) with s*s  (identical math, no Pow op)
      - "cos": use sin^2(z) = 0.5 - 0.5*cos(2z)  (no sin, no pow)
      - "pow": leave the original (baseline)
    """
    if mode == "pow":
        return
    from transformers.models.dac import modeling_dac as _dac

    if mode == "mul":
        def _fwd(self, hidden_states):
            shape = hidden_states.shape
            hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
            s = torch.sin(self.alpha * hidden_states)
            hidden_states = hidden_states + (self.alpha + 1e-9).reciprocal() * (s * s)
            return hidden_states.reshape(shape)
    elif mode == "cos":
        def _fwd(self, hidden_states):
            shape = hidden_states.shape
            hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
            z = self.alpha * hidden_states
            sin_sq = 0.5 - 0.5 * torch.cos(2.0 * z)
            hidden_states = hidden_states + (self.alpha + 1e-9).reciprocal() * sin_sq
            return hidden_states.reshape(shape)
    elif mode == "island":
        # Compute Snake in FP32 with explicit casts. In an FP16 export this puts a
        # hard fp16->fp32->fp16 boundary in the graph that the Myelin fuser cannot
        # cross, splitting the one crashing FP16 kernel into small FP32 Snake
        # kernels (which compile fine) between FP16 convs.
        def _fwd(self, hidden_states):
            shape = hidden_states.shape
            in_dtype = hidden_states.dtype
            h = hidden_states.reshape(shape[0], shape[1], -1).float()
            a = self.alpha.float()
            s = torch.sin(a * h)
            h = h + (a + 1e-9).reciprocal() * (s * s)
            return h.to(in_dtype).reshape(shape)
    else:
        raise ValueError(f"unknown snake mode {mode!r}")

    _dac.Snake1d.forward = _fwd
    print(f"[patch] DAC Snake1d.forward -> mode={mode}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/vocoder.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--opt-len", type=int, default=300, help="representative codes length for tracing")
    ap.add_argument("--snake-mode", default="pow", choices=["pow", "mul", "cos", "island"],
                    help="Snake activation formulation (mul/cos/island dodge the FP16 Myelin crash)")
    ap.add_argument("--half", action="store_true",
                    help="export the vocoder in FP16 (pairs with --snake-mode island)")
    args = ap.parse_args()

    _patch_snake(args.snake_mode)

    from omnivoice import OmniVoice

    print("loading model...", flush=True)
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float32)
    tok = model.audio_tokenizer
    cfg = tok.config
    num_q = cfg.num_quantizers
    codebook_size = cfg.codebook_size
    frame_rate = cfg.frame_rate
    print(f"vocoder: num_quantizers={num_q} codebook_size={codebook_size} "
          f"frame_rate={frame_rate} hop_length={cfg.hop_length}", flush=True)

    wrapper = VocoderDecode(tok).eval().to("cuda:0")
    if args.half:
        wrapper = wrapper.half()
        print("[half] exporting vocoder in FP16", flush=True)

    B, L = 1, args.opt_len
    dummy = torch.randint(0, codebook_size, (B, num_q, L), dtype=torch.long, device="cuda:0")

    with torch.no_grad():
        ref = wrapper(dummy)
    print(f"traced output shape: {tuple(ref.shape)} (samples/token = {ref.shape[-1] / L:.1f})", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"exporting ONNX -> {args.out}", flush=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        args.out,
        input_names=["audio_codes"],
        output_names=["audio_values"],
        dynamic_axes={
            "audio_codes": {0: "batch", 2: "codes_length"},
            "audio_values": {0: "batch", 2: "num_samples"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[done] wrote {args.out} ({size_mb:.1f} MB)", flush=True)

    # Persist vocoder metadata for the TRT builder / runtime.
    import json
    meta = {
        "num_quantizers": int(num_q),
        "codebook_size": int(codebook_size),
        "frame_rate": float(frame_rate),
        "hop_length": int(cfg.hop_length),
        "sampling_rate": int(getattr(model, "sampling_rate", 24000)),
        "samples_per_token": int(round(ref.shape[-1] / L)),
    }
    with open("artifacts/vocoder_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[done] wrote artifacts/vocoder_meta.json", flush=True)


if __name__ == "__main__":
    main()
