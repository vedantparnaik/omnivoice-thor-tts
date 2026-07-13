"""Validate the TensorRT backbone against PyTorch: numerics + per-step latency.

Feeds identical random-but-shape-valid inputs (CFG batch=2, dynamic seq) to the
torch forward and the TRT engine, and reports relative RMS error on the logits
plus a latency comparison. This is the per-step cost that dominates generation
(the backbone runs once per diffusion step).
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.tts_engine import TorchTTSEngine  # noqa: E402
from engine.trt_engine import TRTBackbone  # noqa: E402


def rand_inputs(model, B2, S, dev):
    C = model.config.num_audio_codebook
    V = model.config.audio_vocab_size
    input_ids = torch.randint(0, V, (B2, C, S), dtype=torch.long, device=dev)
    audio_mask = torch.zeros(B2, S, dtype=torch.bool, device=dev)
    audio_mask[:, S // 3:] = True
    attn = torch.ones(B2, 1, S, S, dtype=torch.bool, device=dev)
    return input_ids, audio_mask, attn


def main() -> None:
    dev = "cuda:0"
    print("loading torch engine (fp16)...", flush=True)
    eng = TorchTTSEngine()
    model = eng.model
    torch_fwd = model.forward  # capture original before patching

    bb = TRTBackbone(model, device=dev)
    print(f"backbone engine: {os.path.basename(bb.plan_path)} "
          f"batch={bb.batch} seq=[{bb.min_len},{bb.max_len}]", flush=True)

    for S in (64, 256, 512):
        ii, am, at = rand_inputs(model, bb.batch, S, dev)

        with torch.inference_mode():
            ref = torch_fwd(input_ids=ii, audio_mask=am, attention_mask=at).logits.float()
            out = bb(input_ids=ii, audio_mask=am, attention_mask=at).logits.float()

        diff = (out - ref)
        rel_rms = (diff.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp(min=1e-8)).item()
        # argmax agreement on the audio positions (what sampling actually uses).
        am3 = am[:, None, :].expand(-1, ref.shape[1], -1)
        agree = (out.argmax(-1)[am3] == ref.argmax(-1)[am3]).float().mean().item()

        def bench(fn, n=10):
            with torch.inference_mode():
                fn(); torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(n):
                    fn()
                torch.cuda.synchronize()
            return (time.time() - t0) / n * 1e3

        t_torch = bench(lambda: torch_fwd(input_ids=ii, audio_mask=am, attention_mask=at))
        t_trt = bench(lambda: bb(input_ids=ii, audio_mask=am, attention_mask=at))
        print(f"  S={S:4d}  rel-RMS={rel_rms:.4f}  argmax-agree={agree*100:5.1f}%  "
              f"torch={t_torch:6.2f}ms  trt={t_trt:6.2f}ms  ({t_torch/t_trt:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
