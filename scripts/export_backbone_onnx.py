"""Export the OmniVoice diffusion-LM backbone (single denoising step) to ONNX.

Unlike an autoregressive decoder, the OmniVoice backbone is invoked as a
*full-sequence* forward with **no KV cache** — each of the N diffusion steps
re-runs the whole sequence. That makes a single step cleanly exportable:

    input_ids       (2B, C, S)  long   audio codebook + text token ids
    audio_mask      (2B, S)     bool   True where the position is audio
    attention_mask  (2B, 1, S, S) bool True where attention is allowed
            |
            v   _prepare_embed_inputs -> Qwen3 backbone -> audio_heads
            v
    logits          (2B, C, S, V) float

The N-step unmasking loop (sampling / confidence selection / scatter) stays in
Python and calls this engine once per step. 2B is cond+uncond (CFG); C=8
codebooks, V=1025 vocab.

Exported on CPU (fp32) on purpose: tracing runs several forward passes, and we
keep the GPU idle here — only the TensorRT build touches the GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BackboneStep(nn.Module):
    """ONNX-friendly wrapper: one full-sequence denoising forward."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.C = model.config.num_audio_codebook
        self.V = model.config.audio_vocab_size

    def forward(self, input_ids, audio_mask, attention_mask):
        inputs_embeds = self.model._prepare_embed_inputs(input_ids, audio_mask)
        llm_out = self.model.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            position_ids=None,
        )
        hidden = llm_out[0]
        b, s, _ = hidden.shape
        logits_flat = self.model.audio_heads(hidden)
        audio_logits = logits_flat.view(b, s, self.C, self.V).permute(0, 2, 1, 3)
        return audio_logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/backbone.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--opt-len", type=int, default=256, help="representative seq length for tracing")
    ap.add_argument("--device", default="cpu", help="cpu (safe) or cuda:0")
    args = ap.parse_args()

    from omnivoice import OmniVoice

    print(f"loading model on {args.device} (fp32)...", flush=True)
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", device_map=args.device, dtype=torch.float32
    ).eval()

    C = model.config.num_audio_codebook
    V = model.config.audio_vocab_size
    hidden = model.llm.config.hidden_size
    print(f"backbone: codebooks={C} vocab={V} hidden={hidden} "
          f"llm={model.llm.config.model_type}", flush=True)

    wrapper = BackboneStep(model).eval().to(args.device)

    # Synthetic but structurally-realistic inputs: batch=2 (cond+uncond),
    # first third text (audio_mask False), rest audio (audio_mask True).
    B2, S = 2, args.opt_len
    dev = args.device
    input_ids = torch.randint(0, V, (B2, C, S), dtype=torch.long, device=dev)
    audio_mask = torch.zeros(B2, S, dtype=torch.bool, device=dev)
    audio_mask[:, S // 3:] = True
    attn = torch.zeros(B2, 1, S, S, dtype=torch.bool, device=dev)
    attn[:, :, :, :] = True  # dense (full) attention over the valid window

    with torch.no_grad():
        ref = wrapper(input_ids, audio_mask, attn)
    print(f"traced output shape: {tuple(ref.shape)} (expect (2,{C},{S},{V}))", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"exporting ONNX -> {args.out} (this runs several CPU forward passes)", flush=True)
    torch.onnx.export(
        wrapper,
        (input_ids, audio_mask, attn),
        args.out,
        input_names=["input_ids", "audio_mask", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 2: "seq"},
            "audio_mask": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 2: "seq", 3: "seq"},
            "logits": {0: "batch", 2: "seq"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[done] wrote {args.out} ({size_mb:.1f} MB)", flush=True)

    meta = {
        "num_audio_codebook": int(C),
        "audio_vocab_size": int(V),
        "hidden_size": int(hidden),
        "llm_model_type": str(model.llm.config.model_type),
        "batch_2b": int(B2),
    }
    with open("artifacts/backbone_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[done] wrote artifacts/backbone_meta.json", flush=True)


if __name__ == "__main__":
    main()
