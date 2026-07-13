"""Native TensorRT execution of the OmniVoice vocoder, plus a drop-in engine.

Design
------
* ``TRTRunner`` — thin native TensorRT wrapper. It executes with a dedicated
  ``cudaStream_t`` (via a torch CUDA stream) and uses **pinned / page-locked host
  memory** (``torch.empty(..., pin_memory=True)`` -> ``cudaHostAlloc``) for the
  async device->host copy of the output waveform. GPU inputs are passed
  **zero-copy** by binding the producing tensor's ``data_ptr()`` directly, so no
  extra host<->device copy happens on Jetson's unified memory.
* ``TRTVocoder`` — wraps the vocoder ``.plan`` and mimics the HiggsAudio
  ``decode()`` return signature (``.audio_values``) so it can be patched in.
* ``TRTTTSEngine`` — subclass of the PyTorch engine that swaps the vocoder decode
  for the TensorRT engine: the diffusion backbone still runs in torch, the
  waveform synthesis runs on the compiled TRT engine. Drop-in with the same
  streaming/metrics interface as the baseline.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Optional

import numpy as np

sys.path.insert(0, "/usr/lib/python3.12/dist-packages")
import tensorrt as trt  # noqa: E402
import torch  # noqa: E402

from .tts_engine import GenParams, TorchTTSEngine  # noqa: E402

_ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


class TRTRunner:
    """Execute a serialized TensorRT engine with a dedicated stream + pinned I/O."""

    def __init__(self, plan_path: str, device: str = "cuda:0"):
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(plan_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to load TRT engine: {plan_path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=device)

        self.input_names, self.output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        # cache of pinned host output buffers keyed by numel
        self._pinned: dict = {}

    def _trt_dtype_to_torch(self, name: str):
        m = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64,
            trt.DataType.BOOL: torch.bool,
        }
        return m[self.engine.get_tensor_dtype(name)]

    def infer(self, inputs: dict, return_host: bool = False):
        """Run inference.

        Args:
            inputs: name -> CUDA torch tensor (bound zero-copy via data_ptr).
            return_host: if True, async-copy the output into pinned host memory
                and return a CPU tensor; else return the CUDA output tensor.
        """
        # bind inputs (zero-copy: use the existing device pointers)
        for name in self.input_names:
            t = inputs[name].contiguous()
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())

        # allocate device outputs from the now-resolved dynamic shapes
        outs = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            out = torch.empty(shape, dtype=self._trt_dtype_to_torch(name), device=self.device)
            outs[name] = out
            self.context.set_tensor_address(name, out.data_ptr())

        ok = self.context.execute_async_v3(self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 failed")

        if not return_host:
            self.stream.synchronize()
            return outs

        host = {}
        for name, dev in outs.items():
            key = dev.numel()
            pin = self._pinned.get((name, key))
            if pin is None:
                pin = torch.empty(dev.shape, dtype=dev.dtype, pin_memory=True)
                self._pinned[(name, key)] = pin
            pin.copy_(dev, non_blocking=True)  # async D2H into pinned host memory
            host[name] = pin
        self.stream.synchronize()
        return host


class TRTVocoder:
    """TensorRT vocoder that mimics HiggsAudio ``decode()`` (returns .audio_values)."""

    def __init__(self, plan_path: Optional[str] = None, device: str = "cuda:0"):
        if plan_path is None:
            # prefer fp16 if present, else fp32
            for cand in ("vocoder_fp16.plan", "vocoder_fp32.plan"):
                p = os.path.join(_ARTIFACTS, cand)
                if os.path.exists(p):
                    plan_path = p
                    break
        if plan_path is None or not os.path.exists(plan_path):
            raise FileNotFoundError("no vocoder .plan found in artifacts/")
        self.plan_path = plan_path
        self.runner = TRTRunner(plan_path, device=device)
        self.device = device
        self.in_name = self.runner.input_names[0]
        self.out_name = self.runner.output_names[0]

    def decode(self, audio_codes: torch.Tensor, return_dict: bool = False):
        codes = audio_codes.to(self.device)
        if codes.dtype != torch.int64:
            codes = codes.to(torch.int64)
        outs = self.runner.infer({self.in_name: codes}, return_host=False)
        audio_values = outs[self.out_name].float()
        # Guard against occasional non-finite samples before downstream int16 cast.
        audio_values = torch.nan_to_num(audio_values, nan=0.0, posinf=1.0, neginf=-1.0)
        return SimpleNamespace(audio_values=audio_values)


class TRTBackbone:
    """TensorRT diffusion backbone: one full-sequence denoising forward.

    Mirrors ``OmniVoice.forward``'s call signature used inside the N-step
    unmasking loop and returns an object exposing ``.logits`` (2B, C, S, V).
    The engine is built for a fixed CFG batch (2) and a dynamic sequence window
    ``[min_len, max_len]``; anything outside that (e.g. batch>1 or an unusually
    long chunk) transparently falls back to the original torch forward.
    """

    def __init__(self, model, plan_path: Optional[str] = None, device: str = "cuda:0"):
        if plan_path is None:
            plan_path = os.environ.get("OMNIVOICE_BACKBONE_PLAN") or None
        if plan_path is None:
            # Default BF16 (best fidelity/speed balance). Working alternatives,
            # all selectable via OMNIVOICE_BACKBONE_PLAN:
            #  - backbone_fp16.plan : true FP16 (RMSNorm/reduce pinned FP32 to dodge
            #    the overflow-to-zeros); most faithful (rel-RMS ~0.001).
            #  - backbone_fp32.plan : exact reference, parity speed.
            #  - backbone_int8.plan : INT8 PTQ (calibrated); fastest + smallest but
            #    audibly lower fidelity (quantization perturbs the diffusion path).
            for cand in ("backbone_bf16.plan", "backbone_fp16.plan", "backbone_fp32.plan"):
                p = os.path.join(_ARTIFACTS, cand)
                if os.path.exists(p):
                    plan_path = p
                    break
        if plan_path is None or not os.path.exists(plan_path):
            raise FileNotFoundError("no backbone .plan found in artifacts/")
        self.plan_path = plan_path
        self.device = device
        self.runner = TRTRunner(plan_path, device=device)
        self._orig_forward = model.forward  # bound torch fallback

        # Resolve the dynamic seq window from the engine's profile (dim 2 of input_ids).
        eng = self.runner.engine
        try:
            _, opt, mx = eng.get_tensor_profile_shape("input_ids", 0)
            self.batch = int(opt[0])
            self.min_len = int(eng.get_tensor_profile_shape("input_ids", 0)[0][2])
            self.max_len = int(mx[2])
        except Exception:
            self.batch, self.min_len, self.max_len = 2, 1, 4096

    def __call__(self, input_ids=None, audio_mask=None, attention_mask=None, **kw):
        b, _, s = input_ids.shape
        if b != self.batch or s < self.min_len or s > self.max_len or attention_mask is None:
            return self._orig_forward(
                input_ids=input_ids, audio_mask=audio_mask,
                attention_mask=attention_mask, **kw)
        feed = {
            "input_ids": input_ids.to(self.device, torch.int64).contiguous(),
            "audio_mask": audio_mask.to(self.device, torch.bool).contiguous(),
            "attention_mask": attention_mask.to(self.device, torch.bool).contiguous(),
        }
        outs = self.runner.infer(feed, return_host=False)
        logits = outs[self.runner.output_names[0]].float()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        return SimpleNamespace(logits=logits)


class TRTTTSEngine(TorchTTSEngine):
    """PyTorch backbone + TensorRT vocoder (drop-in for the baseline)."""

    backend = "pytorch+trt-vocoder"

    def __init__(self, model_id: str = "k2-fsa/OmniVoice", device: str = "cuda:0",
                 dtype: str = "float16", plan_path: Optional[str] = None):
        super().__init__(model_id=model_id, device=device, dtype=dtype)
        self.vocoder = TRTVocoder(plan_path=plan_path, device=device)
        # Swap the audio tokenizer's decode for the TensorRT engine.
        self._orig_decode = self.model.audio_tokenizer.decode
        self.model.audio_tokenizer.decode = self.vocoder.decode
        self.backend = f"pytorch+trt-vocoder({os.path.basename(self.vocoder.plan_path)})"


class TRTFullEngine(TRTTTSEngine):
    """Fully TensorRT-accelerated: TRT diffusion backbone + TRT vocoder.

    The N-step unmasking loop stays in Python (sampling / confidence selection /
    scatter run in torch) but each denoising forward is executed by the compiled
    backbone engine, and waveform synthesis by the vocoder engine.
    """

    backend = "trt-backbone+trt-vocoder"

    def __init__(self, model_id: str = "k2-fsa/OmniVoice", device: str = "cuda:0",
                 dtype: str = "float16", plan_path: Optional[str] = None,
                 backbone_plan: Optional[str] = None):
        super().__init__(model_id=model_id, device=device, dtype=dtype, plan_path=plan_path)
        self.backbone = TRTBackbone(self.model, plan_path=backbone_plan, device=device)
        # Route the model's forward (called inside _generate_iterative) through TRT.
        self.model.forward = self.backbone
        self.backend = (f"trt-backbone({os.path.basename(self.backbone.plan_path)})"
                        f"+trt-vocoder({os.path.basename(self.vocoder.plan_path)})")
