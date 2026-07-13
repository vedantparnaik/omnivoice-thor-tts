"""Compile the OmniVoice backbone ONNX into a TensorRT engine.

The backbone has three inputs with a fixed CFG batch of 2 (cond+uncond for B=1)
and a dynamic sequence length:

    input_ids       (2, 8, seq)      int64
    audio_mask      (2, seq)         bool
    attention_mask  (2, 1, seq, seq) bool

We build a single dynamic-seq engine covering short-to-long sentences. FP16 is
attempted first (the transformer is mostly matmul/softmax/RMSNorm); if the
TRT-10.13/Thor Myelin nvrtc path chokes on the RoPE sin/cos fusion — the same
class of failure seen on the vocoder Snake activation — fall back to FP32.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/usr/lib/python3.12/dist-packages")

import tensorrt as trt  # noqa: E402

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

_NORM_KEYS = ("norm", "pow", "sqrt", "variance", "mean", "rms", "reciprocal")


def _pin_norm_layers_fp32(network) -> int:
    """Force normalization/reduce/pow-style layers to FP32.

    RMSNorm computes mean(x^2); the x^2 overflows FP16 (max 65504) on this model,
    collapsing the whole engine to zeros. Keeping just these layers in FP32 while
    the heavy matmuls stay in FP16 is standard mixed-precision practice.
    """
    pinned = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        is_norm = layer.type in (trt.LayerType.NORMALIZATION, trt.LayerType.REDUCE)
        if not is_norm:
            name = (layer.name or "").lower()
            is_norm = any(k in name for k in _NORM_KEYS)
        if not is_norm:
            continue
        try:
            layer.precision = trt.DataType.FLOAT
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.DataType.FLOAT)
            pinned += 1
        except Exception:
            pass
    return pinned


class _BackboneCalibrator(trt.IInt8EntropyCalibrator2):
    """Feeds recorded (input_ids, audio_mask, attention_mask) batches for INT8 PTQ.

    All batches share a fixed calibration length (see capture_backbone_calib.py);
    TensorRT observes activation ranges to pick per-tensor INT8 scales. int64/bool
    inputs are indices/masks (not quantized) but must still be provided.
    """

    def __init__(self, calib_file: str, cache_path: str, device: str = "cuda:0"):
        super().__init__()
        import torch
        data = torch.load(calib_file)
        self.calib_len = int(data["calib_len"])
        self.device = device
        self._t = {
            "input_ids": data["input_ids"].to(device),        # [N,2,8,L] int64
            "audio_mask": data["audio_mask"].to(device),       # [N,2,L]   bool
            "attention_mask": data["attention_mask"].to(device),  # [N,2,1,L,L] bool
        }
        self.n = self._t["input_ids"].shape[0]
        self.idx = 0
        self.cache_path = cache_path
        # persistent per-input device buffers (one batch each)
        self._buf = {k: v[0].clone().contiguous() for k, v in self._t.items()}

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.idx >= self.n:
            return None
        ptrs = []
        for name in names:
            self._buf[name].copy_(self._t[name][self.idx])
            ptrs.append(int(self._buf[name].data_ptr()))
        self.idx += 1
        return ptrs

    def read_calibration_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_path, "wb") as f:
            f.write(cache)


def build(onnx_path: str, out_path: str, precision: str,
          min_len: int, opt_len: int, max_len: int,
          batch: int, num_cb: int, workspace_gb: int, opt_level,
          mixed_bf16: bool = False, keep_norm_fp32: bool = False,
          calib_file: str = None) -> bool:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"parsing {onnx_path} ...", flush=True)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print("  PARSER ERROR:", parser.get_error(i))
        return False
    print(f"network: {network.num_layers} layers, "
          f"{network.num_inputs} inputs, {network.num_outputs} outputs", flush=True)
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"  in[{i}] {t.name} {t.dtype} {tuple(t.shape)}", flush=True)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if opt_level is not None:
        config.builder_optimization_level = opt_level
        print(f"builder_optimization_level={opt_level}", flush=True)

    if precision == "fp32":
        pass
    elif precision == "fp16":
        assert builder.platform_has_fast_fp16
        config.set_flag(trt.BuilderFlag.FP16)
        if mixed_bf16:
            # Let TRT auto-pick BF16 for layers that would overflow in FP16
            # (e.g. RMSNorm x^2) while keeping FP16 elsewhere.
            config.set_flag(trt.BuilderFlag.BF16)
            print("mixed FP16+BF16: builder auto-selects per layer", flush=True)
        if keep_norm_fp32:
            # RMSNorm squares activations (x^2) which overflows FP16's 65504
            # ceiling -> all-zero logits. Pin normalization / reduce / pow / div
            # / sqrt layers to FP32 and force the builder to obey it.
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            pinned = _pin_norm_layers_fp32(network)
            print(f"kept {pinned} norm/reduce/pow layers in FP32", flush=True)
    elif precision == "bf16":
        # BF16 keeps FP32's exponent range (no overflow like FP16) while still
        # using Blackwell tensor cores — the right fit for a transformer whose
        # FP16 build collapses to zeros from layernorm/attention overflow.
        config.set_flag(trt.BuilderFlag.BF16)
    elif precision == "int8":
        assert builder.platform_has_fast_int8
        assert calib_file and os.path.exists(calib_file), "int8 needs --calib-file"
        config.set_flag(trt.BuilderFlag.INT8)
        # BF16 as the fallback for non-INT8 layers — avoids the FP16 RMSNorm
        # overflow that would otherwise re-appear via INT8's mixed-precision path.
        config.set_flag(trt.BuilderFlag.BF16)
        if keep_norm_fp32:
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            print(f"kept {_pin_norm_layers_fp32(network)} norm layers in FP32", flush=True)
        cache = os.path.join(os.path.dirname(out_path), "backbone_int8_calib.cache")
        _calibrator = _BackboneCalibrator(calib_file, cache)
        config.int8_calibrator = _calibrator
        print(f"INT8 PTQ: {_calibrator.n} calib batches @ L={_calibrator.calib_len}", flush=True)

    # One profile covering all three inputs. Batch fixed (CFG=2), seq dynamic.
    profile = builder.create_optimization_profile()
    shapes = {
        "input_ids":      ((batch, num_cb, min_len), (batch, num_cb, opt_len), (batch, num_cb, max_len)),
        "audio_mask":     ((batch, min_len),         (batch, opt_len),         (batch, max_len)),
        "attention_mask": ((batch, 1, min_len, min_len), (batch, 1, opt_len, opt_len), (batch, 1, max_len, max_len)),
    }
    for i in range(network.num_inputs):
        name = network.get_input(i).name
        mn, op, mx = shapes[name]
        profile.set_shape(name, mn, op, mx)
        print(f"profile[{name}]: min={mn} opt={op} max={mx}", flush=True)
    config.add_optimization_profile(profile)

    # INT8 calibration on dynamic shapes needs a fixed-shape calibration profile.
    if precision == "int8":
        L = _calibrator.calib_len
        calib_profile = builder.create_optimization_profile()
        calib_shapes = {
            "input_ids": (batch, num_cb, L),
            "audio_mask": (batch, L),
            "attention_mask": (batch, 1, L, L),
        }
        for i in range(network.num_inputs):
            name = network.get_input(i).name
            s = calib_shapes[name]
            calib_profile.set_shape(name, s, s, s)
        config.set_calibration_profile(calib_profile)
        print(f"calibration profile fixed @ L={L}", flush=True)

    t0 = time.time()
    print(f"building {precision} engine (0.6B transformer — this can take several minutes)...", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print(f"[fail] {precision} engine build returned None", flush=True)
        return False
    build_s = time.time() - t0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(serialized)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[done] wrote {out_path} ({size_mb:.1f} MB) in {build_s:.1f}s", flush=True)

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized)
    inspector = engine.create_engine_inspector()
    try:
        layers = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON)).get("Layers", [])
    except Exception:
        layers = []
    print(f"engine fused layers: {len(layers)} (from {network.num_layers} network layers)", flush=True)
    report = {
        "component": "backbone",
        "precision": precision,
        "engine_mb": round(size_mb, 1),
        "build_s": round(build_s, 1),
        "network_layers": network.num_layers,
        "engine_layers": len(layers),
        "profile": {"min": min_len, "opt": opt_len, "max": max_len, "batch": batch},
    }
    with open("artifacts/trt_backbone_build_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[done] wrote artifacts/trt_backbone_build_report.json", flush=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="artifacts/backbone.onnx")
    ap.add_argument("--out", default="artifacts/backbone.plan")
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16", "int8"])
    ap.add_argument("--calib-file", default="artifacts/backbone_calib.pt",
                    help="int8: recorded calibration batches (from capture_backbone_calib.py)")
    ap.add_argument("--fallback-fp32", action="store_true",
                    help="if fp16 build fails, retry as fp32")
    ap.add_argument("--min-len", type=int, default=32)
    ap.add_argument("--opt-len", type=int, default=384)
    ap.add_argument("--max-len", type=int, default=1200)
    ap.add_argument("--workspace-gb", type=int, default=8)
    ap.add_argument("--opt-level", type=int, default=None)
    ap.add_argument("--mixed-bf16", action="store_true",
                    help="fp16 build: also allow BF16 so TRT auto-avoids overflow")
    ap.add_argument("--keep-norm-fp32", action="store_true",
                    help="fp16 build: pin norm/reduce/pow layers to FP32 (fixes RMSNorm overflow)")
    args = ap.parse_args()

    meta = {}
    if os.path.exists("artifacts/backbone_meta.json"):
        with open("artifacts/backbone_meta.json") as f:
            meta = json.load(f)
    num_cb = int(meta.get("num_audio_codebook", 8))
    batch = int(meta.get("batch_2b", 2))

    ok = build(args.onnx, args.out, args.precision,
               args.min_len, args.opt_len, args.max_len,
               batch, num_cb, args.workspace_gb, args.opt_level,
               mixed_bf16=args.mixed_bf16, keep_norm_fp32=args.keep_norm_fp32,
               calib_file=args.calib_file)

    if not ok and args.precision == "fp16" and args.fallback_fp32:
        print("\n[fallback] fp16 failed — retrying fp32 ...", flush=True)
        out32 = args.out.replace(".plan", "_fp32.plan") if "_fp32" not in args.out else args.out
        ok = build(args.onnx, out32, "fp32",
                   args.min_len, args.opt_len, args.max_len,
                   batch, num_cb, args.workspace_gb, args.opt_level)

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
