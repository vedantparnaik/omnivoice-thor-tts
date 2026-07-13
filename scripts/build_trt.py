"""Compile the vocoder ONNX into a TensorRT engine (FP16, dynamic profile).

Demonstrates the required edge-compilation pipeline:
  - ONNX -> TensorRT network parse
  - FP16 precision kernels
  - dynamic optimization profile for conversational sentence lengths
  - layer-fusion reporting via the engine inspector

Output: artifacts/vocoder_fp16.plan (serialized engine) + a build report.
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
_OPT_LEVEL = None  # set from CLI
_KEEP_SNAKE_FP32 = False  # set from CLI
_STRONGLY_TYPED = False  # set from CLI

# Snake activation = x + (1/a)*sin(a*x)^2 — the sin/pow/reciprocal fusion is what
# crashes Myelin's FP16 codegen on sm_110. Pinning these ops to FP32 lets the rest
# of the vocoder run in FP16 without triggering the bad kernel.
_SNAKE_KEYS = ("sin", "pow", "reciprocal", "snake", "sqrt", "div")


def _pin_snake_layers_fp32(network) -> int:
    pinned = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        name = (layer.name or "").lower()
        if any(k in name for k in _SNAKE_KEYS):
            try:
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.DataType.FLOAT)
                pinned += 1
            except Exception:
                pass
    return pinned


def build(onnx_path: str, out_path: str, precision: str,
          min_len: int, opt_len: int, max_len: int, num_q: int) -> None:
    builder = trt.Builder(TRT_LOGGER)
    net_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if _STRONGLY_TYPED:
        net_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        print("network: STRONGLY_TYPED (precision taken from ONNX dtypes)", flush=True)
    network = builder.create_network(net_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"parsing {onnx_path} ...", flush=True)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print("  PARSER ERROR:", parser.get_error(i))
        raise SystemExit(1)
    print(f"network: {network.num_layers} layers, "
          f"{network.num_inputs} inputs, {network.num_outputs} outputs", flush=True)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB
    if _OPT_LEVEL is not None:
        config.builder_optimization_level = _OPT_LEVEL
        print(f"builder_optimization_level={_OPT_LEVEL}", flush=True)

    if _STRONGLY_TYPED:
        # In strongly-typed mode TRT infers precision purely from the network
        # tensor dtypes (the ONNX was exported FP16 w/ FP32 snake islands), so we
        # must NOT set any precision builder flags.
        pass
    elif precision == "fp32":
        pass  # default TF32/FP32 kernels, no reduced-precision flags
    elif precision == "fp16":
        assert builder.platform_has_fast_fp16
        config.set_flag(trt.BuilderFlag.FP16)
        if _KEEP_SNAKE_FP32:
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            print(f"kept {_pin_snake_layers_fp32(network)} snake layers in FP32", flush=True)
    elif precision == "int8":
        assert builder.platform_has_fast_int8
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # mixed
        if _KEEP_SNAKE_FP32:
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            print(f"kept {_pin_snake_layers_fp32(network)} snake layers in FP32", flush=True)

    # Dynamic optimization profile: batch=1, num_quantizers fixed, length varies.
    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    name = inp.name
    profile.set_shape(name,
                      (1, num_q, min_len),
                      (1, num_q, opt_len),
                      (1, num_q, max_len))
    config.add_optimization_profile(profile)
    print(f"profile[{name}]: min=(1,{num_q},{min_len}) "
          f"opt=(1,{num_q},{opt_len}) max=(1,{num_q},{max_len})", flush=True)

    t0 = time.time()
    print(f"building {precision} engine (this can take a minute)...", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("engine build failed")
    build_s = time.time() - t0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(serialized)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[done] wrote {out_path} ({size_mb:.1f} MB) in {build_s:.1f}s", flush=True)

    # Report fused layers via the engine inspector.
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized)
    inspector = engine.create_engine_inspector()
    info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    try:
        layers = json.loads(info).get("Layers", [])
    except Exception:
        layers = []
    print(f"engine fused layers: {len(layers)} (from {network.num_layers} network layers)", flush=True)
    report = {
        "precision": precision,
        "engine_mb": round(size_mb, 1),
        "build_s": round(build_s, 1),
        "network_layers": network.num_layers,
        "engine_layers": len(layers),
        "profile": {"min": min_len, "opt": opt_len, "max": max_len, "num_quantizers": num_q},
    }
    with open("artifacts/trt_build_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[done] wrote artifacts/trt_build_report.json", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="artifacts/vocoder.onnx")
    ap.add_argument("--out", default="artifacts/vocoder_fp16.plan")
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"])
    ap.add_argument("--min-len", type=int, default=8)
    ap.add_argument("--opt-len", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=1000)
    ap.add_argument("--opt-level", type=int, default=None,
                    help="TRT builder optimization level 0-5 (lower = less aggressive fusion)")
    ap.add_argument("--keep-snake-fp32", action="store_true",
                    help="fp16/int8: pin snake (sin/pow/reciprocal) layers to FP32 to dodge the Myelin crash")
    ap.add_argument("--strongly-typed", action="store_true",
                    help="build a strongly-typed network (precision from ONNX dtypes; different Myelin path)")
    args = ap.parse_args()

    global _OPT_LEVEL, _KEEP_SNAKE_FP32, _STRONGLY_TYPED
    _OPT_LEVEL = args.opt_level
    _KEEP_SNAKE_FP32 = args.keep_snake_fp32
    _STRONGLY_TYPED = args.strongly_typed

    num_q = 8
    meta_path = "artifacts/vocoder_meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            num_q = int(json.load(f).get("num_quantizers", 8))

    build(args.onnx, args.out, args.precision,
          args.min_len, args.opt_len, args.max_len, num_q)


if __name__ == "__main__":
    main()
