"""FastAPI web playground for the OmniVoice edge TTS pipeline.

Endpoints
---------
GET  /                  -> static playground UI
GET  /api/info          -> backend + sample-rate metadata
WS   /ws/tts            -> submit text+params, stream audio chunks + per-chunk metrics
WS   /ws/telemetry      -> live Jetson telemetry (unified mem, power, temp, GPU util)

Audio is streamed as interleaved frames: a JSON text frame describing the next
chunk, immediately followed by a binary frame of little-endian int16 PCM mono at
the engine sample rate. This keeps the browser side dependency-free (Web Audio).

A producer thread runs the (blocking) engine.stream() and hands chunks to the
asyncio event loop via a queue, so chunk N+1 is synthesized while chunk N is
still being sent to the client (the producer/consumer overlap the assignment
asks for).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import GenParams  # noqa: E402
from engine.telemetry import TelemetrySampler  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")

app = FastAPI(title="OmniVoice Thor TTS Playground")

_engine = None
_engine_lock = threading.Lock()
_telemetry: Optional[TelemetrySampler] = None


def get_engine():
    """Lazily construct the configured engine (mock | pytorch | trt)."""
    global _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        backend = os.environ.get("OMNIVOICE_ENGINE", "mock").lower()
        if backend == "mock":
            from engine import MockTTSEngine
            _engine = MockTTSEngine()
        elif backend in ("pytorch", "torch"):
            from engine import TorchTTSEngine
            _engine = TorchTTSEngine()
        elif backend == "trt":
            from engine.trt_engine import TRTTTSEngine  # type: ignore
            _engine = TRTTTSEngine()
        elif backend in ("trt-full", "trtfull"):
            from engine.trt_engine import TRTFullEngine  # type: ignore
            _engine = TRTFullEngine()
        else:
            raise ValueError(f"Unknown OMNIVOICE_ENGINE={backend!r}")
        return _engine


@app.on_event("startup")
def _startup() -> None:
    global _telemetry
    _telemetry = TelemetrySampler(interval_ms=int(os.environ.get("TELEMETRY_MS", "500")))
    _telemetry.start()
    # Preload the engine unless it is the (cheap) mock, so the first request is fast.
    if os.environ.get("OMNIVOICE_ENGINE", "mock").lower() != "mock" and \
       os.environ.get("PRELOAD", "1") == "1":
        threading.Thread(target=get_engine, daemon=True).start()


@app.on_event("shutdown")
def _shutdown() -> None:
    if _telemetry is not None:
        _telemetry.stop()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/info")
def info() -> dict:
    backend = os.environ.get("OMNIVOICE_ENGINE", "mock").lower()
    data = {"configured_backend": backend, "loaded": _engine is not None}
    if _engine is not None:
        data["backend"] = _engine.backend
        data["sampling_rate"] = _engine.sampling_rate
    return data


def _params_from_msg(msg: dict) -> GenParams:
    p = GenParams()
    for k in (
        "language", "instruct", "num_step", "guidance_scale",
        "position_temperature", "class_temperature", "speed", "seed",
        "max_words", "first_chunk_words",
    ):
        if k in msg and msg[k] is not None and msg[k] != "":
            setattr(p, k, msg[k])
    p.num_step = int(p.num_step)
    p.max_words = int(p.max_words)
    p.first_chunk_words = int(p.first_chunk_words)
    p.guidance_scale = float(p.guidance_scale)
    p.position_temperature = float(p.position_temperature)
    p.class_temperature = float(p.class_temperature)
    if p.seed is not None:
        p.seed = int(p.seed)
    return p


def _to_pcm16(audio: np.ndarray) -> bytes:
    a = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (a * 32767.0).astype("<i2").tobytes()


@app.websocket("/ws/tts")
async def ws_tts(ws: WebSocket) -> None:
    await ws.accept()
    loop = asyncio.get_running_loop()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            text = (msg.get("text") or "").strip()
            if not text:
                await ws.send_text(json.dumps({"type": "error", "message": "empty text"}))
                continue
            params = _params_from_msg(msg)
            engine = await loop.run_in_executor(None, get_engine)

            await ws.send_text(json.dumps({
                "type": "start",
                "backend": engine.backend,
                "sample_rate": engine.sampling_rate,
            }))

            # Producer thread -> asyncio queue -> consumer (this coroutine).
            queue: asyncio.Queue = asyncio.Queue(maxsize=8)
            SENTINEL = object()

            def produce() -> None:
                try:
                    for audio, cm in engine.stream(text, params):
                        loop.call_soon_threadsafe(queue.put_nowait, (audio, cm))
                    loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)
                except Exception as e:  # surface engine errors to the client
                    loop.call_soon_threadsafe(queue.put_nowait, ("__error__", str(e)))

            threading.Thread(target=produce, daemon=True).start()

            total_audio = 0.0
            ttfa_ms = None
            peak_vram = None
            while True:
                item = await queue.get()
                if item is SENTINEL:
                    break
                first, second = item
                if isinstance(first, str) and first == "__error__":
                    await ws.send_text(json.dumps({"type": "error", "message": second}))
                    break
                audio, cm = first, second
                if ttfa_ms is None:
                    ttfa_ms = round(cm.cumulative_wall_s * 1000.0, 1)
                total_audio += cm.audio_dur_s
                if cm.vram_alloc_mb is not None:
                    peak_vram = max(peak_vram or 0.0, cm.vram_alloc_mb)
                await ws.send_text(json.dumps({
                    "type": "chunk_meta",
                    "index": cm.index,
                    "text": cm.text,
                    "gen_wall_s": round(cm.gen_wall_s, 3),
                    "audio_dur_s": round(cm.audio_dur_s, 3),
                    "rtf": round(cm.rtf, 3),
                    "ttfa_ms": ttfa_ms,
                    "vram_alloc_mb": cm.vram_alloc_mb,
                    "n_samples": int(len(audio)),
                }))
                await ws.send_bytes(_to_pcm16(audio))

            overall_rtf = None
            if total_audio > 0 and ttfa_ms is not None:
                # recompute overall from cumulative wall of last chunk is not tracked here;
                # client also computes it, this is a convenience summary.
                pass
            await ws.send_text(json.dumps({
                "type": "done",
                "ttfa_ms": ttfa_ms,
                "total_audio_s": round(total_audio, 3),
                "peak_vram_mb": peak_vram,
            }))
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            snap = _telemetry.snapshot() if _telemetry else {}
            await ws.send_text(json.dumps({"type": "telemetry", **snap}))
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return
    except Exception:
        return


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
