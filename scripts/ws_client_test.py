"""Headless test of the playground WebSockets (works with the mock engine)."""

import asyncio
import json
import struct
import sys

import websockets

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8008"


async def test_tts():
    async with websockets.connect(URL + "/ws/tts", max_size=None) as ws:
        await ws.send(json.dumps({
            "text": "Hello from the edge. This is chunk two of the stream. And a final sentence here.",
            "num_step": 32, "first_chunk_words": 8, "max_words": 30, "seed": 7,
        }))
        chunks = 0
        total_samples = 0
        ttfa = None
        pending = None
        while True:
            msg = await ws.recv()
            if isinstance(msg, (bytes, bytearray)):
                n = len(msg) // 2
                total_samples += n
                # sanity: ensure it decodes as int16
                struct.unpack("<%dh" % n, msg)
                chunks += 1
                pending = None
            else:
                d = json.loads(msg)
                if d["type"] == "chunk_meta":
                    pending = d
                    if ttfa is None:
                        ttfa = d["ttfa_ms"]
                    print(f"  chunk {d['index']}: rtf={d['rtf']} gen={d['gen_wall_s']}s "
                          f"audio={d['audio_dur_s']}s '{d['text'][:40]}'")
                elif d["type"] == "start":
                    print(f"  start backend={d['backend']} sr={d['sample_rate']}")
                    sr = d["sample_rate"]
                elif d["type"] == "done":
                    print(f"  done ttfa={d['ttfa_ms']}ms total_audio={d['total_audio_s']}s")
                    break
                elif d["type"] == "error":
                    print("  ERROR", d["message"]); break
        assert chunks >= 2, f"expected multiple chunks, got {chunks}"
        assert total_samples > 0
        print(f"[tts] OK chunks={chunks} samples={total_samples} ttfa={ttfa}ms")


async def test_telemetry():
    async with websockets.connect(URL + "/ws/telemetry") as ws:
        for _ in range(3):
            d = json.loads(await ws.recv())
            print("  tele:", {k: d.get(k) for k in
                              ["ram_used_mb", "gpu_temp_c", "gpu_power_mw", "board_power_mw", "gpu_util_pct"]})
        print("[telemetry] OK")


async def main():
    print("== TTS stream ==")
    await test_tts()
    print("== Telemetry ==")
    await test_telemetry()
    print("ALL OK")


asyncio.run(main())
