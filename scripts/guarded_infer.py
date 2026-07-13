"""Single guarded real inference (used while validating GPU stability).

Runs exactly ONE generation (no warmup loop) to minimise sustained load, and
prints metrics. Args: --num-step N  --text "..."
"""

import argparse
import os
import sys
import time

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import GenParams, TorchTTSEngine  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-step", type=int, default=24)
    ap.add_argument("--text", default="Hello from the Jetson Thor.")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    eng = TorchTTSEngine()
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    audio, sm = eng.synthesize(args.text, GenParams(num_step=args.num_step, seed=args.seed))
    dt = time.time() - t1
    sf.write("artifacts/guarded.wav", audio, eng.sampling_rate)
    print(f"[gen] wall={dt:.3f}s TTFA={sm.ttfa_ms:.0f}ms RTF={sm.overall_rtf:.3f} "
          f"audio={sm.total_audio_s:.2f}s peakVRAM={sm.peak_vram_mb:.0f}MB chunks={sm.n_chunks}",
          flush=True)
    print("OK_GUARDED", flush=True)


if __name__ == "__main__":
    main()
