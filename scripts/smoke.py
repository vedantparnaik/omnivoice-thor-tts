import os
import time

import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice

t0 = time.time()
print("loading model...", flush=True)
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
print(f"loaded in {time.time() - t0:.1f}s; sampling_rate={getattr(model, 'sampling_rate', None)}", flush=True)

text = "Hello from the Jetson Thor. This is a real time text to speech test."

# warmup (kernels/autotune)
_ = model.generate(text, language="English", num_step=16)
torch.cuda.synchronize()

t1 = time.time()
audio = model.generate(text, language="English", num_step=32)[0]
torch.cuda.synchronize()
dt = time.time() - t1
dur = len(audio) / model.sampling_rate
print(f"gen wall={dt:.3f}s audio_dur={dur:.3f}s RTF={dt / dur:.3f}", flush=True)

os.makedirs("artifacts", exist_ok=True)
sf.write("artifacts/smoke.wav", audio, model.sampling_rate)
print("wrote artifacts/smoke.wav; VRAM alloc MB=", round(torch.cuda.memory_allocated() / 1e6, 1), flush=True)
