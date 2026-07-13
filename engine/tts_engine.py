"""OmniVoice TTS engines with streaming + edge benchmarking instrumentation.

``BaseTTSEngine`` implements the streaming/metrics machinery once. Concrete
engines (PyTorch here, TensorRT in Phase 3) only need to implement
``synthesize_chunk`` and expose ``sampling_rate``. This keeps the web API stable
while the underlying inference backend is swapped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .metrics import ChunkMetrics, StreamMetrics
from .text_chunker import split_into_chunks


@dataclass
class GenParams:
    """User-tunable generation parameters exposed by the playground."""

    language: Optional[str] = "English"
    instruct: Optional[str] = None          # voice-design style prompt
    num_step: int = 32                       # diffusion / iterative steps
    guidance_scale: float = 2.0              # classifier-free guidance
    position_temperature: float = 5.0        # position-selection temperature
    class_temperature: float = 0.0           # token-sampling temperature (0=greedy)
    speed: Optional[float] = None
    seed: Optional[int] = None
    # streaming controls
    max_words: int = 50
    first_chunk_words: int = 12

    def gen_kwargs(self) -> dict:
        return dict(
            language=self.language,
            instruct=self.instruct,
            num_step=int(self.num_step),
            guidance_scale=float(self.guidance_scale),
            position_temperature=float(self.position_temperature),
            class_temperature=float(self.class_temperature),
            speed=self.speed,
        )


class BaseTTSEngine:
    """Backend-agnostic streaming + metrics logic."""

    #: audio sample rate in Hz (set by subclasses)
    sampling_rate: int = 24000
    #: short human label for the active backend (shown in the UI)
    backend: str = "base"

    def synthesize_chunk(self, text: str, params: GenParams) -> np.ndarray:
        """Return a 1-D float32 waveform for a single text chunk."""
        raise NotImplementedError

    def _vram_mb(self) -> Optional[float]:
        return None

    def _reset_peak_vram(self) -> None:
        pass

    def _peak_vram_mb(self) -> Optional[float]:
        return None

    def stream(
        self, text: str, params: Optional[GenParams] = None
    ) -> Iterator[Tuple[np.ndarray, ChunkMetrics]]:
        """Yield ``(audio_chunk, ChunkMetrics)`` as each chunk is synthesized."""
        params = params or GenParams()
        chunks = split_into_chunks(
            text, max_words=params.max_words, first_chunk_words=params.first_chunk_words
        )
        self._reset_peak_vram()
        t_start = time.perf_counter()
        for i, chunk_text in enumerate(chunks):
            t0 = time.perf_counter()
            audio = self.synthesize_chunk(chunk_text, params)
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            gen_wall = time.perf_counter() - t0
            audio_dur = len(audio) / self.sampling_rate if len(audio) else 0.0
            m = ChunkMetrics(
                index=i,
                text=chunk_text,
                gen_wall_s=gen_wall,
                audio_dur_s=audio_dur,
                rtf=(gen_wall / audio_dur) if audio_dur > 0 else float("inf"),
                cumulative_wall_s=time.perf_counter() - t_start,
                vram_alloc_mb=self._vram_mb(),
            )
            yield audio, m

    def synthesize(
        self, text: str, params: Optional[GenParams] = None
    ) -> Tuple[np.ndarray, StreamMetrics]:
        """Non-streaming convenience wrapper. Returns full audio + metrics."""
        params = params or GenParams()
        sm = StreamMetrics()
        audio_parts: List[np.ndarray] = []
        t_start = time.perf_counter()
        for audio, cm in self.stream(text, params):
            if sm.ttfa_ms is None:
                sm.ttfa_ms = cm.cumulative_wall_s * 1000.0
            audio_parts.append(audio)
            sm.chunks.append(cm)
            sm.total_audio_s += cm.audio_dur_s
        sm.total_wall_s = time.perf_counter() - t_start
        sm.peak_vram_mb = self._peak_vram_mb()
        sm.finalize()
        full = np.concatenate(audio_parts) if audio_parts else np.zeros(0, np.float32)
        return full, sm


class TorchTTSEngine(BaseTTSEngine):
    """PyTorch FP16 OmniVoice backend (baseline)."""

    backend = "pytorch-fp16"

    def __init__(
        self,
        model_id: str = "k2-fsa/OmniVoice",
        device: str = "cuda:0",
        dtype: str = "float16",
    ):
        import torch
        from omnivoice import OmniVoice

        self._torch = torch
        torch_dtype = getattr(torch, dtype)
        self.device = device
        self.model = OmniVoice.from_pretrained(
            model_id, device_map=device, dtype=torch_dtype
        )
        self.model.eval()
        self.sampling_rate = int(getattr(self.model, "sampling_rate", 24000))

    def synthesize_chunk(self, text: str, params: GenParams) -> np.ndarray:
        torch = self._torch
        if params.seed is not None:
            torch.manual_seed(int(params.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(params.seed))
        with torch.inference_mode():
            out = self.model.generate(text, **params.gen_kwargs())
        torch.cuda.synchronize()
        return np.asarray(out[0], dtype=np.float32).reshape(-1)

    def _vram_mb(self) -> Optional[float]:
        return round(self._torch.cuda.memory_allocated() / 1e6, 1)

    def _reset_peak_vram(self) -> None:
        self._torch.cuda.reset_peak_memory_stats()

    def _peak_vram_mb(self) -> Optional[float]:
        return round(self._torch.cuda.max_memory_allocated() / 1e6, 1)


class MockTTSEngine(BaseTTSEngine):
    """GPU-free stand-in used to develop/test the server + UI + streaming.

    Produces a short sine tone per chunk (pitch varies by chunk) with a small
    artificial compute delay proportional to requested ``num_step``, so TTFA/RTF
    and the producer/consumer overlap behave realistically without the model.
    """

    backend = "mock-cpu"

    def __init__(self, sampling_rate: int = 24000, rtf: float = 0.25):
        import time as _t

        self._time = _t
        self.sampling_rate = sampling_rate
        self._rtf = rtf

    def synthesize_chunk(self, text: str, params: GenParams) -> np.ndarray:
        words = max(1, len(text.split()))
        audio_dur = min(15.0, 0.35 * words)  # ~0.35s per word
        n = int(audio_dur * self.sampling_rate)
        t = np.arange(n, dtype=np.float32) / self.sampling_rate
        freq = 180.0 + 40.0 * (hash(text) % 8)
        wave = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        env = np.minimum(1.0, np.minimum(t * 20, (audio_dur - t) * 20)).astype(np.float32)
        wave *= np.clip(env, 0.0, 1.0)
        # emulate compute time scaled by diffusion steps and target RTF
        step_factor = params.num_step / 32.0
        self._time.sleep(audio_dur * self._rtf * step_factor)
        return wave
