"""Metric containers for edge TTS benchmarking.

RTF (Real-Time Factor) = wall_seconds / audio_seconds. Lower is better; < 1.0
means faster than real time. TTFA (Time-To-First-Audio) is the latency from
request start to the first audio samples being ready to play.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class ChunkMetrics:
    index: int
    text: str
    gen_wall_s: float
    audio_dur_s: float
    rtf: float
    cumulative_wall_s: float
    vram_alloc_mb: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StreamMetrics:
    ttfa_ms: Optional[float] = None
    total_wall_s: float = 0.0
    total_audio_s: float = 0.0
    overall_rtf: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    n_chunks: int = 0
    chunks: List[ChunkMetrics] = field(default_factory=list)

    def finalize(self) -> "StreamMetrics":
        self.n_chunks = len(self.chunks)
        if self.total_audio_s > 0:
            self.overall_rtf = self.total_wall_s / self.total_audio_s
        return self

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["chunks"] = [c.to_dict() for c in self.chunks]
        return d
