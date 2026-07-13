"""Live Jetson telemetry via ``tegrastats`` (plus optional nvidia-smi GPU util).

``tegrastats`` on Thor emits lines like::

    07-01-2026 12:22:55 RAM 8739/125771MB (lfb 1x4MB) CPU [...] cpu@40C tj@41C
    gpu@41C ... VDD_GPU 12037mW/7676mW VDD_CPU_SOC_MSS 10484mW/9314mW
    VIN_SYS_5V0 8696mW/7461mW VIN 34842mW/30476mW

We parse unified-memory footprint, temperatures and power rails. GPU utilisation
percentage is not present in Thor's tegrastats output, so it is sampled
separately (best-effort) from ``nvidia-smi`` if available.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from typing import Dict, Optional

_RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
_TEMP_RE = re.compile(r"(\w+)@([\d.]+)C")
_POWER_RE = re.compile(r"(VDD_\w+|VIN(?:_\w+)?) (\d+)mW/(\d+)mW")


def parse_tegrastats_line(line: str) -> Dict:
    out: Dict = {"ts": time.time()}

    m = _RAM_RE.search(line)
    if m:
        out["ram_used_mb"] = int(m.group(1))
        out["ram_total_mb"] = int(m.group(2))

    temps = {name: float(val) for name, val in _TEMP_RE.findall(line)}
    if temps:
        out["temps_c"] = temps
        # convenient top-level fields
        if "gpu" in temps:
            out["gpu_temp_c"] = temps["gpu"]
        if "cpu" in temps:
            out["cpu_temp_c"] = temps["cpu"]

    power = {}
    for name, cur, avg in _POWER_RE.findall(line):
        power[name] = {"cur_mw": int(cur), "avg_mw": int(avg)}
    if power:
        out["power_mw"] = power
        if "VDD_GPU" in power:
            out["gpu_power_mw"] = power["VDD_GPU"]["cur_mw"]
        if "VIN" in power:
            out["board_power_mw"] = power["VIN"]["cur_mw"]

    return out


class TelemetrySampler:
    """Background sampler exposing the latest telemetry snapshot."""

    def __init__(self, interval_ms: int = 500):
        self.interval_ms = interval_ms
        self._latest: Dict = {}
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._smi_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._have_tegrastats = shutil.which("tegrastats") is not None
        self._have_smi = shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        if self._have_tegrastats:
            self._thread = threading.Thread(target=self._run_tegrastats, daemon=True)
            self._thread.start()
        if self._have_smi:
            self._smi_thread = threading.Thread(target=self._run_smi, daemon=True)
            self._smi_thread.start()

    def _run_tegrastats(self) -> None:
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception:
            self._have_tegrastats = False
            return
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            sample = parse_tegrastats_line(line)
            with self._lock:
                self._latest.update(sample)

    def _run_smi(self) -> None:
        while not self._stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3,
                )
                val = r.stdout.strip().splitlines()[0].strip()
                if val and val.upper() not in ("N/A", "[N/A]"):
                    with self._lock:
                        self._latest["gpu_util_pct"] = float(val)
            except Exception:
                pass
            self._stop.wait(max(0.5, self.interval_ms / 1000.0))

    def snapshot(self) -> Dict:
        with self._lock:
            return dict(self._latest)

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    s = TelemetrySampler(interval_ms=500)
    s.start()
    try:
        for _ in range(6):
            time.sleep(1)
            print(s.snapshot())
    finally:
        s.stop()
