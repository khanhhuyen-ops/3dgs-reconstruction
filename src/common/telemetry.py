"""Sample device-wide VRAM for external processes; distinguish it from tensor peaks."""
from __future__ import annotations

import shutil
import subprocess
import threading
import time


class VramMonitor:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.samples: list[float] = []
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.started = 0.0

    def _sample(self) -> None:
        binary = shutil.which("nvidia-smi")
        if not binary:
            return
        while not self.stop_event.is_set():
            try:
                result = subprocess.run([binary, "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"], capture_output=True, text=True, timeout=5, check=False)
                if result.returncode == 0:
                    self.samples.append(float(result.stdout.strip()))
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self.stop_event.wait(0.5)

    def __enter__(self) -> VramMonitor:
        self.started = time.perf_counter()
        self.thread.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop_event.set()
        self.thread.join(timeout=6)

    def report(self) -> dict:
        return {"runtime_seconds": time.perf_counter() - self.started,
                "peak_device_memory_used_mib_sampled": max(self.samples) if self.samples else None,
                "vram_samples": len(self.samples), "sampling_interval_seconds": 0.5,
                "memory_scope": "device-wide sampled VRAM; includes other processes; not exact process allocation peak"}

