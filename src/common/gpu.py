"""Verify CUDA by executing actual operations; never fall back to CPU."""
from __future__ import annotations

import time
import os

from src.common.io import PipelineError


def require_cuda():
    try:
        import torch
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != "0":
            raise RuntimeError("GPU policy requires physical GPU 0; unset CUDA_VISIBLE_DEVICES or set it to 0")
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        torch.cuda.set_device(0)
        return torch
    except (ImportError, RuntimeError) as exc:
        raise PipelineError("STOP_GPU_UNAVAILABLE", f"CUDA required on cuda:0: {exc}") from exc


def audit_cuda() -> dict:
    started = time.perf_counter()
    try:
        torch = require_cuda()
        torch.cuda.reset_peak_memory_stats(0)
        a = torch.arange(16, dtype=torch.float32, device="cuda:0").reshape(4, 4)
        b = a @ a
        torch.cuda.synchronize(0)
        if b.device.type != "cuda" or b.sum().item() != 3920:
            raise RuntimeError("CUDA computation verification failed")
        report = {"status": "SUCCESS", "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
                  "device": str(b.device), "name": torch.cuda.get_device_name(0), "result_sum": b.sum().item(),
                  "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                  "runtime_seconds": time.perf_counter() - started}
        del a, b
        torch.cuda.empty_cache()
        return report
    except (PipelineError, RuntimeError) as exc:
        return {"status": "STOP_GPU_UNAVAILABLE", "error": str(exc), "runtime_seconds": time.perf_counter() - started}
