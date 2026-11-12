"""Stream thumbnails through CUDA and select temporally distributed keyframes."""
from __future__ import annotations

import csv
import math
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.gpu import require_cuda
from src.common.io import PipelineError, code_hash, fingerprint, staging_directory, write_json
from src.preprocessing.extract_frames import validate_manifest


def score_frames(directory: Path, frames: list[dict], pre: dict, seed: int = 42) -> tuple[list[dict], dict]:
    torch = require_cuda()
    import torch.nn.functional as functional

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device="cuda:0").reshape(1, 1, 3, 3)
    rows = []
    try:
        with torch.inference_mode():
            for start in range(0, len(frames), pre["batch_size"]):
                batch = frames[start:start + pre["batch_size"]]
                arrays = []
                for row in batch:
                    with Image.open(directory / row["filename"]) as image:
                        image = image.convert("RGB")
                        width = min(pre["thumbnail_width"], image.width)
                        image = image.resize((width, max(4, round(image.height * width / image.width))), Image.Resampling.LANCZOS)
                        arrays.append(np.asarray(image).copy())
                data = torch.from_numpy(np.stack(arrays)).to(device="cuda:0", dtype=torch.float32).permute(0, 3, 1, 2) / 255
                gray = (data * torch.tensor([0.299, 0.587, 0.114], device="cuda:0").view(1, 3, 1, 1)).sum(1, keepdim=True)
                blur = functional.conv2d(gray, kernel).flatten(1).var(1, unbiased=False)
                exposure = ((gray < 0.02) | (gray > 0.98)).float().flatten(1).mean(1)
                descriptors = functional.adaptive_avg_pool2d(gray, (18, 32)).flatten(1)
                if any(x.device.type != "cuda" for x in (data, blur, exposure, descriptors)):
                    raise PipelineError("STOP_GPU_UNAVAILABLE", "Quality tensor left CUDA")
                for row, sharpness, fraction, descriptor in zip(batch, blur.cpu().tolist(), exposure.cpu().tolist(), descriptors.cpu().tolist()):
                    score = sharpness * (1 - fraction)
                    rows.append({**row, "blur_score": sharpness, "exposure_fraction": fraction,
                                 "selection_score": score, "descriptor": descriptor})
                del data, gray, blur, exposure, descriptors
        torch.cuda.synchronize(0)
        report = {"status": "SUCCESS", "device": "cuda:0", "backend": "PyTorch CUDA convolution/reductions",
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(0), "runtime_seconds": time.perf_counter() - started}
        return rows, report
    except torch.cuda.OutOfMemoryError as exc:
        raise PipelineError("STOP_GPU_UNAVAILABLE", "CUDA OOM: reduce batch_size/thumbnail_width in a new experiment; CPU fallback disabled") from exc
    finally:
        del kernel
        torch.cuda.empty_cache()


def choose_keyframes(rows: list[dict], pre: dict) -> tuple[list[dict], list[dict], str]:
    """Temporal bins + CUDA similarity; temporal gaps are an overlap heuristic."""
    torch = require_cuda()
    if len(rows) < pre["min_selected_frames"]:
        raise PipelineError("STOP_INPUT_INVALID", "Too few candidates; never duplicate frames to meet a minimum")
    first, last = rows[0]["timestamp_sec"], rows[-1]["timestamp_sec"]
    eligible = []
    rejections = {}
    for index, row in enumerate(rows):
        if row["blur_score"] < pre["min_blur_score"]:
            rejections[index] = "below blur threshold"
        elif row["exposure_fraction"] > pre["max_exposure_fraction"]:
            rejections[index] = "extreme exposure"
        else:
            eligible.append(index)
    mode = "quality_temporal_bins"
    if len(eligible) < pre["min_selected_frames"]:
        # Algorithmic fallback still uses the CUDA scores/similarity just computed.
        eligible = list(range(len(rows)))
        rejections.clear()
        mode = "uniform_temporal_fallback_unreliable_quality"
    target = min(pre["target_selected_frames"], len(eligible))
    descriptors = torch.tensor([r["descriptor"] for r in rows], dtype=torch.float32, device="cuda:0")
    selected: list[int] = []
    hashes = set()

    def add(index: int, reason: str) -> bool:
        if index in selected:
            return True
        if rows[index]["sha256"] in hashes:
            rejections[index] = "duplicate file hash"
            return False
        if selected:
            mse = (descriptors[selected] - descriptors[index]).square().mean(1)
            if mse.min().item() < pre["duplicate_mse"]:
                rejections[index] = "near duplicate thumbnail (CUDA MSE)"
                return False
        selected.append(index)
        hashes.add(rows[index]["sha256"])
        rows[index]["reason"] = reason
        rejections.pop(index, None)
        return True

    try:
        with torch.inference_mode():
            for bin_index in range(target):
                midpoint = first + (bin_index + 0.5) * (last - first) / target
                candidates = [i for i in eligible if min(target - 1, math.floor((rows[i]["timestamp_sec"] - first) / max(last-first, 1e-12) * target)) == bin_index]
                if mode.startswith("uniform"):
                    candidates.sort(key=lambda i: (abs(rows[i]["timestamp_sec"] - midpoint), i))
                else:
                    candidates.sort(key=lambda i: (-rows[i]["selection_score"], abs(rows[i]["timestamp_sec"] - midpoint), i))
                for candidate in candidates:
                    if add(candidate, mode):
                        break
            # Include boundary coverage and bridge long gaps while respecting max frames.
            while len(selected) < pre["max_selected_frames"]:
                ordered = sorted(selected)
                anchors = [None, *ordered, None]
                gaps = []
                for a, b in zip(anchors, anchors[1:]):
                    left = first if a is None else rows[a]["timestamp_sec"]
                    right = last if b is None else rows[b]["timestamp_sec"]
                    if right - left > pre["max_gap_seconds"]:
                        gaps.append((right - left, left, right))
                if not gaps:
                    break
                _, left, right = max(gaps)
                candidates = [i for i in eligible if i not in selected and left <= rows[i]["timestamp_sec"] <= right]
                candidates.sort(key=lambda i: (abs(rows[i]["timestamp_sec"] - (left+right)/2), i))
                if not any(add(i, "temporal bridge (overlap heuristic)") for i in candidates):
                    break
        selected.sort()
        if len(selected) < pre["min_selected_frames"]:
            raise PipelineError("STOP_INPUT_INVALID", f"Only {len(selected)} distinct keyframes; minimum={pre['min_selected_frames']}")
        times = [first, *[rows[i]["timestamp_sec"] for i in selected], last]
        if max(b-a for a, b in zip(times, times[1:])) > pre["max_gap_seconds"] + 1e-6:
            raise PipelineError("STOP_INPUT_INVALID", "Temporal gap exceeds max_gap_seconds; increase frame budget/change capture; overlap unverified")
        rejected = [{"frame_id": r["frame_id"], "timestamp_sec": r["timestamp_sec"],
                     "blur_score": r["blur_score"], "exposure_fraction": r["exposure_fraction"],
                     "reason": rejections.get(i, "not chosen within temporal bin")} for i, r in enumerate(rows) if i not in selected]
        return [{k: v for k, v in rows[i].items() if k != "descriptor"} for i in selected], rejected, mode
    finally:
        descriptors = None
        torch.cuda.empty_cache()


def select_keyframes(extracted: Path, destination: Path, pre: dict, root: Path, seed: int, resume: bool = False) -> dict:
    started = time.perf_counter()
    manifest = validate_manifest(extracted)
    torch = require_cuda()
    key = fingerprint({"extraction_key": manifest["cache_key"], "frames": manifest["frames"], "config": pre,
                       "seed": seed, "torch": torch.__version__, "code": code_hash()})
    if destination.exists():
        if not resume:
            raise PipelineError("FAILED", "Selected frames exist; use --resume or a new path")
        selected = validate_manifest(destination, key)
        if len({r["sha256"] for r in selected["frames"]}) != len(selected["frames"]):
            raise PipelineError("FAILED", "Duplicate selected frame checksums")
        return selected
    temporary = staging_directory(destination)
    try:
        rows, gpu_report = score_frames(extracted, manifest["frames"], pre, seed)
        chosen, rejected, mode = choose_keyframes(rows, pre)
        gpu_report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        gpu_report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
        for row in chosen:
            shutil.copy2(extracted / row["filename"], temporary / row["filename"])
            row["source_path"] = row["path"]
            row["path"] = (destination / row["filename"]).relative_to(root).as_posix()
        result = {"status": "SUCCESS", "cache_key": key, "source_sha256": manifest["source_sha256"],
                  "source_video": manifest["source_video"], "extraction_cache_key": manifest["cache_key"], "code_hash": code_hash(),
                  "seed": seed, "config": pre, "selection_algorithm": mode,
                  "overlap_status": "temporal/appearance heuristic; must be verified with two-view geometry",
                  "gpu": gpu_report, "runtime_seconds": time.perf_counter() - started, "frames": chosen, "rejected_frames": rejected}
        write_json(temporary / "manifest.json", result)
        columns = ["frame_id", "filename", "timestamp_sec", "source_video_sha256", "blur_score", "exposure_fraction", "selection_score", "reason"]
        with (temporary / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(chosen)
        validate_manifest(temporary, key)
        temporary.rename(destination)
        return result
    except Exception as exc:
        write_json(temporary / "failure.json", {"status": "FAILED", "error": str(exc)})
        raise
