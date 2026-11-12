"""Extract source frames using real decoded PTS, including VFR inputs."""
from __future__ import annotations

import re
import time
from fractions import Fraction
from pathlib import Path

from PIL import Image

from src.common.io import PipelineError, code_hash, fingerprint, read_json, run_process, sha256, staging_directory, write_json
from src.preprocessing.video_probe import probe_video


def parse_showinfo(output: str) -> list[dict]:
    timebase = re.search(r"config in time_base:\s*(\d+/\d+)", output)
    if not timebase:
        raise PipelineError("STOP_INPUT_INVALID", "FFmpeg did not report frame time_base")
    base = Fraction(timebase.group(1))
    rows = []
    for match in re.finditer(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:", output):
        rows.append({"emitted_index": int(match[1]), "source_pts": int(match[2]),
                     "source_time_base": str(base), "timestamp_sec": float(int(match[2]) * base)})
    if not rows or [r["emitted_index"] for r in rows] != list(range(len(rows))):
        raise PipelineError("STOP_INPUT_INVALID", "Missing or discontinuous FFmpeg timestamps")
    if any(b["timestamp_sec"] <= a["timestamp_sec"] for a, b in zip(rows, rows[1:])):
        raise PipelineError("STOP_INPUT_INVALID", "Decoded timestamps are not strictly increasing")
    return rows


def validate_manifest(directory: Path, cache_key: str | None = None) -> dict:
    manifest = read_json(directory / "manifest.json")
    if cache_key and manifest["cache_key"] != cache_key:
        raise PipelineError("FAILED", f"Cache key changed; use a new experiment directory: {directory}")
    if manifest.get("status") != "SUCCESS" or not manifest.get("frames"):
        raise PipelineError("FAILED", "Incomplete manifest")
    previous = -float("inf")
    names = set()
    for row in manifest["frames"]:
        filename = row["filename"]
        if Path(filename).name != filename or filename in names:
            raise PipelineError("FAILED", "Unsafe or repeated manifest filename")
        names.add(filename)
        path = directory / filename
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise PipelineError("FAILED", f"Missing/corrupt frame; existing output preserved: {path}")
        with Image.open(path) as image:
            image.load()
            if image.size != (row["width"], row["height"]):
                raise PipelineError("FAILED", f"Image dimensions disagree with manifest: {path}")
        if row["timestamp_sec"] <= previous:
            raise PipelineError("FAILED", "Non-monotonic manifest timestamps")
        previous = row["timestamp_sec"]
    return manifest


def extract_frames(video: Path, destination: Path, pre: dict, tools: dict, root: Path, resume: bool = False) -> dict:
    started = time.perf_counter()
    metadata = probe_video(video, tools["ffprobe"])
    version = run_process([tools["ffmpeg"], "-version"], timeout=15)["output"].splitlines()[0]
    input_hash = sha256(video)
    key = fingerprint({"source_sha256": input_hash, "config": pre, "ffmpeg": version, "code": code_hash()})
    if destination.exists():
        if not resume:
            raise PipelineError("FAILED", "Extraction exists; use --resume or a new path")
        return validate_manifest(destination, key)
    temporary = staging_directory(destination)
    fps, width = pre["candidate_fps"], int(pre["max_width"])
    # select preserves the original frame PTS; fps would invent a new timeline.
    select = f"select='isnan(prev_selected_t)+gt(floor((t-start_t)*{fps}),floor((prev_selected_t-start_t)*{fps}))'"
    filters = f"{select},scale=w='trunc(min(iw,{width})/2)*2':h=-2:flags=lanczos,showinfo"
    command = [tools["ffmpeg"], "-hide_banner", "-nostdin", "-n", "-copyts"]
    if pre.get("max_duration_seconds") is not None:
        # An output -t can drop the final frame *after* showinfo has logged it.
        # Bound the input and filter before showinfo, retaining source PTS.
        command += ["-t", str(pre["max_duration_seconds"])]
        filters = f"trim=duration={pre['max_duration_seconds']}," + filters
    command += ["-i", str(video), "-map", "0:v:0", "-an", "-sn"]
    command += ["-vf", filters, "-fps_mode", "passthrough", "-q:v", str(pre["jpeg_quality"]), str(temporary / "frame_%06d.jpg")]
    try:
        result = run_process(command, temporary / "ffmpeg.log", timeout=7200)
        timestamps = parse_showinfo(result["output"])
        images = sorted(temporary.glob("frame_*.jpg"))
        if len(images) != len(timestamps):
            raise PipelineError("STOP_INPUT_INVALID", "Emitted frame count and PTS count disagree")
        oriented_width, oriented_height = metadata["width"], metadata["height"]
        if int(metadata["rotation_degrees"]) % 180:
            oriented_width, oriented_height = oriented_height, oriented_width
        frames = []
        for index, (path, stamp) in enumerate(zip(images, timestamps), 1):
            with Image.open(path) as image:
                image.load()
                w, h = image.size
            frames.append({**stamp, "frame_id": index, "filename": path.name,
                           "path": (destination / path.name).relative_to(root).as_posix(),
                           "source_video_sha256": input_hash, "video_time_sec": stamp["timestamp_sec"] - metadata["start_time"],
                           "width": w, "height": h, "sha256": sha256(path), "blur_score": None, "selection_score": None,
                           "reason": "first source frame in temporal sampling bin",
                           "image_transform": {"autorotate_degrees": metadata["rotation_degrees"],
                                               "oriented_source_size": [oriented_width, oriented_height],
                                               "crop": None, "oriented_to_output": [[w/oriented_width, 0, 0], [0, h/oriented_height, 0], [0, 0, 1]]}})
        if sha256(video) != input_hash:
            raise PipelineError("STOP_INPUT_INVALID", "Source video changed during extraction")
        manifest = {"status": "SUCCESS", "cache_key": key, "source_video": video.relative_to(root).as_posix(),
                    "source_sha256": input_hash, "ffmpeg_version": version, "metadata": metadata,
                    "sampling_fps": fps, "config": pre, "command": command, "code_hash": code_hash(),
                    "timestamp_convention": "decoded source PTS; no synthetic fps interpolation",
                    "decode_backend": "CPU (FFmpeg decode/resize permitted)", "runtime_seconds": time.perf_counter() - started,
                    "frames": frames}
        write_json(temporary / "manifest.json", manifest)
        validate_manifest(temporary, key)
        temporary.rename(destination)
        return manifest
    except Exception as exc:
        write_json(temporary / "failure.json", {"status": "FAILED", "error": str(exc), "command": command})
        raise
