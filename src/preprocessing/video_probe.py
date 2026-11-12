"""Probe video without using reference images or ground truth."""
from __future__ import annotations

import json
import math
from pathlib import Path

from src.common.io import PipelineError, run_process


def probe_video(video: Path, ffprobe: str) -> dict:
    if not video.is_file() or video.stat().st_size == 0:
        raise PipelineError("STOP_INPUT_INVALID", f"Missing or empty video: {video}")
    try:
        result = run_process([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json", str(video)], timeout=60)
        data = json.loads(result["output"])
        stream = data["streams"][0]
        duration = float(stream.get("duration", data["format"].get("duration", 0)))
        if not math.isfinite(duration) or duration <= 0 or min(stream["width"], stream["height"]) < 2:
            raise ValueError("Invalid duration or dimensions")
        rotation = float(stream.get("tags", {}).get("rotate", 0))
        for side in stream.get("side_data_list", []):
            rotation = float(side.get("rotation", rotation))
        if rotation % 90:
            raise ValueError("Only right-angle display rotations are supported")
        return {"codec": stream.get("codec_name"), "width": stream["width"], "height": stream["height"],
                "duration_seconds": duration, "nominal_fps": stream.get("r_frame_rate"),
                "average_fps": stream.get("avg_frame_rate"), "time_base": stream["time_base"],
                "start_time": float(stream.get("start_time", 0)), "frame_count": stream.get("nb_frames"),
                "rotation_degrees": rotation, "raw_probe": data}
    except (PipelineError, ValueError, KeyError, IndexError) as exc:
        raise PipelineError("STOP_INPUT_INVALID", f"Video probe failed: {exc}") from exc

