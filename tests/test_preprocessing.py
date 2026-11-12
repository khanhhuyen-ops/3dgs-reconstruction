"""Run real FFmpeg VFR extraction and CUDA thumbnail selection when available."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.config import ROOT, load_config, resolve_tool
from src.common.io import PipelineError, run_process, sha256
from src.preprocessing.extract_frames import extract_frames, parse_showinfo, validate_manifest
from src.preprocessing.select_keyframes import select_keyframes


def test_integer_pts_not_nominal_fps() -> None:
    output = "config in time_base: 1/90000, frame_rate: 25/1\nn: 0 pts: 12345 pts_time:0.137\nn: 1 pts: 73456 pts_time:0.816\n"
    parsed = parse_showinfo(output)
    assert parsed[0]["timestamp_sec"] == 12345/90000
    assert parsed[1]["timestamp_sec"] == 73456/90000
    with pytest.raises(PipelineError):
        parse_showinfo(output + "n: 2 pts: 1 pts_time:0.0\n")


def test_vfr_extraction_resume_and_cuda_selection(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/sfm_ignatius.yaml")
    tools = {name: resolve_tool(name, config) for name in ("ffmpeg", "ffprobe")}
    if not all(tools.values()):
        pytest.skip("FFmpeg/ffprobe not installed")
    video = tmp_path / "source video.mp4"
    command = [tools["ffmpeg"], "-hide_banner", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=12:duration=4",
               "-vf", "select='not(mod(n,5))+not(mod(n,7))'", "-fps_mode", "vfr", "-c:v", "mpeg4", "-q:v", "2", str(video)]
    run_process(command, timeout=30)
    before = sha256(video)
    pre = {**config["preprocessing"], "candidate_fps": 3.0, "max_width": 320, "target_selected_frames": 6,
           "min_selected_frames": 4, "max_selected_frames": 8, "min_blur_score": 0, "duplicate_mse": 1e-8, "max_gap_seconds": 1.5}
    extracted = tmp_path / "frames"
    result = extract_frames(video, extracted, pre, tools, tmp_path)
    probe = run_process([tools["ffprobe"], "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_entries", "frame=pts_time", "-of", "json", str(video)], timeout=30)
    source_pts = [float(row["pts_time"]) for row in json.loads(probe["output"])["frames"]]
    assert all(any(abs(row["timestamp_sec"] - pts) < 1e-5 for pts in source_pts) for row in result["frames"])
    assert all(row["width"] == 160 for row in result["frames"])
    assert sha256(video) == before
    assert extract_frames(video, extracted, pre, tools, tmp_path, resume=True)["cache_key"] == result["cache_key"]
    bounded = extract_frames(video, tmp_path / "bounded", {**pre, "max_duration_seconds": 1.3}, tools, tmp_path)
    assert bounded["frames"][-1]["timestamp_sec"] < 1.3
    with pytest.raises(PipelineError):
        extract_frames(video, extracted, {**pre, "candidate_fps": 1}, tools, tmp_path, resume=True)
    import torch
    if torch.cuda.is_available():
        selected = tmp_path / "selected"
        selection = select_keyframes(extracted, selected, pre, tmp_path, 42)
        assert 4 <= len(selection["frames"]) <= 8
        assert selection["gpu"]["device"] == "cuda:0"
        assert len({row["sha256"] for row in selection["frames"]}) == len(selection["frames"])
        rerun = select_keyframes(extracted, tmp_path / "selected_repeat", pre, tmp_path, 42)
        assert [r["frame_id"] for r in rerun["frames"]] == [r["frame_id"] for r in selection["frames"]]
    else:
        with pytest.raises(PipelineError, match="CUDA required"):
            select_keyframes(extracted, tmp_path / "selected", pre, tmp_path, 42)
    (extracted / result["frames"][0]["filename"]).write_bytes(b"corrupt")
    with pytest.raises(PipelineError):
        validate_manifest(extracted)

