"""Repository-relative configuration; reject unsafe paths and policy changes."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml

from src.common.io import PipelineError

ROOT = Path(__file__).resolve().parents[2]


def repo_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes repository: {value}")
    return path


def load_config(path: Path, root: Path = ROOT) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    policy = config["gpu_policy"]
    if policy.get("mode") != "strict" or policy.get("allow_cpu_fallback") is not False:
        raise ValueError("This implementation requires GPU-strict without CPU fallback")
    for key in ("require_gpu_features", "require_gpu_matching", "require_gpu_ba", "verify_runtime_backend"):
        if policy.get(key) is not True:
            raise ValueError(f"GPU policy requires {key}=true")
    if policy.get("device") != "cuda:0" or policy.get("gpu_index") != 0:
        raise ValueError("Only cuda:0 / GPU index 0 is supported")
    if config["input"].get("source_mode") != "video":
        raise ValueError("Use a separate implementation/config for image_set input")
    video = repo_path(root, config["input"]["video"])
    destinations = [repo_path(root, v) for v in config["paths"].values()]
    for destination in destinations:
        if destination == root.resolve() or "raw" in [p.lower() for p in destination.relative_to(root).parts]:
            raise ValueError(f"Protected output destination: {destination}")
        if video == destination or video.is_relative_to(destination):
            raise ValueError("Output contains input video")
    for i, first in enumerate(destinations):
        for second in destinations[i + 1:]:
            if first.is_relative_to(second) or second.is_relative_to(first):
                raise ValueError("Artifact directories must be disjoint")
    pre = config["preprocessing"]
    if not 2 <= pre["min_selected_frames"] <= pre["target_selected_frames"] <= pre["max_selected_frames"]:
        raise ValueError("Invalid selected-frame limits")
    for key in ("candidate_fps", "max_width", "batch_size", "thumbnail_width", "max_gap_seconds"):
        if pre[key] <= 0:
            raise ValueError(f"Expected positive {key}")
    if pre.get("max_duration_seconds") is not None and pre["max_duration_seconds"] <= 0:
        raise ValueError("Expected positive max_duration_seconds")
    if config["sfm"]["implementation"] != "colmap":
        raise ValueError("V1 is gated on a validated V0")
    if config["sfm"]["matching"] not in ("sequential", "exhaustive"):
        raise ValueError("matching must be sequential or exhaustive")
    if config["sfm"]["mapper_backend"] != "CASPAR":
        raise ValueError("Only GPU-exclusive CASPAR is currently verified by this wrapper")
    return config


def resolve_tool(name: str, config: dict) -> str | None:
    """Prefer explicit config/env and PATH; use local portable tools if present."""
    configured = os.environ.get(f"{name.upper()}_BIN", config["tools"][name])
    found = shutil.which(configured)
    if found:
        candidate = Path(found)
    elif Path(configured).is_file():
        candidate = Path(configured).resolve()
    else:
        candidates = list((ROOT / ".tools").glob(f"**/{name}.exe"))
        candidate = sorted(candidates)[0] if candidates else None
    if candidate and candidate.suffix.lower() in (".bat", ".cmd"):
        executable = candidate.parent / "bin" / "colmap.exe"
        if executable.is_file():
            candidate = executable
        else:
            raise PipelineError("FAILED", "Set COLMAP_BIN to colmap.exe; batch wrappers require shell-specific quoting")
    return str(candidate) if candidate else None

