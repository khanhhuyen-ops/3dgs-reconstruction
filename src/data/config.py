"""Phase01 YAML configuration; isolated from the protected SfM configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

import yaml

from src.common.config import ROOT, repo_path

DEFAULTS = {
    "schema_version": "prepared-v1",
    "dataset_name": "scene",
    "source": {
        "mode": "video",
        "image_root": "",
        "model_dir": "",
        "format": "auto",
        "comparison_model_dir": None,
        "manifest": None,
        "ba_history": "unknown",
        "discovery_roots": [],
        "strict_protocol_evidence": None,
    },
    "output": {"dataset_root": "", "report_root": "", "file_mode": "copy"},
    "images": {
        "undistort": "auto",
        "max_width": None,
        "format": "preserve",
        "jpeg_quality": 95,
    },
    "split": {
        "strategy": "interleaved",
        "seed": 42,
        "ratios": [0.75, 0.125, 0.125],
        "period": 8,
        "test_offset": 0,
        "val_offset": 4,
        "protocol": "fixed_pose_transductive",
    },
    "validation": {
        "quaternion_tolerance": 1e-6,
        "rotation_tolerance": 1e-9,
        "warn_reprojection_px": 4.0,
        "max_reprojection_p95_px": 10.0,
        "min_registered_images": 3,
        "min_points": 1,
        "min_track_length": 2,
        "min_observations_per_image": 1,
    },
    "filter": {
        "duplicate_tracks": "reject",
        "outside_image_tracks": "reject",
        "max_point_error_px": None,
    },
    "normalization": {"enabled": True},
    "training_points": {"min_train_views": 2, "min_parallax_degrees": 0.0},
}


def _merge(defaults: dict, provided: dict, prefix: str = "") -> dict:
    result = deepcopy(defaults)
    if not isinstance(provided, dict):
        raise ValueError(f"Expected configuration mapping at {prefix}")
    for key, value in provided.items():
        if key not in defaults:
            raise ValueError(f"Unknown configuration field: {prefix}{key}")
        result[key] = (
            _merge(defaults[key], value, prefix + key + ".")
            if isinstance(defaults[key], dict)
            else value
        )
    return result


def load_config(path: Path, root: Path = ROOT) -> dict:
    config = _merge(DEFAULTS, yaml.safe_load(path.read_text(encoding="utf-8")))
    if config["schema_version"] != "prepared-v1" or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", config["dataset_name"]
    ):
        raise ValueError(
            "Expected schema_version=prepared-v1 and nonempty dataset_name"
        )
    source, output = config["source"], config["output"]
    if source["mode"] not in ("video", "image_set") or source["format"] not in (
        "auto",
        "bin",
        "txt",
    ):
        raise ValueError("source.mode must be video/image_set, format auto/bin/txt")
    for group, key in [
        (source, "image_root"),
        (source, "model_dir"),
        (output, "dataset_root"),
        (output, "report_root"),
    ]:
        if not group[key]:
            raise ValueError(f"Missing required path: {key}")
        repo_path(root, group[key])
    inputs = [repo_path(root, source[key]) for key in ("image_root", "model_dir")]
    for key in ("comparison_model_dir", "manifest", "strict_protocol_evidence"):
        if source[key]:
            inputs.append(repo_path(root, source[key]))
    for value in source["discovery_roots"]:
        repo_path(root, value)
    # Never read GT as a preparation input, including aliases/symlinks.
    if any("ground_truth" in [x.lower() for x in p.parts] for p in inputs):
        raise ValueError("Ground truth is evaluator-only")
    destinations = [
        repo_path(root, output[key]) for key in ("dataset_root", "report_root")
    ]
    for dest in destinations:
        parts = [x.lower() for x in dest.relative_to(root.resolve()).parts]
        if (
            dest == root.resolve()
            or "raw" in parts
            or any(
                dest == p or dest.is_relative_to(p) or p.is_relative_to(dest)
                for p in inputs
            )
        ):
            raise ValueError(f"Protected or overlapping output path: {dest}")
    if destinations[0].is_relative_to(destinations[1]) or destinations[
        1
    ].is_relative_to(destinations[0]):
        raise ValueError("Dataset and report directories must be disjoint")
    if output["file_mode"] not in ("copy", "hardlink", "symlink"):
        raise ValueError("output.file_mode must be copy/hardlink/symlink")
    images = config["images"]
    if images["undistort"] not in ("auto", "required", "never") or images[
        "format"
    ] not in ("preserve", "png", "jpg"):
        raise ValueError("Invalid image undistortion/format policy")
    if images["max_width"] is not None and (
        not isinstance(images["max_width"], int) or images["max_width"] < 2
    ):
        raise ValueError("images.max_width must be null or integer >=2")
    if not 1 <= images["jpeg_quality"] <= 100:
        raise ValueError("jpeg_quality must be 1..100")
    split = config["split"]
    if split["strategy"] not in ("interleaved", "random", "temporal_block"):
        raise ValueError("Unsupported split strategy")
    if split["protocol"] not in ("fixed_pose_transductive", "strict_train_only"):
        raise ValueError("Unknown split protocol")
    if (
        len(split["ratios"]) != 3
        or any(v <= 0 for v in split["ratios"])
        or abs(sum(split["ratios"]) - 1) > 1e-9
    ):
        raise ValueError(
            "split.ratios must contain positive train/val/test ratios summing to one"
        )
    if (
        split["period"] < 3
        or not 0 <= split["test_offset"] < split["period"]
        or not 0 <= split["val_offset"] < split["period"]
        or split["test_offset"] == split["val_offset"]
    ):
        raise ValueError("Invalid interleaved period/offsets")
    if config["filter"]["duplicate_tracks"] not in ("reject", "drop_track"):
        raise ValueError("Duplicate track policy must be reject/drop_track")
    if config["filter"]["outside_image_tracks"] not in ("reject", "drop_track"):
        raise ValueError("Outside-image track policy must be reject/drop_track")
    if (
        config["filter"]["max_point_error_px"] is not None
        and config["filter"]["max_point_error_px"] <= 0
    ):
        raise ValueError("max_point_error_px must be positive/null")
    if any(v <= 0 for v in config["validation"].values()):
        raise ValueError("Validation thresholds must be positive")
    if (
        config["training_points"]["min_train_views"] < 2
        or not 0 <= config["training_points"]["min_parallax_degrees"] < 180
    ):
        raise ValueError("Require >=2 train views and parallax in [0,180)")
    return config
