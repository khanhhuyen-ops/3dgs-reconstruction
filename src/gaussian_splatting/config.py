"""Versioned B0 profiles, separate from protected SfM/data configurations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from src.common.config import ROOT, repo_path

DEFAULTS = {
    "method": "B0_Original3DGS",
    "profile": "smoke",
    "experiment": "b0",
    "seed": 42,
    "dataset": "datasets/tanks_and_temples/Ignatius/prepared/legacy_v1/ignatius-legacy-v1-6a3308c22df11122",
    "output_root": "outputs/tanks_and_temples/Ignatius/research/B0_3DGS",
    "device": "cuda:0",
    "allow_cpu_fallback": False,
    "iterations": 1200,
    "resolution": 4,
    "sh_degree": 2,
    "white_background": False,
    "train_view_limit": None,
    "initial_point_limit": None,
    "cpu_camera_cache": 12,
    "max_gaussians": None,
    "torch_reserved_budget_gib": 5.0,
    "checkpoint_interval": 1000,
    "validation_interval": 1000,
    "log_interval": 50,
    "gradient_check_interval": 100,
    "validation_view_limit": 4,
    "optimizer": {
        "position_lr_init": 0.00016,
        "position_lr_final": 0.0000016,
        "position_lr_delay_mult": 0.01,
        "position_lr_max_steps": 30000,
        "feature_lr": 0.0025,
        "opacity_lr": 0.05,
        "scaling_lr": 0.005,
        "rotation_lr": 0.001,
        "percent_dense": 0.01,
        "lambda_dssim": 0.2,
    },
    "density": {
        "densify_from_iter": 500,
        "densify_until_iter": 15000,
        "densification_interval": 100,
        "opacity_reset_interval": 3000,
        "densify_grad_threshold": 0.0002,
    },
}


def load_config(path: Path) -> dict:
    provided = yaml.safe_load(path.read_text(encoding="utf-8"))
    result = deepcopy(DEFAULTS)
    for key, value in provided.items():
        if key not in result:
            raise ValueError(f"Unknown B0 config field {key}")
        if isinstance(result[key], dict):
            if set(value) - set(result[key]):
                raise ValueError(f"Unknown B0 {key} fields")
            result[key].update(value)
        else:
            result[key] = value
    if (
        result["method"] != "B0_Original3DGS"
        or result["device"] != "cuda:0"
        or result["allow_cpu_fallback"] is not False
    ):
        raise ValueError(
            "Only Original3DGS B0 on cuda:0 without CPU fallback is supported"
        )
    if result["profile"] not in ("smoke", "low_memory", "research_quality"):
        raise ValueError("Unknown experimental profile")
    if result["resolution"] not in (1, 2, 4, 8) or not 0 <= result["sh_degree"] <= 3:
        raise ValueError("Use resolution divisor 1/2/4/8 and SH degree 0..3")
    for key in (
        "iterations",
        "cpu_camera_cache",
        "checkpoint_interval",
        "validation_interval",
        "log_interval",
        "gradient_check_interval",
    ):
        if not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f"{key} must be positive integer")
    for key in ("initial_point_limit", "train_view_limit", "max_gaussians"):
        if result[key] is not None and (
            not isinstance(result[key], int) or result[key] < 4
        ):
            raise ValueError(f"{key} must be null or integer >=4")
    if not 0 <= result["optimizer"]["lambda_dssim"] <= 1 or any(
        v <= 0 for k, v in result["optimizer"].items() if k != "lambda_dssim"
    ):
        raise ValueError("Invalid native optimizer parameters")
    if (
        any(v <= 0 for v in result["density"].values())
        or result["density"]["densify_from_iter"]
        >= result["density"]["densify_until_iter"]
    ):
        raise ValueError("Invalid density-control schedule")
    source = repo_path(ROOT, result["dataset"])
    out = repo_path(ROOT, result["output_root"])
    if (
        "ground_truth" in source.parts
        or "raw" in out.parts
        or not out.is_relative_to(ROOT / "outputs")
        or source.is_relative_to(out)
        or out.is_relative_to(source)
    ):
        raise ValueError("Invalid/protected dataset/output path")
    return result
