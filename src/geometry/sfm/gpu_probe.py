"""Test GPU BA on a small, clearly labelled synthetic diagnostic model."""
from __future__ import annotations

import re
import uuid
from pathlib import Path

import numpy as np

from src.common.io import PipelineError, run_process, sha256, write_json
from src.geometry.sfm.metrics import model_metrics
from src.geometry.sfm.model_io import read_model


def cli_options(help_text: str) -> set[str]:
    return set(re.findall(r"--([A-Za-z][A-Za-z0-9_.]*)", help_text))


def inspect_colmap(binary: str, output: Path, matching: str = "sequential") -> dict:
    result = {"binary": binary, "binary_sha256": sha256(Path(binary)), "help": {}, "options": {}}
    for command in (None, "feature_extractor", f"{matching}_matcher", "mapper", "bundle_adjuster", "model_converter"):
        name = command or "version"
        report = run_process([binary, *([command] if command else []), "-h"], output / f"{name}_help.log", timeout=30)
        result["help"][name] = report["output"]
        result["options"][name] = sorted(cli_options(report["output"]))
    result["version"] = next((line for line in result["help"]["version"].splitlines() if "COLMAP " in line), "unknown")
    return result


def write_ba_fixture(directory: Path, seed: int = 42, point_count: int = 48) -> None:
    """Known poses/projections; perturbed XYZ makes this a nontrivial BA test."""
    directory.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    truth = np.column_stack((rng.uniform(-1, 1, point_count), rng.uniform(-0.65, 0.65, point_count), rng.uniform(4, 7, point_count)))
    initial = truth + rng.normal(0, 0.025, truth.shape)
    centers = [-0.6, 0, 0.6]
    (directory / "cameras.txt").write_text("1 SIMPLE_RADIAL 640 480 500 320 240 0\n", encoding="utf-8")
    image_rows = []
    for image_id, center in enumerate(centers, 1):
        image_rows.append(f"{image_id} 1 0 0 0 {-center} 0 0 1 diagnostic_{image_id}.jpg")
        xy = truth[:, :2].copy()
        xy[:, 0] -= center
        xy = xy / truth[:, 2:] * 500 + [320, 240]
        image_rows.append(" ".join(f"{x:.12f} {y:.12f} {i}" for i, (x, y) in enumerate(xy, 1)))
    (directory / "images.txt").write_text("\n".join(image_rows) + "\n", encoding="utf-8")
    point_rows = []
    for index, (x, y, z) in enumerate(initial):
        track = " ".join(f"{image_id} {index}" for image_id in range(1, 4))
        point_rows.append(f"{index+1} {x:.12f} {y:.12f} {z:.12f} 128 128 128 0 {track}")
    (directory / "points3D.txt").write_text("\n".join(point_rows) + "\n", encoding="utf-8")
    write_json(directory / "DIAGNOSTIC_ONLY.json", {"synthetic": True, "purpose": "BA backend probe, never scene reconstruction/benchmark", "seed": seed, "points": point_count})


def probe_gpu_ba(capabilities: dict, output: Path) -> dict:
    """Fail closed unless a GPU-exclusive CASPAR solve actually reduces error."""
    required = {"Mapper.ba_local_backend", "Mapper.ba_global_backend", "Mapper.ba_gpu_index"}
    report = {"status": "BLOCKED_GPU_BA", "verified": False, "backend": "UNAVAILABLE",
              "colmap_binary_sha256": capabilities["binary_sha256"],
              "ceres_cuda": "Not verified: generic use_gpu flags may permit CPU solver/fallback",
              "remediation": "Build COLMAP with CASPAR_ENABLED=ON and rerun preflight; mapper remains disabled until a nontrivial GPU BA probe passes."}
    if not required.issubset(capabilities["options"]["mapper"]) or "BundleAdjustment.backend" not in capabilities["options"]["bundle_adjuster"]:
        report["reason"] = "Installed CLI cannot select GPU-exclusive BA for local/global mapping"
        return report
    probe_dir = output / f"ba_probe_{uuid.uuid4().hex[:8]}"
    before, optimized, text = probe_dir / "synthetic_input", probe_dir / "optimized", probe_dir / "optimized_txt"
    write_ba_fixture(before)
    optimized.mkdir()
    text.mkdir()
    command = [capabilities["binary"], "bundle_adjuster", "--input_path", str(before), "--output_path", str(optimized),
               "--BundleAdjustment.backend", "CASPAR", "--BundleAdjustment.refine_focal_length", "0",
               "--BundleAdjustment.refine_extra_params", "0", "--BundleAdjustment.refine_sensor_from_rig", "0"]
    ba_options = capabilities["options"]["bundle_adjuster"]
    if "BundleAdjustmentCaspar.gpu_index" in ba_options:
        command += ["--BundleAdjustmentCaspar.gpu_index", "0"]
    if "log_level" in ba_options:
        command += ["--log_level", "2"]
    report.update({"probe_directory": str(probe_dir), "command": command})
    try:
        baseline = model_metrics(read_model(before), 3)["mean_reprojection_error_px"]
        result = run_process(command, probe_dir / "bundle_adjuster.log", timeout=120)
        output_lower = result["output"].lower()
        if any(marker in output_lower for marker in ("built without", "fallback", "falling back", "no data to optimize", "no residuals")):
            raise ValueError("Backend unavailable/fallback/no-op in BA log")
        run_process([capabilities["binary"], "model_converter", "--input_path", str(optimized), "--output_path", str(text), "--output_type", "TXT"], probe_dir / "converter.log", timeout=30)
        after = model_metrics(read_model(text), 3)["mean_reprojection_error_px"]
        positive_log = "caspar" in output_lower and ("solver" in output_lower or "bundle adjustment" in output_lower)
        if not positive_log or not after < baseline * 0.9:
            raise ValueError(f"No convincing nontrivial CASPAR execution: before={baseline}, after={after}")
        report.update({"status": "SUCCESS", "verified": True, "backend": "CASPAR", "reason": "GPU-exclusive backend, successful command, solver log and reduced real residuals",
                       "before_error_px": baseline, "after_error_px": after, "runtime_seconds": result["runtime_seconds"]})
    except (PipelineError, ValueError, KeyError, OSError) as exc:
        report["reason"] = str(exc)
    write_json(probe_dir / "report.json", report)
    return report

