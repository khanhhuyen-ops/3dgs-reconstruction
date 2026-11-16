"""Validate/downloaded Kaggle outputs and import once without overwriting local runs."""

from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import math
from workflow import (
    ROOT,
    bundle_id,
    inspect_checkpoint,
    inside,
    read_json,
    sha256,
    verify,
    write_json,
)


def import_results(folder: Path, root: Path = ROOT):
    m = verify(folder)
    if m["schema"] != "phase02-kaggle-output-v1":
        raise ValueError("Expected result bundle")
    run = folder / "run"
    config = read_json(run / "resolved_config.json")
    parent = inside(inside(root, config["output_root"]), m["parent_run"])
    parent_config = read_json(parent / "resolved_config.json")
    allowed = {"experiment", "checkpoint_interval", "torch_reserved_budget_gib"}
    if any(config[k] != parent_config[k] for k in config if k not in allowed):
        raise ValueError("Optimization settings differ from local parent experiment")
    dataset = inside(root, config["dataset"])
    if (
        sha256(dataset / "metadata.json") != m["dataset_metadata_sha256"]
        or read_json(dataset / "splits.json")["hash"] != m["split_hash"]
    ):
        raise ValueError("Local dataset/split is incompatible")
    pointer = read_json(run / "checkpoints/latest.json")
    checkpoint = inside(run / "checkpoints", pointer["path"])
    if sha256(checkpoint) != pointer["sha256"]:
        raise ValueError("Output pointer checksum mismatch")
    check = inspect_checkpoint(checkpoint, config)
    if (
        check["iteration"] != m["iteration"]
        or check["identity"]["metadata"] != m["dataset_metadata_sha256"]
        or check["identity"]["split"] != m["split_hash"]
    ):
        raise ValueError("Checkpoint/experiment identity mismatch")
    target = inside(root, m["run_relative"])
    expected_parent = inside(root, config["output_root"])
    if (
        target.parent != expected_parent
        or not target.is_relative_to(root / "outputs")
        or target.name != config["experiment"]
    ):
        raise ValueError("Invalid output destination")
    ply = None
    if m["complete"]:
        result = read_json(run / "training_result.json")
        metrics = read_json(run / "metrics/test_030000.json")
        test_ids = read_json(dataset / "splits.json")["image_ids"]["test"]
        if (
            result["status"] != "SUCCESS"
            or result["iterations"] != 30000
            or m["iteration"] != 30000
        ):
            raise ValueError("Incomplete training labelled final")
        if (
            metrics["status"] != "SUCCESS"
            or metrics["split_hash"] != m["split_hash"]
            or metrics["view_count"] != len(test_ids)
            or [r["image_id"] for r in metrics["per_view"]] != test_ids
        ):
            raise ValueError("Held-out evaluation incomplete/mismatched")
        if any(
            metrics["aggregate_arithmetic_mean_per_view"].get(k) is None
            for k in ("psnr", "ssim", "lpips")
        ):
            raise ValueError("Missing actual RGB metrics")
        for row in [
            metrics["aggregate_arithmetic_mean_per_view"],
            *metrics["per_view"],
        ]:
            if not all(
                isinstance(row[k], (int, float)) and math.isfinite(row[k])
                for k in ("psnr", "ssim", "lpips")
            ):
                raise ValueError("Nonfinite/incomplete RGB metrics")
        prefix = Path(m["run_relative"])
        rel_ply = Path(result["export"]["path"]).relative_to(prefix)
        ply = inside(run, rel_ply.as_posix())
        if (
            sha256(ply) != result["export"]["sha256"]
            or result["export"]["render_reload_max_abs_difference"] > 1e-5
        ):
            raise ValueError("Native PLY export validation mismatch")
        from plyfile import PlyData

        parsed = PlyData.read(ply)
        required = {"x", "y", "z", "nx", "ny", "nz", "opacity"}
        required.update(f"f_dc_{i}" for i in range(3))
        required.update(
            f"f_rest_{i}" for i in range(3 * ((config["sh_degree"] + 1) ** 2 - 1))
        )
        required.update(f"scale_{i}" for i in range(3))
        required.update(f"rot_{i}" for i in range(4))
        if len(parsed["vertex"]) != check["gaussians"] or not required <= set(
            parsed["vertex"].data.dtype.names
        ):
            raise ValueError("Invalid Gaussian PLY attributes/count")
        import numpy as np

        if not all(
            np.isfinite(parsed["vertex"].data[name]).all()
            for name in parsed["vertex"].data.dtype.names
        ):
            raise ValueError("Nonfinite Gaussian PLY attributes")
        if metrics["checkpoint_export_sha256"] != result["export"]["sha256"]:
            raise ValueError("Metrics refer to another Gaussian model")
    receipt = {
        "bundle_id": bundle_id(m),
        "manifest_sha256": sha256(folder / "manifest.json"),
        "status": "IMPORTED_FINAL" if m["complete"] else "IMPORTED_PARTIAL",
        "checkpoint": check,
    }
    if target.exists():
        if (
            read_json(target / "kaggle_import.json")["bundle_id"]
            != receipt["bundle_id"]
        ):
            raise FileExistsError("Unrelated existing run; refusing overwrite")
        for rel, expected in m["files"].items():
            if rel.startswith("run/") and sha256(inside(target, rel[4:])) != expected:
                raise ValueError("Previously imported artifact changed")
        return target
    temporary = target.with_name("." + target.name + ".importing")
    shutil.copytree(run, temporary)
    write_json(temporary / "kaggle_import.json", receipt)
    write_json(
        temporary / "blender_handoff.json",
        {
            "gaussian_ply": str(target / ply.relative_to(run)) if ply else None,
            "coordinate_system": "Original COLMAP; no pose/XYZ normalization applied",
            "artifact_type": "Gaussian splats",
            "local_comparison_candidates": [
                str(p)
                for p in sorted(
                    expected_parent.glob(
                        "ignatius-legacy-*/point_cloud/iteration_030000/point_cloud.ply"
                    )
                )
            ],
            "instruction": "Import Gaussian PLY with the installed Gaussian Splatting addon; keep object transforms identity. No addon installation required.",
        },
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary.rename(target)
    return target


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    args = p.parse_args()
    print(import_results(args.results.resolve()))
