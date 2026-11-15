"""Independent native COLMAP readers; never substitute mocks for native loaders."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from src.common.io import read_json, sha256
from src.data.colmap import read_model


def check_compatibility(directory: Path, references: Path | None = None) -> dict:
    model = read_model(directory / "sparse/0")
    meta = read_json(directory / "metadata.json")
    report = {}
    registry = read_json(references / "registry.json") if references else {}
    for method in ("3dgs", "2dgs", "pgsr"):
        problems = []
        for c in model.cameras.values():
            cx, cy = c.params[2:4] if c.model == "PINHOLE" else c.params[1:3]
            if c.model not in ("PINHOLE", "SIMPLE_PINHOLE"):
                problems.append(f"Unsupported camera {c.model}")
            if abs(cx - c.width / 2) > 1e-6 or abs(cy - c.height / 2) > 1e-6:
                problems.append(
                    "Native FoV-only loader discards off-center principal point; adapter required"
                )
        names = [Path(im.name).name.split(".")[0] for im in model.images.values()]
        if method != "3dgs" and (
            len(set(names)) != len(names)
            or any("/" in im.name for im in model.images.values())
        ):
            problems.append(
                "Native basename handling requires deterministic name adapter"
            )
        if any(row["mask_path"] for row in meta["images"]):
            problems.append(
                "Undistortion validity masks require explicit loss-mask adapter"
            )
        item = {
            "input_specification": "VERIFIED"
            if method in registry and not problems
            else "UNVERIFIED"
            if method not in registry
            else "ADAPTER_REQUIRED",
            "native_colmap_reader": "NOT_RUN",
            "full_native_scene_loader": "NOT_RUN",
            "full_loader_reason": "Local renderer/scene dependencies are not installed; no dependency stubs or training launched",
            "split_adapter": "src.data.loader.apply_native_split (native SceneInfo integration untested)",
            "problems": problems,
            "source": registry.get(method),
        }
        if method in registry:
            try:
                source = references / method / "scene/colmap_loader.py"
                if (
                    sha256(source)
                    != registry[method]["files"]["scene/colmap_loader.py"]["sha256"]
                ):
                    raise ValueError("Reference source hash mismatch")
                spec = importlib.util.spec_from_file_location(
                    f"phase01_{method}_colmap", source
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sub = directory / ("sparse" if method == "pgsr" else "sparse/0")
                cameras = module.read_intrinsics_binary(str(sub / "cameras.bin"))
                images = module.read_extrinsics_binary(str(sub / "images.bin"))
                xyz, rgb, errors = module.read_points3D_binary(
                    str(sub / "points3D.bin")
                )
                if (
                    set(cameras) != set(model.cameras)
                    or set(images) != set(model.images)
                    or len(xyz) != len(model.points)
                ):
                    raise ValueError("Native counts/IDs disagree")
                for iid, im in model.images.items():
                    native = images[iid]
                    if (
                        native.name != im.name
                        or native.camera_id != im.camera_id
                        or not np.array_equal(native.qvec, im.qvec)
                        or not np.array_equal(native.tvec, im.tvec)
                        or not np.array_equal(native.xys, im.xys)
                        or not np.array_equal(native.point3D_ids, im.point3D_ids)
                    ):
                        raise ValueError(f"Native pose/observation mismatch: {iid}")
                for cid, c in model.cameras.items():
                    native = cameras[cid]
                    if (
                        native.model != c.model
                        or native.width != c.width
                        or native.height != c.height
                        or not np.array_equal(native.params, c.params)
                    ):
                        raise ValueError(f"Native intrinsics mismatch: {cid}")
                points = sorted(model.points.values(), key=lambda p: p.id)
                if (
                    not np.array_equal(xyz, np.array([p.xyz for p in points]))
                    or not np.array_equal(rgb, np.array([p.rgb for p in points]))
                    or not np.array_equal(
                        np.asarray(errors).reshape(-1),
                        np.array([p.error for p in points]),
                    )
                ):
                    raise ValueError("Native point positions/colors/errors disagree")
                item["native_colmap_reader"] = "PASSED"
                item["counts"] = {
                    "images": len(images),
                    "cameras": len(cameras),
                    "points": len(xyz),
                }
            except (OSError, ValueError, ImportError, AssertionError) as exc:
                item["native_colmap_reader"] = "FAILED"
                item["problems"].append(str(exc))
        report[method] = item
    return {
        "status": "FAILED"
        if any(
            r["native_colmap_reader"] == "FAILED" or r["problems"]
            for r in report.values()
        )
        else "PARTIAL",
        "methods": report,
        "full_scene_loader_compatibility_verified": False,
        "training_started": False,
        "specification_note": "Pinned native input formats verified separately from full scene/renderer imports",
    }
