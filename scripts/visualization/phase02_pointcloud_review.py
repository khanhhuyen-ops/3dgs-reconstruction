"""Create reproducible point-cloud inspection artifacts and a separate Blender scene.

CPU parsing/preview generation only. Training artifacts are never rewritten.
Blender Gaussian previews use the installed KIRI addon and are not metric renders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(source: Path, output: Path, tag: str, expected: str | None = None) -> Path:
    """Inspect every original record; sample only the explicitly labelled viewport PLY."""
    from plyfile import PlyData, PlyElement

    source, output = source.resolve(), output.resolve()
    digest = file_hash(source)
    if expected and digest != expected:
        raise ValueError("Source PLY checksum differs from the published export")
    target = output / tag
    target.mkdir(parents=True, exist_ok=False)
    records = PlyData.read(source, mmap="r")["vertex"].data
    names = records.dtype.names
    if not len(records) or not {"x", "y", "z"} <= set(names):
        raise ValueError("Empty point cloud or missing XYZ")
    for name in names:
        if not np.isfinite(records[name]).all():
            raise ValueError(f"Nonfinite point attribute: {name}")
    xyz = np.column_stack([records[k] for k in ("x", "y", "z")]).astype(np.float32)
    gaussian = {"f_dc_0", "f_dc_1", "f_dc_2", "opacity"} <= set(names)
    if gaussian:
        rgb = np.column_stack([records[f"f_dc_{i}"] for i in range(3)])
        rgb = np.clip(0.5 + 0.28209479177387814 * rgb, 0, 1)
        # Stable sigmoid; original logits remain untouched in both source and preview.
        alpha = np.exp(-np.logaddexp(0, -records["opacity"]))
    else:
        rgb = np.column_stack([records[k] for k in ("red", "green", "blue")]) / 255
        alpha = np.ones(len(records), dtype=np.float32)
    rgba = np.column_stack([rgb, alpha]).astype(np.float32)
    np.savez(target / "all_centers.npz", xyz=xyz, rgba=rgba)
    indices = np.sort(
        np.random.default_rng(20261117).choice(
            len(records), min(200000, len(records)), replace=False
        )
    )
    preview = target / "viewport_sample.ply"
    PlyData(
        [PlyElement.describe(np.ascontiguousarray(records[indices]), "vertex")],
        text=False,
        byte_order="<",
    ).write(preview)
    report = {
        "status": "SUCCESS",
        "source": source.relative_to(ROOT).as_posix(),
        "source_sha256": digest,
        "point_count": len(records),
        "attributes": list(names),
        "all_attributes_finite": True,
        "bbox_min": xyz.min(axis=0).tolist(),
        "bbox_max": xyz.max(axis=0).tolist(),
        "xyz_percentiles_1_50_99": np.percentile(xyz, [1, 50, 99], axis=0).tolist(),
        "opacity_percentiles_0_50_100": np.percentile(alpha, [0, 50, 100]).tolist(),
        "gaussian": gaussian,
        "all_centers": (target / "all_centers.npz").relative_to(ROOT).as_posix(),
        "viewport_sample": preview.relative_to(ROOT).as_posix(),
        "sample_count": len(indices),
        "sample_seed": 20261117,
        "sample_policy": "Uniform without replacement; visualization only, no opacity/crop filtering",
        "coordinate_policy": "Original XYZ; identity object transform; no geometric alignment",
        "preview_semantics": "KIRI SH-DC viewport preview, not official CUDA rendering/evaluation",
    }
    if gaussian:
        report["log_scale_range"] = [
            float(min(records[f"scale_{i}"].min() for i in range(3))),
            float(max(records[f"scale_{i}"].max() for i in range(3))),
        ]
        rotations = np.column_stack([records[f"rot_{i}"] for i in range(4)])
        norms = np.linalg.norm(rotations, axis=1)
        report["quaternion_norm_min_max"] = [float(norms.min()), float(norms.max())]
    (target / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    return target / "inspection.json"


def build_blender(reports: list[str], dataset: str, destination: str) -> dict:
    """Keep existing scene objects; add all centers plus bounded native splat previews."""
    import bpy
    from mathutils import Matrix

    destination = ROOT / destination
    if destination.exists():
        raise FileExistsError(destination)
    original_scene = bpy.context.scene
    before = {
        o.name: (o.as_pointer(), tuple(v for row in o.matrix_world for v in row))
        for o in bpy.data.objects
    }
    scene = bpy.data.scenes.new("Phase02_PointCloud_Review_20261117")
    bpy.context.window.scene = scene
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 50
    scene.world = bpy.data.worlds.new("Phase02_Review_World")
    scene.world.color = (0, 0, 0)
    results = []
    previews = []
    for report_path in reports:
        report = json.loads((ROOT / report_path).read_text())
        tag = Path(report_path).parent.name
        collection = bpy.data.collections.new(tag)
        scene.collection.children.link(collection)
        with np.load(ROOT / report["all_centers"], allow_pickle=False) as arrays:
            mesh = bpy.data.meshes.new(tag + "_all_centers")
            mesh.vertices.add(report["point_count"])
            mesh.vertices.foreach_set("co", arrays["xyz"].ravel())
            colors = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
            colors.data.foreach_set("color", arrays["rgba"].ravel())
            mesh.update()
            obj = bpy.data.objects.new(tag + "_ALL_XYZ", mesh)
            collection.objects.link(obj)
            obj["source_sha256"] = report["source_sha256"]
            obj["source_path"] = report["source"]
            obj["artifact_role"] = "All original Gaussian centers, no faces; RGB from SH DC"
            obj.hide_render = True
            obj.hide_set(True)
            assert len(mesh.vertices) == report["point_count"]
            assert np.allclose(mesh.vertices[0].co, arrays["xyz"][0], atol=1e-7)
        preview_name = None
        if report["gaussian"]:
            for other in scene.objects:
                other.select_set(False)
            scene.sna_dgs_scene_properties.import_proxy = False
            scene.sna_dgs_scene_properties.import_uv = False
            outcome = bpy.ops.sna.dgs_render_import_ply_e0a3a(
                filepath=str(ROOT / report["viewport_sample"])
            )
            if "FINISHED" not in outcome:
                raise RuntimeError("Installed Gaussian addon could not import the native sample")
            preview_obj = bpy.context.active_object
            preview_obj.name = tag + "_SPLAT_PREVIEW"
            collection.objects.link(preview_obj)
            for previous in list(preview_obj.users_collection):
                if previous != collection:
                    previous.objects.unlink(preview_obj)
            assert len(preview_obj.data.vertices) == report["sample_count"]
            assert np.allclose(np.array(preview_obj.matrix_world), np.eye(4))
            required = {"f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "rot_0"}
            assert required <= set(preview_obj.data.attributes.keys())
            preview_obj["source_point_count"] = report["point_count"]
            preview_obj["sample_policy"] = report["sample_policy"]
            preview_obj["source_sha256"] = report["source_sha256"]
            preview_obj.modifiers["KIRI_3DGS_Render_GN"].show_viewport = True
            for mod in preview_obj.modifiers:
                if "Sorter" in mod.name:
                    mod.show_viewport = False
                    mod.show_render = False
            preview_obj.hide_render = bool(previews)
            preview_obj.hide_set(bool(previews))
            previews.append(preview_obj)
            preview_name = preview_obj.name
        results.append({"tag": tag, "all_centers_object": obj.name,
                        "full_point_count": len(mesh.vertices), "splat_preview": preview_name,
                        "preview_count": report["sample_count"]})
    metadata = json.loads((ROOT / dataset / "metadata.json").read_text())
    split = json.loads((ROOT / dataset / "splits.json").read_text())
    first_test = split["image_ids"]["test"][0]
    record = next(r for r in metadata["images"] if r["image_id"] == first_test)
    data = bpy.data.cameras.new("Phase02_COLMAP_TestCamera")
    camera = bpy.data.objects.new("Phase02_COLMAP_TestCamera", data)
    scene.collection.objects.link(camera)
    camera.matrix_world = Matrix(record["Twc"]) @ Matrix.Diagonal((1, -1, -1, 1))
    data.sensor_width = 36
    data.lens = record["K"][0][0] * data.sensor_width / record["width"]
    data.clip_start = 0.01
    data.clip_end = 10000
    scene.camera = camera
    # The addon imports with camera updates disabled. Supply the actual COLMAP
    # camera before enabling quads; its stored default projection is unrelated.
    view = camera.matrix_world.inverted()
    projection = camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(), x=960, y=540
    )
    for preview_obj in previews:
        bpy.context.view_layer.objects.active = preview_obj
        owner = preview_obj.sna_dgs_object_properties
        mode = next(
            item.identifier
            for item in owner.bl_rna.properties["update_mode"].enum_items
            if item.name == "Enable Camera Updates"
        )
        owner.update_mode = mode
        mod = preview_obj.modifiers["KIRI_3DGS_Render_GN"]
        menu = next(n for n in mod.node_group.nodes if hasattr(n, "enum_items"))
        mode_names = [item.name for item in menu.enum_items]
        mod["Socket_50"] = mode_names.index("Enable Camera Updates")
        mod["Socket_54"] = True
        for index in range(16):
            mod[f"Socket_{2 + index}"] = float(view[index % 4][index // 4])
            mod[f"Socket_{18 + index}"] = float(projection[index % 4][index // 4])
        mod["Socket_34"] = 960.0
        mod["Socket_35"] = 540.0
        preview_obj.update_tag(refresh={"DATA"})
    scene["camera_convention"] = "COLMAP Twc @ diag(1,-1,-1,1); original XYZ preserved"
    scene["coordinate_units"] = "Arbitrary monocular SfM units; no metric scale alignment"
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            area.spaces.active.region_3d.view_perspective = "CAMERA"
            area.spaces.active.overlay.show_overlays = False
            area.spaces.active.shading.type = "MATERIAL"
    if previews:
        for obj in scene.objects:
            obj.select_set(False)
        previews[0].select_set(True)
        bpy.context.view_layer.objects.active = previews[0]
    for name, state in before.items():
        obj = bpy.data.objects[name]
        assert (obj.as_pointer(), tuple(v for row in obj.matrix_world for v in row)) == state
    receipt = {"status": "SUCCESS", "scene": scene.name,
               "original_scene": original_scene.name, "preserved_objects": list(before),
               "clouds": results, "camera_image_id": first_test,
               "scope": "All XYZ imported; bounded native Gaussian samples for visual inspection only"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(destination), check_existing=False)
    destination.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return receipt


def verify_cuda_ply(run: Path, output: Path) -> dict:
    """Reload the unchanged Kaggle PLY and render one frozen test camera on CUDA."""
    from types import SimpleNamespace
    import time

    from src.gaussian_splatting.native import initialize

    config = json.loads((run / "resolved_config.json").read_text())
    result = json.loads((run / "training_result.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    environment = initialize()
    import torch
    from PIL import Image
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from src.gaussian_splatting.data import Dataset
    from src.gaussian_splatting.validation import validate_model

    ply = run / "point_cloud/iteration_030000/point_cloud.ply"
    if file_hash(ply) != result["export"]["sha256"]:
        raise ValueError("Untrusted PLY checksum")
    data = Dataset(config, output)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = GaussianModel(config["sh_degree"])
    model.load_ply(str(ply))
    model.active_sh_degree = config["sh_degree"]
    validation = validate_model(model)
    iid = data.ids["test"][0]
    camera = data.camera(iid)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.full((3,), float(config["white_background"]), device="cuda:0")
    with torch.no_grad():
        rgb = render(camera, model, pipe, bg)["render"].clamp(0, 1)
        if rgb.device != torch.device("cuda:0") or not torch.isfinite(rgb).all():
            raise ValueError("CUDA RGB render failed")
        prediction = np.rint(rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(prediction).save(output / "windows_cuda_render.png")
    original_png = run / f"renders/test/iteration_030000/{iid:06d}_render.png"
    with Image.open(original_png) as image:
        reference = np.array(image.convert("RGB"), dtype=np.int16)
    error = np.abs(prediction.astype(np.int16) - reference)
    report = {
        "status": "SUCCESS", "scope": "One-view PLY inference only; no optimizer resume or training",
        "environment": environment, "model": validation, "image_id": iid,
        "png_difference_mean_8bit": float(error.mean()),
        "png_difference_max_8bit": int(error.max()),
        "png_pixels_equal_fraction": float((error == 0).mean()),
        "tensor_device": str(rgb.device), "source_ply_sha256": file_hash(ply),
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
        "elapsed_seconds": time.perf_counter() - started,
        "comparison_scope": "Quantized PNG from Kaggle GPU vs Windows GPU; not float bitwise parity",
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    print(prepare(args.source, args.output, args.tag, args.expected_sha256))
