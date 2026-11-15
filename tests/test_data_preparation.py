"""Synthetic fixtures prove adapter invariants, never scene reconstruction quality."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import struct

import numpy as np
from PIL import Image as PILImage
import pytest

from src.common.io import read_json, sha256
from src.data.colmap import (
    Camera,
    Image,
    Model,
    Point,
    discover_models,
    read_model,
    semantic_hashes,
    validate_trivial_rigs,
    write_model,
)
from src.data.config import DEFAULTS, load_config
from src.data.geometry import (
    intrinsics,
    normalization,
    normalize_geometry,
    pose,
    project,
    undistort_xy,
)
from src.data.image_processing import prepare_images, remap_radial, transfer
from src.data.loader import PreparedDataset, verify_dataset
from src.data.preparation import prepare, qualify
from src.data.splits import make_split
from src.data.training_data import derive_training_model
from src.data.validation import validate_model


@pytest.fixture
def scene(tmp_path: Path) -> tuple[Model, dict, Path]:
    root = tmp_path
    images_root = root / "source/images"
    images_root.mkdir(parents=True)
    camera = Camera(7, "PINHOLE", 48, 32, np.array([35.0, 36.0, 24.0, 16.0]))
    xyz = np.array([[-0.3, -0.2, 4.0], [0.1, 0.15, 4.2], [0.25, -0.1, 3.5]])
    ids = [101, 501, 9001]
    images = {}
    for index in range(12):
        iid = index * 3 + 11
        angle = (index - 5) * 0.01
        q = np.array([np.cos(angle / 2), 0, np.sin(angle / 2), 0])
        t = np.array([(index - 5) * 0.1, 0.0, 0.0])
        tcw = pose(q, t)
        xy = project(camera, xyz @ tcw[:3, :3].T + tcw[:3, 3])
        name = f"frame_{index:03d}.png"
        images[iid] = Image(iid, q, t, 7, name, xy, np.array(ids, dtype=np.int64))
        PILImage.new("RGB", (48, 32), (30 + index * 5, 100, 170)).save(
            images_root / name
        )
    points = {
        pid: Point(
            pid,
            xyz[i],
            np.array([255, 0, 0]),
            0.0,
            np.array([(iid, i) for iid in images]),
        )
        for i, pid in enumerate(ids)
    }
    model = Model({7: camera}, images, points)
    write_model(model, root / "source/model")
    config = deepcopy(DEFAULTS)
    config["dataset_name"] = "unit-fixture-not-scene"
    config["source"].update(
        mode="image_set", image_root="source/images", model_dir="source/model"
    )
    config["output"].update(
        dataset_root="derived/prepared", report_root="derived/reports"
    )
    return model, config, root


@pytest.mark.parametrize("format", ["bin", "txt"])
def test_official_format_roundtrip_preserves_ids_and_empty_observations(scene, format):
    model, _, root = scene
    im = next(iter(model.images.values()))
    model.images[im.id] = replace(
        im,
        name="name with spaces.png",
        xys=np.empty((0, 2)),
        point3D_ids=np.empty(0, np.int64),
    )
    out = root / format
    write_model(model, out, format)
    loaded = read_model(out)
    assert semantic_hashes(model) == semantic_hashes(loaded)
    assert loaded.images[im.id].xys.shape == (0, 2)
    assert set(loaded.points) == {101, 501, 9001}
    with pytest.raises(FileExistsError):
        write_model(model, out, format)


def test_partial_binary_never_falls_back_and_truncation_rejected(scene):
    model, _, root = scene
    out = root / "partial"
    write_model(model, out, "txt")
    (out / "cameras.bin").write_bytes(b"\0" * 8)
    with pytest.raises(FileNotFoundError, match="images.bin"):
        read_model(out)
    assert discover_models(root / "source")[0]["formats"]["bin"] == [
        "cameras",
        "images",
        "points3D",
    ]
    source = root / "source/model/images.bin"
    source.write_bytes(source.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated|Truncated"):
        read_model(source.parent)


def test_quaternion_pose_inverse_projection_and_similarity():
    q = np.array([np.sqrt(0.5), 0, np.sqrt(0.5), 0])
    t = pose(q, np.array([0.0, 0.0, 3.0]))
    np.testing.assert_allclose(
        t[:3, :3] @ np.array([0.0, 0.0, 1.0]), [1, 0, 0], atol=1e-14
    )
    np.testing.assert_allclose(np.linalg.inv(t)[:3, 3], -t[:3, :3].T @ t[:3, 3])
    camera = Camera(1, "PINHOLE", 100, 80, np.array([40.0, 50.0, 50.0, 40.0]))
    xyz = np.array([[1.0, 0.2, 2.0], [2.0, 0.0, 1.0]])
    norm = normalization(np.array([[-1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 2.0, 1.0]]))
    newxyz, newt = normalize_geometry(xyz, t, norm)
    np.testing.assert_allclose(
        project(camera, xyz @ t[:3, :3].T + t[:3, 3]),
        project(camera, newxyz @ newt[:3, :3].T + newt[:3, 3]),
        atol=1e-10,
    )
    homogeneous = (
        np.column_stack((newxyz, np.ones(len(newxyz)))) @ np.array(norm["inverse"]).T
    )
    np.testing.assert_allclose(homogeneous[:, :3], xyz, atol=1e-12)
    with pytest.raises(ValueError, match="Degenerate"):
        normalization(np.zeros((3, 3)))


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("focal", "invalid_intrinsics"),
        ("quaternion", "invalid_pose"),
        ("nanpoint", "invalid_point"),
        ("rgb", "invalid_point"),
        ("track", "broken_track"),
        ("observation", "broken_observation"),
        ("duplicate", "duplicate_image_tracks"),
        ("negative", "negative_depth"),
        ("size", "image_size"),
        ("missing", "image_file"),
        ("corrupt", "image_file"),
        ("name", "image_file"),
        ("duplicate_name", "duplicate_filename"),
        ("model", "unsupported_camera"),
    ],
)
def test_qualification_rejects_corruption(scene, mutation, code):
    model, _, root = scene
    model = deepcopy(model)
    im = next(iter(model.images.values()))
    p = model.points[101]
    if mutation == "focal":
        model.cameras[7].params[1] = -1
    elif mutation == "quaternion":
        im.qvec *= 2
    elif mutation == "nanpoint":
        p.xyz[0] = np.nan
    elif mutation == "rgb":
        p.rgb[0] = 256
    elif mutation == "track":
        p.track[0, 1] = 9999
    elif mutation == "observation":
        im.point3D_ids[0] = 99999
    elif mutation == "duplicate":
        im.xys = np.vstack((im.xys, im.xys[0]))
        im.point3D_ids = np.append(im.point3D_ids, 101)
        p.track = np.vstack((p.track, [im.id, 3]))
    elif mutation == "negative":
        p.xyz[2] = -10
    elif mutation == "size":
        model.cameras[7].width = 49
    elif mutation == "missing":
        (root / "source/images" / im.name).unlink()
    elif mutation == "corrupt":
        (root / "source/images" / im.name).write_bytes(b"not image")
    elif mutation == "name":
        im.name = "../escape.png"
    elif mutation == "duplicate_name":
        list(model.images.values())[1].name = im.name
    elif mutation == "model":
        model.cameras[7].model = "OPENCV"
    report = validate_model(model, root / "source/images")
    assert report["status"] == "FAILED"
    assert code in report["error_counts"]


def test_derived_duplicate_filter_clears_reciprocity_without_source_edits(scene):
    model, config, root = scene
    im = next(iter(model.images.values()))
    im.xys = np.vstack((im.xys, im.xys[0]))
    im.point3D_ids = np.append(im.point3D_ids, 101)
    model.points[101].track = np.vstack((model.points[101].track, [im.id, 3]))
    write_model(model, root / "duplicate")
    config["source"]["model_dir"] = "duplicate"
    _, rejected = qualify(config, root)
    assert rejected["status"] == "FAILED"
    before = {p.name: sha256(p) for p in (root / "duplicate").iterdir()}
    config["filter"]["duplicate_tracks"] = "drop_track"
    accepted, report = qualify(config, root)
    assert report["status"] == "SUCCESS" and 101 not in accepted.points
    assert np.count_nonzero(accepted.images[im.id].point3D_ids == -1) == 2
    assert before == {p.name: sha256(p) for p in (root / "duplicate").iterdir()}


def test_distortion_resize_parity_and_pixel_center_mapping(scene):
    model, config, root = scene
    c = Camera(7, "SIMPLE_RADIAL", 48, 32, np.array([35.0, 24.0, 16.0, -0.02]))
    model.cameras[7] = c
    xyz = np.array([p.xyz for p in model.points.values()])
    for im in model.images.values():
        t = pose(im.qvec, im.tvec)
        im.xys = project(c, xyz @ t[:3, :3].T + t[:3, 3])
    options = {**config["images"], "max_width": 24, "format": "png"}
    new, records = prepare_images(
        model, root / "source/images", root / "transformed", options, "copy"
    )
    for im in new.images.values():
        t = pose(im.qvec, im.tvec)
        np.testing.assert_allclose(
            im.xys,
            project(new.cameras[7], xyz @ t[:3, :3].T + t[:3, 3]),
            atol=1e-4,
            rtol=0,
        )
    assert records[11]["transform"]["scale_xy"] == [0.5, 0.5]
    # Identity distortion must map every first/last pixel center exactly.
    pixels = np.arange(48 * 32 * 3, dtype=np.uint8).reshape(32, 48, 3)
    zero = replace(c, params=np.array([35.0, 24.0, 16.0, 0.0]))
    target = Camera(7, "PINHOLE", 48, 32, np.array([35.0, 35.0, 24.0, 16.0]))
    warped, mask = remap_radial(pixels, zero, target)
    np.testing.assert_array_equal(warped, pixels)
    assert np.all(mask == 255)
    with pytest.raises(ValueError, match="requires undistortion"):
        prepare_images(
            model,
            root / "source/images",
            root / "bad",
            {**options, "undistort": "never"},
            "copy",
        )


@pytest.mark.parametrize("strategy", ["interleaved", "random", "temporal_block"])
def test_split_is_disjoint_deterministic_complete_and_chronological(scene, strategy):
    model, config, _ = scene
    policy = {**config["split"], "strategy": strategy}
    a = make_split(model, policy, "image_set")
    b = make_split(model, policy, "image_set")
    assert a == b
    sets = [set(v) for v in a["image_ids"].values()]
    assert set.union(*sets) == set(model.images)
    assert all(not sets[i] & sets[j] for i, j in [(0, 1), (0, 2), (1, 2)])
    frames = {
        im.name: {"timestamp_sec": float(100 - index), "frame_id": index}
        for index, im in enumerate(model.images.values())
    }
    video = make_split(model, policy, "video", frames)
    assert video["ordered_image_ids"] == list(reversed(model.images))
    with pytest.raises(ValueError, match="timestamp"):
        make_split(model, policy, "video")
    with pytest.raises(ValueError, match="BLOCKED_INPUT"):
        make_split(model, {**policy, "protocol": "strict_train_only"}, "image_set")


def test_train_colors_and_tracks_never_read_heldout_rgb(scene):
    model, config, root = scene
    split = make_split(model, config["split"], "image_set")
    train, obs, _ = derive_training_model(
        model, root / "source/images", split, config["training_points"]
    )
    for name in ("val", "test"):
        for iid in split["image_ids"][name]:
            # Corrupt held-out images entirely: deriving training points must still work.
            (root / "source/images" / model.images[iid].name).write_bytes(
                b"heldout must not be decoded"
            )
    again, _, _ = derive_training_model(
        model, root / "source/images", split, config["training_points"]
    )
    assert semantic_hashes(train) == semantic_hashes(again)
    assert set(obs["image_id"]) <= set(split["image_ids"]["train"])
    expected = np.mean(
        [30 + list(model.images).index(iid) * 5 for iid in split["image_ids"]["train"]]
    )
    assert train.points[101].rgb[0] == round(expected)
    assert all(
        set(p.track[:, 0]) <= set(split["image_ids"]["train"])
        for p in train.points.values()
    )


def test_end_to_end_export_loader_resume_and_integrity(scene):
    model, config, root = scene
    before = {
        p.relative_to(root).as_posix(): sha256(p)
        for p in (root / "source").rglob("*")
        if p.is_file()
    }
    dry = prepare(config, root, dry_run=True)
    assert dry["status"] == "DRY_RUN" and not (root / "derived").exists()
    result = prepare(config, root)
    output = root / result["dataset_path"]
    assert result["readiness"]["status"] == "SUCCESS"
    dataset = PreparedDataset(output)
    batch = dataset[0]
    assert batch["rgb"].shape == (3, 32, 48) and batch["rgb"].dtype == np.float32
    assert batch["mask"].all() and len(dataset) == 9
    assert set(dataset.initialization_points()["ids"]) == set(model.points)
    with pytest.raises(FileExistsError):
        prepare(config, root)
    assert prepare(config, root, resume=True)["resumed"] is True
    assert before == {
        p.relative_to(root).as_posix(): sha256(p)
        for p in (root / "source").rglob("*")
        if p.is_file()
    }
    damaged = output / read_json(output / "metadata.json")["images"][0]["path"]
    damaged.write_bytes(b"corrupt")
    assert verify_dataset(output)["status"] == "FAILED"
    with pytest.raises(ValueError, match="Resume rejected"):
        prepare(config, root, resume=True)


@pytest.mark.parametrize("mode", ["copy", "hardlink", "symlink"])
def test_copy_link_modes_and_overwrite_protection(tmp_path, mode):
    src = tmp_path / "source.txt"
    src.write_text("preserve")
    dst = tmp_path / "destination.txt"
    try:
        transfer(src, dst, mode)
    except OSError as exc:
        if mode == "symlink":
            pytest.skip(f"Windows symlink privilege unavailable: {exc}")
        raise
    assert dst.read_text() == "preserve"
    with pytest.raises(FileExistsError):
        transfer(src, dst, mode)


def test_binary_and_text_rig_pose_equivalence_and_rejection(scene):
    model, _, root = scene
    directory = root / "source/model"
    (directory / "rigs.bin").write_bytes(struct.pack("<QIIiI", 1, 1, 1, 0, 7))
    with (directory / "frames.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(model.images)))
        for im in model.images.values():
            f.write(
                struct.pack("<II7dIiIQ", im.id, 1, *im.qvec, *im.tvec, 1, 0, 7, im.id)
            )
    assert validate_trivial_rigs(directory, model)["status"] == "verified_trivial"
    # A multi-sensor rig cannot silently masquerade as an ordinary camera model.
    (directory / "rigs.bin").write_bytes(struct.pack("<QIIiI", 1, 1, 2, 0, 7))
    with pytest.raises(ValueError, match="Nontrivial"):
        validate_trivial_rigs(directory, model)


def test_config_protects_raw_gt_and_unknown_fields(scene):
    _, config, root = scene
    import yaml

    path = root / "config.yaml"
    for key, value, match in [
        ("dataset_root", "source/images/derived", "Protected"),
        ("dataset_root", "datasets/scene/raw/x", "Protected"),
    ]:
        variant = deepcopy(config)
        variant["output"][key] = value
        path.write_text(yaml.safe_dump(variant), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_config(path, root)
    variant = deepcopy(config)
    variant["source"]["image_root"] = "datasets/scene/raw/ground_truth"
    path.write_text(yaml.safe_dump(variant), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluator"):
        load_config(path, root)
    variant = deepcopy(config)
    variant["silently_ignored_key"] = True
    path.write_text(yaml.safe_dump(variant), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path, root)


def test_mismatched_authoritative_exports_rejected(scene):
    model, config, root = scene
    model.images[11].tvec[0] += 0.1
    write_model(model, root / "alternate", "txt")
    config["source"]["comparison_model_dir"] = "alternate"
    with pytest.raises(ValueError, match="disagreement"):
        qualify(config, root)


def test_radial_inverse_restores_nonzero_distortion():
    c = Camera(1, "SIMPLE_RADIAL", 1920, 1080, np.array([1159.0, 960.0, 540.0, -0.028]))
    xyz = np.array([[-0.7, -0.3, 1.0], [0.8, 0.4, 1.0], [0, 0, 1.0]])
    uv = project(c, xyz)
    pinhole = replace(
        c, model="PINHOLE", params=np.array([1159.0, 1159.0, 960.0, 540.0])
    )
    np.testing.assert_allclose(
        undistort_xy(c, uv), project(pinhole, xyz), atol=1e-4, rtol=0
    )
    np.testing.assert_allclose(intrinsics(c), intrinsics(pinhole))


def test_interrupted_preparation_reuses_only_verified_images(scene, monkeypatch):
    _, config, root = scene
    from src.data import preparation, image_processing

    derive = preparation.derive_training_model

    def interrupted(*args, **kwargs):
        raise RuntimeError("Injected unit-test interruption after images")

    monkeypatch.setattr(preparation, "derive_training_model", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        prepare(config, root)
    partial = next((root / config["output"]["dataset_root"]).glob(".*.partial-*"))
    assert read_json(partial / "state.json")["status"] == "FAILED"
    monkeypatch.setattr(preparation, "derive_training_model", derive)
    cached = []
    original = image_processing.transfer

    def record_transfer(source, destination, mode):
        if source.is_relative_to(partial):
            cached.append(source)
        return original(source, destination, mode)

    monkeypatch.setattr(image_processing, "transfer", record_transfer)
    result = prepare(config, root, resume=True)
    assert result["status"] == "SUCCESS" and len(cached) == 12
    assert read_json(partial / "state.json")["status"] == "FAILED"


def test_radial_outside_image_tracks_require_explicit_derived_filter(scene):
    model, config, root = scene
    c = Camera(7, "SIMPLE_RADIAL", 48, 32, np.array([35.0, 24.0, 16.0, -0.05]))
    model.cameras[7] = c
    xyz = np.array([p.xyz for p in model.points.values()])
    for im in model.images.values():
        t = pose(im.qvec, im.tvec)
        im.xys = project(c, xyz @ t[:3, :3].T + t[:3, 3])
    model.images[11].xys[0] = [47.99, 16.0]
    write_model(model, root / "radial_model")
    config["source"]["model_dir"] = "radial_model"
    config["images"]["format"] = "png"
    config["validation"]["max_reprojection_p95_px"] = 100.0
    with pytest.raises(ValueError, match="outside transformed image"):
        prepare(config, root)
    config["filter"]["outside_image_tracks"] = "drop_track"
    result = prepare(config, root)
    metadata = read_json(root / result["dataset_path"] / "metadata.json")
    assert metadata["transformed_point_filter"]["outside_image_point_ids"] == [101]
    assert metadata["initialization"]["points"] == 2


def test_diagnostic_probe_model_never_qualifies_as_scene(scene):
    model, config, root = scene
    write_model(model, root / "audits/probe")
    config["source"]["model_dir"] = "audits/probe"
    with pytest.raises(ValueError, match="Diagnostic/probe"):
        qualify(config, root)
