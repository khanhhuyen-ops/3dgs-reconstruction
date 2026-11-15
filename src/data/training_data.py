"""Train-only initialization geometry/appearance and observation tables."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from src.data.colmap import Model
from src.data.geometry import camera_centers, pose, project
from src.data.image_processing import bilinear
from src.data.validation import filter_points, image_path


def derive_training_model(
    model: Model, images_root: Path, split: dict, options: dict
) -> tuple[Model, dict, dict]:
    train = set(split["image_ids"]["train"])
    centers = camera_centers(model)
    points, rejected = {}, {"insufficient_train_views": [], "insufficient_parallax": []}
    for p in model.points.values():
        track = np.array(
            [(iid, idx) for iid, idx in p.track if int(iid) in train], dtype=np.int64
        ).reshape(-1, 2)
        if len(track) < options["min_train_views"]:
            rejected["insufficient_train_views"].append(p.id)
            continue
        if options["min_parallax_degrees"] > 0:
            rays = np.array([p.xyz - centers[int(i)] for i, _ in track])
            norms = np.linalg.norm(rays, axis=1, keepdims=True)
            if np.any(norms < 1e-12):
                raise ValueError(f"Point {p.id} coincides with a camera center")
            rays /= norms
            angle = np.degrees(np.arccos(np.clip((rays @ rays.T).min(), -1, 1)))
            if angle < options["min_parallax_degrees"]:
                rejected["insufficient_parallax"].append(p.id)
                continue
        points[p.id] = replace(p, track=track, rgb=np.zeros(3, dtype=np.int64))
    if not points:
        raise ValueError("No points have enough unique train observations/parallax")
    ordered = sorted(points)
    row_for = {pid: i for i, pid in enumerate(ordered)}
    sums, counts = np.zeros((len(points), 3)), np.zeros(len(points), dtype=int)
    obs = {
        k: []
        for k in ("image_id", "point2D_idx", "point3D_id", "xy", "z", "residual_px")
    }
    residual_sums = np.zeros(len(points))
    for iid in split["image_ids"]["train"]:
        im = model.images[iid]
        indices = np.flatnonzero(np.isin(im.point3D_ids, ordered))
        pids = im.point3D_ids[indices]
        rows = np.array([row_for[int(pid)] for pid in pids], dtype=int)
        with PILImage.open(image_path(images_root, im.name)) as source:
            pixels = np.array(source.convert("RGB"))
        colors, valid = bilinear(pixels, im.xys[indices])
        if not valid.all():
            raise ValueError(
                f"Training observations outside image footprint: image {iid}"
            )
        sums[rows] += colors
        counts[rows] += 1
        t = pose(im.qvec, im.tvec)
        xc = (
            np.array([points[int(pid)].xyz for pid in pids]).reshape(-1, 3)
            @ t[:3, :3].T
            + t[:3, 3]
        )
        residual = np.linalg.norm(
            project(model.cameras[im.camera_id], xc) - im.xys[indices], axis=1
        )
        residual_sums[rows] += residual
        obs["image_id"].extend([iid] * len(indices))
        obs["point2D_idx"].extend(indices.tolist())
        obs["point3D_id"].extend(pids.tolist())
        obs["xy"].extend(im.xys[indices].tolist())
        obs["z"].extend(xc[:, 2].tolist())
        obs["residual_px"].extend(residual.tolist())
    for pid, row in row_for.items():
        if counts[row] < options["min_train_views"]:
            raise ValueError(f"Train appearance support mismatch: point {pid}")
        points[pid] = replace(
            points[pid],
            rgb=np.clip(np.rint(sums[row] / counts[row]), 0, 255).astype(np.int64),
            error=float(residual_sums[row] / counts[row]),
        )
    base = filter_points(model, set(model.points) - set(points))
    images = {
        iid: replace(
            im,
            point3D_ids=im.point3D_ids.copy()
            if iid in train
            else np.full_like(im.point3D_ids, -1),
        )
        for iid, im in base.images.items()
    }
    return (
        Model(base.cameras, images, points),
        {
            k: np.array(
                v, dtype=np.float64 if k in ("xy", "z", "residual_px") else np.int64
            )
            for k, v in obs.items()
        },
        rejected,
    )


def write_point_cloud(model: Model, path: Path) -> None:
    """Native Gaussian loaders expect XYZ, zero normals and uint8 RGB in PLY."""
    dtype = np.dtype(
        [(k, "<f4") for k in ("x", "y", "z", "nx", "ny", "nz")]
        + [(k, "u1") for k in ("red", "green", "blue")]
    )
    array = np.zeros(len(model.points), dtype=dtype)
    points = sorted(model.points.values(), key=lambda p: p.id)
    xyz, rgb = np.array([p.xyz for p in points]), np.array([p.rgb for p in points])
    for i, k in enumerate(("x", "y", "z")):
        array[k] = xyz[:, i]
    for i, k in enumerate(("red", "green", "blue")):
        array[k] = rgb[:, i]
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex "
        + str(len(points))
        + "\n"
    )
    header += (
        "".join(
            "property "
            + ("float" if k in ("x", "y", "z", "nx", "ny", "nz") else "uchar")
            + " "
            + k
            + "\n"
            for k in dtype.names
        )
        + "end_header\n"
    )
    with path.open("xb") as f:
        f.write(header.encode("ascii"))
        f.write(array.tobytes())
