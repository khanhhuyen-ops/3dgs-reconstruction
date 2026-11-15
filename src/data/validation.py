"""Full reciprocal-track and image validation, with explicit qualification errors."""

from __future__ import annotations

from collections import Counter
from pathlib import Path, PureWindowsPath

import numpy as np
from PIL import Image as PILImage

from src.common.io import sha256
from src.data.colmap import MODEL_IDS, Model
from src.data.geometry import SUPPORTED, camera_centers, intrinsics, pose, project

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def image_path(root: Path, name: str) -> Path:
    if (
        "\\" in name
        or PureWindowsPath(name).drive
        or "\0" in name
        or "\n" in name
        or "\r" in name
    ):
        raise ValueError(f"Unsafe/nonportable image name: {name!r}")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"Image path escapes image root: {name!r}")
    return path


def distribution(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(a.size),
        "min": float(a.min()),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def validate_model(
    model: Model, images_root: Path, thresholds: dict | None = None
) -> dict:
    """Decode one image at a time; compute residuals per image, never all pixels."""
    limits = {
        "quaternion_tolerance": 1e-6,
        "rotation_tolerance": 1e-9,
        "warn_reprojection_px": 4.0,
        "max_reprojection_p95_px": 10.0,
        "min_registered_images": 3,
        "min_points": 1,
        "min_track_length": 2,
        "min_observations_per_image": 1,
        **(thresholds or {}),
    }
    errors, warnings, counts = [], [], Counter()

    def error(code: str, detail: str) -> None:
        counts[code] += 1
        if counts[code] <= 8:
            errors.append({"code": code, "detail": detail})

    if (
        len(model.images) < limits["min_registered_images"]
        or len(model.points) < limits["min_points"]
    ):
        error("insufficient_geometry", "Too few registered images or sparse points")
    camera_ok = set()
    for c in model.cameras.values():
        if c.model not in SUPPORTED:
            error("unsupported_camera", f"camera {c.id}: {c.model}")
            continue
        if (
            c.params.shape != (MODEL_IDS[c.model][1],)
            or not np.isfinite(c.params).all()
            or min(c.width, c.height) <= 0
        ):
            error("invalid_intrinsics", f"camera {c.id}: dimensions/parameters")
            continue
        k = intrinsics(c)
        if (
            k[0, 0] <= 0
            or k[1, 1] <= 0
            or not (0 <= k[0, 2] <= c.width and 0 <= k[1, 2] <= c.height)
        ):
            error("invalid_intrinsics", f"camera {c.id}: focal lengths/principal point")
            continue
        if not (
            0.25 * c.width <= k[0, 2] <= 0.75 * c.width
            and 0.25 * c.height <= k[1, 2] <= 0.75 * c.height
        ):
            warnings.append(f"camera {c.id}: unusual principal point")
        camera_ok.add(c.id)
    image_records, transforms, filenames = {}, {}, set()
    all_paths = (
        {
            p.relative_to(images_root).as_posix()
            for p in images_root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        }
        if images_root.is_dir()
        else set()
    )
    for im in model.images.values():
        if im.name.casefold() in filenames:
            error("duplicate_filename", im.name)
        filenames.add(im.name.casefold())
        if im.camera_id not in model.cameras:
            error("missing_camera", f"image {im.id}: camera {im.camera_id}")
        if (
            im.xys.shape != (len(im.point3D_ids), 2)
            or not np.isfinite(im.xys).all()
            or np.any(im.point3D_ids < -1)
        ):
            error("invalid_observations", f"image {im.id}")
        if (
            not np.isfinite(im.tvec).all()
            or not np.isfinite(im.qvec).all()
            or abs(np.linalg.norm(im.qvec) - 1) > limits["quaternion_tolerance"]
        ):
            error("invalid_pose", f"image {im.id}: quaternion/translation")
        else:
            t = pose(im.qvec, im.tvec)
            r = t[:3, :3]
            if (
                not np.allclose(
                    r @ r.T, np.eye(3), atol=limits["rotation_tolerance"], rtol=0
                )
                or abs(np.linalg.det(r) - 1) > limits["rotation_tolerance"]
            ):
                error("invalid_pose", f"image {im.id}: rotation")
            transforms[im.id] = t
        try:
            path = image_path(images_root, im.name)
            with PILImage.open(path) as source:
                source.load()
                size = source.size
                if source.getexif().get(274, 1) != 1:
                    raise ValueError(
                        "Nonidentity EXIF orientation requires explicit camera-consistent conversion"
                    )
            c = model.cameras.get(im.camera_id)
            if c is not None and size != (c.width, c.height):
                error("image_size", f"{im.name}: {size} != {(c.width, c.height)}")
            image_records[im.id] = {
                "name": im.name,
                "width": size[0],
                "height": size[1],
                "sha256": sha256(path),
            }
        except (OSError, ValueError) as exc:
            error("image_file", f"{im.name}: {exc}")
    duplicate_ids, bad_point_ids, track_lengths, reported = [], [], [], []
    # Per-image expected associations also catch observations absent from point tracks.
    reciprocal = {iid: {} for iid in model.images}
    for p in model.points.values():
        bad = False
        if (
            p.xyz.shape != (3,)
            or not np.isfinite(p.xyz).all()
            or p.rgb.shape != (3,)
            or np.any(p.rgb < 0)
            or np.any(p.rgb > 255)
            or not np.isfinite(p.error)
            or p.error < 0
        ):
            error("invalid_point", f"point {p.id}")
            bad = True
        else:
            reported.append(p.error)
        track_lengths.append(len(p.track))
        unique_images = set(p.track[:, 0].tolist())
        if len(unique_images) != len(p.track):
            duplicate_ids.append(p.id)
        if len(unique_images) < limits["min_track_length"]:
            error("short_track", f"point {p.id}")
            bad = True
        for iid, idx in p.track:
            im = model.images.get(int(iid))
            if (
                im is None
                or idx < 0
                or idx >= len(im.point3D_ids)
                or im.point3D_ids[idx] != p.id
            ):
                error("broken_track", f"point {p.id} -> ({iid},{idx})")
                bad = True
            elif int(idx) in reciprocal[int(iid)]:
                error("duplicate_track_entry", f"point {p.id} -> ({iid},{idx})")
                bad = True
            else:
                reciprocal[int(iid)][int(idx)] = p.id
        if bad:
            bad_point_ids.append(p.id)
    if duplicate_ids:
        error(
            "duplicate_image_tracks",
            f"{len(duplicate_ids)} tracks contain multiple observations of the same image",
        )
    residuals, depths, support = [], [], {}
    for iid, im in model.images.items():
        indices = np.flatnonzero(im.point3D_ids >= 0)
        good_indices, xyz = [], []
        for idx in indices:
            pid = int(im.point3D_ids[idx])
            if reciprocal[iid].get(int(idx)) != pid:
                error("broken_observation", f"image {iid} observation {idx} -> {pid}")
            if pid in model.points and np.isfinite(model.points[pid].xyz).all():
                good_indices.append(int(idx))
                xyz.append(model.points[pid].xyz)
        support[iid] = len(good_indices)
        if support[iid] < limits["min_observations_per_image"]:
            error("image_support", f"image {iid}: {support[iid]} observations")
        if xyz and iid in transforms and im.camera_id in camera_ok:
            t = transforms[iid]
            xc = np.array(xyz) @ t[:3, :3].T + t[:3, 3]
            depths.extend(xc[:, 2].tolist())
            if np.any(xc[:, 2] <= 0):
                error("negative_depth", f"image {iid}")
            positive = xc[:, 2] > 0
            values = np.linalg.norm(
                project(model.cameras[im.camera_id], xc[positive])
                - im.xys[np.array(good_indices)[positive]],
                axis=1,
            )
            if not np.isfinite(values).all():
                error("nonfinite_projection", f"image {iid}")
            else:
                residuals.extend(values.tolist())
    residual_stats = distribution(residuals)
    if residuals and residual_stats["p95"] > limits["max_reprojection_p95_px"]:
        error(
            "reprojection_threshold", f"p95={residual_stats['p95']:.6g}px exceeds limit"
        )
    if residuals and residual_stats["max"] > limits["warn_reprojection_px"]:
        warnings.append(
            "Some reprojection residuals exceed warning threshold; none were silently removed"
        )
    valid_xyz = np.array(
        [p.xyz for p in model.points.values() if np.isfinite(p.xyz).all()]
    ).reshape(-1, 3)
    bbox = (
        [valid_xyz.min(axis=0).tolist(), valid_xyz.max(axis=0).tolist()]
        if len(valid_xyz)
        else None
    )
    centers = np.array([np.linalg.inv(t)[:3, 3] for t in transforms.values()]).reshape(
        -1, 3
    )
    if len(centers) and np.linalg.norm(np.ptp(centers, axis=0)) < 1e-12:
        error("degenerate_camera_centers", "All camera centers coincide")
    sphere_center = np.median(valid_xyz, axis=0) if len(valid_xyz) else np.zeros(3)
    extent = (
        float(np.linalg.norm(np.ptp(valid_xyz, axis=0))) if len(valid_xyz) else None
    )
    volume = float(np.prod(np.ptp(valid_xyz, axis=0))) if len(valid_xyz) else 0
    camera_ids = Counter(im.camera_id for im in model.images.values())
    return {
        "status": "FAILED" if errors else "SUCCESS",
        "errors": errors,
        "error_counts": dict(counts),
        "warnings": warnings,
        "statistics": {
            "registered_images": len(model.images),
            "camera_count": len(model.cameras),
            "image_to_camera": {
                str(im.id): im.camera_id for im in model.images.values()
            },
            "camera_image_counts": dict(camera_ids),
            "camera_models": sorted({c.model for c in model.cameras.values()}),
            "points": len(model.points),
            "finite_points": len(valid_xyz),
            "track_length": distribution(track_lengths),
            "reported_point_error_px": distribution(reported),
            "reprojection_error_px": residual_stats,
            "observed_depth": distribution(depths),
            "positive_depth_fraction": float(np.mean(np.array(depths) > 0))
            if depths
            else None,
            "per_image_support": support,
            "bbox": bbox,
            "approximate_extent": extent,
            "bounding_sphere": {
                "center": sphere_center.tolist(),
                "radius": float(np.linalg.norm(valid_xyz - sphere_center, axis=1).max())
                if len(valid_xyz)
                else None,
            },
            "point_density_per_bbox_volume": len(valid_xyz) / volume
            if volume > 0
            else None,
            "camera_centers": {
                str(i): c.tolist() for i, c in camera_centers(model).items()
            }
            if len(transforms) == len(model.images)
            else {},
            "camera_center_axis_range": np.ptp(centers, axis=0).tolist()
            if len(centers)
            else None,
            "camera_distance_from_median": distribution(
                np.linalg.norm(centers - np.median(centers, axis=0), axis=1)
            )
            if len(centers)
            else distribution([]),
            "registered_fraction": len(image_records) / len(all_paths)
            if all_paths
            else None,
            "unregistered_images": sorted(
                all_paths - {im.name for im in model.images.values()}
            ),
        },
        "duplicate_track_point_ids": duplicate_ids,
        "invalid_point_ids": bad_point_ids,
        "images": image_records,
        "thresholds": limits,
    }


def filter_points(model: Model, rejected: set[int]) -> Model:
    """Derived copy: remove complete tracks and clear all reciprocal associations."""
    from dataclasses import replace

    images = {}
    for im in model.images.values():
        ids = im.point3D_ids.copy()
        ids[np.isin(ids, list(rejected))] = -1
        images[im.id] = replace(im, point3D_ids=ids)
    return Model(
        dict(model.cameras),
        images,
        {pid: p for pid, p in model.points.items() if pid not in rejected},
    )
