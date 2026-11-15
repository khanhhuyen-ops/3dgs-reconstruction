"""Stream image transforms with matching K/keypoints in COLMAP pixel centers."""

from __future__ import annotations

import os
import shutil
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from src.common.io import read_json, sha256, write_json
from src.data.colmap import Camera, Model
from src.data.geometry import intrinsics, project, undistort_xy
from src.data.validation import image_path


def transfer(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(os.path.relpath(source, destination.parent))
    else:
        raise ValueError(f"Unknown transfer mode: {mode}")


def bilinear(pixels: np.ndarray, xys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates use first-pixel center (.5,.5); clamp edges within footprint."""
    h, w = pixels.shape[:2]
    valid = (
        (xys[:, 0] >= 0)
        & (xys[:, 0] <= w)
        & (xys[:, 1] >= 0)
        & (xys[:, 1] <= h)
        & np.isfinite(xys).all(axis=1)
    )
    xy = np.clip(xys - 0.5, [0, 0], [w - 1, h - 1])
    lo = np.floor(xy).astype(int)
    hi = np.minimum(lo + 1, [w - 1, h - 1])
    t = xy - lo
    a, b = t[:, 0:1], t[:, 1:2]
    values = (
        (1 - a) * (1 - b) * pixels[lo[:, 1], lo[:, 0]]
        + a * (1 - b) * pixels[lo[:, 1], hi[:, 0]]
        + (1 - a) * b * pixels[hi[:, 1], lo[:, 0]]
        + a * b * pixels[hi[:, 1], hi[:, 0]]
    )
    values[~valid] = 0
    return values, valid


def remap_radial(
    pixels: np.ndarray, source: Camera, target: Camera
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse image warp in 64-row tiles; only one RGB image resides in RAM."""
    out = np.empty((target.height, target.width, 3), np.uint8)
    mask = np.empty((target.height, target.width), np.uint8)
    k = intrinsics(target)
    for start in range(0, target.height, 64):
        stop = min(start + 64, target.height)
        xx, yy = np.meshgrid(
            np.arange(target.width) + 0.5, np.arange(start, stop) + 0.5
        )
        uv = np.column_stack((xx.ravel(), yy.ravel()))
        xy = (uv - k[:2, 2]) / np.diag(k)[:2]
        source_xy = project(source, np.column_stack((xy, np.ones(len(xy)))))
        rgb, valid = bilinear(pixels, source_xy)
        out[start:stop] = (
            np.clip(np.rint(rgb), 0, 255)
            .astype(np.uint8)
            .reshape(stop - start, target.width, 3)
        )
        mask[start:stop] = (valid.reshape(stop - start, target.width) * 255).astype(
            np.uint8
        )
    return out, mask


def prepare_images(
    model: Model,
    source_root: Path,
    destination: Path,
    options: dict,
    mode: str,
    *,
    cached_directory: Path | None = None,
) -> tuple[Model, dict]:
    cameras, images, records = {}, {}, {}
    progress = {}
    cached = (
        read_json(cached_directory / "images_progress.json")
        if cached_directory and (cached_directory / "images_progress.json").is_file()
        else {}
    )
    for c in model.cameras.values():
        if c.model == "SIMPLE_RADIAL" and options["undistort"] == "never":
            raise ValueError(f"Camera {c.id} requires undistortion; policy=never")
        width = min(c.width, options["max_width"] or c.width)
        height = max(1, round(c.height * width / c.width))
        k = intrinsics(c).copy()
        k[0] *= width / c.width
        k[1] *= height / c.height
        cameras[c.id] = Camera(
            c.id,
            "PINHOLE",
            width,
            height,
            np.array([k[0, 0], k[1, 1], k[0, 2], k[1, 2]]),
        )
    names = set()
    for im in model.images.values():
        src = model.cameras[im.camera_id]
        target = cameras[im.camera_id]
        original = image_path(source_root, im.name)
        name = (
            str(Path(im.name).with_suffix("." + options["format"])).replace("\\", "/")
            if options["format"] != "preserve"
            else im.name
        )
        if name.casefold() in names:
            raise ValueError(
                f"Image format conversion causes duplicate filename: {name}"
            )
        names.add(name.casefold())
        dst = image_path(destination / "images", name)
        modified = (
            src.model == "SIMPLE_RADIAL"
            or (src.width, src.height) != (target.width, target.height)
            or options["format"] != "preserve"
        )
        mask_path = None
        previous = cached.get(str(im.id))
        if previous:
            if (
                previous["source_sha256"] != sha256(original)
                or previous["name"] != name
            ):
                raise ValueError(f"Partial cache source identity mismatch: {im.name}")
            cached_image = cached_directory / "images" / name
            if not cached_image.is_file() or sha256(cached_image) != previous["sha256"]:
                raise ValueError(f"Corrupt partial image cache: {cached_image}")
            transfer(cached_image, dst, "copy")
            mask_path = previous["mask_path"]
            if mask_path:
                cached_mask = cached_directory / mask_path
                if sha256(cached_mask) != previous["mask_sha256"]:
                    raise ValueError(f"Corrupt partial mask cache: {cached_mask}")
                transfer(cached_mask, destination / mask_path, "copy")
        elif not modified:
            transfer(original, dst, mode)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            with PILImage.open(original) as opened:
                rgb = opened.convert("RGB")
                if src.model == "SIMPLE_RADIAL":
                    pixels, mask = remap_radial(np.array(rgb), src, target)
                    rgb.close()
                    rgb = PILImage.fromarray(pixels)
                    if np.any(mask == 0):
                        mask_path = f"masks/{im.id}.png"
                        (destination / "masks").mkdir(parents=True, exist_ok=True)
                        PILImage.fromarray(mask).save(destination / mask_path)
                elif rgb.size != (target.width, target.height):
                    resized = rgb.resize(
                        (target.width, target.height), PILImage.Resampling.BILINEAR
                    )
                    rgb.close()
                    rgb = resized
                kwargs = (
                    {"quality": options["jpeg_quality"], "subsampling": 0}
                    if dst.suffix.lower() in (".jpg", ".jpeg")
                    else {}
                )
                rgb.save(dst, **kwargs)
                rgb.close()
        xy = undistort_xy(src, im.xys)
        xy *= np.array([target.width / src.width, target.height / src.height])
        images[im.id] = replace(im, name=name, xys=xy)
        records[im.id] = {
            "source_name": im.name,
            "name": name,
            "mask_path": mask_path,
            "transform": {
                "algorithm": "inverse_SIMPLE_RADIAL_bilinear_v1"
                if src.model == "SIMPLE_RADIAL"
                else "pinhole_resize_v1",
                "source_model": src.model,
                "source_params": src.params.tolist(),
                "source_size": [src.width, src.height],
                "target_size": [target.width, target.height],
                "pixel_centers": "COLMAP first pixel at (0.5,0.5); Pillow sampling index = coordinate - 0.5",
                "scale_xy": [target.width / src.width, target.height / src.height],
                "crop": None,
                "undistorted": True,
                "file_mode": "derived_copy" if modified else mode,
            },
        }
        progress[str(im.id)] = {
            "source_sha256": sha256(original),
            "name": name,
            "sha256": sha256(dst),
            "mask_path": mask_path,
            "mask_sha256": sha256(destination / mask_path) if mask_path else None,
        }
        write_json(destination / "images_progress.json", progress)
        if len(progress) % 16 == 0 or len(progress) == len(model.images):
            logging.getLogger(__name__).info(
                "Prepared images %d/%d", len(progress), len(model.images)
            )
    return Model(cameras, images, dict(model.points)), records
