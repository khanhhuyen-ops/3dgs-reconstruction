"""Float64 camera diagnostics and reversible downstream similarity transforms."""

from __future__ import annotations

import numpy as np

from src.data.colmap import Camera
from src.geometry.sfm.model_io import quaternion_rotation

SUPPORTED = {"SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL"}


def pose(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_rotation(qvec.tolist())
    matrix[:3, 3] = tvec
    return matrix


def intrinsics(camera: Camera) -> np.ndarray:
    if camera.model not in SUPPORTED:
        raise ValueError(
            f"Projection unsupported for camera {camera.id}: {camera.model}"
        )
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
    else:
        fx, cx, cy = camera.params[:3]
        fy = fx
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def project(camera: Camera, camera_points: np.ndarray) -> np.ndarray:
    xy = np.asarray(camera_points, dtype=np.float64)[:, :2] / camera_points[:, 2:3]
    if camera.model == "SIMPLE_RADIAL":
        xy = xy * (1 + camera.params[3] * np.sum(xy * xy, axis=1, keepdims=True))
    k = intrinsics(camera)
    return xy * np.diag(k)[:2] + k[:2, 2]


def undistort_xy(camera: Camera, xys: np.ndarray) -> np.ndarray:
    """Invert SIMPLE_RADIAL on its monotonic branch; preserve COLMAP centers."""
    if camera.model != "SIMPLE_RADIAL":
        return xys.copy()
    k = intrinsics(camera)
    xy = (xys - k[:2, 2]) / np.diag(k)[:2]
    rd = np.linalg.norm(xy, axis=1)
    radius = rd.copy()
    distortion = camera.params[3]
    for _ in range(40):
        derivative = 1 + 3 * distortion * radius**2
        if np.any(derivative <= 0):
            raise ValueError("SIMPLE_RADIAL inverse outside its monotonic domain")
        step = (radius * (1 + distortion * radius**2) - rd) / derivative
        radius -= step
        if np.max(np.abs(step), initial=0) < 1e-13:
            break
    if np.max(np.abs(radius * (1 + distortion * radius**2) - rd), initial=0) > 1e-10:
        raise ValueError("SIMPLE_RADIAL inverse failed to converge")
    ratio = np.divide(radius, rd, out=np.ones_like(rd), where=rd > 0)
    return xy * ratio[:, None] * np.diag(k)[:2] + k[:2, 2]


def camera_centers(model) -> dict[int, np.ndarray]:
    return {
        im.id: np.linalg.inv(pose(im.qvec, im.tvec))[:3, 3]
        for im in model.images.values()
    }


def normalization(centers: np.ndarray, enabled: bool = True) -> dict:
    center = np.median(centers, axis=0) if enabled else np.zeros(3)
    radius = (
        float(np.percentile(np.linalg.norm(centers - center, axis=1), 90))
        if enabled
        else 1.0
    )
    if radius <= 1e-12 or not np.isfinite(radius):
        raise ValueError("Degenerate train-camera radius; cannot normalize scene")
    scale = 1 / radius
    transform = np.eye(4)
    transform[:3, :3] *= scale
    transform[:3, 3] = -scale * center
    return {
        "applied_to_canonical_model": False,
        "center": center.tolist(),
        "scale": scale,
        "transform": transform.tolist(),
        "inverse": np.linalg.inv(transform).tolist(),
        "fit": "median/percentile90 train camera centers" if enabled else "identity",
        "depth": "axial z in arbitrary SfM units; normalized z = scale * original z",
    }


def normalize_geometry(
    xyz: np.ndarray, tcw: np.ndarray, normal: dict
) -> tuple[np.ndarray, np.ndarray]:
    c, s = np.array(normal["center"]), normal["scale"]
    out = tcw.copy()
    out[:3, 3] = s * (tcw[:3, 3] + tcw[:3, :3] @ c)
    return s * (xyz - c), out
