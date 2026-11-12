"""Read COLMAP TXT poses/observations without assuming a model directory ID."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from src.common.io import write_json


def quaternion_rotation(q: list[float]) -> np.ndarray:
    values = np.asarray(q, dtype=np.float64)
    if not np.isfinite(values).all() or abs(np.linalg.norm(values) - 1) > 1e-3:
        raise ValueError("Invalid unit quaternion")
    w, x, y, z = values / np.linalg.norm(values)
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                     [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])


def read_model(directory: Path) -> dict:
    cameras, images, points = {}, {}, {}
    for line in (directory / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tokens = line.split()
        camera_id, model, width, height = int(tokens[0]), tokens[1], int(tokens[2]), int(tokens[3])
        params = [float(x) for x in tokens[4:]]
        counts = {"SIMPLE_RADIAL": 4, "PINHOLE": 4, "SIMPLE_PINHOLE": 3}
        if model not in counts or len(params) != counts[model] or not np.isfinite(params).all() or min(width, height) <= 0 or params[0] <= 0:
            raise ValueError(f"Unsupported/invalid camera: {line}")
        if model == "PINHOLE" and params[1] <= 0:
            raise ValueError("Invalid fy")
        if camera_id in cameras:
            raise ValueError("Duplicate camera ID")
        cameras[camera_id] = {"id": camera_id, "model": model, "width": width, "height": height, "params": params}
    lines = iter((directory / "images.txt").read_text(encoding="utf-8").splitlines())
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        tokens = line.split(maxsplit=9)
        image_id, q, t, camera_id = int(tokens[0]), [float(v) for v in tokens[1:5]], [float(v) for v in tokens[5:8]], int(tokens[8])
        observations = next(lines).split()
        if len(observations) % 3 or not np.isfinite(t).all() or image_id in images or camera_id not in cameras:
            raise ValueError("Invalid image/observation row")
        rotation = quaternion_rotation(q)
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, t
        inverse = np.linalg.inv(transform)
        images[image_id] = {"id": image_id, "name": tokens[9], "camera_id": camera_id, "qvec_wxyz": q, "tvec": t,
                            "world_to_camera": transform.tolist(), "camera_to_world": inverse.tolist(),
                            "center": inverse[:3, 3].tolist(),
                            "observations": [[float(observations[i]), float(observations[i+1]), int(observations[i+2])] for i in range(0, len(observations), 3)]}
    for line in (directory / "points3D.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tokens = line.split()
        point_id = int(tokens[0])
        xyz, rgb, error = [float(v) for v in tokens[1:4]], [int(v) for v in tokens[4:7]], float(tokens[7])
        track = [int(v) for v in tokens[8:]]
        if not np.isfinite([*xyz, error]).all() or error < 0 or len(track) % 2 or point_id in points:
            raise ValueError("Invalid point3D")
        points[point_id] = {"id": point_id, "xyz": xyz, "rgb": rgb, "reported_error_px": error,
                            "track": list(zip(track[::2], track[1::2]))}
    return {"cameras": cameras, "images": images, "points": points}


def project(camera: dict, camera_points: np.ndarray) -> np.ndarray:
    """CPU diagnostics only; this is not a custom SfM solver."""
    normalized = camera_points[:, :2] / camera_points[:, 2:3]
    params = camera["params"]
    if camera["model"] == "PINHOLE":
        return normalized * params[:2] + params[2:4]
    f, cx, cy = params[:3]
    if camera["model"] == "SIMPLE_RADIAL":
        normalized = normalized * (1 + params[3] * np.sum(normalized**2, axis=1, keepdims=True))
    return normalized * f + [cx, cy]


def export_poses(model: dict, output: Path) -> None:
    write_json(output, {"pose_convention": "COLMAP world-to-camera, right handed: x right, y down, z forward",
                        "pixel_convention": "COLMAP pixel centers; preserve fitted distortion parameters",
                        "scale": "monocular arbitrary global similarity; not metric meters; no GT alignment",
                        "depth_units": "arbitrary reconstruction units", "alignment": None,
                        "cameras": list(model["cameras"].values()),
                        "images": [{k: v for k, v in image.items() if k != "observations"} for image in model["images"].values()]})


def validate_ply(path: Path, expected_vertices: int) -> None:
    """Parse ASCII or binary little-endian COLMAP vertex properties and XYZ."""
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Not PLY")
        header = []
        for _ in range(100):
            line = stream.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        else:
            raise ValueError("Invalid PLY header")
        if f"element vertex {expected_vertices}" not in header or expected_vertices <= 0:
            raise ValueError("PLY vertex count disagrees with model")
        properties = [line.split()[1:] for line in header if line.startswith("property ")]
        names = [p[1] for p in properties]
        if not all(x in names for x in ("x", "y", "z")):
            raise ValueError("PLY missing XYZ")
        types = {"float": "f", "float32": "f", "double": "d", "uchar": "B", "uint8": "B"}
        binary = "format binary_little_endian 1.0" in header
        if not binary and "format ascii 1.0" not in header:
            raise ValueError("Unsupported PLY encoding")
        codec = struct.Struct("<" + "".join(types[p[0]] for p in properties)) if binary else None
        for _ in range(expected_vertices):
            values = codec.unpack(stream.read(codec.size)) if codec else [float(v) for v in stream.readline().split()]
            if len(values) != len(names) or not np.isfinite([values[names.index(x)] for x in ("x", "y", "z")]).all():
                raise ValueError("Invalid/truncated PLY vertex")

