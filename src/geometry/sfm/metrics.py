"""Recompute observation residuals and check real reconstruction topology."""
from __future__ import annotations

import numpy as np

from src.geometry.sfm.model_io import project


def model_metrics(model: dict, input_frames: int) -> dict:
    images, points = model["images"], model["points"]
    if len(images) < 2 or not points or input_frames < len(images):
        raise ValueError("Need >=2 registered images, nonempty points, valid input count")
    centers = np.asarray([image["center"] for image in images.values()])
    if not np.isfinite(centers).all() or np.max(np.linalg.norm(centers - centers[0], axis=1)) < 1e-8:
        raise ValueError("Degenerate camera baseline")
    residuals, track_lengths = [], []
    associations = set()
    for point_id, point in points.items():
        track = point["track"]
        if len(track) < 2 or len({image_id for image_id, _ in track}) != len(track):
            raise ValueError("Point must have >=2 distinct camera observations")
        track_lengths.append(len(track))
        for image_id, index in track:
            image = images[image_id]
            if (image_id, index) in associations or not 0 <= index < len(image["observations"]):
                raise ValueError("Invalid or reused observation index")
            associations.add((image_id, index))
            observation = image["observations"][index]
            if observation[2] != point_id:
                raise ValueError("Nonreciprocal 2D/3D track")
            matrix = np.asarray(image["world_to_camera"])
            xyz = matrix[:3, :3] @ point["xyz"] + matrix[:3, 3]
            if not np.isfinite(xyz).all() or xyz[2] <= 0:
                raise ValueError("Nonfinite point or negative depth")
            projected = project(model["cameras"][image["camera_id"]], xyz[None])[0]
            error = float(np.linalg.norm(projected - observation[:2]))
            if not np.isfinite(error):
                raise ValueError("Nonfinite reprojection error")
            residuals.append(error)
    for image_id, image in images.items():
        for index, observation in enumerate(image["observations"]):
            if observation[2] != -1 and (image_id, index) not in associations:
                raise ValueError("Image references untracked point")
    return {"input_frames": input_frames, "registered_images": len(images),
            "registration_ratio": len(images)/input_frames, "points3d_count": len(points),
            "mean_track_length": float(np.mean(track_lengths)), "median_track_length": float(np.median(track_lengths)),
            "mean_reprojection_error_px": float(np.mean(residuals)), "median_reprojection_error_px": float(np.median(residuals)),
            "p95_reprojection_error_px": float(np.percentile(residuals, 95)), "observation_count": len(residuals),
            "reprojection_metric_definition": "Euclidean pixel residual per tracked observation; fitted distortion included",
            "positive_depth_fraction": 1.0}
