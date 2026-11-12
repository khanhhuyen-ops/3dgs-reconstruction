"""Known geometry fixtures exercise conventions, residuals and rejection gates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.geometry.sfm.gpu_probe import write_ba_fixture
from src.geometry.sfm.metrics import model_metrics
from src.geometry.sfm.model_io import export_poses, quaternion_rotation, read_model, validate_ply


def test_geometry_residuals_and_camera_centers(tmp_path: Path) -> None:
    write_ba_fixture(tmp_path / "fixture")
    model = read_model(tmp_path / "fixture")
    measured = model_metrics(model, 4)
    assert measured["registered_images"] == 3
    assert measured["registration_ratio"] == 0.75
    assert measured["points3d_count"] == 48
    assert measured["mean_track_length"] == 3
    assert 0.1 < measured["mean_reprojection_error_px"] < 10
    np.testing.assert_allclose(model["images"][1]["center"], [-0.6, 0, 0])
    export_poses(model, tmp_path / "poses.json")
    assert (tmp_path / "poses.json").stat().st_size > 0
    model["points"][1]["xyz"][2] = -1
    with pytest.raises(ValueError, match="negative depth"):
        model_metrics(model, 4)


def test_broken_tracks_and_quaternion_rejected(tmp_path: Path) -> None:
    write_ba_fixture(tmp_path / "fixture")
    model = read_model(tmp_path / "fixture")
    model["points"][1]["track"][0] = (1, 99999)
    with pytest.raises(ValueError, match="observation"):
        model_metrics(model, 3)
    with pytest.raises(ValueError):
        quaternion_rotation([0, 0, 0, 0])


def test_ply_truncation_and_nan_rejected(tmp_path: Path) -> None:
    ply = tmp_path / "test.ply"
    header = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    ply.write_text(header + "1 2 3\n")
    validate_ply(ply, 1)
    for bad in ("nan 2 3\n", ""):
        ply.write_text(header + bad)
        with pytest.raises(ValueError):
            validate_ply(ply, 1)

