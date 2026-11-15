"""Lazy CPU dataset loading and a strict, machine-readable artifact gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from src.common.io import fingerprint, read_json, sha256
from src.data.colmap import read_model, semantic_hashes
from src.data.geometry import intrinsics, pose
from src.data.validation import image_path, validate_model
from src.geometry.sfm.model_io import validate_ply


class PreparedDataset:
    """Load only the requested split, one RGB/mask at a time; no GPU allocation."""

    def __init__(self, directory: Path, split: str = "train", *, verify: bool = True):
        self.directory = directory
        if verify:
            report = verify_dataset(directory)
            if report["status"] != "SUCCESS":
                raise ValueError(f"Invalid prepared dataset: {report['errors']}")
        self.metadata = read_json(directory / "metadata.json")
        manifest = read_json(directory / "splits.json")
        if split not in ("train", "val", "test"):
            raise ValueError("Explicit split must be train/val/test")
        by_id = {row["image_id"]: row for row in self.metadata["images"]}
        self.records = [by_id[iid] for iid in manifest["image_ids"][split]]
        self.split = split

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        row = self.records[index]
        with PILImage.open(self.directory / row["path"]) as image:
            rgb = (
                np.array(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
                / 255
            )
        mask = np.ones((1, row["height"], row["width"]), dtype=bool)
        if row["mask_path"]:
            with PILImage.open(self.directory / row["mask_path"]) as image:
                mask = np.array(image, dtype=np.uint8)[None] > 0
        return {
            "image_id": row["image_id"],
            "camera_id": row["camera_id"],
            "rgb": rgb,
            "mask": mask,
            "K": np.array(row["K"], dtype=np.float64),
            "Tcw": np.array(row["Tcw"], dtype=np.float64),
            "Twc": np.array(row["Twc"], dtype=np.float64),
            "normalization": self.metadata["normalization"],
        }

    def initialization_points(self) -> dict:
        model = read_model(self.directory / "sparse/0")
        points = sorted(model.points.values(), key=lambda p: p.id)
        return {
            "ids": np.array([p.id for p in points], dtype=np.int64),
            "xyz": np.array([p.xyz for p in points], dtype=np.float64),
            "rgb": np.array([p.rgb for p in points], dtype=np.float32) / 255,
        }


def verify_dataset(directory: Path) -> dict:
    """Verify bytes, cameras, split coverage, train-only points and observation data."""
    errors = []
    report = {
        "status": "FAILED",
        "errors": errors,
        "native_scene_loaders": "NOT_TESTED",
        "training_started": False,
        "gpu_compute": "NOT_RUN (CPU data preparation)",
        "phase02_training_authorized": False,
    }
    try:
        meta = read_json(directory / "metadata.json")
        if (
            meta["schema_version"] != "prepared-v1"
            or meta["geometry_transform_applied"] is not False
            or meta["source"]["ground_truth_used"] is not False
        ):
            raise ValueError("Invalid schema/transform/GT contract")
        hashes = read_json(directory / "integrity.json")
        required = {
            "metadata.json",
            "splits.json",
            "normalization.json",
            "train_observations.npz",
            "qualification.json",
            "sparse/0/cameras.bin",
            "sparse/0/images.bin",
            "sparse/0/points3D.bin",
            "sparse/0/points3D.ply",
        }
        if not required <= set(hashes):
            raise ValueError("Integrity manifest omits required artifacts")
        for relative, expected in hashes.items():
            # Root-relative containment permits declared image symlinks only.
            path = directory / relative
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError(f"Unsafe artifact path: {relative}")
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Artifact missing or hash mismatch: {relative}")
        split = read_json(directory / "splits.json")
        split_body = {k: v for k, v in split.items() if k != "hash"}
        if (
            fingerprint(split_body) != split["hash"]
            or split["hash"] != meta["split_hash"]
        ):
            raise ValueError("Split fingerprint mismatch")
        sets = [set(split["image_ids"][k]) for k in ("train", "val", "test")]
        ids = {row["image_id"] for row in meta["images"]}
        if (
            any(not s for s in sets)
            or any(
                len(s) != len(split["image_ids"][k])
                for s, k in zip(sets, ("train", "val", "test"))
            )
            or any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2)))
            or set.union(*sets) != ids
        ):
            raise ValueError("Split is overlapping, incomplete, empty or duplicated")
        model = read_model(directory / "sparse/0")
        if set(model.images) != ids or len(meta["images"]) != len(ids):
            raise ValueError("Manifest/model image IDs mismatch")
        for row in meta["images"]:
            im = model.images[row["image_id"]]
            c = model.cameras[im.camera_id]
            if (
                row["path"] != "images/" + im.name
                or row["camera_id"] != im.camera_id
                or (row["width"], row["height"]) != (c.width, c.height)
            ):
                raise ValueError(f"Manifest image/camera mismatch: {im.id}")
            if sha256(image_path(directory / "images", im.name)) != row["sha256"]:
                raise ValueError(f"Image hash mismatch: {im.name}")
            if (
                not np.allclose(pose(im.qvec, im.tvec), row["Tcw"], atol=1e-10, rtol=0)
                or not np.allclose(intrinsics(c), row["K"], atol=1e-10, rtol=0)
                or not np.allclose(
                    np.array(row["Tcw"]) @ np.array(row["Twc"]),
                    np.eye(4),
                    atol=1e-10,
                    rtol=0,
                )
            ):
                raise ValueError(f"Camera geometry mismatch: image {im.id}")
        quality = validate_model(
            model,
            directory / "images",
            {**meta["config"]["validation"], "min_observations_per_image": 0},
        )
        if quality["status"] != "SUCCESS":
            raise ValueError(
                f"Prepared model quality errors: {quality['error_counts']}"
            )
        if semantic_hashes(read_model(directory / "sparse")) != semantic_hashes(model):
            raise ValueError("PGSR sparse/ mirror disagrees with sparse/0/")
        track_rows = set()
        for p in model.points.values():
            if (
                not set(p.track[:, 0]) <= sets[0]
                or len(p.track) < meta["config"]["training_points"]["min_train_views"]
            ):
                raise ValueError(f"Non-train or insufficient point support: {p.id}")
            track_rows.update((int(iid), int(idx), p.id) for iid, idx in p.track)
        with np.load(
            directory / "train_observations.npz", allow_pickle=False
        ) as observations:
            n = len(observations["image_id"])
            if any(
                len(observations[k]) != n
                for k in ("point2D_idx", "point3D_id", "xy", "z", "residual_px")
            ) or observations["xy"].shape != (n, 2):
                raise ValueError("Observation table shapes disagree")
            actual = set(
                zip(
                    observations["image_id"].tolist(),
                    observations["point2D_idx"].tolist(),
                    observations["point3D_id"].tolist(),
                )
            )
            if (
                actual != track_rows
                or len(actual) != n
                or not np.isfinite(observations["xy"]).all()
                or not np.isfinite(observations["z"]).all()
                or np.any(observations["z"] <= 0)
                or not np.isfinite(observations["residual_px"]).all()
            ):
                raise ValueError(
                    "Train observation table does not match reciprocal model tracks"
                )
        normal = read_json(directory / "normalization.json")
        if (
            normal != meta["normalization"]
            or normal["scale"] <= 0
            or not np.allclose(
                np.array(normal["transform"]) @ np.array(normal["inverse"]),
                np.eye(4),
                atol=1e-10,
            )
        ):
            raise ValueError("Normalization/inverse mismatch")
        validate_ply(directory / "sparse/0/points3D.ply", len(model.points))
        samples = {}
        for subset in ("train", "val", "test"):
            sample = PreparedDataset(directory, subset, verify=False)[0]
            if (
                sample["rgb"].dtype != np.float32
                or sample["rgb"].shape[0] != 3
                or not np.isfinite(sample["rgb"]).all()
                or sample["rgb"].min() < 0
                or sample["rgb"].max() > 1
            ):
                raise ValueError(f"CPU loader RGB contract failed: {subset}")
            samples[subset] = {
                "image_id": sample["image_id"],
                "rgb_shape": list(sample["rgb"].shape),
                "dtype": "float32",
                "device": "CPU",
            }
        report.update(
            status="SUCCESS",
            dataset_id=meta["dataset_id"],
            split_counts={k: len(v) for k, v in split["image_ids"].items()},
            points=len(model.points),
            observations=len(track_rows),
            source_integrity_verified_at_preparation=meta["source_integrity_verified"],
            protocol=split["policy"]["protocol"],
            convention=meta["pose_convention"],
            canonical_geometry_preserved=True,
            project_cpu_loader="VERIFIED",
            project_loader_samples=samples,
            readiness_scope="Prepared artifact integrity/geometry only; native scene loaders reported separately",
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
    return report


def apply_native_split(scene_info, directory: Path, method: str) -> dict:
    """Apply frozen three-way splits to an actual native SceneInfo before training.

    Call the native reader with eval=False, then this function. It verifies the
    complete camera set and point cloud; recompute native train-only normalization
    after selection in Phase02. No training state is constructed here.
    """
    meta = read_json(directory / "metadata.json")
    split = read_json(directory / "splits.json")
    cameras = list(scene_info.train_cameras) + list(scene_info.test_cameras)
    if method not in ("3dgs", "2dgs", "pgsr"):
        raise ValueError("Unknown method")
    by_name = {camera.image_name: camera for camera in cameras}
    expected = {
        row["image_id"]: row["name"]
        if method == "3dgs"
        else Path(row["name"]).name.split(".")[0]
        for row in meta["images"]
    }
    if (
        len(by_name) != len(cameras)
        or len(set(expected.values())) != len(expected)
        or set(by_name) != set(expected.values())
    ):
        raise ValueError("Native camera names/coverage disagree with manifest")
    init = PreparedDataset(directory, verify=False).initialization_points()
    # Native readers may enumerate points differently: compare sorted XYZ/RGB rows.
    actual = np.column_stack(
        (
            np.asarray(scene_info.point_cloud.points),
            np.asarray(scene_info.point_cloud.colors),
        )
    )
    target = np.column_stack((init["xyz"], init["rgb"]))

    def key(a: np.ndarray) -> np.ndarray:
        return np.lexsort(tuple(a[:, i] for i in reversed(range(a.shape[1]))))

    if actual.shape != target.shape or not np.allclose(
        actual[key(actual)], target[key(target)], atol=1e-5, rtol=1e-6
    ):
        raise ValueError(
            "Native loader point cloud differs from train-only initialization"
        )
    return {
        name: [by_name[expected[iid]] for iid in ids]
        for name, ids in split["image_ids"].items()
    }
