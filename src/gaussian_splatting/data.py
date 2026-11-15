"""Real native SceneInfo parsing with frozen splits and lazy native cameras."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.common.config import ROOT
from src.common.io import read_json, sha256, write_json
from src.data.loader import verify_dataset
from src.data.colmap import read_model


class Dataset:
    def __init__(self, config: dict, report_dir: Path, *, verify: bool = True):
        self.config = config
        self.path = ROOT / config["dataset"]
        if verify:
            report = verify_dataset(self.path)
            write_json(report_dir / "data_readiness.json", report)
            if report["status"] != "SUCCESS":
                raise ValueError(f"Prepared dataset failed: {report['errors']}")
        self.metadata = read_json(self.path / "metadata.json")
        self.split = read_json(self.path / "splits.json")
        from scene.dataset_readers import readColmapSceneInfo, getNerfppNorm

        with (
            (report_dir / "native_loader.log").open("w", encoding="utf-8") as f,
            redirect_stdout(f),
        ):
            native = readColmapSceneInfo(str(self.path), None, False)
        infos = list(native.train_cameras) + list(native.test_cameras)
        by_name = {info.image_name: info for info in infos}
        self.records = {row["image_id"]: row for row in self.metadata["images"]}
        self.infos = {}
        if (
            len(by_name) != len(infos)
            or len(infos) != len(self.records)
            or native.point_cloud is None
        ):
            raise ValueError("Native SceneInfo is incomplete or has duplicated names")
        for iid, row in self.records.items():
            # The pinned 2023 reader uses basename.split(".")[0], unlike 2026 main.
            name = Path(row["name"]).name.split(".")[0]
            if name not in by_name or row["mask_path"] is not None:
                raise ValueError("Native name/mask adapter required")
            info = by_name[name]
            tcw = np.eye(4)
            tcw[:3, :3] = info.R.T
            tcw[:3, 3] = info.T
            k = np.array(row["K"])
            if (
                not np.allclose(tcw, row["Tcw"], atol=1e-9)
                or abs(k[0, 2] - row["width"] / 2) > 1e-6
                or abs(k[1, 2] - row["height"] / 2) > 1e-6
            ):
                raise ValueError(f"Native camera projection mismatch: {iid}")
            if info.image.size != (row["width"], row["height"]):
                raise ValueError("Native image dimensions disagree")
            self.infos[iid] = info
            info.image.close()
        self.ids = {k: list(v) for k, v in self.split["image_ids"].items()}
        if (
            config["train_view_limit"]
            and len(self.ids["train"]) > config["train_view_limit"]
        ):
            positions = np.linspace(
                0, len(self.ids["train"]) - 1, config["train_view_limit"], dtype=int
            )
            self.ids["train"] = [self.ids["train"][i] for i in positions]
        self.native_normalization = getNerfppNorm(
            [self.infos[i] for i in self.ids["train"]]
        )
        self.extent = float(self.native_normalization["radius"])
        if not np.isfinite(self.extent) or self.extent <= 0:
            raise ValueError("Degenerate native train-camera extent")
        # Native getNerfppNorm affects spatial LR/ADC extent only. No pose/XYZ transform.
        source_model = read_model(self.path / "sparse/0")
        point_ids = np.array(sorted(source_model.points))
        if len(point_ids) != len(native.point_cloud.points):
            raise ValueError("Native initial point count mismatch")
        source_xyz = np.array([source_model.points[int(i)].xyz for i in point_ids])
        source_rgb = (
            np.array([source_model.points[int(i)].rgb for i in point_ids]) / 255.0
        )
        if not np.allclose(
            native.point_cloud.points, source_xyz, atol=1e-5, rtol=1e-6
        ) or not np.allclose(native.point_cloud.colors, source_rgb, atol=1e-7):
            raise ValueError(
                "Native XYZ/colors disagree with train-only initialization"
            )
        selection = np.arange(len(point_ids))
        if (
            config["initial_point_limit"]
            and len(selection) > config["initial_point_limit"]
        ):
            selection = np.sort(
                np.random.default_rng(config["seed"]).choice(
                    selection, config["initial_point_limit"], replace=False
                )
            )
        from utils.graphics_utils import BasicPointCloud

        self.pcd = BasicPointCloud(
            native.point_cloud.points[selection],
            native.point_cloud.colors[selection],
            native.point_cloud.normals[selection],
        )
        self.point_ids = point_ids[selection]
        self.cache = OrderedDict()
        self.report = {
            "status": "SUCCESS",
            "native_scene_loader": "readColmapSceneInfo (actual, unmodified)",
            "native_revision": "414b553ef1d3032ea634b7b7a1d121173f36592c",
            "source_mode": self.metadata["source"]["mode"],
            "dataset_id": self.metadata["dataset_id"],
            "metadata_sha256": sha256(self.path / "metadata.json"),
            "split_hash": self.split["hash"],
            "split_counts": {k: len(v) for k, v in self.ids.items()},
            "source_split_counts": {
                k: len(v) for k, v in self.split["image_ids"].items()
            },
            "initial_points": len(self.pcd.points),
            "point_selection": "seeded source-point subset"
            if len(selection) != len(point_ids)
            else "all train-only initialization points",
            "native_train_extent": self.extent,
            "native_nerf_normalization": {
                k: v.tolist() if hasattr(v, "tolist") else v
                for k, v in self.native_normalization.items()
            },
            "project_normalization_applied": False,
            "pose_transform_applied": False,
            "protocol": "fixed_pose_transductive; source BA accessed held-out RGB",
            "camera_sampling": "native random-without-replacement stack over selected train IDs only",
        }
        write_json(report_dir / "dataset_integration.json", self.report)
        np.save(
            report_dir / "initial_point_ids.npy", self.point_ids, allow_pickle=False
        )

    def camera(self, iid: int):
        if iid in self.cache:
            self.cache.move_to_end(iid)
            return self.cache[iid]
        from utils.camera_utils import loadCam

        info = self.infos[iid]
        with Image.open(info.image_path) as image:
            camera = loadCam(
                SimpleNamespace(
                    resolution=self.config["resolution"], data_device="cpu"
                ),
                iid,
                info._replace(image=image),
                1.0,
            )
        row = self.records[iid]
        sx, sy = camera.image_width / row["width"], camera.image_height / row["height"]
        if not np.isclose(
            camera.FoVx, 2 * np.arctan(camera.image_width / (2 * row["K"][0][0] * sx))
        ) or not np.isclose(
            camera.FoVy, 2 * np.arctan(camera.image_height / (2 * row["K"][1][1] * sy))
        ):
            raise ValueError("Resize/intrinsics consistency failed")
        if (
            camera.original_image.device.type != "cpu"
            or camera.world_view_transform.device.type != "cuda"
        ):
            raise ValueError("Unexpected native camera devices")
        self.cache[iid] = camera
        while len(self.cache) > self.config["cpu_camera_cache"]:
            self.cache.popitem(last=False)
        return camera
