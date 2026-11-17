"""Read-only prepared-scene adapter with native PGSR cameras and train-only neighbors."""
from collections import OrderedDict
from pathlib import Path
import math

import numpy as np

from src.common.io import read_json, write_json, sha256
from src.data.loader import verify_dataset
from src.data.colmap import read_model
from .config import ROOT


class SceneAdapter:
    def __init__(self, config: dict, run: Path):
        import torch
        from scene.cameras import Camera, process_image
        from utils.graphics_utils import BasicPointCloud

        self.config = config
        self.directory = ROOT / config["dataset"]
        readiness = verify_dataset(self.directory)
        write_json(run / "data_readiness.json", readiness)
        if readiness["status"] != "SUCCESS":
            raise ValueError(readiness)
        self.metadata = read_json(self.directory / "metadata.json")
        self.split = read_json(self.directory / "splits.json")
        self.ids = {k: list(v) for k, v in self.split["image_ids"].items()}
        if config["train_view_limit"]:
            # Explicit development subset, contiguous accepted IDs for real neighbors.
            self.ids["train"] = self.ids["train"][:config["train_view_limit"]]
        self.cache = OrderedDict()
        cache = self.cache
        capacity = config["cpu_camera_cache"]

        class CachedCamera(Camera):
            def get_image(cam):
                if cam.image_name not in cache:
                    rgb, gray, _ = process_image(cam.image_path, cam.resolution, cam.ncc_scale)
                    cache[cam.image_name] = (rgb, gray)
                    while len(cache) > capacity:
                        cache.popitem(last=False)
                cache.move_to_end(cam.image_name)
                rgb, gray = cache[cam.image_name]
                return rgb.to("cuda:0"), gray.to("cuda:0")

        records = {r["image_id"]: r for r in self.metadata["images"]}
        self.cameras = {}
        self.train = []
        # Train UIDs index only the train exposure table; held-out views have no exposure row.
        for split in ("train", "val", "test"):
            for uid, iid in enumerate(self.ids[split]):
                row = records[iid]
                k = np.asarray(row["K"])
                tcw = np.asarray(row["Tcw"])
                width, height = round(row["width"]/config["resolution"]), round(row["height"]/config["resolution"])
                if not np.allclose(k[:2, 2], [row["width"]/2, row["height"]/2]):
                    raise ValueError("Centered native PGSR camera adapter required")
                camera = CachedCamera(
                    colmap_id=row["camera_id"], R=tcw[:3, :3].T.copy(), T=tcw[:3, 3].copy(),
                    FoVx=2*math.atan(row["width"]/(2*k[0, 0])),
                    FoVy=2*math.atan(row["height"]/(2*k[1, 1])),
                    image_width=width, image_height=height,
                    image_path=str(self.directory / row["path"]),
                    image_name=Path(row["name"]).stem, uid=uid if split=="train" else -1,
                    ncc_scale=config["ncc_scale"], preload_img=False, data_device="cpu",
                )
                camera.image_id = iid
                if not np.allclose(camera.world_view_transform.T.cpu().numpy(), tcw, atol=1e-6):
                    raise ValueError("Native PGSR world-to-camera parity failed")
                if abs(camera.Fx - k[0, 0]*width/row["width"]) > 1e-5:
                    raise ValueError("Native PGSR resized focal parity failed")
                self.cameras[iid] = camera
                if split == "train":
                    self.train.append(camera)
        centers = torch.stack([c.camera_center for c in self.train])
        center = centers.mean(0)
        self.extent = float((centers-center).norm(dim=1).max())*1.1
        if not self.extent > 0:
            raise ValueError("Degenerate train camera extent")
        # Same distance-first / angle-second policy as pinned scene/__init__.py.
        rays = torch.stack([torch.tensor(c.R, dtype=torch.float32, device="cuda:0")[:, 2] for c in self.train])
        rays = torch.nn.functional.normalize(rays, dim=-1)
        distances = (centers[:, None]-centers[None]).norm(dim=-1).cpu().numpy()
        angles = (torch.acos((rays@rays.T).clamp(-1, 1))*180/3.14159).cpu().numpy()
        graph = []
        for index, camera in enumerate(self.train):
            order = np.lexsort((angles[index], distances[index]))
            mask = ((angles[index, order] < config["multi_view_max_angle"])
                    & (distances[index, order] > config["multi_view_min_dis"])
                    & (distances[index, order] < config["multi_view_max_dis"]))
            selected = order[mask][:config["multi_view_num"]].tolist()
            camera.nearest_id = selected
            camera.nearest_names = [self.train[i].image_name for i in selected]
            graph.append({"image_id": camera.image_id, "neighbors": [self.train[i].image_id for i in selected]})
        if not any(c.nearest_id for c in self.train):
            raise ValueError("No eligible real train neighbors; geometry smoke/training cannot proceed")
        write_json(run / "neighbors.json", {"policy": "native distance-first + angle gates; acos dot clipped for roundoff", "train_only": True, "graph": graph})
        model = read_model(self.directory / "sparse/0")
        points = sorted(model.points.values(), key=lambda p: p.id)
        indices = np.arange(len(points))
        if config["initial_point_limit"] and len(points) > config["initial_point_limit"]:
            indices = np.sort(np.random.default_rng(config["seed"]).choice(indices, config["initial_point_limit"], replace=False))
        xyz = np.array([points[i].xyz for i in indices])
        rgb = np.array([points[i].rgb for i in indices])/255.
        self.pcd = BasicPointCloud(xyz, rgb, np.zeros_like(xyz))
        np.save(run / "initial_point_ids.npy", np.array([points[i].id for i in indices]))
        self.identity = {"metadata": sha256(self.directory / "metadata.json"), "split": self.split["hash"]}
        write_json(run / "dataset_integration.json", {
            "status": "SUCCESS", "split_counts": {k: len(v) for k,v in self.ids.items()},
            "initial_points": len(xyz), "native_camera_projection": "VERIFIED",
            "train_extent": self.extent, "initialization": "Frozen train-only prepared SfM; native destructive reader bypassed",
            "source_modified": False, "identity": self.identity,
        })

    def camera(self, image_id: int):
        return self.cameras[image_id]
