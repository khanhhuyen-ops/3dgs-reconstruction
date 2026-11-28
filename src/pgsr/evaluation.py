"""PGSR inference reload, frozen held-out RGB metrics and geometry export contract."""
from pathlib import Path
from types import SimpleNamespace
import json
import time

import numpy as np

from src.common.io import read_json, write_json, sha256
from .config import ROOT


def image(path, value):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(np.clip(value, 0, 1)*255).astype(np.uint8)).save(path)


def evaluate(config, run: Path, environment: dict):
    import torch
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from .data import SceneAdapter
    from .geometry import valid_geometry
    from .validation import validate_model
    from src.evaluation.rendering_metrics import Metrics

    output = run / "evaluation"
    output.mkdir(exist_ok=False)
    data = SceneAdapter(config, output)
    result = read_json(run / "training_result.json")
    identity = read_json(run / "identity.json")
    if any(identity[k] != v for k,v in data.identity.items()):
        raise ValueError("Evaluation dataset differs from the frozen training input")
    export = result["export"]
    # Relative-to-project export path survives moving the complete run across hosts.
    ply = run / "point_cloud" / f"iteration_{result['iterations']:06d}" / "point_cloud.ply"
    if sha256(ply) != export["sha256"]:
        raise ValueError("PGSR PLY checksum differs from the training export")
    model = GaussianModel(config["sh_degree"])
    model.load_ply(str(ply))
    model.active_sh_degree = export["active_sh_degree"]
    model.use_app = False
    model_report = validate_model(model)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.full((3,), float(config["white_background"]), device="cuda:0")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        camera = data.camera(export["reference_image_id"])
        package = render(camera, model, pipe, bg, app_model=None)
        errors = {}
        with np.load(ply.parent / "reload_reference.npz") as reference:
            for key,native_key in (("rgb","render"),("depth","plane_depth"),("normal","rendered_normal")):
                current = package[native_key].cpu().numpy()
                if current.shape != reference[key].shape or not np.isfinite(current).all():
                    raise ValueError("Invalid reloaded PGSR render")
                errors[key] = float(np.max(np.abs(current-reference[key])))
                if errors[key] > 1e-5:
                    raise ValueError(f"PGSR PLY reload parity failed: {key} {errors[key]}")
        write_json(output / "reload_validation.json", {"status":"SUCCESS", "max_abs_errors":errors,
                   "model":model_report,"reference_image_id":camera.image_id,
                   "environment":environment,"exposure_disabled":True})
        del package
        metrics = Metrics()
        if metrics.lpips is None:
            raise RuntimeError("LPIPS unavailable: "+str(metrics.lpips_error))
        rows = []
        for iid in data.ids["test"]:
            camera = data.camera(iid)
            target, _ = camera.get_image()
            prediction = render(camera, model, pipe, bg, app_model=None,
                                return_plane=False, return_depth_normal=False)["render"].clamp(0,1)
            values = metrics(prediction, target)
            rows.append({"image_id":iid, **values})
            image(output / "test_rgb" / f"{iid:06d}.png", prediction.permute(1,2,0).cpu().numpy())
            del prediction, target
        expected_ids = data.split["image_ids"]["test"]
        if [row["image_id"] for row in rows] != expected_ids or any(
            row[key] is None or not np.isfinite(row[key])
            for row in rows
            for key in ("psnr", "ssim", "lpips")
        ):
            raise ValueError("Incomplete/nonfinite frozen held-out evaluation")
        report = {"status":"SUCCESS", "method":"B2_PGSR", "iterations":result["iterations"],
                  "count":len(rows),"split":"test","resolution_divisor":config["resolution"],
                  "mean":{k:float(np.mean([r[k] for r in rows])) for k in ("psnr","ssim","lpips")},
                  "views":rows,"weight_hashes":metrics.weight_hashes,"test_exposure_fit":False,
                  "fair_full_resolution_comparison":config["resolution"]==1 and config["train_view_limit"] is None,
                  "note":"Development subsets are diagnostics, not full baseline results",
                  "dataset_identity":data.identity}
        write_json(output / "rendering_metrics.json", report)
        del metrics
        torch.cuda.empty_cache()
        geometry = []
        folder = output / "geometry"
        folder.mkdir()
        for index,camera in enumerate(data.train):
            package = render(camera, model, pipe, bg, app_model=None)
            valid = valid_geometry(package,camera,config["depth_epsilon"])
            normal = package["rendered_normal"]
            unit = torch.nn.functional.normalize(normal,dim=0,eps=config["normal_epsilon"])
            depth = torch.where(valid,package["plane_depth"].squeeze(),0)
            unit = torch.where(valid[None],unit,0)
            d = depth.cpu().numpy().astype(np.float32)
            n = unit.permute(1,2,0).cpu().numpy().astype(np.float32)
            mask = valid.cpu().numpy()
            path = folder / f"{camera.image_id:06d}.npz"
            np.savez_compressed(path,depth=d,normal=n,valid=mask,
                                K=camera.get_k().cpu().numpy(),Tcw=camera.world_view_transform.T.cpu().numpy(),
                                image_id=camera.image_id)
            values = d[mask]
            normal_lengths = np.linalg.norm(n[mask],axis=-1)
            row = {"image_id":camera.image_id,"path":path.relative_to(output).as_posix(),"sha256":sha256(path),
                   "valid_pixels":int(mask.sum()),"total_pixels":mask.size,"valid_ratio":float(mask.mean()),
                   "depth_quantiles":np.quantile(values,[0,.01,.5,.99,1]).tolist() if len(values) else None,
                   "unit_normal_max_error":float(np.max(np.abs(normal_lengths-1))) if len(values) else None}
            geometry.append(row)
            if index < 3:
                lo,hi = np.quantile(values,[.01,.99]) if len(values) else (0,1)
                preview = np.where(mask,np.clip((d-lo)/max(hi-lo,1e-8),0,1),0)
                image(output/"geometry_preview"/f"{camera.image_id:06d}_depth.png",preview)
                image(output/"geometry_preview"/f"{camera.image_id:06d}_normal.png",np.where(mask[...,None],n*.5+.5,0))
            del package, normal, unit, depth, valid
        contract = {"schema":"pgsr-depth-normal-v1","status":"SUCCESS", "views":geometry,
                    "count":len(geometry),"split":"train","depth":"positive axial camera Z, arbitrary SfM units",
                    "normal":"unit camera-frame, camera-facing, HWC xyz; invalid entries zero",
                    "valid":"finite positive depth, finite nonzero accumulated normal, nonparallel plane ray; not a calibrated confidence or foreground mask",
                    "raw_plane":"native positive distance delta and unnormalized camera-facing N satisfy N^T x + delta = 0",
                    "camera":"Tcw world-to-camera, x-right y-down z-forward, K at exported resolution",
                    "world_normalization":"identity; original prepared scene coordinates",
                    "mesh_evaluation":"NOT_RUN_PHASE04","dataset_identity":data.identity,
                    "mean_valid_ratio":float(np.mean([r["valid_ratio"] for r in geometry])),
                    "peak_torch_reserved_bytes":torch.cuda.max_memory_reserved(),"seconds":time.perf_counter()-started}
        if not geometry or not all(r["valid_pixels"]>0 and r["unit_normal_max_error"]<1e-5 for r in geometry):
            raise ValueError("Invalid/empty exported PGSR geometry")
        write_json(output/"geometry_contract.json",contract)
    write_json(output/"result.json",{"status":"SUCCESS","rendering":report["mean"],"geometry_views":len(geometry),
               "ply_reload_verified":True,"training_status":result["status"],"environment":environment})
    return {"status":"SUCCESS","run":str(run),"rendering":report["mean"],"geometry_views":len(geometry)}
