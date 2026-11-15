"""Frozen-checkpoint held-out rendering, PNG bundles and per-view RGB metrics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.common.config import ROOT
from src.common.io import read_json, sha256, write_json
from src.gaussian_splatting.data import Dataset
from src.gaussian_splatting.validation import validate_model
from src.evaluation.rendering_metrics import Metrics


def evaluate(
    config: dict,
    run: Path,
    environment: dict,
    *,
    subset: str = "test",
    learned: bool = True,
) -> dict:
    import torch
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from src.gaussian_splatting.runner import memory

    trained = read_json(run / "training_result.json")
    if trained["status"] != "SUCCESS" or trained["iterations"] != config["iterations"]:
        raise ValueError(
            "Final held-out evaluation requires completed, frozen training configuration"
        )
    ply = ROOT / trained["export"]["path"]
    if sha256(ply) != trained["export"]["sha256"]:
        raise ValueError("Frozen trained PLY has changed")
    data = Dataset(config, run)
    model = GaussianModel(config["sh_degree"])
    model.load_ply(str(ply))
    model.active_sh_degree = trained["export"]["active_sh_degree"]
    validate_model(model)
    pipe = SimpleNamespace(
        convert_SHs_python=False, compute_cov3D_python=False, debug=False
    )
    bg = torch.tensor(
        [1.0, 1.0, 1.0] if config["white_background"] else [0.0, 0.0, 0.0],
        device="cuda",
    )
    folder = run / "renders" / subset / f"iteration_{trained['iterations']:06d}"
    bundle_path = folder / "manifest.json"
    if bundle_path.exists():
        bundle = read_json(bundle_path)
        if (
            bundle["ply_sha256"] != sha256(ply)
            or [row["image_id"] for row in bundle["images"]] != data.ids[subset]
        ):
            raise ValueError("Cached render bundle identity mismatch")
        for row in bundle["images"]:
            for field in ("prediction", "reference", "comparison"):
                if sha256(folder / row[field]) != row[field + "_sha256"]:
                    raise ValueError(f"Corrupt render bundle image {row['image_id']}")
    else:
        if folder.exists():
            raise FileExistsError(f"Incomplete existing render bundle: {folder}")
        folder.mkdir(parents=True)
        rows = []
        with torch.no_grad():
            for iid in data.ids[subset]:
                camera = data.camera(iid)
                prediction = render(camera, model, pipe, bg)["render"].clamp(0, 1)
                if (
                    not torch.isfinite(prediction).all()
                    or prediction.shape != camera.original_image.shape
                ):
                    raise ValueError(f"Invalid RGB render: image {iid}")
                pred = np.rint(prediction.permute(1, 2, 0).cpu().numpy() * 255).astype(
                    np.uint8
                )
                reference = np.rint(
                    camera.original_image.permute(1, 2, 0).numpy() * 255
                ).astype(np.uint8)
                names = {
                    "prediction": f"{iid:06d}_render.png",
                    "reference": f"{iid:06d}_reference.png",
                    "comparison": f"{iid:06d}_comparison.png",
                }
                Image.fromarray(pred).save(folder / names["prediction"])
                Image.fromarray(reference).save(folder / names["reference"])
                Image.fromarray(np.concatenate([reference, pred], axis=1)).save(
                    folder / names["comparison"]
                )
                rows.append(
                    {
                        "image_id": iid,
                        "source_name": data.records[iid]["name"],
                        "width": camera.image_width,
                        "height": camera.image_height,
                        **names,
                        **{k + "_sha256": sha256(folder / v) for k, v in names.items()},
                    }
                )
        bundle = {
            "status": "SUCCESS",
            "split": subset,
            "images": rows,
            "ply_sha256": sha256(ply),
            "split_hash": data.split["hash"],
            "resolution_divisor": config["resolution"],
            "color_space": "sRGB [0,1]; rounded 8-bit PNG for reproducible metrics",
            "exposure_fitting": False,
            "background": "white" if config["white_background"] else "black",
            "protocol": "fixed-pose transductive SfM; no strict train-only pose-independence claim",
        }
        write_json(bundle_path, bundle)
    if not learned:
        return {
            "status": "SUCCESS",
            "scope": "RENDER_ONLY",
            "bundle": bundle_path.relative_to(ROOT).as_posix(),
            "views": len(bundle["images"]),
        }
    metrics = Metrics(learned=True)
    values = []
    with torch.no_grad():
        for row in bundle["images"]:
            tensors = []
            for name in ("prediction", "reference"):
                with Image.open(folder / row[name]) as image:
                    tensors.append(
                        torch.from_numpy(
                            np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255
                        ).to("cuda:0")
                    )
            values.append({"image_id": row["image_id"], **metrics(*tensors)})
    aggregates = {
        key: float(np.mean([v[key] for v in values]))
        if all(v[key] is not None for v in values)
        else None
        for key in ("psnr", "ssim", "lpips")
    }
    report = {
        "status": "SUCCESS" if metrics.lpips is not None else "PARTIAL",
        "split": subset,
        "view_count": len(values),
        "excluded_views": [],
        "per_view": values,
        "aggregate_arithmetic_mean_per_view": aggregates,
        "lpips_unavailable_reason": metrics.lpips_error,
        "metric_definition": {
            "psnr": "-10log10 MSE RGB, per-view then arithmetic mean; identity infinity as null+flag",
            "ssim": "pinned official utils.loss_utils.ssim, window11 sigma1.5, zero padding5, mean RGB/pixels",
            "lpips": "pinned official lpipsPyTorch AlexNet v0.1; [0,1] mapped to [-1,1], CUDA, no test exposure fitting",
        },
        "weight_sha256": metrics.weight_hashes,
        "weight_sources": {
            "alexnet-owt-7be5be79.pth": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
            "alex.pth": "https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/alex.pth",
        },
        "native_revision": environment["native_revision"],
        "bundle": bundle_path.relative_to(ROOT).as_posix(),
        "checkpoint_export_sha256": sha256(ply),
        "dataset_id": data.metadata["dataset_id"],
        "split_hash": data.split["hash"],
        "training_iterations": trained["iterations"],
        "profile": config["profile"],
        "memory": memory(),
        "protocol": bundle["protocol"],
        "quality_label": "functionality test only"
        if config["profile"] == "smoke"
        else "resource-constrained baseline; not paper-equivalent quality",
    }
    output = run / "metrics" / f"{subset}_{trained['iterations']:06d}.json"
    if output.exists():
        raise FileExistsError(f"Metrics already exist; refusing overwrite: {output}")
    write_json(output, report)
    return report
