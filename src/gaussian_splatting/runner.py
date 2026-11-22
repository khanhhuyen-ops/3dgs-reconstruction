"""Original3DGS training/rendering entrypoint with native algorithm and frozen data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import numpy as np
import yaml

from src.common.config import ROOT, repo_path
from src.common.io import PipelineError, fingerprint, read_json, sha256, write_json
from src.common.telemetry import VramMonitor
from src.gaussian_splatting import checkpoint
from src.gaussian_splatting.config import load_config
from src.gaussian_splatting.native import initialize
from src.gaussian_splatting.validation import validate_model, cuda_unit_fixture

LOG = logging.getLogger(__name__)


def code_identity() -> str:
    paths = sorted((ROOT / "src/gaussian_splatting").glob("*.py")) + sorted(
        (ROOT / "src/evaluation").glob("*.py")
    )
    return fingerprint({p.relative_to(ROOT).as_posix(): sha256(p) for p in paths})


def seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def memory() -> dict:
    import torch

    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device_free_bytes": free,
        "device_total_bytes": total,
    }


def append_json(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")


def density_event(
    model, options, extent: float, cap: int | None, max_screen_size: int | None
) -> dict:

    before = len(model.get_xyz)
    grads = model.xyz_gradient_accum / model.denom
    # Same zero-visibility rule as the official implementation, not loss sanitization.
    grads[grads.isnan()] = 0
    projected = int((grads.flatten() >= options.densify_grad_threshold).sum()) + before
    event = {"before": before, "projected_upper_final_count": projected, "cap": cap}
    if cap is not None and projected > cap:
        return {
            **event,
            "status": "SKIPPED_RESOURCE_CAP",
            "after": before,
            "cloned": 0,
            "split_parents": 0,
            "split_children": 0,
            "pruned": 0,
        }
    original_clone, original_split = model.densify_and_clone, model.densify_and_split
    counts = {}

    def clone(*args, **kwargs):
        n = len(model.get_xyz)
        original_clone(*args, **kwargs)
        counts["cloned"] = len(model.get_xyz) - n

    def split(*args, **kwargs):
        n = len(model.get_xyz)
        original_split(*args, **kwargs)
        counts["split_parents"] = len(model.get_xyz) - n
        counts["after_split"] = len(model.get_xyz)

    model.densify_and_clone = clone
    model.densify_and_split = split
    try:
        model.densify_and_prune(
            options.densify_grad_threshold, 0.005, extent, max_screen_size
        )
    finally:
        model.densify_and_clone = original_clone
        model.densify_and_split = original_split
    return {
        **event,
        "status": "SUCCESS",
        "after": len(model.get_xyz),
        "cloned": counts["cloned"],
        "split_parents": counts["split_parents"],
        "split_children": 2 * counts["split_parents"],
        "pruned": counts["after_split"] - len(model.get_xyz),
    }


def quick_validation(model, data, pipe, background, limit: int | None) -> dict:
    import torch
    from gaussian_renderer import render
    from src.evaluation.rendering_metrics import psnr_value
    from utils.loss_utils import ssim

    ids = data.ids["val"][:limit] if limit else data.ids["val"]
    values = []
    with torch.no_grad():
        for iid in ids:
            camera = data.camera(iid)
            out = render(camera, model, pipe, background)["render"].clamp(0, 1)
            target = camera.original_image.to("cuda:0")
            values.append(
                {
                    "image_id": iid,
                    **psnr_value(out, target),
                    "ssim": float(ssim(out, target)),
                }
            )
    return {
        "split": "val",
        "image_ids": ids,
        "per_view": values,
        "mean_psnr": float(np.mean([v["psnr"] for v in values]))
        if all(v["psnr"] is not None for v in values)
        else None,
    }


def export_and_verify(model, data, pipe, background, run: Path, iteration: int) -> dict:
    import torch
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from plyfile import PlyData

    folder = run / "point_cloud" / f"iteration_{iteration:06d}"
    folder.mkdir(parents=True)
    ply = folder / "point_cloud.ply"
    model.save_ply(str(ply))
    vertices = PlyData.read(ply)["vertex"]
    required = {
        "x",
        "y",
        "z",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
    }
    if len(vertices) != len(model.get_xyz) or not required <= set(
        vertices.data.dtype.names
    ):
        raise ValueError("Invalid Gaussian PLY schema/count")
    if any(not np.isfinite(vertices[name]).all() for name in vertices.data.dtype.names):
        raise ValueError("Nonfinite Gaussian PLY values")
    reloaded = GaussianModel(model.max_sh_degree)
    reloaded.load_ply(str(ply))
    reloaded.active_sh_degree = model.active_sh_degree
    cam = data.camera(data.ids["val"][0])
    with torch.no_grad():
        a = render(cam, model, pipe, background)["render"]
        b = render(cam, reloaded, pipe, background)["render"]
        difference = float((a - b).abs().max())
    if difference > 1e-5:
        raise ValueError(f"Gaussian PLY render-reload mismatch {difference}")
    report = {
        "status": "SUCCESS",
        "path": ply.relative_to(ROOT).as_posix(),
        "sha256": sha256(ply),
        "gaussians": len(vertices),
        "active_sh_degree": model.active_sh_degree,
        "max_sh_degree": model.max_sh_degree,
        "render_reload_max_abs_difference": difference,
        "artifact_type": "Gaussian splats; not triangle mesh",
    }
    write_json(folder / "validation.json", report)
    return report


def train(
    config: dict,
    run: Path,
    environment: dict,
    *,
    resume: bool = False,
    stop_after: int | None = None,
    initialize_only: bool = False,
) -> dict:
    import torch
    from src.gaussian_splatting.data import Dataset
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import l1_loss, ssim

    seed_all(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    data = Dataset(config, run)
    if (
        config["max_gaussians"] is not None
        and len(data.pcd.points) > config["max_gaussians"]
    ):
        raise PipelineError(
            "BLOCKED_RESOURCE",
            "Initial point count exceeds declared Gaussian cap; choose an explicit point-subset profile",
        )
    options = SimpleNamespace(
        iterations=config["iterations"], **config["optimizer"], **config["density"]
    )
    pipe = SimpleNamespace(
        convert_SHs_python=False, compute_cov3D_python=False, debug=False
    )
    background = torch.tensor(
        [1.0, 1.0, 1.0] if config["white_background"] else [0.0, 0.0, 0.0],
        device="cuda",
    )
    identity = {
        "config": fingerprint(config),
        "metadata": sha256(data.path / "metadata.json"),
        "split": data.split["hash"],
        "native_revision": environment["native_revision"],
        "extension_hashes": environment["extension_hashes"],
        "code": code_identity(),
    }
    model = GaussianModel(config["sh_degree"])
    first, optimizer_steps, sampler = 0, 0, []
    if resume:
        saved = checkpoint.load_latest(run / "checkpoints", model, options, identity)
        first = saved["iteration"]
        optimizer_steps = saved["optimizer_steps"]
        sampler = saved["sampler_remaining_ids"]
        LOG.info(
            "Restored iteration %d, optimizer steps %d, sampler remaining %d",
            first,
            optimizer_steps,
            len(sampler),
        )
        write_json(
            run / f"resume_{first:06d}.json",
            {
                "status": "SUCCESS",
                "iteration": first,
                "optimizer_steps": optimizer_steps,
                "native_state_restored": True,
                "rng_and_sampler_restored": True,
                "model": validate_model(model),
            },
        )
    else:
        model.create_from_pcd(data.pcd, data.extent)
        model.training_setup(options)
        write_json(
            run / "initialization.json",
            {
                "status": "SUCCESS",
                **validate_model(model),
                "positions_from": "Phase01 train-only sparse PLY",
                "scale": "native distCUDA2 -> log(sqrt(max(mean_squared_3NN,1e-7)))",
                "opacity_initial": 0.1,
                "rotation_initial": "identity Hamilton WXYZ",
                "sh": "RGB2SH DC, higher coefficients zero",
            },
        )
    if initialize_only:
        cam = data.camera(data.ids["train"][0])
        out = render(cam, model, pipe, background)["render"]
        target = cam.original_image.to("cuda:0")
        loss = 0.8 * l1_loss(out, target) + 0.2 * (1 - ssim(out, target))
        loss.backward()
        check = validate_model(model, gradients=True)
        if not torch.isfinite(loss) or not any(
            check["gradient_norms"][k] > 0 for k in ("xyz", "scaling", "opacity")
        ):
            raise ValueError("Real initialization/backward test failed")
        return {
            "status": "SUCCESS",
            "scope": "REAL_DATA_INITIALIZATION_RENDER_BACKWARD",
            "loss": float(loss),
            "model": check,
            "memory": memory(),
            "dataset": data.report,
        }
    until = min(config["iterations"], stop_after or config["iterations"])
    if first >= until:
        raise ValueError("Resume must continue beyond saved iteration")
    elapsed_start = time.perf_counter()
    last_time = elapsed_start
    last_loss = None
    iteration = first
    with VramMonitor() as monitor:
        try:
            for iteration in range(first + 1, until + 1):
                model.update_learning_rate(iteration)
                if iteration % 1000 == 0:
                    model.oneupSHdegree()
                if not sampler:
                    sampler = data.ids["train"].copy()
                iid = sampler.pop(random.randint(0, len(sampler) - 1))
                camera = data.camera(iid)
                target = camera.original_image.to("cuda:0")
                package = render(camera, model, pipe, background)
                image = package["render"]
                l1 = l1_loss(image, target)
                structural = 1 - ssim(image, target)
                loss = (
                    1 - options.lambda_dssim
                ) * l1 + options.lambda_dssim * structural
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite photometric loss")
                loss.backward()
                if (
                    iteration == first + 1
                    or iteration % config["gradient_check_interval"] == 0
                ):
                    validate_model(model, gradients=True)
                with torch.no_grad():
                    if iteration < options.densify_until_iter:
                        visible = package["visibility_filter"]
                        radii = package["radii"]
                        model.max_radii2D[visible] = torch.max(
                            model.max_radii2D[visible], radii[visible]
                        )
                        model.add_densification_stats(
                            package["viewspace_points"], visible
                        )
                        if (
                            iteration > options.densify_from_iter
                            and iteration % options.densification_interval == 0
                        ):
                            event = density_event(
                                model,
                                options,
                                data.extent,
                                config["max_gaussians"],
                                20
                                if iteration > options.opacity_reset_interval
                                else None,
                            )
                            append_json(
                                run / "density.jsonl",
                                {"iteration": iteration, **event, **memory()},
                            )
                        if iteration % options.opacity_reset_interval == 0 or (
                            config["white_background"]
                            and iteration == options.densify_from_iter
                        ):
                            model.reset_opacity()
                            append_json(
                                run / "density.jsonl",
                                {"iteration": iteration, "event": "opacity_reset"},
                            )
                    # Exactly the selected native ADC -> Adam order; native skips final Adam.
                    if iteration < config["iterations"]:
                        model.optimizer.step()
                        optimizer_steps += 1
                    model.optimizer.zero_grad(set_to_none=True)
                    last_loss = float(loss)
                    if (
                        iteration % config["log_interval"] == 0
                        or iteration == first + 1
                    ):
                        torch.cuda.synchronize()
                        now = time.perf_counter()
                        row = {
                            "iteration": iteration,
                            "image_id": iid,
                            "loss": last_loss,
                            "l1": float(l1),
                            "dssim": float(structural),
                            "loss_schema_version": "gaussian-loss-v1",
                            "losses": {
                                "total": last_loss,
                                "raw": {
                                    "l1": float(l1),
                                    "ssim_dissimilarity": float(structural),
                                },
                                "coefficients": {
                                    "l1": 1 - options.lambda_dssim,
                                    "ssim_dissimilarity": options.lambda_dssim,
                                },
                                "weighted": {
                                    "l1": float(l1) * (1 - options.lambda_dssim),
                                    "ssim_dissimilarity": float(structural)
                                    * options.lambda_dssim,
                                },
                                "activation": {
                                    "l1": True,
                                    "ssim_dissimilarity": True,
                                },
                            },
                            "train_psnr": float(
                                -10 * torch.log10((image - target).square().mean())
                            ),
                            "gaussians": len(model.get_xyz),
                            "active_sh_degree": model.active_sh_degree,
                            "optimizer_steps": optimizer_steps,
                            "elapsed_segment_seconds": now - elapsed_start,
                            "seconds_since_last_log": now - last_time,
                            **memory(),
                        }
                        append_json(run / "training.jsonl", row)
                        LOG.info(
                            "iteration %d/%d loss %.6f gaussians %d allocated %.0f MiB",
                            iteration,
                            config["iterations"],
                            last_loss,
                            len(model.get_xyz),
                            row["allocated_bytes"] / 1048576,
                        )
                        last_time = now
                    del package, image, target, loss, l1, structural
                    if (
                        torch.cuda.memory_reserved()
                        > config["torch_reserved_budget_gib"] * 1024**3
                    ):
                        raise PipelineError(
                            "BLOCKED_RESOURCE",
                            "Measured reserved VRAM exceeds configured budget; preserve checkpoints and use a declared retry profile",
                        )
                    if iteration % config["validation_interval"] == 0:
                        write_json(
                            run / f"validation_{iteration:06d}.json",
                            quick_validation(
                                model,
                                data,
                                pipe,
                                background,
                                config["validation_view_limit"],
                            ),
                        )
                    if (
                        iteration % config["checkpoint_interval"] == 0
                        or iteration == until
                    ):
                        validate_model(model)
                        checkpoint.save(
                            run / "checkpoints",
                            model,
                            iteration,
                            sampler,
                            identity,
                            optimizer_steps,
                        )
            export = export_and_verify(model, data, pipe, background, run, until)
        except (RuntimeError, ValueError, PipelineError) as exc:
            write_json(
                run / "failure.json",
                {
                    "status": getattr(
                        exc,
                        "status",
                        "BLOCKED_RESOURCE"
                        if isinstance(exc, torch.cuda.OutOfMemoryError)
                        else "FAILED",
                    ),
                    "iteration": iteration,
                    "reason": str(exc),
                    "last_checkpoint": read_json(run / "checkpoints/latest.json")
                    if (run / "checkpoints/latest.json").exists()
                    else None,
                    "memory": memory(),
                },
            )
            raise
    previous = (
        read_json(run / "training_result.json")
        if resume and (run / "training_result.json").exists()
        else {}
    )
    report = {
        "status": "SUCCESS"
        if until == config["iterations"]
        else "INTERRUPTED_FOR_RESUME_TEST",
        "iterations": until,
        "optimizer_steps": optimizer_steps,
        "initial_gaussians": len(data.pcd.points),
        "final_gaussians": len(model.get_xyz),
        "final_sampled_training_loss": last_loss,
        "training_seconds_total": previous.get("training_seconds_total", 0)
        + time.perf_counter()
        - elapsed_start,
        "memory": memory(),
        "device_wide_memory": monitor.report(),
        "export": export,
        "dataset": data.report,
        "native_final_iteration_skips_adam": True,
        "current_segment_start_iteration": first + 1,
    }
    write_json(run / "training_result.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pinned B0 Original3DGS; CUDA only, frozen Phase01 split"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["preflight", "dataset", "initialize", "train", "render", "evaluate"],
        default="train",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()
    config = load_config(args.config)
    existing = args.resume or args.run
    if existing:
        run = repo_path(ROOT, str(existing))
        if (
            not run.is_relative_to(repo_path(ROOT, config["output_root"]))
            or read_json(run / "resolved_config.json") != config
        ):
            raise ValueError("Run/config/output-root mismatch")
    else:
        run = repo_path(ROOT, config["output_root"]) / (
            config["experiment"]
            + "_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + uuid.uuid4().hex[:6]
        )
        run.mkdir(parents=True)
        write_json(run / "resolved_config.json", config)
        (run / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        write_json(
            run / "run_manifest.json",
            {
                "status": "IN_PROGRESS",
                "method": config["method"],
                "profile": config["profile"],
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "display_timezone": "Asia/Saigon",
                "source_mode": "legacy image_set, historical CPU BA (not video GPU-strict SfM)",
                "git_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "code_hash": code_identity(),
                "configuration_hash": fingerprint(config),
            },
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run / "run.log", encoding="utf-8"),
        ],
    )
    write_json(
        run / f"invocation_{uuid.uuid4().hex[:8]}.json",
        {"argv": sys.argv, "stage": args.stage, "cwd": "."},
    )
    LOG.info("Run: %s", run.relative_to(ROOT))
    try:
        environment = initialize()
        write_json(run / "environment.json", environment)
        if args.stage == "preflight":
            report = {
                "status": "SUCCESS",
                "cuda_unit_fixture": cuda_unit_fixture(config),
                "environment": environment,
            }
        elif args.stage == "dataset":
            from src.gaussian_splatting.data import Dataset

            data = Dataset(config, run)
            camera = data.camera(data.ids["train"][0])
            report = {
                "status": "SUCCESS",
                "dataset": data.report,
                "sample_shape": list(camera.original_image.shape),
                "sample_device": str(camera.original_image.device),
            }
        elif args.stage in ("initialize", "train"):
            report = train(
                config,
                run,
                environment,
                resume=bool(args.resume),
                stop_after=args.stop_after,
                initialize_only=args.stage == "initialize",
            )
        else:
            if existing is None:
                raise ValueError(
                    "render/evaluate requires --run with a trained checkpoint"
                )
            from src.evaluation.evaluate_baseline import evaluate

            report = evaluate(
                config,
                run,
                environment,
                subset=args.split,
                learned=args.stage == "evaluate",
            )
        write_json(run / f"{args.stage}_result.json", report)
        if args.stage == "train":
            manifest = read_json(run / "run_manifest.json")
            manifest.update(
                status=report["status"],
                iterations=report["iterations"],
                training_started=True,
            )
            write_json(run / "run_manifest.json", manifest)
        print(
            json.dumps(
                {"status": report["status"], "run": run.relative_to(ROOT).as_posix()}
            )
        )
        return 0
    except (OSError, ValueError, RuntimeError, PipelineError) as exc:
        status = getattr(exc, "status", "FAILED")
        write_json(
            run / f"{args.stage}_failure.json", {"status": status, "reason": str(exc)}
        )
        LOG.exception("%s", status)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
