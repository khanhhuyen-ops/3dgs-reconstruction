"""Audited native PGSR optimization order, immutable experiments and full-state resume."""
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import logging
import random
import time
import uuid

import numpy as np

from src.common.io import read_json, write_json, fingerprint, sha256
from src.common.telemetry import VramMonitor
from .config import ROOT, load
from .native import initialize, code_identity, REVISION

LOG = logging.getLogger(__name__)


def memory():
    import torch
    return {"allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False)+"\n")


def options(config):
    return SimpleNamespace(**config["optimizer"], iterations=config["iterations"])


def export_model(run, model, data, pipe, bg, iteration):
    import torch
    from gaussian_renderer import render
    folder = run / "point_cloud" / f"iteration_{iteration:06d}"
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "point_cloud.ply"
    model.save_ply(str(path))
    # Save float reference now; reload is verified in a separate inference process,
    # avoiding a second large Gaussian store beside the training optimizer.
    with torch.no_grad():
        camera = data.camera(data.ids["val"][0])
        package = render(camera, model, pipe, bg, app_model=None)
        np.savez_compressed(folder / "reload_reference.npz",
                            rgb=package["render"].cpu().numpy(),
                            depth=package["plane_depth"].cpu().numpy(),
                            normal=package["rendered_normal"].cpu().numpy())
    report = {"status": "EXPORTED_PENDING_SEPARATE_RELOAD", "path": path.relative_to(ROOT).as_posix(),
              "sha256": sha256(path), "gaussians": len(model.get_xyz),
              "active_sh_degree": model.active_sh_degree,
              "reference_image_id": camera.image_id, "artifact_type": "PGSR Gaussian splats, not a triangle mesh"}
    write_json(folder / "validation.json", report)
    return report


def train(config, run, environment, resume=False, stop_after=None):
    import torch
    from scene.gaussian_model import GaussianModel
    from scene.app_model import AppModel
    from gaussian_renderer import render
    from .data import SceneAdapter
    from .geometry import objectives
    from .validation import validate_model
    from . import checkpoint

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    segment = run / "segments" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+uuid.uuid4().hex[:6])
    segment.mkdir(parents=True)
    data = SceneAdapter(config, segment)
    identity = {"method":"B2_PGSR", "config":fingerprint(config), **data.identity,
                "native_revision":REVISION, "code":code_identity()}
    if resume:
        if read_json(run / "identity.json") != identity:
            raise ValueError("PGSR resume identity changed")
    else:
        write_json(run / "identity.json", identity)
    model = GaussianModel(config["sh_degree"])
    app = AppModel(len(data.train)).train().cuda()
    opt = options(config)
    sampler, first, optimizer_steps = [], 0, 0
    if resume:
        payload = checkpoint.restore(run, model, app, opt, identity, environment, data.ids["train"])
        first, optimizer_steps = payload["iteration"], payload["optimizer_steps"]
        sampler = payload["sampler_remaining_ids"]
        del payload
    else:
        model.create_from_pcd(data.pcd, data.extent)
        model.training_setup(opt)
    target = stop_after or config["iterations"]
    if not first < target <= config["iterations"]:
        raise ValueError("Target must exceed the published checkpoint and not exceed native total iterations")
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.full((3,), float(config["white_background"]), device="cuda:0")
    previous = read_json(run / "training_result.json") if (run / "training_result.json").exists() else {}
    coverage = {k: previous.get("geometry_coverage_totals", {}).get(k, 0)
                for k in ("geometry_valid_pixels", "multi_valid_pixels", "ncc_valid_patches")}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    completed, status = first, "INTERRUPTED_FOR_RESUME_TEST"
    write_json(run / "run_manifest.json", {"method":"B2_PGSR", "status":"TRAINING",
               "iterations":first, "training_started":True, "identity":identity})
    with VramMonitor() as monitor:
        try:
            for iteration in range(first+1, target+1):
                model.update_learning_rate(iteration)
                if iteration % 1000 == 0:
                    model.oneupSHdegree()
                if not sampler:
                    sampler = data.ids["train"].copy()
                iid = sampler.pop(random.randint(0, len(sampler)-1))
                camera = data.camera(iid)
                rgb, gray = camera.get_image()
                model.use_app = bool(opt.exposure_compensation and iteration > config["exposure_from_iter"])
                geometry_active = iteration > opt.single_view_weight_from_iter
                bg = torch.rand(3, device="cuda") if opt.random_background else background
                package = render(camera, model, pipe, bg, app_model=app,
                                 return_plane=geometry_active, return_depth_normal=geometry_active)
                near, near_package, near_gray = None, None, None
                if iteration > opt.multi_view_weight_from_iter and camera.nearest_id:
                    near = data.train[random.sample(camera.nearest_id, 1)[0]]
                    _, near_gray = near.get_image()
                    near_package = render(near, model, pipe, bg, app_model=app,
                                          return_plane=True, return_depth_normal=False)
                loss, terms, coefficients, counts, photometric = objectives(
                    model, app, camera, near, package, near_package, rgb, gray, near_gray, opt, config, iteration)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite PGSR objective")
                loss.backward()
                gradients = {}
                if iteration % config["gradient_check_interval"] == 0 or iteration == target:
                    gradients = validate_model(model, gradients=True)["gradient_norms"]
                    if app.appear_ab.grad is not None and not torch.isfinite(app.appear_ab.grad).all():
                        raise ValueError("Nonfinite exposure gradient")
                for key in coverage:
                    coverage[key] += counts.get(key, 0)
                # Native PGSR order: collect statistics -> AbsGS ADC -> multi-view
                # visibility trim -> opacity reset -> Adam (except final iteration).
                with torch.no_grad():
                    if iteration < opt.densify_until_iter:
                        visible = package["visibility_filter"]
                        observed = (package["out_observe"] > 0) & visible
                        model.max_radii2D[observed] = torch.maximum(model.max_radii2D[observed], package["radii"][observed])
                        model.add_densification_stats(package["viewspace_points"], package["viewspace_points_abs"], visible)
                        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            before = len(model.get_xyz)
                            if before > opt.max_all_points:
                                raise RuntimeError("BLOCKED_RESOURCE: native point cap exceeded before ADC")
                            model.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold,
                                                   opt.opacity_cull_threshold, data.extent,
                                                   20 if iteration > opt.opacity_reset_interval else None)
                            append(run / "density.jsonl", {"iteration":iteration,"event":"densify_and_prune",
                                   "before":before,"after":len(model.get_xyz)})
                    if opt.use_multi_view_trim and iteration % config["trim_interval"] == 0 and iteration < opt.densify_until_iter:
                        observations = torch.zeros_like(model.get_opacity)
                        for view in data.train:
                            observation = render(view, model, pipe, bg, app_model=app, return_plane=False, return_depth_normal=False)["out_observe"]
                            observations[observation > 0] += 1
                        prune = (observations < 2).squeeze(-1)
                        if prune.any():
                            before = len(model.get_xyz)
                            model.prune_points(prune)
                            append(run / "density.jsonl", {"iteration":iteration,"event":"visibility_prune",
                                   "before":before,"after":len(model.get_xyz),"pruned":before-len(model.get_xyz)})
                    if iteration < opt.densify_until_iter:
                        if iteration % opt.opacity_reset_interval == 0 or (config["white_background"] and iteration==opt.densify_from_iter):
                            model.reset_opacity()
                    if iteration < opt.iterations:
                        model.optimizer.step()
                        app.optimizer.step()
                        optimizer_steps += 1
                        model.optimizer.zero_grad(set_to_none=True)
                        app.optimizer.zero_grad(set_to_none=True)
                completed = iteration
                if not len(model.get_xyz):
                    raise ValueError("PGSR visibility trimming removed all Gaussians")
                if iteration % config["log_interval"] == 0 or iteration == target:
                    if (photometric["l1_appearance"] is None
                            and "app_image" in package):
                        with torch.no_grad():
                            photometric["l1_appearance"] = (
                                package["app_image"] - rgb).abs().mean()
                    raw_terms = {k:float(v.detach()) for k,v in terms.items()}
                    weighted_terms = {k:raw_terms[k]*coefficients[k] for k in terms}
                    photometric_values = {
                        key: (float(value.detach()) if hasattr(value, "detach") else value)
                        for key, value in photometric.items()
                    }
                    row = {"iteration":iteration,"image_id":iid,"neighbor_image_id":near.image_id if near else None,
                           "loss":float(loss.detach()),"raw":raw_terms,
                           "coefficients":coefficients,"weighted":weighted_terms,
                           "loss_schema_version":"gaussian-loss-v1",
                           "losses":{"total":float(loss.detach()),"raw":raw_terms,
                                     "coefficients":coefficients,"weighted":weighted_terms,
                                     "photometric":photometric_values,
                                     "activation":{"flatten":True,
                                                   "single":iteration>opt.single_view_weight_from_iter,
                                                   "multi_geo":iteration>opt.multi_view_weight_from_iter and near is not None,
                                                   "multi_ncc":iteration>opt.multi_view_weight_from_iter and near is not None,
                                                   "exposure":bool(photometric_values["exposure_gate_used"])},
                                     "coverage":counts},
                           "gaussians":len(model.get_xyz),"active_sh_degree":model.active_sh_degree,
                           "optimizer_steps":optimizer_steps,
                           "learning_rates":{group["name"]:group["lr"] for group in model.optimizer.param_groups},
                           "gradient_norms":gradients,"elapsed_segment_seconds":time.perf_counter()-started,
                           "position_lr":model.optimizer.param_groups[0]["lr"],**counts,**memory()}
                    append(run / "training.jsonl", row)
                    LOG.info(
                        "iteration %d/%d loss_total %.9g raw_rgb %.9g raw_flatten %.9g "
                        "raw_single %.9g raw_multi_geo %.9g raw_multi_ncc %.9g "
                        "weighted_rgb %.9g weighted_flatten %.9g weighted_single %.9g "
                        "weighted_multi_geo %.9g weighted_multi_ncc %.9g points %d "
                        "geo_pixels %d ncc_patches %d l1_selected %.9g dssim %.9g "
                        "l1_uncompensated %.9g l1_appearance %s exposure_gate %s",
                        iteration, config["iterations"], row["loss"],
                        raw_terms["rgb"], raw_terms["flatten"], raw_terms["single"],
                        raw_terms["multi_geo"], raw_terms["multi_ncc"],
                        weighted_terms["rgb"], weighted_terms["flatten"],
                        weighted_terms["single"], weighted_terms["multi_geo"],
                        weighted_terms["multi_ncc"], row["gaussians"],
                        counts["multi_valid_pixels"], counts["ncc_valid_patches"],
                        photometric_values["l1_selected"], photometric_values["ssim_dissimilarity"],
                        photometric_values["l1_uncompensated"], photometric_values["l1_appearance"],
                        photometric_values["exposure_gate_used"])
                del package, near_package, loss, terms, photometric, rgb, gray, near_gray
                if torch.cuda.memory_reserved() > config["torch_reserved_budget_gib"]*1024**3:
                    raise RuntimeError("BLOCKED_RESOURCE: measured reserved VRAM exceeds the declared profile")
                session_stop = time.perf_counter()-started > config["session_seconds"]-600
                if iteration % config["checkpoint_interval"] == 0 or iteration==target or session_stop:
                    validate_model(model)
                    checkpoint.save(run, model, app, iteration, sampler, identity, environment, config, optimizer_steps)
                if session_stop:
                    status = "PARTIAL_SESSION_BUDGET"
                    break
            if completed > opt.multi_view_weight_from_iter and not coverage["ncc_valid_patches"]:
                raise ValueError("BLOCKED_GEOMETRY_COVERAGE: no valid NCC patches executed")
            if completed == config["iterations"]:
                status = "SUCCESS"
            exported = export_model(run, model, data, pipe, background, completed)
        except BaseException as exc:
            failure_status = ("INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else
                              "BLOCKED_RESOURCE" if "RESOURCE" in str(exc) or isinstance(exc,torch.cuda.OutOfMemoryError) else "FAILED")
            write_json(run / "run_manifest.json", {"method":"B2_PGSR", "status":failure_status,
                       "iterations":completed, "training_started":True, "identity":identity})
            write_json(run / "failure.json", {"status":failure_status,
                       "iteration_in_progress":locals().get("iteration"),"reason":str(exc),
                       "published_checkpoint":read_json(run / "checkpoints/latest.json") if (run / "checkpoints/latest.json").exists() else None,
                       "memory":memory()})
            raise
        report = {"status":status,"iterations":completed,"optimizer_steps":optimizer_steps,
                  "initial_gaussians":len(data.pcd.points),"final_gaussians":len(model.get_xyz),
                  "training_seconds_total":previous.get("training_seconds_total",0)+time.perf_counter()-started,
                  "current_segment_start_iteration":first+1,"current_segment_seconds":time.perf_counter()-started,
                  "timing_scope":"Measured PGSR segments in this run, loop/checkpoint/export; excludes setup, data qualification and any unreported interrupted tail",
                  "geometry_coverage_totals":coverage,"memory":memory(),"device_wide_memory":monitor.report(),
                  "export":exported,"dataset_identity":data.identity,"exposure_rows":len(app.appear_ab)}
        write_json(run / "training_result.json", report)
        write_json(segment / "result.json", report)
        write_json(run / "run_manifest.json", {"method":"B2_PGSR","status":status,"iterations":completed,
                   "training_started":True,"identity":identity})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["preflight","train","evaluate"], default="train")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load(args.config)
    if args.dry_run:
        print(json.dumps(config,indent=2))
        return 0
    environment = initialize()
    import torch
    torch.set_num_threads(4)
    from .validation import cuda_fixture
    if args.run:
        run = args.run.resolve()
        if not run.is_relative_to((ROOT/config["output_root"]).resolve()):
            raise ValueError("PGSR run must stay in its own configured output root")
    else:
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:6]
        run = ROOT/config["output_root"]/(config["experiment"]+"_"+suffix)
    if args.resume or args.stage=="evaluate":
        if read_json(run/"resolved_config.json") != config:
            raise ValueError("Resolved configuration changed; use a new declared experiment")
    else:
        run.mkdir(parents=True,exist_ok=False)
        write_json(run/"resolved_config.json", config)
        write_json(run/"environment.json", environment)
        write_json(run/"run_manifest.json", {"method":"B2_PGSR","status":"INITIALIZING","training_started":False})
    logging.basicConfig(level=logging.INFO,handlers=[logging.StreamHandler(),logging.FileHandler(run/"run.log",encoding="utf-8")])
    print("PGSR_RUN="+run.relative_to(ROOT).as_posix(),flush=True)
    if args.stage=="evaluate":
        from .evaluation import evaluate
        print(evaluate(config,run,environment))
    else:
        cuda_fixture(run/"preflight"/uuid.uuid4().hex[:8])
        torch.cuda.empty_cache()
        if args.stage=="train":
            print(train(config,run,environment,args.resume,args.stop_after))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
