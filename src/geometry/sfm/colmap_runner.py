"""COLMAP V0 with verified CUDA workloads and a fail-closed GPU BA gate."""
from __future__ import annotations

import logging
import re
import sqlite3
import time
import uuid
from pathlib import Path

from src.common.config import repo_path
from src.common.io import PipelineError, code_hash, fingerprint, read_json, run_process, sha256, write_json
from src.common.telemetry import VramMonitor
from src.geometry.sfm.metrics import model_metrics
from src.geometry.sfm.model_io import export_poses, read_model, validate_ply
from src.preprocessing.extract_frames import validate_manifest

LOGGER = logging.getLogger(__name__)


def choose_option(options: list[str], *choices: str) -> str:
    for choice in choices:
        if choice in options:
            return "--" + choice
    raise PipelineError("STOP_GPU_UNAVAILABLE", f"Installed CLI lacks required option: {' / '.join(choices)}")


def feature_commands(config: dict, capabilities: dict, database: Path, images: Path) -> tuple[list[str], list[str]]:
    binary, sfm = capabilities["binary"], config["sfm"]
    extraction_options = capabilities["options"]["feature_extractor"]
    matching_options = capabilities["options"][sfm["matching"] + "_matcher"]
    extraction = [binary, "feature_extractor", "--database_path", str(database), "--image_path", str(images),
                  "--ImageReader.camera_model", sfm["camera_model"], "--ImageReader.single_camera", str(int(sfm["single_camera"])),
                  choose_option(extraction_options, "FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"), "1",
                  choose_option(extraction_options, "FeatureExtraction.gpu_index", "SiftExtraction.gpu_index"), "0",
                  choose_option(extraction_options, "FeatureExtraction.max_image_size", "SiftExtraction.max_image_size"), str(sfm["max_image_size"]),
                  choose_option(extraction_options, "SiftExtraction.max_num_features"), str(sfm["max_num_features"])]
    if "FeatureExtraction.type" in extraction_options:
        extraction += ["--FeatureExtraction.type", "SIFT"]
    matching = [binary, sfm["matching"] + "_matcher", "--database_path", str(database),
                choose_option(matching_options, "FeatureMatching.use_gpu", "SiftMatching.use_gpu"), "1",
                choose_option(matching_options, "FeatureMatching.gpu_index", "SiftMatching.gpu_index"), "0",
                choose_option(matching_options, "FeatureMatching.max_num_matches", "SiftMatching.max_num_matches"), str(sfm["max_num_matches"])]
    if "FeatureMatching.type" in matching_options:
        matching += ["--FeatureMatching.type", "SIFT_BRUTEFORCE"]
    if "FeatureMatching.skip_geometric_verification" in matching_options:
        matching += ["--FeatureMatching.skip_geometric_verification", "0"]
    if sfm["matching"] == "sequential":
        matching += [choose_option(matching_options, "SequentialMatching.overlap"), str(sfm["sequential_overlap"])]
    for command, options in ((extraction, extraction_options), (matching, matching_options)):
        if "default_random_seed" in options:
            command += ["--default_random_seed", str(config["project"]["seed"])]
        if "log_level" in options:
            command += ["--log_level", "2"]
    return extraction, matching


def database_stats(database: Path, expected_names: list[str]) -> dict:
    """Verify exact frame provenance plus populated descriptor/geometry blobs."""
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        names = sorted(row[0] for row in connection.execute("SELECT name FROM images"))
        if names != sorted(expected_names):
            raise ValueError("Database images differ from selected video frames")
        stats = {"images": len(names)}
        signatures = {}
        for table in ("keypoints", "descriptors"):
            rows = connection.execute(f"SELECT image_id, rows, cols, data FROM {table} ORDER BY image_id").fetchall()
            if len(rows) != len(names) or any(r[1] <= 0 or not r[3] for r in rows):
                raise ValueError(f"Missing or empty {table}")
            import hashlib
            signatures[table] = [(r[0], r[1], r[2], hashlib.sha256(r[3]).hexdigest()) for r in rows]
            stats[table + "_rows"] = sum(r[1] for r in rows)
        stats["feature_signature"] = fingerprint(signatures)
        geometries = connection.execute("SELECT pair_id, rows, cols, config, data FROM two_view_geometries WHERE rows > 0 AND config > 0 ORDER BY pair_id").fetchall()
        stats["verified_pairs"] = len(geometries)
        stats["verified_inliers"] = sum(row[1] for row in geometries)
        stats["verified_geometry_signature"] = fingerprint([(r[0], r[1], r[2], r[3], r[4].hex() if r[4] else None) for r in geometries])
    return stats


def require_runtime_log(output: str, stage: str) -> None:
    lower = output.lower()
    if re.search(r"fall(?:ing)?\s*back|fallback|sift.*cpu|cpu.*sift", lower):
        raise PipelineError("STOP_GPU_UNAVAILABLE", f"CPU/fallback reported in {stage} log")
    # The explicit CUDA SiftGPU implementation logs this after backend construction.
    if not re.search(r"(?:sift.*(?:cuda|gpu)|(?:cuda|gpu).*sift)", lower):
        raise PipelineError("STOP_GPU_UNAVAILABLE", f"No positive SIFT CUDA backend evidence in {stage} log")


def mapper_command(config: dict, capabilities: dict, database: Path, images: Path, sparse: Path, ba_report: dict) -> list[str]:
    if not ba_report.get("verified") or ba_report.get("backend") != "CASPAR" or ba_report.get("colmap_binary_sha256") != capabilities["binary_sha256"]:
        raise PipelineError("BLOCKED_GPU_BA", "Mapper forbidden: current binary has no verified GPU-exclusive BA backend")
    if config["sfm"]["mapper_backend"] != "CASPAR" or config["sfm"]["camera_model"] not in ("SIMPLE_RADIAL", "PINHOLE"):
        raise PipelineError("BLOCKED_GPU_BA", "Strict mapper currently supports verified CASPAR with SIMPLE_RADIAL/PINHOLE only")
    options = capabilities["options"]["mapper"]
    command = [capabilities["binary"], "mapper", "--database_path", str(database), "--image_path", str(images), "--output_path", str(sparse),
               choose_option(options, "Mapper.ba_local_backend"), "CASPAR", choose_option(options, "Mapper.ba_global_backend"), "CASPAR",
               choose_option(options, "Mapper.ba_gpu_index"), "0", choose_option(options, "Mapper.ba_refine_sensor_from_rig"), "0"]
    # COLMAP's structure-less registration can invoke Ceres independently of
    # the mapper's local/global BA backend. Keep the runtime log gate as well.
    for option in ("Mapper.structure_less_registration_fallback", "Mapper.structure_less_registration_only"):
        if option in options:
            command += ["--" + option, "0"]
    if "log_level" in options:
        command += ["--log_level", "2"]
    for key in ("default_random_seed", "Mapper.random_seed"):
        if key in options:
            command += ["--" + key, str(config["project"]["seed"])]
    return command


def run_colmap(config: dict, root: Path, output: Path, selected: Path, environment: dict, resume: bool) -> dict:
    started = time.perf_counter()
    manifest = validate_manifest(selected)
    if manifest.get("code_hash") != code_hash() or manifest.get("config") != config["preprocessing"] or manifest.get("seed") != config["project"]["seed"]:
        raise PipelineError("FAILED", "Selection provenance changed; rerun preprocessing into new paths")
    if sha256(repo_path(root, config["input"]["video"])) != manifest["source_sha256"]:
        raise PipelineError("STOP_INPUT_INVALID", "Video hash differs from selected-frame provenance")
    capabilities = environment.get("colmap")
    if not capabilities or environment["cuda"]["status"] != "SUCCESS":
        raise PipelineError("STOP_GPU_UNAVAILABLE", "COLMAP CUDA/PyTorch preflight failed")
    if "with CUDA" not in capabilities["version"]:
        raise PipelineError("STOP_GPU_UNAVAILABLE", "COLMAP build does not advertise CUDA; OpenGL SIFT is insufficient")
    database = output / "database.db"
    state_path = output / "colmap_state.json"
    key = fingerprint({"selected": manifest["cache_key"], "frames": manifest["frames"], "sfm": config["sfm"],
                       "seed": config["project"]["seed"], "binary": capabilities["binary_sha256"], "code": code_hash()})
    state = read_json(state_path) if state_path.exists() else {"cache_key": key, "stages": {}}
    if state["cache_key"] != key:
        raise PipelineError("FAILED", "COLMAP cache key changed; preserve DB and use a new output path")
    if database.exists() and not state_path.exists():
        raise PipelineError("FAILED", "Untracked existing database; refusing to modify")
    if state["stages"] and not resume:
        raise PipelineError("FAILED", "COLMAP checkpoint exists; use --resume")
    write_json(state_path, state)
    commands = list(state.get("commands", []))
    gpu = read_json(output / "gpu_report.json")
    metrics = {"status": "RUNNING", "input_frames": len(manifest["frames"]), "registered_images": None,
               "registration_ratio": None, "points3d_count": None, "mean_track_length": None,
               "mean_reprojection_error_px": None, "median_reprojection_error_px": None, "runtime_seconds": {},
               "gpu": gpu, "source_video_sha256": manifest["source_sha256"], "selected_cache_key": manifest["cache_key"]}
    expected = [r["filename"] for r in manifest["frames"]]
    extraction, matching = feature_commands(config, capabilities, database, selected)

    def execute_stage(name: str, command: list[str]) -> dict:
        log = output / "logs" / f"{name}_{uuid.uuid4().hex[:8]}.log"
        record = {"command": command, "log": str(log.relative_to(output)), "status": "FAILED", "returncode": None}
        with VramMonitor() as monitor:
            try:
                result = run_process(command, log, timeout=7200)
                record.update({"status": "SUCCESS", "returncode": result["returncode"]})
            finally:
                record.update(monitor.report())
                commands.append(record)
                state["commands"] = commands
                write_json(output / "commands.json", commands)
                write_json(state_path, state)
        return {**record, "output": result["output"]}

    try:
        if "features" in state["stages"]:
            stats = database_stats(database, expected)
            if stats["feature_signature"] != state["stages"]["features"]["feature_signature"]:
                raise PipelineError("FAILED", "Cached feature blobs changed")
        else:
            LOGGER.info("Running CUDA SIFT on %d selected frames", len(expected))
            result = execute_stage("features", extraction)
            require_runtime_log(result["output"], "features")
            stats = database_stats(database, expected)
            state["stages"]["features"] = {**stats, "runtime": {k: v for k, v in result.items() if k != "output"}, "gpu_verified": True}
            write_json(state_path, state)
        gpu["feature_runtime"] = state["stages"]["features"]["runtime"]
        if "matching" in state["stages"]:
            stats = database_stats(database, expected)
            if stats["verified_geometry_signature"] != state["stages"]["matching"]["verified_geometry_signature"]:
                raise PipelineError("FAILED", "Cached verified matches changed")
        else:
            LOGGER.info("Running CUDA %s matching and geometric verification", config["sfm"]["matching"])
            result = execute_stage("matching", matching)
            require_runtime_log(result["output"], "matching")
            stats = database_stats(database, expected)
            if stats["verified_pairs"] < 1:
                raise PipelineError("STOP_MAPPING_INVALID", "No verified two-view geometry")
            state["stages"]["matching"] = {**stats, "runtime": {k: v for k, v in result.items() if k != "output"}, "gpu_verified": True}
            write_json(state_path, state)
        gpu["matching_runtime"] = state["stages"]["matching"]["runtime"]
        metrics["database"] = database_stats(database, expected)
        write_json(output / "gpu_report.json", gpu)
        if not environment["gpu_ba"].get("verified"):
            raise PipelineError("BLOCKED_GPU_BA", environment["gpu_ba"]["reason"])
        if "mapping" in state["stages"]:
            record = state["stages"]["mapping"]
            for relative, checksum in record["artifact_hashes"].items():
                path = output / relative
                if not path.is_file() or sha256(path) != checksum:
                    raise PipelineError("FAILED", f"Mapped artifact changed/missing: {relative}")
            model = read_model(output / record["model_text"])
            metrics.update(model_metrics(model, len(expected)))
            validate_ply(output / record["ply"], metrics["points3d_count"])
        else:
            sparse = output / ("sparse_" + uuid.uuid4().hex[:8])
            sparse.mkdir()
            command = mapper_command(config, capabilities, database, selected, sparse, environment["gpu_ba"])
            result = execute_stage("mapping", command)
            lower = result["output"].lower()
            if "caspar" not in lower or "fallback" in lower or "falling back" in lower or "creating ceres bundle adjuster" in lower:
                raise PipelineError("BLOCKED_GPU_BA", "Mapper log did not confirm CASPAR without CPU fallback")
            models = []
            for candidate in sorted(sparse.iterdir()):
                if not candidate.is_dir() or not all((candidate / (name + ".bin")).is_file() for name in ("cameras", "images", "points3D")):
                    continue
                text = candidate / "txt"
                text.mkdir()
                execute_stage("convert_txt", [capabilities["binary"], "model_converter", "--input_path", str(candidate), "--output_path", str(text), "--output_type", "TXT"])
                try:
                    model = read_model(text)
                    candidate_metrics = model_metrics(model, len(expected))
                    models.append((candidate_metrics["registered_images"], candidate, text, model, candidate_metrics))
                except (ValueError, OSError, KeyError) as exc:
                    LOGGER.warning("Invalid model %s: %s", candidate.name, exc)
            if not models:
                raise PipelineError("STOP_MAPPING_INVALID", "Mapper returned no valid reconstruction")
            _, best, text, model, measured = max(models, key=lambda item: item[0])
            if measured["registered_images"] < config["sfm"]["min_registered_images"]:
                raise PipelineError("STOP_MAPPING_INVALID", "Insufficient registered images")
            if measured["mean_reprojection_error_px"] > config["sfm"].get("max_mean_reprojection_error_px", 4.0):
                raise PipelineError("STOP_MAPPING_INVALID", "Mean reprojection error exceeds configured quality gate")
            if not {i["name"] for i in model["images"].values()}.issubset(set(expected)):
                raise PipelineError("STOP_MAPPING_INVALID", "Mapped images are not selected video frames")
            ply = best / "sparse.ply"
            execute_stage("convert_ply", [capabilities["binary"], "model_converter", "--input_path", str(best), "--output_path", str(ply), "--output_type", "PLY"])
            validate_ply(ply, measured["points3d_count"])
            export_poses(model, best / "camera_poses.json")
            artifacts = [p for p in best.rglob("*") if p.is_file()]
            state["stages"]["mapping"] = {"model_text": text.relative_to(output).as_posix(), "ply": ply.relative_to(output).as_posix(),
                                            "artifact_hashes": {p.relative_to(output).as_posix(): sha256(p) for p in artifacts},
                                            "runtime": {k: v for k, v in result.items() if k != "output"}}
            write_json(state_path, state)
            metrics.update(measured)
        metrics["status"] = "SUCCESS"
    except (PipelineError, ValueError, OSError, KeyError) as exc:
        metrics["status"] = getattr(exc, "status", "FAILED")
        metrics["error"] = str(exc)
    finally:
        metrics["runtime_seconds"] = {name: stage["runtime"]["runtime_seconds"] for name, stage in state["stages"].items()}
        metrics["runtime_seconds"]["this_invocation"] = time.perf_counter() - started
        metrics["gpu"] = gpu
        write_json(output / "gpu_report.json", gpu)
        write_json(output / "metrics.json", metrics)
    return metrics
