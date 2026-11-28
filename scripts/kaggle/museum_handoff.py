"""Build and import isolated Museum handoffs from verified inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import yaml

from src.common.config import ROOT, load_config, repo_path
from src.common.io import (
    PipelineError, read_json, sha256, staging_directory, write_json,
)
from src.data.config import load_config as load_data_config
from src.data.loader import verify_dataset
from src.data.preparation import prepare

VIDEO = "datasets/tanks_and_temples/museum/Museum.mp4"
VIDEO_SHA256 = "f407eca1d989587c5c350fce42156c235b6e17dedc08c1013fa739d1088d83f2"
OUTPUTS = "outputs/tanks_and_temples/Museum"
METHODS = {"B0": "B0_3DGS", "B2": "B2_PGSR"}
MODELS = {"B0": "B0_Original3DGS", "B2": "B2_PGSR"}


def verified_sfm(path: Path) -> dict:
    metrics = read_json(path)
    ba = metrics.get("gpu", {}).get("ba", {})
    if (
        metrics.get("status") != "SUCCESS"
        or metrics.get("source_video_sha256") != VIDEO_SHA256
        or ba.get("status") != "SUCCESS"
        or ba.get("backend") != "CASPAR"
        or ba.get("verified") is not True
    ):
        raise PipelineError("BLOCKED_GPU_BA", metrics.get("error", "Museum GPU BA evidence is missing"))
    return metrics


def qualify_duplicate_tracks(sfm: dict, output: Path, config: dict) -> tuple[Path, Path]:
    """Qualify a derived copy of a successful GPU solve rejected for duplicate tracks."""
    from src.data.colmap import read_model, validate_trivial_rigs, write_model
    from src.data.training_data import write_point_cloud
    from src.data.validation import filter_points, validate_model
    from src.geometry.sfm.metrics import model_metrics
    from src.geometry.sfm.model_io import export_poses, read_model as read_sfm_model, validate_ply

    metrics_path = output / "metrics.json"
    metrics = read_json(metrics_path)
    ba = metrics["gpu"]["ba"]
    sfm_status = metrics["status"]
    if (
        sfm_status != "STOP_MAPPING_INVALID"
        or metrics["source_video_sha256"] != VIDEO_SHA256
        or ba.get("verified") is not True
        or ba.get("backend") != "CASPAR"
        or config["filter"]["duplicate_tracks"] != "drop_track"
    ):
        raise PipelineError(sfm_status, "No qualified Museum reconstruction or approved track filter")
    commands = read_json(output / "commands.json")
    [mapper] = [entry for entry in commands if entry["command"][1] == "mapper"]
    [converter] = [
        entry for entry in commands
        if entry["command"][1] == "model_converter" and "TXT" in entry["command"]
    ]
    if mapper["returncode"] != 0 or converter["returncode"] != 0:
        raise ValueError("Museum mapper or TXT conversion did not complete")

    def argument(entry: dict, option: str) -> str:
        command = entry["command"]
        return command[command.index("--" + option) + 1]

    mapper_log = output / mapper["log"]
    log = mapper_log.read_text(encoding="utf-8").lower()
    if (
        "caspar solver setup" not in log
        or any(s in log for s in ("fallback", "falling back", "creating ceres bundle adjuster"))
        or argument(mapper, "Mapper.ba_local_backend") != "CASPAR"
        or argument(mapper, "Mapper.ba_global_backend") != "CASPAR"
        or sha256(Path(mapper["command"][0])) != ba["colmap_binary_sha256"]
    ):
        raise PipelineError("BLOCKED_GPU_BA", "Mapper log or binary lacks exclusive CASPAR evidence")
    raw = Path(argument(converter, "output_path")).resolve()
    if not raw.is_relative_to(output) or raw.parent != Path(argument(converter, "input_path")).resolve():
        raise ValueError("Unexpected original Museum model path")
    if not raw.is_relative_to(Path(argument(mapper, "output_path")).resolve()):
        raise ValueError("TXT conversion does not descend from the verified mapper")
    model = read_model(raw, "txt")
    validate_trivial_rigs(raw, model)
    selected = repo_path(ROOT, sfm["paths"]["selected_frames"])
    frames = read_json(selected / "manifest.json")
    if (
        frames["source_sha256"] != VIDEO_SHA256
        or frames["cache_key"] != metrics["selected_cache_key"]
        or not {image.name for image in model.images.values()} <= {row["filename"] for row in frames["frames"]}
    ):
        raise ValueError("Mapped images differ from verified Museum frames")
    before = validate_model(model, selected, config["validation"])
    removed = set(before["duplicate_track_point_ids"])
    if not removed or set(before["error_counts"]) != {"duplicate_image_tracks"}:
        raise ValueError(f"Cannot recover mapper model with errors {before['error_counts']}")
    artifact_hashes = {name: sha256(raw / name) for name in ("cameras.txt", "images.txt", "points3D.txt")}
    accepted = filter_points(model, removed)
    after = validate_model(accepted, selected, config["validation"])
    if after["status"] != "SUCCESS":
        raise ValueError(f"Derived track qualification failed: {after['error_counts']}")
    destination = output / "qualified_duplicate_tracks_v1"
    staging = staging_directory(destination)
    write_model(accepted, staging / "sparse", "txt")
    write_model(accepted, staging / "sparse", "bin")
    exported = read_sfm_model(staging / "sparse")
    measured = model_metrics(exported, len(frames["frames"]))
    if (
        measured["registered_images"] < sfm["sfm"]["min_registered_images"]
        or measured["mean_reprojection_error_px"] > sfm["sfm"].get("max_mean_reprojection_error_px", 4.0)
    ):
        raise ValueError("Derived tracks failed SfM registration or reprojection thresholds")
    write_point_cloud(accepted, staging / "sparse.ply")
    validate_ply(staging / "sparse.ply", measured["points3d_count"])
    export_poses(exported, staging / "camera_poses.json")
    provenance = {
        "source_model": raw.relative_to(ROOT).as_posix(), "original_hashes": artifact_hashes,
        "source_metrics_sha256": sha256(metrics_path), "mapper_log_sha256": sha256(mapper_log),
        "original_status": sfm_status, "original_points": len(model.points),
        "removed_duplicate_track_ids": sorted(removed),
        "policy": "Drop whole tracks with repeated image IDs, with no change to camera poses or retained XYZ",
        "derived_model_hashes": {
            path.name: sha256(path) for path in sorted((staging / "sparse").iterdir())
        },
    }
    write_json(staging / "metrics.json", {
        **metrics, **measured, "status": "SUCCESS", "error": None,
        "qualification": provenance, "mapper_runtime_seconds": mapper["runtime_seconds"],
    })
    staging.rename(destination)
    return destination / "sparse", destination / "metrics.json"


def prepare_scene(sfm_config: Path, *, resume: bool = False) -> tuple[Path, Path]:
    """Bind the prepared-data template to the actual validated mapper model."""
    sfm = load_config(sfm_config)
    if sfm["input"]["video"] != VIDEO or sfm["input"]["scene"].lower() != "museum":
        raise ValueError("Requires a Museum video SfM configuration")
    sfm_output = repo_path(ROOT, sfm["paths"]["output"])
    metrics_path = sfm_output / "metrics.json"
    if not metrics_path.is_file():
        raise PipelineError("BLOCKED_INPUT", f"Missing {metrics_path}")
    config = load_data_config(ROOT / "configs/data/museum.yaml")
    if read_json(metrics_path)["status"] == "STOP_MAPPING_INVALID":
        qualified = sfm_output / "qualified_duplicate_tracks_v1"
        if resume and qualified.is_dir():
            metrics_path = qualified / "metrics.json"
            model = qualified / "sparse"
            for name, digest in read_json(metrics_path)["qualification"]["derived_model_hashes"].items():
                if sha256(model / name) != digest:
                    raise ValueError(f"Derived model changed: {name}")
        else:
            model, metrics_path = qualify_duplicate_tracks(sfm, sfm_output, config)
    else:
        verified_sfm(metrics_path)
        state = read_json(sfm_output / "colmap_state.json")
        mapping = state["stages"]["mapping"]
        model = (sfm_output / mapping["model_text"]).resolve()
        if not model.is_relative_to(sfm_output):
            raise ValueError("Mapped model escapes the SfM output")
        for filename in ("cameras.txt", "images.txt", "points3D.txt"):
            artifact = model / filename
            recorded = mapping["artifact_hashes"][artifact.relative_to(sfm_output).as_posix()]
            if sha256(artifact) != recorded:
                raise ValueError(f"Mapped model file changed: {artifact}")
    verified_sfm(metrics_path)
    manifest = repo_path(ROOT, sfm["paths"]["selected_frames"]) / "manifest.json"
    if read_json(manifest)["source_sha256"] != VIDEO_SHA256:
        raise ValueError("Selected frames belong to another source video")
    config["source"].update(
        image_root=sfm["paths"]["selected_frames"],
        model_dir=model.relative_to(ROOT).as_posix(),
        format="txt",
        manifest=manifest.relative_to(ROOT).as_posix(),
        ba_history="Verified CUDA/CASPAR GPU-exclusive Museum video SfM",
        discovery_roots=[sfm["paths"]["output"]],
    )
    config["output"]["report_root"] = (
        f"{OUTPUTS}/data_preparation/{sfm_output.name}"
    )
    report = prepare(config, resume=resume)
    dataset = repo_path(ROOT, report["dataset_path"])
    report_dir = repo_path(ROOT, config["output"]["report_root"])
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / f"{dataset.name}.json", {"config": config, **report})
    return dataset, metrics_path


def validate_scene(dataset: Path, metrics: Path) -> dict:
    verified_sfm(metrics)
    readiness = verify_dataset(dataset)
    if readiness["status"] != "SUCCESS":
        raise ValueError(readiness)
    metadata = read_json(dataset / "metadata.json")
    if (
        metadata["dataset_id"] != dataset.name
        or not dataset.name.startswith("museum-video-v1-")
        or metadata["source"]["mode"] != "video"
        or metadata["source"]["manifest"]["video_sha256"] != VIDEO_SHA256
    ):
        raise ValueError("Prepared scene lacks Museum video provenance")
    return {
        "dataset_id": dataset.name,
        "video_sha256": VIDEO_SHA256,
        "split_hash": metadata["split_hash"],
        "integrity_sha256": sha256(dataset / "integrity.json"),
        "sfm_metrics_sha256": sha256(metrics),
    }


def build_scene(dataset: Path, metrics: Path, destination: Path) -> dict:
    receipt = validate_scene(dataset, metrics)
    staging = staging_directory(destination)
    scene = staging / "scene" / dataset.name
    # Copy only the verified logical dataset, dereferencing declared image links.
    for name in [*read_json(dataset / "integrity.json"), "integrity.json"]:
        target = scene / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dataset / name, target)
    shutil.copyfile(metrics, staging / "metrics.json")
    write_json(staging / "scene.json", receipt)
    if validate_scene(scene, staging / "metrics.json") != receipt:
        raise ValueError("Prepared Museum package changed during copy")
    staging.rename(destination)
    return {"status": "SUCCESS", **receipt, "scene_receipt_sha256": sha256(destination / "scene.json")}


def build_source(destination: Path) -> dict:
    staging = staging_directory(destination)
    archive = staging / "museum_source.bundle"
    suffixes = {".py", ".sh", ".yaml", ".yml", ".json"}
    files = sorted(
        path
        for directory in ("src", "configs", "scripts")
        for path in (ROOT / directory).rglob("*")
        if path.suffix in suffixes and path.is_file() and not path.is_symlink()
    )
    hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in files}
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as zip_file:
        for path in files:
            zip_file.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as zip_file:
        if set(zip_file.namelist()) != set(hashes):
            raise ValueError("Invalid Museum capsule")
        for name, digest in hashes.items():
            if hashlib.sha256(zip_file.read(name)).hexdigest() != digest:
                raise ValueError(f"Source changed during packaging: {name}")
    manifest = {
        "schema": "museum-kaggle-source-v1",
        "status": "SUCCESS",
        "archive": archive.name,
        "source_sha256": sha256(archive),
        "files": hashes,
        "prepared_scene_included": False,
    }
    write_json(staging / "source_manifest.json", manifest)
    staging.rename(destination)
    return {key: value for key, value in manifest.items() if key != "files"}


def pack_scene(source: Path, destination: Path) -> dict:
    receipt = read_json(source / "scene.json")
    dataset = source / "scene" / receipt["dataset_id"]
    if not dataset.resolve().is_relative_to((source / "scene").resolve()):
        raise ValueError("Invalid prepared scene path")
    if validate_scene(dataset, source / "metrics.json") != receipt:
        raise ValueError("Prepared scene receipt differs")
    staging = staging_directory(destination)
    archive = staging / "museum_scene.bundle"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as packed:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                packed.write(path, path.relative_to(source).as_posix())
    with zipfile.ZipFile(archive) as packed:
        if packed.testzip() is not None:
            raise ValueError("Prepared scene archive failed CRC verification")
    result = {
        "status": "SUCCESS", "archive": archive.name,
        "scene_archive_sha256": sha256(archive),
        "scene_receipt_sha256": sha256(source / "scene.json"),
    }
    write_json(staging / "scene_manifest.json", result)
    staging.rename(destination)
    return result


def bind_profile(dataset: Path, method: str, profile: str, destination: Path) -> dict:
    if profile not in {"smoke", "low_memory", "research_quality"}:
        raise ValueError("Unknown Museum resource profile")
    relative = dataset.resolve().relative_to(ROOT)
    if relative.parts[:6] != ("datasets", "tanks_and_temples", "museum", "prepared", "video_v1", dataset.name):
        raise ValueError("Expected a canonical prepared Museum dataset")
    metadata = read_json(dataset / "metadata.json")
    if metadata["source"]["manifest"]["video_sha256"] != VIDEO_SHA256:
        raise ValueError("Unexpected training video")
    name = {"B0": "b0", "B2": "pgsr"}[method]
    config = yaml.safe_load((ROOT / f"configs/training/museum/{name}_{profile}.yaml").read_text(encoding="utf-8"))
    config["dataset"] = relative.as_posix()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return config


def import_run(bundle: Path) -> dict:
    receipt = read_json(bundle / "scene.json")
    dataset_id = receipt["dataset_id"]
    if Path(dataset_id).name != dataset_id:
        raise ValueError("Unsafe scene path")
    dataset = bundle / "scene" / dataset_id
    if validate_scene(dataset, bundle / "metrics.json") != receipt:
        raise ValueError("Scene receipt mismatch")
    inputs = read_json(bundle / "training_input.json")
    if (
        inputs["scene_receipt_sha256"] != sha256(bundle / "scene.json")
        or inputs["video_sha256"] != VIDEO_SHA256
    ):
        raise ValueError("Training input differs from the attached Museum scene")
    method = inputs["method"]
    run = bundle / "run"
    config = read_json(run / "resolved_config.json")
    identity = read_json(run / "run_manifest.json").get("identity", {})
    if (
        config["method"] != MODELS[method]
        or config["dataset"] != f"datasets/tanks_and_temples/museum/prepared/video_v1/{dataset_id}"
        or (identity and identity.get("metadata") != sha256(dataset / "metadata.json"))
    ):
        raise ValueError("Model/data identity mismatch")
    pointer_path = run / "checkpoints/latest.json"
    pointer = read_json(pointer_path) if pointer_path.is_file() else None
    if pointer:
        name = pointer["path"]
        if Path(name).name != name or sha256(run / "checkpoints" / name) != pointer["sha256"]:
            raise ValueError("Checkpoint integrity mismatch")
    run_id = inputs["run_id"]
    if not run_id.startswith("museum-") or Path(run_id).name != run_id:
        raise ValueError("Invalid Museum run ID")
    destination = ROOT / OUTPUTS / "research" / METHODS[method] / run_id
    if destination.exists():
        raise FileExistsError(destination)
    canonical = repo_path(ROOT, config["dataset"])
    if not canonical.exists():
        dataset_stage = staging_directory(canonical)
        shutil.copytree(dataset, dataset_stage, dirs_exist_ok=True)
        dataset_stage.rename(canonical)
    if validate_scene(canonical, bundle / "metrics.json") != receipt:
        raise ValueError("Local prepared scene differs from the run input")
    staging = staging_directory(destination)
    for name in ("scene.json", "training_input.json", "metrics.json"):
        shutil.copyfile(bundle / name, staging / name)
    shutil.copytree(run, staging, dirs_exist_ok=True)
    if (bundle / "losses").is_dir():
        shutil.copytree(bundle / "losses", staging / "losses")
    staging.rename(destination)
    return {
        "status": "IMPORTED",
        "run": destination.relative_to(ROOT).as_posix(),
        "method": MODELS[method],
        "iteration": pointer["iteration"] if pointer else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    source = stages.add_parser("source")
    source.add_argument("--destination", type=Path, required=True)
    scene = stages.add_parser("scene")
    scene.add_argument("--sfm-config", type=Path, required=True)
    scene.add_argument("--destination", type=Path, required=True)
    scene.add_argument("--resume", action="store_true")
    pack = stages.add_parser("pack-scene")
    pack.add_argument("source", type=Path)
    pack.add_argument("--destination", type=Path, required=True)
    profile = stages.add_parser("profile")
    profile.add_argument("--dataset", type=Path, required=True)
    profile.add_argument("--method", choices=METHODS, required=True)
    profile.add_argument("--profile", required=True)
    profile.add_argument("--destination", type=Path, required=True)
    importer = stages.add_parser("import-run")
    importer.add_argument("bundle", type=Path)
    args = parser.parse_args()
    try:
        if args.stage == "source":
            report = build_source(args.destination)
        elif args.stage == "scene":
            dataset, metrics = prepare_scene(args.sfm_config, resume=args.resume)
            report = build_scene(dataset, metrics, args.destination)
        elif args.stage == "pack-scene":
            report = pack_scene(args.source, args.destination)
        elif args.stage == "profile":
            report = bind_profile(args.dataset, args.method, args.profile, args.destination)
        else:
            report = import_run(args.bundle)
    except (OSError, ValueError, KeyError, PipelineError) as error:
        print(json.dumps({"status": getattr(error, "status", "FAILED"), "reason": str(error)}))
        return 2
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
