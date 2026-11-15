"""Qualification, transactional preparation and content-verified resume."""

from __future__ import annotations

import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.common.config import ROOT, repo_path
from src.common.io import (
    PipelineError,
    fingerprint,
    read_json,
    sha256,
    staging_directory,
    write_json,
)
from src.data.colmap import (
    discover_models,
    read_model,
    semantic_hashes,
    validate_trivial_rigs,
    write_model,
)
from src.data.geometry import camera_centers, intrinsics, normalization, pose
from src.data.image_processing import prepare_images
from src.data.splits import make_split
from src.data.training_data import derive_training_model, write_point_cloud
from src.data.validation import filter_points, validate_model

LOG = logging.getLogger(__name__)


def implementation_hash() -> str:
    data_files = sorted(Path(__file__).parent.glob("*.py"))
    dependencies = [
        ROOT / "src/common/io.py",
        ROOT / "src/common/config.py",
        ROOT / "src/geometry/sfm/model_io.py",
    ]
    return fingerprint(
        {p.relative_to(ROOT).as_posix(): sha256(p) for p in data_files + dependencies}
    )


def model_file_hashes(directory: Path) -> dict:
    return (
        {
            p.name: sha256(p)
            for p in sorted(directory.iterdir())
            if p.is_file()
            and (
                p.name == "project.ini"
                or p.stem in ("cameras", "images", "points3D", "rigs", "frames")
                and p.suffix in (".bin", ".txt")
            )
        }
        if directory.is_dir()
        else {}
    )


def qualify(config: dict, root: Path = ROOT) -> tuple[object, dict]:
    source = config["source"]
    directory = repo_path(root, source["model_dir"])
    report = {
        "status": "BLOCKED_INPUT",
        "source_mode": source["mode"],
        "model_dir": source["model_dir"],
        "image_root": source["image_root"],
        "ba_history": source["ba_history"],
        "source_hashes": model_file_hashes(directory),
        "config": config,
        "implementation_hash": implementation_hash(),
    }
    try:
        if any(
            part.lower() in ("audits", "synthetic_input", "input_roundtrip")
            for part in directory.parts
        ):
            raise ValueError(
                "Diagnostic/probe models cannot qualify as real scene inputs"
            )
        model = read_model(directory, source["format"])
    except FileNotFoundError as exc:
        report["reason"] = str(exc)
        return None, report
    report["rigs"] = validate_trivial_rigs(directory, model)
    semantics = semantic_hashes(model)
    comparisons = []
    # Both encodings in one directory must agree; explicit paired TXT export too.
    if all(
        (directory / f"{stem}.{ext}").is_file()
        for stem in ("cameras", "images", "points3D")
        for ext in ("bin", "txt")
    ):
        comparisons.append((directory, "txt" if source["format"] != "txt" else "bin"))
    if source["comparison_model_dir"]:
        comparisons.append((repo_path(root, source["comparison_model_dir"]), "auto"))
    for other_path, fmt in comparisons:
        other = read_model(other_path, fmt)
        if semantic_hashes(other) != semantics:
            raise ValueError(
                f"Authoritative BIN/TXT disagreement at {other_path}; no automatic selection"
            )
        validate_trivial_rigs(other_path, other)
    report["comparison"] = {
        "status": "equal" if comparisons else "single_authoritative_format",
        "semantic_hashes": semantics,
        "files": {
            p.relative_to(root).as_posix(): model_file_hashes(p) for p, _ in comparisons
        },
    }
    LOG.info("Validating %d images / %d points", len(model.images), len(model.points))
    original = validate_model(
        model, repo_path(root, source["image_root"]), config["validation"]
    )
    report["original"] = original
    rejected = {"duplicate_tracks": [], "point_error": []}
    if config["filter"]["duplicate_tracks"] == "drop_track":
        rejected["duplicate_tracks"] = original["duplicate_track_point_ids"]
    limit = config["filter"]["max_point_error_px"]
    if limit is not None:
        rejected["point_error"] = [
            p.id for p in model.points.values() if p.error > limit
        ]
    # Structural corruption is not an outlier-filtering problem.
    if set(original["error_counts"]) - {
        "duplicate_image_tracks",
        "reprojection_threshold",
        "image_support",
    }:
        report["status"] = "FAILED"
        report["reason"] = "Source has structural validation errors"
        return None, report
    removed = set(rejected["duplicate_tracks"]) | set(rejected["point_error"])
    accepted = filter_points(model, removed) if removed else model
    report["filter"] = {
        "policy": config["filter"],
        "rejected_ids": rejected,
        "removed_points": len(removed),
    }
    report["qualification"] = (
        validate_model(
            accepted, repo_path(root, source["image_root"]), config["validation"]
        )
        if removed
        else original
    )
    report["status"] = report["qualification"]["status"]
    if report["status"] != "SUCCESS":
        report["reason"] = "Source did not pass qualification; see error_counts"
        return None, report
    return accepted, report


def audit_sources(config: dict, root: Path = ROOT) -> dict:
    """Inspect every candidate explicitly; probe/fixture paths are diagnostic only."""
    records = []
    for value in config["source"]["discovery_roots"]:
        for candidate in discover_models(repo_path(root, value)):
            path = Path(candidate["path"])
            candidate["path"] = path.relative_to(root).as_posix()
            candidate["diagnostic_only"] = any(
                x in path.parts for x in ("audits", "fixtures", "fixture")
            )
            try:
                model = read_model(path)
                image_root = repo_path(root, config["source"]["image_root"])
                if candidate["diagnostic_only"]:
                    candidate.update(
                        status="DIAGNOSTIC_NOT_SCENE",
                        images=len(model.images),
                        points=len(model.points),
                    )
                else:
                    candidate["validation"] = validate_model(
                        model, image_root, config["validation"]
                    )
                    candidate["rigs"] = validate_trivial_rigs(path, model)
            except (OSError, ValueError) as exc:
                candidate.update(status="FAILED", reason=str(exc))
            records.append(candidate)
    return {
        "models": records,
        "selection": "Only source.model_dir is selected; no fallback",
    }


def source_frames(config: dict, root: Path) -> tuple[dict, dict | None]:
    manifest = config["source"]["manifest"]
    if not manifest:
        if config["source"]["mode"] == "video":
            raise ValueError(
                "Video source requires manifest with PTS/timestamps and image hashes"
            )
        return {}, None
    path = repo_path(root, manifest)
    data = read_json(path)
    records = {}
    for row in data["frames"]:
        name = row["filename"]
        if name in records:
            raise ValueError(f"Duplicate source-manifest filename: {name}")
        records[name] = row
    return records, {
        "path": manifest,
        "sha256": sha256(path),
        "video_sha256": data.get("source_sha256"),
    }


def prepare(
    config: dict, root: Path = ROOT, *, resume: bool = False, dry_run: bool = False
) -> dict:
    started = time.perf_counter()
    model, qualification = qualify(config, root)
    if model is None:
        raise PipelineError(
            qualification["status"], qualification.get("reason", "Qualification failed")
        )
    frames, manifest_ref = source_frames(config, root)
    image_records = qualification["qualification"]["images"]
    if config["source"]["mode"] == "video":
        for im in model.images.values():
            frame = frames.get(im.name)
            if not frame or frame.get("sha256") != image_records[im.id]["sha256"]:
                raise ValueError(f"Video manifest/image hash mismatch: {im.name}")
    source = {
        "mode": config["source"]["mode"],
        "model_dir": config["source"]["model_dir"],
        "image_root": config["source"]["image_root"],
        "model_hashes": qualification["source_hashes"],
        "comparison": qualification["comparison"],
        "manifest": manifest_ref,
        "ba_history": config["source"]["ba_history"],
        "image_hashes": {str(k): v["sha256"] for k, v in image_records.items()},
        "ground_truth_used": False,
    }
    cache_key = fingerprint(
        {
            "source": source,
            "config": config,
            "implementation_hash": implementation_hash(),
        }
    )
    dataset_id = config["dataset_name"] + "-" + cache_key[:16]
    destination = repo_path(root, config["output"]["dataset_root"]) / dataset_id
    LOG.info("Dataset destination: %s", destination)
    if destination.exists():
        if not resume:
            raise FileExistsError(
                f"Existing dataset; use --resume to verify without overwriting: {destination}"
            )
        from src.data.loader import verify_dataset

        result = verify_dataset(destination)
        if (
            result["status"] != "SUCCESS"
            or read_json(destination / "metadata.json")["cache_key"] != cache_key
        ):
            raise ValueError(
                "Resume rejected: dataset content/config/code fingerprint mismatch"
            )
        return {
            "status": "SUCCESS",
            "dataset_path": destination.relative_to(root).as_posix(),
            "resumed": True,
            "readiness": result,
        }
    split = make_split(model, config["split"], config["source"]["mode"], frames)
    centers = camera_centers(model)
    normal = normalization(
        np.array([centers[i] for i in split["image_ids"]["train"]]),
        config["normalization"]["enabled"],
    )
    if dry_run:
        return {
            "status": "DRY_RUN",
            "dataset_path": destination.relative_to(root).as_posix(),
            "qualified": True,
            "cache_key": cache_key,
            "split_counts": {k: len(v) for k, v in split["image_ids"].items()},
        }
    cached_directory = None
    if resume:
        candidates = []
        for previous in destination.parent.glob(f".{destination.name}.partial-*"):
            if (previous / "state.json").is_file() and read_json(
                previous / "state.json"
            ).get("cache_key") == cache_key:
                candidates.append(previous)
        if candidates:
            cached_directory = max(candidates, key=lambda p: p.stat().st_mtime_ns)
            LOG.info(
                "Reusing verified per-image artifacts from %s; failed run stays intact",
                cached_directory,
            )
    stage = staging_directory(destination)
    write_json(
        stage / "state.json",
        {"status": "IN_PROGRESS", "cache_key": cache_key, "config": config},
    )
    try:
        LOG.info("Preparing image copies/undistortion one image at a time")
        prepared, transforms = prepare_images(
            model,
            repo_path(root, config["source"]["image_root"]),
            stage,
            config["images"],
            config["output"]["file_mode"],
            cached_directory=cached_directory,
        )
        outside = set()
        outside_observations = 0
        for im in prepared.images.values():
            c = prepared.cameras[im.camera_id]
            bad = (
                (im.xys[:, 0] < 0)
                | (im.xys[:, 0] > c.width)
                | (im.xys[:, 1] < 0)
                | (im.xys[:, 1] > c.height)
            )
            bad &= im.point3D_ids >= 0
            outside_observations += int(bad.sum())
            outside.update(int(pid) for pid in im.point3D_ids[bad])
        if outside and config["filter"]["outside_image_tracks"] == "reject":
            raise ValueError(
                f"{len(outside)} point tracks contain {outside_observations} observations outside transformed image; explicit derived drop_track policy required"
            )
        transformed_filter = {
            "outside_image_point_ids": sorted(outside),
            "outside_image_observations": outside_observations,
            "policy": config["filter"]["outside_image_tracks"],
        }
        if outside:
            prepared = filter_points(prepared, outside)
        quality = validate_model(prepared, stage / "images", config["validation"])
        if quality["status"] != "SUCCESS":
            raise ValueError(
                f"Transformed dataset validation failed: {quality['error_counts']}"
            )
        train, observations, rejected = derive_training_model(
            prepared, stage / "images", split, config["training_points"]
        )
        train_limits = {**config["validation"], "min_observations_per_image": 0}
        training_quality = validate_model(train, stage / "images", train_limits)
        if training_quality["status"] != "SUCCESS":
            raise ValueError(
                f"Train-only model validation failed: {training_quality['error_counts']}"
            )
        if any(
            training_quality["statistics"]["per_image_support"][i]
            < config["validation"]["min_observations_per_image"]
            for i in split["image_ids"]["train"]
        ):
            raise ValueError(
                "A train image has insufficient accepted training-point support"
            )
        write_model(prepared, stage / "qualified_model")
        write_model(train, stage / "sparse/0")
        write_point_cloud(train, stage / "sparse/0/points3D.ply")
        # PGSR official snapshot reads sparse/ directly. Same bytes, no second geometry.
        from src.data.image_processing import transfer

        for name in ("cameras.bin", "images.bin", "points3D.bin", "points3D.ply"):
            transfer(stage / "sparse/0" / name, stage / "sparse" / name, "copy")
        np.savez_compressed(stage / "train_observations.npz", **observations)
        write_json(stage / "splits.json", split)
        write_json(stage / "normalization.json", normal)
        write_json(stage / "qualification.json", qualification)
        image_metadata = []
        for im in prepared.images.values():
            t = pose(im.qvec, im.tvec)
            c = prepared.cameras[im.camera_id]
            record = transforms[im.id]
            frame = frames.get(record["source_name"], {})
            image_metadata.append(
                {
                    "image_id": im.id,
                    "camera_id": im.camera_id,
                    "path": "images/" + im.name,
                    "name": im.name,
                    "sha256": sha256(stage / "images" / im.name),
                    "width": c.width,
                    "height": c.height,
                    "K": intrinsics(c).tolist(),
                    "Tcw": t.tolist(),
                    "Twc": np.linalg.inv(t).tolist(),
                    "frame_id": frame.get("frame_id"),
                    "timestamp_sec": frame.get("timestamp_sec"),
                    "source_frame": frame or None,
                    **record,
                }
            )
        metadata = {
            "schema_version": "prepared-v1",
            "dataset_id": dataset_id,
            "cache_key": cache_key,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "display_timezone": "Asia/Saigon",
            "source": source,
            "config": config,
            "implementation_hash": implementation_hash(),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.system(),
            },
            "command": sys.argv,
            "pose_convention": "COLMAP Hamilton WXYZ; world-to-camera; x right/y down/z forward",
            "units": "arbitrary monocular SfM units; no GT alignment",
            "geometry_transform_applied": False,
            "normalization": normal,
            "split_hash": split["hash"],
            "images": image_metadata,
            "statistics": quality["statistics"],
            "training_statistics": training_quality["statistics"],
            "training_point_filter": rejected,
            "transformed_point_filter": transformed_filter,
            "initialization": {
                "points": len(train.points),
                "colors": "mean bilinear train RGB only",
                "errors": "recomputed mean train-only pixel residual",
                "track_support": "train image IDs only",
                "scales": "Phase02 native CUDA nearest-neighbor distances; not initialized in Phase01",
                "sh": "Phase02 RGB2SH DC=(RGB/255-0.5)/0.28209479177387814; higher coefficients zero per pinned baseline",
            },
            "native_loader_policy": "Use project manifest adapter; native implicit splits do not implement project val/test contract",
            "runtime_seconds": time.perf_counter() - started,
        }
        # Verify source model and every registered source image after preparation.
        if (
            model_file_hashes(repo_path(root, config["source"]["model_dir"]))
            != source["model_hashes"]
        ):
            raise ValueError("Protected model changed during preparation")
        for im in model.images.values():
            if (
                sha256(repo_path(root, config["source"]["image_root"]) / im.name)
                != source["image_hashes"][str(im.id)]
            ):
                raise ValueError(f"Protected source image changed: {im.name}")
        metadata["source_integrity_verified"] = True
        write_json(stage / "metadata.json", metadata)
        artifacts = {
            p.relative_to(stage).as_posix(): sha256(p)
            for p in sorted(stage.rglob("*"))
            if p.is_file() and p.name != "state.json"
        }
        write_json(stage / "integrity.json", artifacts)
        from src.data.loader import verify_dataset

        readiness = verify_dataset(stage)
        write_json(stage / "readiness.json", readiness)
        if readiness["status"] != "SUCCESS":
            raise ValueError(f"Readiness failed: {readiness['errors']}")
        write_json(
            stage / "state.json",
            {"status": "SUCCESS", "cache_key": cache_key, "dataset_id": dataset_id},
        )
        stage.rename(destination)
    except Exception as exc:
        write_json(
            stage / "state.json",
            {
                "status": "FAILED",
                "reason": str(exc),
                "cache_key": cache_key,
                "partial_output": True,
            },
        )
        raise
    return {
        "status": "SUCCESS",
        "dataset_path": destination.relative_to(root).as_posix(),
        "resumed": False,
        "readiness": readiness,
    }
