"""Package a published full checkpoint and exact prepared dataset; no training."""

from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import zipfile
from workflow import (
    ROOT,
    copy_file,
    newest_checkpoint,
    read_json,
    seal,
    sha256,
    write_json,
)


def prepare(run: Path, destination: Path):
    if destination.exists():
        raise FileExistsError(destination)
    config = read_json(run / "resolved_config.json")
    checkpoint, validation = newest_checkpoint(run, config)
    dataset = ROOT / config["dataset"]
    from src.data.loader import verify_dataset

    readiness = verify_dataset(dataset)
    if readiness["status"] != "SUCCESS":
        raise ValueError(readiness)
    from src.gaussian_splatting.runner import code_identity
    from src.gaussian_splatting.native import REVISION

    identity = validation["identity"]
    if identity["code"] != code_identity() or identity["native_revision"] != REVISION:
        raise ValueError("Current source does not match saved checkpoint")
    if (
        identity["metadata"] != sha256(dataset / "metadata.json")
        or identity["split"] != read_json(dataset / "splits.json")["hash"]
    ):
        raise ValueError("Dataset/checkpoint mismatch")
    payload = (
        __import__("workflow")
        .torch_module()
        .load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    )
    if not set(payload["sampler_remaining_ids"]) <= set(
        read_json(dataset / "splits.json")["image_ids"]["train"]
    ):
        raise ValueError("Sampler contains held-out IDs")
    destination.mkdir(parents=True)
    hashes = read_json(dataset / "integrity.json")
    for rel in [*hashes, "integrity.json"]:
        from workflow import inside

        copy_file(inside(dataset, rel), destination / "scene" / rel)
    target = destination / "checkpoint" / checkpoint.name
    copy_file(checkpoint, target)
    if sha256(target) != validation["sha256"]:
        raise ValueError("Checkpoint copy changed")
    write_json(destination / "config/train_config.json", config)
    copy_file(dataset / "splits.json", destination / "config/split_manifest.json")
    copy_file(
        ROOT / "configs/training/dependency_lock.json",
        destination / "config/dependency_lock.json",
    )
    # Source capsule excludes paper, outputs, external binaries, secrets and raw data.
    source_paths = []
    for folder in ("src", "scripts/kaggle", "configs/training"):
        source_paths += [
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file()
            and p.suffix in (".py", ".json", ".yaml", ".sh")
            and "__pycache__" not in p.parts
        ]
    with zipfile.ZipFile(destination / "source.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(source_paths):
            z.write(p, p.relative_to(ROOT).as_posix())
    manifest = seal(
        destination,
        {
            "schema": "phase02-kaggle-input-v1",
            "parent_run": run.name,
            "source_base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_capsule": "source.zip",
            "source_code_hash": code_identity(),
            "checkpoint": "checkpoint/" + checkpoint.name,
            "checkpoint_validation": validation,
            "dataset_id": readiness["dataset_id"],
            "dataset_metadata_sha256": identity["metadata"],
            "split_hash": identity["split"],
            "trusted_project_checkpoint_only": True,
        },
    )
    write_json(
        destination.parent / (destination.name + "_verification.json"),
        {
            "status": "SUCCESS",
            "bundle": str(destination),
            "manifest_sha256": sha256(destination / "manifest.json"),
            "checkpoint": validation,
            "dataset": readiness,
            "kaggle_executed": False,
        },
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.run.resolve(), args.destination.resolve())
    print(
        {
            "status": "SUCCESS",
            "iteration": result["checkpoint_validation"]["iteration"],
            "files": len(result["files"]),
        }
    )
