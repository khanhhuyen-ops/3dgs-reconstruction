"""Kaggle stages; importing this module never starts training."""

from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid
from workflow import (
    ROOT,
    fingerprint,
    inspect_checkpoint,
    read_json,
    seal,
    sha256,
    torch_module,
    verify,
    write_json,
)


def prepare(bundle: Path, previous: Path | None = None):
    m = verify(bundle)
    config = read_json(bundle / "config/train_config.json")
    parent = bundle / m["checkpoint"]
    if previous:
        old = verify(previous)
        if old["schema"] != "phase02-kaggle-output-v1" or old[
            "input_manifest_sha256"
        ] != sha256(bundle / "manifest.json"):
            raise ValueError("Previous Kaggle session belongs to another input")
        old_config = read_json(previous / "run/resolved_config.json")
        old_cp = (
            previous
            / "run/checkpoints"
            / read_json(previous / "run/checkpoints/latest.json")["path"]
        )
        validation = inspect_checkpoint(old_cp, old_config)
        local_iteration = m["checkpoint_validation"]["iteration"]
        if validation["iteration"] <= local_iteration:
            raise ValueError("Previous session is not newer than local checkpoint")
        parent, config = old_cp, old_config
    original = deepcopy(config)
    torch = torch_module()
    from src.gaussian_splatting.native import initialize
    from src.gaussian_splatting.runner import code_identity

    environment = initialize()
    if code_identity() != m["source_code_hash"]:
        raise ValueError("Training code differs from source capsule")
    validation = inspect_checkpoint(parent, original)
    if (
        validation["identity"]["code"] != code_identity()
        or validation["identity"]["native_revision"] != environment["native_revision"]
    ):
        raise ValueError("Incompatible training algorithm revision")
    if (
        validation["identity"]["metadata"] != m["dataset_metadata_sha256"]
        or validation["identity"]["split"] != m["split_hash"]
    ):
        raise ValueError("Dataset provenance mismatch")
    dataset = ROOT / config["dataset"]
    if dataset.exists():
        if dataset.resolve() != (bundle / "scene").resolve():
            raise FileExistsError(dataset)
    else:
        dataset.parent.mkdir(parents=True, exist_ok=True)
        dataset.symlink_to((bundle / "scene").resolve(), target_is_directory=True)
    from src.data.loader import verify_dataset

    readiness = verify_dataset(dataset)
    if readiness["status"] != "SUCCESS":
        raise ValueError(readiness)
    # Engineering changes only: preserve every optimization/data setting.
    config["checkpoint_interval"] = 1000
    config["torch_reserved_budget_gib"] = min(
        12.0, environment["total_vram_bytes"] / 1024**3 * 0.8
    )
    config["experiment"] = "b0-kaggle-" + uuid.uuid4().hex[:10]
    allowed = {"checkpoint_interval", "torch_reserved_budget_gib", "experiment"}
    if any(config[k] != original[k] for k in config if k not in allowed):
        raise ValueError("Optimization config changed during migration")
    run = ROOT / config["output_root"] / config["experiment"]
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "resolved_config.json", config)
    import yaml

    (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    data = torch.load(parent, map_location="cpu", weights_only=False, mmap=True)
    old_identity = deepcopy(data["identity"])
    data["identity"] = {
        **old_identity,
        "config": fingerprint(config),
        "code": code_identity(),
        "extension_hashes": environment["extension_hashes"],
    }
    # cuda:0 is the only used device. Other CUDA RNG streams are not part of computation.
    data["rng"]["torch_cuda"] = [
        data["rng"]["torch_cuda"][0]
    ] * torch.cuda.device_count()
    destination = run / "checkpoints" / parent.name
    destination.parent.mkdir()
    torch.save(data, destination)
    write_json(
        destination.parent / "latest.json",
        {
            "path": destination.name,
            "sha256": sha256(destination),
            "iteration": data["iteration"],
        },
    )
    inspect_checkpoint(destination, config, data["identity"])
    write_json(
        run / "migration.json",
        {
            "status": "SUCCESS",
            "parent_sha256": sha256(parent),
            "original_identity": old_identity,
            "effective_identity": data["identity"],
            "config_changes": {
                k: {"before": original[k], "after": config[k]}
                for k in allowed
                if original[k] != config[k]
            },
            "input_manifest_sha256": sha256(bundle / "manifest.json"),
            "parent_iteration": data["iteration"],
            "parent_gaussians": validation["gaussians"],
            "workflow_hashes": {
                p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")
            },
            "timing_scope": "Training duration in this run covers this Kaggle session only; local/earlier session timing is not silently added",
            "rng_note": "GPU0 stream preserved; unused streams replicated to target device count; no bitwise cross-GPU guarantee",
        },
    )
    write_json(run / "environment.json", environment)
    write_json(
        run / "run_manifest.json",
        {"status": "PREPARED_FOR_CUDA_RESTORE", "training_started": False},
    )
    write_json(
        ROOT / "kaggle_active_run.json", {"run": run.relative_to(ROOT).as_posix()}
    )
    del data
    restore_verify(run)
    return run


def restore_verify(run):
    from src.gaussian_splatting import checkpoint
    from src.gaussian_splatting.validation import validate_model
    from scene.gaussian_model import GaussianModel

    config = read_json(run / "resolved_config.json")
    identity = read_json(run / "migration.json")["effective_identity"]
    options = SimpleNamespace(
        iterations=config["iterations"], **config["optimizer"], **config["density"]
    )
    model = GaussianModel(config["sh_degree"])
    payload = checkpoint.load_latest(run / "checkpoints", model, options, identity)
    report = validate_model(model)
    if not set(payload["sampler_remaining_ids"]) <= set(
        read_json(ROOT / config["dataset"] / "splits.json")["image_ids"]["train"]
    ):
        raise ValueError("Restored sampler contains held-out images")
    write_json(
        run / "cuda_restore.json",
        {"status": "SUCCESS", "iteration": payload["iteration"], **report},
    )
    del model, payload
    torch_module().cuda.empty_cache()


def train(run: Path, until: int):
    from src.gaussian_splatting.native import initialize
    from src.gaussian_splatting import runner, checkpoint

    config = read_json(run / "resolved_config.json")
    start = read_json(run / "checkpoints/latest.json")["iteration"]
    if not start < until <= 30000 or config["iterations"] != 30000:
        raise ValueError("Session target must exceed saved iteration and be <=30000")
    import logging

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(run / "run.log")],
    )
    original_save = checkpoint.save

    def retained_save(*args, **kwargs):
        result = original_save(*args, **kwargs)
        # Delete only older checkpoints CREATED IN THIS NEW Kaggle run; parent input is immutable.
        files = sorted((run / "checkpoints").glob("iteration_*.pth"))
        for p in files[:-2]:
            p.unlink()
        return result

    checkpoint.save = retained_save
    try:
        report = runner.train(config, run, initialize(), resume=True, stop_after=until)
        write_json(run / "training_result.json", report)
        write_json(
            run / "run_manifest.json",
            {
                "status": report["status"],
                "iterations": report["iterations"],
                "training_started": True,
            },
        )
    finally:
        checkpoint.save = original_save


def export(run: Path, output: Path, bundle: Path):
    if output.exists():
        raise FileExistsError(output)
    migration = read_json(run / "migration.json")
    if migration["input_manifest_sha256"] != sha256(bundle / "manifest.json"):
        raise ValueError("Wrong original input")
    config = read_json(run / "resolved_config.json")
    pointer = read_json(run / "checkpoints/latest.json")
    cp = run / "checkpoints" / pointer["path"]
    if sha256(cp) != pointer["sha256"]:
        raise ValueError("Corrupt output checkpoint")
    check = inspect_checkpoint(cp, config)
    needed = sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
    output.parent.mkdir(parents=True, exist_ok=True)
    used = sum(p.stat().st_size for p in output.parent.rglob("*") if p.is_file())
    if (
        needed * 1.1 > shutil.disk_usage(output.parent).free
        or used + needed > 18 * 1024**3
    ):
        raise ValueError(
            "Insufficient export space / 18GiB output safety budget; save the existing recovery bundle first"
        )
    shutil.copytree(run, output / "run")
    return seal(
        output,
        {
            "schema": "phase02-kaggle-output-v1",
            "parent_run": read_json(bundle / "manifest.json")["parent_run"],
            "input_manifest_sha256": migration["input_manifest_sha256"],
            "dataset_metadata_sha256": migration["effective_identity"]["metadata"],
            "split_hash": migration["effective_identity"]["split"],
            "run_relative": run.relative_to(ROOT).as_posix(),
            "checkpoint_validation": check,
            "iteration": check["iteration"],
            "training_complete": check["iteration"] == config["iterations"],
            "complete": check["iteration"] == config["iterations"]
            and (run / "metrics/test_030000.json").exists()
            and read_json(run / "metrics/test_030000.json")["status"] == "SUCCESS",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "train", "evaluate", "export"])
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--previous", type=Path)
    p.add_argument("--until", type=int, default=30000)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.stage == "prepare":
        print(prepare(args.input, args.previous))
    else:
        run = ROOT / read_json(ROOT / "kaggle_active_run.json")["run"]
        if args.stage == "train":
            train(run, args.until)
        elif args.stage == "export":
            if args.output is None:
                raise ValueError("--output required")
            print(export(run, args.output, args.input)["iteration"])
        else:
            from src.gaussian_splatting.native import initialize
            from src.evaluation.evaluate_baseline import evaluate

            env = initialize()
            config = read_json(run / "resolved_config.json")
            report = evaluate(config, run, env)
            write_json(run / "evaluate_result.json", report)
