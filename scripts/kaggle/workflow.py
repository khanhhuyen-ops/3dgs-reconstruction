"""Portable, integrity-checked Phase02 handoff. CPU checkpoint inspection only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.io import fingerprint, read_json, sha256, write_json  # noqa: E402

HANDLES = []


def torch_module():
    if os.name == "nt":
        for p in (
            Path(sys.base_prefix) / "Library/bin",
            Path(sys.base_prefix) / "DLLs",
        ):
            if p.is_dir():
                HANDLES.append(os.add_dll_directory(str(p)))
    import torch

    return torch


def inside(root: Path, relative: str) -> Path:
    if "\\" in relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Unsafe relative path: {relative}")
    p = root / relative
    if not p.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes bundle: {relative}")
    return p


def seal(folder: Path, metadata: dict) -> dict:
    files = {
        p.relative_to(folder).as_posix(): sha256(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    # Include nested render manifests as well.
    files.update(
        {
            p.relative_to(folder).as_posix(): sha256(p)
            for p in folder.rglob("manifest.json")
            if p.parent != folder
        }
    )
    result = {**metadata, "files": files}
    write_json(folder / "manifest.json", result)
    return result


def verify(folder: Path) -> dict:
    m = read_json(folder / "manifest.json")
    if m.get("schema") not in ("phase02-kaggle-input-v1", "phase02-kaggle-output-v1"):
        raise ValueError("Unsupported bundle schema")
    for rel, expected in m["files"].items():
        p = inside(folder, rel)
        if not p.is_file() or sha256(p) != expected:
            raise ValueError(f"Checksum mismatch/missing: {rel}")
    actual = {
        p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()
    }
    if actual != set(m["files"]) | {"manifest.json"}:
        raise ValueError("Bundle contains unlisted files")
    return m


def extract_source(archive: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            inside(destination, info.filename)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Source archive symlink rejected")
        z.extractall(destination)


def inspect_checkpoint(path: Path, config: dict, identity: dict | None = None) -> dict:
    """Only call for a trusted project checkpoint after verifying its SHA256."""
    torch = torch_module()
    data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if data.get("schema") != "b0-checkpoint-v1":
        raise ValueError("Not a full B0 training checkpoint")
    if identity is not None and data["identity"] != identity:
        raise ValueError("Checkpoint provenance mismatch")
    if data["identity"]["config"] != fingerprint(config):
        raise ValueError("Checkpoint config mismatch")
    iteration = data["iteration"]
    if not 0 < iteration <= config["iterations"] or data["optimizer_steps"] != min(
        iteration, config["iterations"] - 1
    ):
        raise ValueError("Iteration/optimizer step mismatch")
    cap = data["native_capture"]
    if len(cap) != 12:
        raise ValueError("Unexpected native capture layout")
    n = len(cap[1])
    sh = (config["sh_degree"] + 1) ** 2 - 1
    shapes = [
        (n, 3),
        (n, 1, 3),
        (n, sh, 3),
        (n, 3),
        (n, 4),
        (n, 1),
        (n,),
        (n, 1),
        (n, 1),
    ]
    if not n or not 0 <= cap[0] <= config["sh_degree"] or cap[11] <= 0:
        raise ValueError("Invalid model/SH/spatial extent")
    for tensor, shape in zip(cap[1:10], shapes):
        if tuple(tensor.shape) != shape or tensor.dtype != torch.float32:
            raise ValueError("Invalid Gaussian/ADC shape or dtype")
        for part in tensor.split(32768):
            if not torch.isfinite(part).all():
                raise ValueError("Nonfinite Gaussian/ADC tensor")
    optimizer = cap[10]
    expected = {
        "xyz": cap[1],
        "f_dc": cap[2],
        "f_rest": cap[3],
        "scaling": cap[4],
        "rotation": cap[5],
        "opacity": cap[6],
    }
    if {g["name"] for g in optimizer["param_groups"]} != set(expected):
        raise ValueError("Missing Adam groups")
    for group in optimizer["param_groups"]:
        if len(group["params"]) != 1:
            raise ValueError("Unexpected Adam parameter layout")
        state = optimizer["state"].get(group["params"][0])
        if not state or "step" not in state:
            raise ValueError("Missing Adam state")
        for key in ("exp_avg", "exp_avg_sq"):
            if state[key].shape != expected[group["name"]].shape:
                raise ValueError("Adam row state mismatch")
            for part in state[key].split(32768):
                if not torch.isfinite(part).all():
                    raise ValueError("Nonfinite Adam state")
    if (
        set(data["rng"]) != {"python", "numpy", "torch_cpu", "torch_cuda"}
        or not data["rng"]["torch_cuda"]
    ):
        raise ValueError("Missing RNG state")
    ids = data["sampler_remaining_ids"]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sampler IDs")
    return {
        "status": "SUCCESS",
        "iteration": iteration,
        "gaussians": n,
        "optimizer_steps": data["optimizer_steps"],
        "active_sh_degree": cap[0],
        "optimizer_and_adc_present": True,
        "rng_and_sampler_present": True,
        "identity": data["identity"],
        "sha256": sha256(path),
        "scope": "CPU serialization/state validation; CUDA restore must run on Kaggle",
    }


def newest_checkpoint(run: Path, config: dict):
    pointer = read_json(run / "checkpoints/latest.json")
    path = inside(run / "checkpoints", pointer["path"])
    if sha256(path) != pointer["sha256"]:
        raise ValueError("Published checkpoint checksum mismatch")
    report = inspect_checkpoint(path, config)
    if report["iteration"] != pointer["iteration"]:
        raise ValueError("Published iteration mismatch")
    return path, report


def copy_file(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def bundle_id(m: dict) -> str:
    return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()[:20]
