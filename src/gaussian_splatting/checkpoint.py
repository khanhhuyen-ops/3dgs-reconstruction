"""Native capture/restore plus RNG/sampler state, atomic immutable checkpoint files."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from src.common.io import read_json, sha256, write_json


def save(
    directory: Path,
    model,
    iteration: int,
    sampler: list[int],
    identity: dict,
    optimizer_steps: int,
) -> Path:
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"iteration_{iteration:06d}.pth"
    if path.exists():
        raise FileExistsError(f"Refusing checkpoint overwrite: {path}")
    payload = {
        "schema": "b0-checkpoint-v1",
        "identity": identity,
        "iteration": iteration,
        "optimizer_steps": optimizer_steps,
        "native_capture": model.capture(),
        "sampler_remaining_ids": sampler.copy(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        },
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as f:
        torch.save(payload, f)
    temporary.rename(path)
    reopened = torch.load(path, map_location="cuda:0", weights_only=False)
    if reopened["iteration"] != iteration or reopened["identity"] != identity:
        raise ValueError("Checkpoint serialization validation failed")
    del reopened
    write_json(
        directory / "latest.json",
        {"path": path.name, "sha256": sha256(path), "iteration": iteration},
    )
    return path


def load_latest(directory: Path, model, options, identity: dict) -> dict:
    import torch

    pointer = read_json(directory / "latest.json")
    path = directory / pointer["path"]
    if (
        path.parent.resolve() != directory.resolve()
        or sha256(path) != pointer["sha256"]
    ):
        raise ValueError("Checkpoint identity/integrity mismatch")
    data = torch.load(path, map_location="cuda:0", weights_only=False)
    if data["schema"] != "b0-checkpoint-v1" or data["identity"] != identity:
        raise ValueError("Checkpoint dataset/config/native implementation mismatch")
    model.restore(data["native_capture"], options)
    random.setstate(data["rng"]["python"])
    np.random.set_state(data["rng"]["numpy"])
    torch.set_rng_state(data["rng"]["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([state.cpu() for state in data["rng"]["torch_cuda"]])
    return data
