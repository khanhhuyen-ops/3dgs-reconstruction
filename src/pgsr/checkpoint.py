"""Atomic PGSR-native state plus exposure optimizer, RNG, sampler and provenance."""
from pathlib import Path
import random
import numpy as np

from src.common.io import read_json, write_json, sha256


def inspect(path: Path, identity: dict | None = None) -> dict:
    import torch
    data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if data.get("schema") != "pgsr-full-checkpoint-v1":
        raise ValueError("Expected full PGSR checkpoint; B0/PLY cannot resume PGSR")
    if identity is not None and data["identity"] != identity:
        raise ValueError("PGSR checkpoint optimization/data/source identity mismatch")
    capture = data["native_capture"]
    if len(capture) != 16 or len(capture[1]) == 0:
        raise ValueError("Invalid native PGSR capture")
    n = len(capture[1])
    shapes = [(n, 3), (n, 6), (n, 1, 3), (n, (data["sh_degree"]+1)**2-1, 3),
              (n, 3), (n, 4), (n, 1)]
    for value, shape in zip(capture[1:8], shapes):
        if tuple(value.shape) != shape or value.dtype != torch.float32:
            raise ValueError("PGSR parameter shape/dtype mismatch")
    def finite(value):
        if isinstance(value, torch.Tensor):
            flat = value.reshape(-1)
            if any(not torch.isfinite(c).all() for c in flat.split(32768)):
                raise ValueError("Nonfinite checkpoint tensor")
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                finite(item)
    finite(capture)
    finite(data["app_state"])
    finite(data["app_optimizer"])
    optimizer = capture[14]
    if len(optimizer["param_groups"]) != 7:
        raise ValueError("PGSR Adam groups missing")
    for group, parameter in zip(optimizer["param_groups"], capture[1:8]):
        # Native group order: xyz, knn_f, f_dc, f_rest, opacity, scaling, rotation.
        name = group["name"]
        by_name = {"xyz": capture[1], "knn_f": capture[2], "f_dc": capture[3],
                   "f_rest": capture[4], "scaling": capture[5], "rotation": capture[6], "opacity": capture[7]}
        parameter = by_name[name]
        state = optimizer["state"].get(group["params"][0], {})
        if name != "knn_f" and data["optimizer_steps"] and not state:
            raise ValueError(f"Missing native Adam state: {name}")
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state and state[key].shape != parameter.shape:
                raise ValueError("PGSR optimizer row surgery mismatch")
    if data["app_state"]["appear_ab"].shape[1] != 2 or "state" not in data["app_optimizer"]:
        raise ValueError("Missing exposure state")
    if len(set(data["sampler_remaining_ids"])) != len(data["sampler_remaining_ids"]):
        raise ValueError("Duplicated remaining sampler entries")
    if set(data["rng"]) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("Incomplete RNG checkpoint")
    expected_steps = data["iteration"] - int(data["iteration"] == data["total_iterations"])
    if data["optimizer_steps"] != expected_steps:
        raise ValueError("Native final-step accounting mismatch")
    return {"status": "SUCCESS", "iteration": data["iteration"], "gaussians": n,
            "optimizer_steps": data["optimizer_steps"], "exposure_rows": len(data["app_state"]["appear_ab"]),
            "active_sh_degree": capture[0], "full_state": True, "identity": data["identity"]}


def save(run, model, app, iteration, sampler, identity, environment, config, optimizer_steps):
    import torch
    folder = run / "checkpoints"
    folder.mkdir(exist_ok=True)
    path = folder / f"iteration_{iteration:06d}.pth"
    if path.exists():
        raise FileExistsError(path)
    payload = {
        "schema": "pgsr-full-checkpoint-v1", "identity": identity, "runtime": environment,
        "iteration": iteration, "optimizer_steps": optimizer_steps,
        "total_iterations": config["iterations"], "sh_degree": config["sh_degree"],
        "native_capture": model.capture(), "use_app": model.use_app,
        "app_state": app.state_dict(), "app_optimizer": app.optimizer.state_dict(),
        "sampler_remaining_ids": list(sampler),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()},
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as f:
        torch.save(payload, f)
    temporary.rename(path)
    inspection = inspect(path, identity)
    write_json(folder / "latest.json", {"path": path.name, "iteration": iteration,
                                        "sha256": sha256(path), "validation": inspection})
    # Retain only this run's last two fully published checkpoints.
    for older in sorted(folder.glob("iteration_*.pth"))[:-2]:
        older.unlink()
    return path


def restore(run, model, app, options, identity, environment, train_ids):
    import torch
    pointer = read_json(run / "checkpoints/latest.json")
    path = run / "checkpoints" / pointer["path"]
    if path.parent.resolve() != (run / "checkpoints").resolve() or sha256(path) != pointer["sha256"]:
        raise ValueError("Checkpoint pointer/hash mismatch")
    inspect(path, identity)
    payload = torch.load(path, map_location="cuda:0", weights_only=False)
    if not set(payload["sampler_remaining_ids"]) <= set(train_ids):
        raise ValueError("Checkpoint sampler contains a held-out/nonexistent camera")
    # Build bytes may differ across OS/SM/compiler. Optimization/source/data identity
    # remains strict; native CUDA smoke must pass before any process reaches restore.
    previous = payload["runtime"]
    if any(previous[k] != environment[k] for k in ("torch", "cuda_runtime", "native_revision")):
        raise ValueError("Unsupported runtime migration")
    model.restore(payload["native_capture"], options)
    model.use_app = payload["use_app"]
    app.load_state_dict(payload["app_state"])
    app.optimizer.load_state_dict(payload["app_optimizer"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch_cpu"].cpu())
    states = payload["rng"]["torch_cuda"]
    torch.cuda.set_rng_state_all([states[0].cpu()] * torch.cuda.device_count())
    write_json(run / f"resume_{payload['iteration']:06d}_{len(list(run.glob('resume_*.json'))):02d}.json", {
        "status": "SUCCESS", "iteration": payload["iteration"], "parent_sha256": pointer["sha256"],
        "source_identity_unchanged": True, "previous_extension_hashes": previous["extension_hashes"],
        "current_extension_hashes": environment["extension_hashes"],
        "rng_note": "cuda:0 stream restored; unused streams copied, no cross-GPU float bitwise guarantee",
    })
    return payload
