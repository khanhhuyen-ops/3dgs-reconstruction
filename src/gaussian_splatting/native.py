"""Load the unmodified 2023 reference implementation and verify its identity."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import platform
import subprocess
import sys

from src.common.config import ROOT
from src.common.io import PipelineError, sha256

REVISION = "414b553ef1d3032ea634b7b7a1d121173f36592c"
SUBMODULES = {
    "submodules/diff-gaussian-rasterization": "59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d",
    "submodules/simple-knn": "44f764299fa305faf6ec5ebd99939e0508331503",
    "submodules/diff-gaussian-rasterization/third_party/glm": "5c46b9c07008ae65cb81ab79cd677ecc1934b903",
}
DLL_HANDLES = []


def initialize() -> dict:
    """DLL setup changes this process only; parent Conda environment stays read-only."""
    if os.name == "nt":
        for path in (
            Path(sys.base_prefix) / "Library/bin",
            Path(sys.base_prefix) / "DLLs",
        ):
            if path.is_dir():
                DLL_HANDLES.append(os.add_dll_directory(str(path)))
    native = ROOT / "external/gaussian-splatting"
    if not native.is_dir():
        raise PipelineError(
            "BLOCKED_BACKEND",
            "Run scripts/setup_3dgs.ps1 and obtain the pinned official checkout",
        )

    def git(*args: str, cwd: Path = native) -> str:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()

    if git("rev-parse", "HEAD") != REVISION:
        raise ValueError("Unexpected Original3DGS revision")
    if git("diff", "--name-only", "--ignore-submodules=all"):
        raise ValueError("Official implementation has unrecorded modifications")
    for name, revision in SUBMODULES.items():
        if git("rev-parse", "HEAD", cwd=native / name) != revision or git(
            "diff", "--name-only", cwd=native / name
        ):
            raise ValueError(f"Unexpected native source/changes: {name}")
    sys.path.insert(0, str(native)) if str(native) not in sys.path else None
    from src.common.gpu import require_cuda

    torch = require_cuda()
    if torch.__version__ != "2.4.1+cu121" or torch.version.cuda != "12.1":
        raise PipelineError(
            "BLOCKED_BACKEND",
            "Use project .tools/b0_env (torch2.4.1+cu121); never CPU fallback",
        )
    for name in ("diff_gaussian_rasterization._C", "simple_knn._C"):
        module = importlib.import_module(name)
        if (
            not Path(module.__file__)
            .resolve()
            .is_relative_to(Path(sys.prefix).resolve())
        ):
            raise ValueError(
                f"{name} must be installed in the project-local environment"
            )
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": "cuda:0",
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "native_revision": REVISION,
        "submodules": SUBMODULES,
        "extension_hashes": {
            name: sha256(Path(importlib.import_module(name).__file__))
            for name in ("diff_gaussian_rasterization._C", "simple_knn._C")
        },
        "determinism": "RNG states saved; CUDA rasterizer atomic reductions may be nondeterministic",
        "optional_depth_exposure_antialiasing": False,
    }
