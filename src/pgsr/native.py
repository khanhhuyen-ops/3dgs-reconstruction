"""Load pinned PGSR CUDA modules in a dedicated process and environment."""
import importlib
import os
from pathlib import Path
import platform
import subprocess
import sys

from src.common.io import sha256, fingerprint
from .config import ROOT

REVISION = "de24f1a38b350387e8d8fe381b2cd70c1ae946e7"
HANDLES = []


def code_identity() -> str:
    files = sorted((ROOT / "src/pgsr").glob("*.py"))
    return fingerprint({p.name: p.read_text(encoding="utf-8").replace("\r\n", "\n") for p in files})


def initialize() -> dict:
    source = ROOT / "external/PGSR"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dirty = subprocess.check_output(["git", "diff", "--name-only"], cwd=source, text=True).strip()
    generated_metadata = [p for p in dirty.splitlines() if ".egg-info/" in p]
    source_changes = [p for p in dirty.splitlines() if ".egg-info/" not in p]
    if revision != REVISION or source_changes:
        raise ValueError("PGSR checkout revision/clean tracked source mismatch")
    for name in ("scene", "gaussian_renderer", "utils"):
        loaded = sys.modules.get(name)
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(source):
            raise RuntimeError("Run PGSR in a separate process; B0/native namespaces cannot be mixed")
    sys.path.insert(0, str(source))
    if os.name == "nt":
        for p in (Path(sys.base_prefix)/"Library/bin", Path(sys.base_prefix)/"DLLs"):
            if p.is_dir():
                HANDLES.append(os.add_dll_directory(str(p)))
    from src.common.gpu import require_cuda
    torch = require_cuda()
    if torch.__version__ != "2.4.1+cu121" or torch.version.cuda != "12.1":
        raise RuntimeError("Use the separate pinned PGSR environment: torch2.4.1+cu121")
    hashes = {}
    for name in ("diff_plane_rasterization._C", "simple_knn._C"):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(f"PGSR extension must be installed in its own venv: {name}")
        hashes[name] = sha256(Path(module.__file__))
    import pytorch3d
    from pytorch3d.transforms import quaternion_to_matrix
    q = torch.tensor([[1., 0., 0., 0.]], device="cuda:0", requires_grad=True)
    matrix = quaternion_to_matrix(q)
    assert matrix.is_cuda and torch.isfinite(matrix).all()
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "device": "cuda:0", "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
            "native_revision": revision, "extension_hashes": hashes,
            "regenerated_upstream_packaging_metadata": generated_metadata,
            "pytorch3d_version": pytorch3d.__version__,
            "pytorch3d_usage": "Official pure-Torch CUDA transforms; unused PyTorch3D extension not built",
            "code_identity": code_identity()}
