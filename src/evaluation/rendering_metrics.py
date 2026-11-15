"""Pinned official RGB metric components and explicit JSON handling of infinite PSNR."""

from __future__ import annotations

import math
from pathlib import Path

from src.common.io import sha256
from src.common.config import ROOT

WEIGHT_HASHES = {
    "alexnet-owt-7be5be79.pth": "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
    "alex.pth": "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
}


def psnr_value(prediction, target) -> dict:
    import torch

    if (
        prediction.shape != target.shape
        or prediction.ndim != 3
        or prediction.shape[0] != 3
        or not torch.isfinite(prediction).all()
        or not torch.isfinite(target).all()
    ):
        raise ValueError("RGB metric requires finite equal [3,H,W] tensors")
    if (
        prediction.min() < 0
        or prediction.max() > 1
        or target.min() < 0
        or target.max() > 1
    ):
        raise ValueError("RGB metric range must be [0,1]")
    mse = float((prediction - target).square().mean())
    return {
        "mse": mse,
        "psnr": -10 * math.log10(mse) if mse > 0 else None,
        "psnr_infinite": mse == 0,
    }


class Metrics:
    def __init__(self, *, learned: bool = True):
        self.lpips = None
        self.lpips_error = None
        self.weight_hashes = {}
        if learned:
            import torch

            torch.hub.set_dir(str(ROOT / ".tools/b0_weights"))

            if not torch.cuda.is_available():
                raise RuntimeError("LPIPS requires CUDA; no CPU fallback")
            try:
                for name, expected in WEIGHT_HASHES.items():
                    cached = ROOT / ".tools/b0_weights/checkpoints" / name
                    if cached.is_file() and sha256(cached) != expected:
                        raise ValueError(
                            f"LPIPS pretrained-weight checksum mismatch: {name}"
                        )
                from lpipsPyTorch.modules.lpips import LPIPS

                self.lpips = LPIPS(net_type="alex", version="0.1").eval().to("cuda:0")
                for path in sorted(
                    (Path(torch.hub.get_dir()) / "checkpoints").glob("*")
                ):
                    if path.name in ("alexnet-owt-7be5be79.pth", "alex.pth"):
                        self.weight_hashes[path.name] = sha256(path)
                if set(self.weight_hashes) != {"alexnet-owt-7be5be79.pth", "alex.pth"}:
                    raise ValueError("Cannot establish LPIPS weight provenance")
                if self.weight_hashes != WEIGHT_HASHES:
                    raise ValueError(
                        "Downloaded LPIPS weights differ from locked official weights"
                    )
            except (OSError, RuntimeError, ValueError) as exc:
                self.lpips = None
                self.lpips_error = str(exc)

    def __call__(self, prediction, target) -> dict:
        import torch
        from utils.loss_utils import ssim

        values = psnr_value(prediction, target)
        with torch.no_grad():
            values["ssim"] = float(ssim(prediction, target))
            values["lpips"] = (
                float(self.lpips(prediction[None] * 2 - 1, target[None] * 2 - 1))
                if self.lpips is not None
                else None
            )
        if any(
            v is not None and not math.isfinite(v)
            for v in [values["psnr"], values["ssim"], values["lpips"]]
        ):
            raise ValueError("Nonfinite metric")
        return values
