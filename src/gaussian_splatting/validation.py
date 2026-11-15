"""CUDA model/optimizer validation, and an explicitly synthetic kernel unit test."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def validate_model(model, *, gradients: bool = False) -> dict:
    import torch

    n = len(model.get_xyz)
    if n <= 0:
        raise ValueError("Empty Gaussian model")
    groups = {
        "xyz": model._xyz,
        "f_dc": model._features_dc,
        "f_rest": model._features_rest,
        "opacity": model._opacity,
        "scaling": model._scaling,
        "rotation": model._rotation,
    }
    norms = {}
    for name, value in groups.items():
        if (
            value.shape[0] != n
            or value.device != torch.device("cuda:0")
            or value.dtype != torch.float32
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid CUDA FP32 Gaussian parameters: {name}")
        if gradients:
            if value.grad is None or not torch.isfinite(value.grad).all():
                raise ValueError(f"Missing/nonfinite gradient: {name}")
            norms[name] = float(value.grad.norm())
    if model._features_dc.shape != (n, 1, 3) or model._features_rest.shape != (
        n,
        (model.max_sh_degree + 1) ** 2 - 1,
        3,
    ):
        raise ValueError("Invalid SH shapes")
    if (
        not torch.isfinite(model.get_scaling).all()
        or torch.any(model.get_scaling <= 0)
        or not torch.allclose(
            model.get_rotation.norm(dim=1), torch.ones(n, device="cuda"), atol=1e-5
        )
    ):
        raise ValueError("Invalid activated scale/rotation")
    if torch.any(model.get_opacity < 0) or torch.any(model.get_opacity > 1):
        raise ValueError("Invalid activated opacity")
    for group in model.optimizer.param_groups if model.optimizer else []:
        parameter = group["params"][0]
        state = model.optimizer.state.get(parameter, {})
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state and (
                state[key].shape != parameter.shape
                or not state[key].is_cuda
                or not torch.isfinite(state[key]).all()
            ):
                raise ValueError(f"Invalid Adam row state: {group['name']}.{key}")
    return {
        "gaussians": n,
        "active_sh_degree": model.active_sh_degree,
        "max_sh_degree": model.max_sh_degree,
        "gradient_norms": norms,
        "opacity_min": float(model.get_opacity.min()),
        "opacity_mean": float(model.get_opacity.mean()),
        "opacity_max": float(model.get_opacity.max()),
        "scale_min": float(model.get_scaling.min()),
        "scale_max": float(model.get_scaling.max()),
    }


def cuda_unit_fixture(config: dict) -> dict:
    """Known small anisotropic scene: kernel/gradient evidence only, never a reconstruction."""
    import torch
    from scene.gaussian_model import GaussianModel
    from scene.cameras import Camera
    from gaussian_renderer import render
    from utils.graphics_utils import BasicPointCloud
    from utils.loss_utils import l1_loss, ssim

    xyz = np.array(
        [
            [-0.4, -0.3, 3],
            [0.2, -0.2, 3.1],
            [-0.2, 0.3, 2.8],
            [0.4, 0.2, 3.3],
            [0.0, 0.0, 3.2],
            [0.1, 0.4, 3.4],
        ],
        np.float32,
    )
    colors = np.array(
        [
            [0.8, 0.2, 0.1],
            [0.1, 0.8, 0.2],
            [0.2, 0.1, 0.8],
            [0.7, 0.3, 0.4],
            [0.4, 0.7, 0.3],
            [0.3, 0.4, 0.7],
        ],
        np.float32,
    )
    g = GaussianModel(2)
    g.create_from_pcd(BasicPointCloud(xyz, colors, np.zeros_like(xyz)), 1.0)
    g.training_setup(
        SimpleNamespace(
            iterations=config["iterations"], **config["optimizer"], **config["density"]
        )
    )
    with torch.no_grad():
        g._scaling[:] = torch.tensor([-0.9, -1.3, -1.7], device="cuda")
        g._rotation[:, 1:] = torch.tensor([0.1, 0.2, 0.05], device="cuda")
    g.active_sh_degree = 1
    camera = Camera(
        1,
        np.eye(3),
        np.zeros(3),
        1.0,
        1.0,
        torch.zeros(3, 64, 64),
        None,
        "CUDA_UNIT_FIXTURE",
        1,
        data_device="cpu",
    )
    pipe = SimpleNamespace(
        convert_SHs_python=False, compute_cov3D_python=False, debug=False
    )
    out = render(camera, g, pipe, torch.zeros(3, device="cuda"))["render"]
    if (
        out.shape != (3, 64, 64)
        or not out.is_cuda
        or not torch.isfinite(out).all()
        or out.min() < 0
        or out.max() > 1.0001
    ):
        raise ValueError("CUDA rasterizer forward contract failed")
    target = torch.linspace(0.1, 0.8, 64 * 64 * 3, device="cuda").reshape(3, 64, 64)
    loss = 0.8 * l1_loss(out, target) + 0.2 * (1 - ssim(out, target))
    loss.backward()
    report = validate_model(g, gradients=True)
    if any(
        report["gradient_norms"][key] <= 0
        for key in ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")
    ):
        raise ValueError(
            "Relevant CUDA gradients are zero on the anisotropic unit fixture"
        )
    before = g._xyz.detach().clone()
    g.optimizer.step()
    torch.cuda.synchronize()
    if torch.equal(before, g._xyz):
        raise ValueError("CUDA Adam did not update positions")
    return {
        "status": "SUCCESS",
        "scope": "CUDA_UNIT_FIXTURE (synthetic unit test, not project reconstruction)",
        "renderer": "official diff_gaussian_rasterization CUDA",
        "initialization_knn": "official distCUDA2 CUDA",
        "loss": float(loss),
        "optimizer_update_verified": True,
        **report,
    }
