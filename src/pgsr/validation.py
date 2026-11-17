"""Actual CUDA plane/gradient/exposure/warp tests, explicitly synthetic unit fixtures."""
from pathlib import Path
from types import SimpleNamespace
import math
import argparse
import tempfile

import numpy as np

from src.common.io import write_json


def validate_model(model, gradients=False):
    import torch
    n = len(model.get_xyz)
    if n < 1:
        raise ValueError("Empty PGSR model")
    norms = {}
    for name in ("xyz", "knn_f", "features_dc", "features_rest", "scaling", "rotation", "opacity"):
        if name == "knn_f" and model.optimizer is None:
            continue  # Native inference PLY does not serialize the unused KNN feature.
        value = getattr(model, "_"+name)
        if value.shape[0] != n or value.device != torch.device("cuda:0") or not torch.isfinite(value).all():
            raise ValueError(f"Invalid CUDA PGSR tensor: {name}")
        if gradients and name != "knn_f":
            if value.grad is None or not torch.isfinite(value.grad).all():
                raise ValueError(f"Missing/nonfinite PGSR gradient: {name}")
            norms[name] = float(value.grad.norm())
    if not torch.isfinite(model.get_scaling).all() or not (model.get_scaling > 0).all():
        raise ValueError("Invalid PGSR activated scales")
    if not torch.allclose(model.get_rotation.norm(dim=-1), torch.ones(n, device="cuda"), atol=1e-5):
        raise ValueError("Invalid normalized Gaussian quaternion")
    for group in model.optimizer.param_groups if model.optimizer else []:
        param = group["params"][0]
        state = model.optimizer.state.get(param, {})
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state and (state[key].shape != param.shape or not state[key].is_cuda or not torch.isfinite(state[key]).all()):
                raise ValueError("Invalid PGSR Adam state surgery")
    return {"gaussians": n, "active_sh_degree": model.active_sh_degree,
            "gradient_norms": norms, "min_scale": float(model.get_scaling.min()),
            "max_scale": float(model.get_scaling.max())}


def cuda_fixture(output: Path) -> dict:
    from .native import initialize
    environment = initialize()
    import torch
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from scene.app_model import AppModel
    from gaussian_renderer import render
    from utils.graphics_utils import BasicPointCloud, patch_warp
    from PIL import Image
    from .config import DEFAULTS
    from .geometry import valid_geometry, plane_homography, warp_patch, edge_weight, objectives
    from . import checkpoint

    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((40, 40, 3), 127, dtype=np.uint8)).save(output / "unit_camera.png")
    options = SimpleNamespace(**DEFAULTS["optimizer"], iterations=10)
    options.single_view_weight_from_iter = 0
    options.multi_view_weight_from_iter = 0
    options.multi_view_sample_num = 128
    coordinates = np.linspace(-1., 1., 8)
    x, y = np.meshgrid(coordinates, coordinates)
    xyz = np.column_stack([x.ravel(), y.ravel(), np.full(64, 2.)])
    colors = np.column_stack([.5+.3*np.sin(3*x.ravel()), .5+.3*np.sin(4*y.ravel()), np.full(64, .4)])
    model = GaussianModel(3)
    model.create_from_pcd(BasicPointCloud(xyz, colors, np.zeros_like(xyz)), 1.)
    model.training_setup(options)
    with torch.no_grad():
        model._scaling[:] = torch.tensor([math.log(.2), math.log(.2), math.log(.002)], device="cuda")
        model._opacity[:] = math.log(.8/.2)
    cameras = [Camera(1, np.eye(3), np.array([t, 0., 0.]), math.pi/2, math.pi/2,
                      40, 40, str(output / "unit_camera.png"), "unit", i,
                      preload_img=False, data_device="cpu") for i,t in enumerate((0., -.08))]
    reference, neighbor = cameras
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    bg = torch.zeros(3, device="cuda")
    package = render(reference, model, pipe, bg)
    covered = valid_geometry(package, reference) & (package["rendered_normal"].norm(dim=0)>.05)
    assert int(covered.sum()) > 100
    error = (package["plane_depth"].squeeze()[covered]-2.).abs().max().item()
    assert error < 1e-4, error
    assert all(torch.isfinite(package[k]).all() for k in ("render", "plane_depth", "rendered_normal", "depth_normal"))
    assert (package["rendered_normal"][2][covered] < 0).all()
    # Unique min-scale derivative, plane-depth and normal derivatives use actual kernels.
    analytic_loss = package["render"].mean()+package["plane_depth"].squeeze()[covered].mean()
    analytic_loss = analytic_loss+package["rendered_normal"][0][covered].mean()+100*model.get_scaling[:, 2].mean()
    analytic_loss.backward()
    gradients = validate_model(model, gradients=True)["gradient_norms"]
    assert all(gradients[k]>0 for k in ("xyz", "scaling", "rotation", "opacity", "features_dc"))
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    uv = torch.tensor([[[12., 12.], [20., 20.], [27., 24.]]], device="cuda")
    normal = torch.tensor([[0., 0., -1.]], device="cuda")
    distance = torch.tensor([2.], device="cuda")
    h = plane_homography(reference, neighbor, normal, distance)
    warped, valid = warp_patch(h, uv)
    expected = uv.clone()
    expected[..., 0] -= .08*reference.Fx/2
    homography_error = float((warped-expected).abs().max())
    assert homography_error < 1e-4 and valid.all()
    assert torch.allclose(warped, patch_warp(h, uv), atol=1e-5)
    assert torch.allclose(h, plane_homography(reference, neighbor, normal*.3, distance*.3), atol=1e-6)
    undefined, valid_undefined = warp_patch(torch.zeros_like(h), uv)
    assert not valid_undefined.any() and torch.isfinite(undefined).all()
    constant_weights = edge_weight(torch.ones(3, 40, 40, device="cuda"))
    assert torch.isfinite(constant_weights).all() and constant_weights[10, 10]==1
    app = AppModel(2)
    model.use_app = True
    package = render(reference, model, pipe, bg, app_model=app)
    near_package = render(neighbor, model, pipe, bg, app_model=app)
    grid_y, grid_x = torch.meshgrid(torch.arange(40, device="cuda"), torch.arange(40, device="cuda"), indexing="ij")
    gray = (.5+.2*torch.sin(grid_x.float()*.45)+.2*torch.cos(grid_y.float()*.37))[None]
    gray_near = (.5+.2*torch.sin((grid_x.float()+1)*.45)+.2*torch.cos(grid_y.float()*.37))[None]
    target = (package["render"].detach()*.8+.02).clamp(0, 1)
    total, losses, _, counts = objectives(model, app, reference, neighbor, package, near_package,
                                         target, gray, gray_near, options, DEFAULTS, 2)
    assert counts["geometry_valid_pixels"]>0 and counts["multi_valid_pixels"]>0
    assert counts["ncc_valid_patches"]>0 and counts["exposure_gate_used"]
    total.backward()
    assert torch.isfinite(app.appear_ab.grad).all() and app.appear_ab.grad.abs().sum()>0
    validate_model(model, gradients=True)
    app.optimizer.step()
    app.optimizer.zero_grad(set_to_none=True)
    model.optimizer.zero_grad(set_to_none=True)
    # Capture after model's first native Adam step; total schedule remains10.
    identity = {"unit_fixture": "PGSR_SYNTHETIC_CUDA_ONLY"}
    path = checkpoint.save(output, model, app, 1, [0, 1], identity, environment,
                           {"iterations":10, "sh_degree":3}, 1)
    restored_model = GaussianModel(3)
    restored_app = AppModel(2)
    restored = checkpoint.restore(output, restored_model, restored_app, options,
                                  identity, environment, [0, 1])
    assert restored["iteration"]==1 and torch.equal(restored_app.appear_ab, app.appear_ab)
    assert torch.equal(restored_model._xyz, model._xyz)
    assert restored_app.optimizer.state_dict()["state"]
    report = {"status":"SUCCESS", "scope":"SYNTHETIC_CUDA_UNIT, not project reconstruction/training",
              "environment":environment, "plane_depth_max_error":error,
              "homography_max_error":homography_error, "kernel_gradient_norms":gradients,
              "objective_terms":{k:float(v.detach()) for k,v in losses.items()},
              "objective_counts":counts, "exposure_update_verified":True,
              "full_checkpoint_restore":checkpoint.inspect(path, identity),
              "constant_image_guard":True, "undefined_warp_mask":True}
    write_json(output / "cuda_unit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    location = args.output or Path(tempfile.mkdtemp(prefix="pgsr-cuda-unit-"))
    print(cuda_fixture(location))
