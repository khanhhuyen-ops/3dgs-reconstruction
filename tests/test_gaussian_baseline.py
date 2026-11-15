"""B0 CPU contracts and actual CUDA kernel/state tests; no synthetic scene metrics."""

from __future__ import annotations

from copy import deepcopy
import random
from types import SimpleNamespace

import numpy as np
import pytest

from src.common.config import ROOT
from src.common.io import read_json
from src.gaussian_splatting.config import DEFAULTS, load_config


def test_profiles_preserve_method_and_reference_schedules():
    low = load_config(ROOT / "configs/training/b0_low_memory.yaml")
    ref = load_config(ROOT / "configs/training/b0_research_quality.yaml")
    assert low["method"] == ref["method"] == "B0_Original3DGS"
    assert (
        ref["iterations"] == 30000
        and ref["sh_degree"] == 3
        and ref["max_gaussians"] is None
    )
    assert ref["density"] == DEFAULTS["density"]
    assert ref["optimizer"] == low["optimizer"] == DEFAULTS["optimizer"]
    assert low["resolution"] == 4 and low["max_gaussians"] == 250000


@pytest.mark.parametrize(
    "key,value",
    [
        ("device", "cpu"),
        ("allow_cpu_fallback", True),
        ("resolution", 3),
        ("sh_degree", 4),
        ("iterations", 0),
        ("typo", 2),
    ],
)
def test_invalid_config_fails(tmp_path, key, value):
    import yaml

    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({key: value}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_psnr_known_values_and_infinity_policy():
    import torch
    from src.evaluation.rendering_metrics import psnr_value

    black = torch.zeros(3, 16, 16)
    white = torch.ones_like(black)
    assert psnr_value(black, white)["psnr"] == 0.0
    assert psnr_value(black, black)["psnr"] is None
    assert psnr_value(black, black)["psnr_infinite"]
    assert abs(psnr_value(black, white * 0.5)["psnr"] - 6.020599913279624) < 1e-10
    with pytest.raises(ValueError):
        psnr_value(black, white * 2)


@pytest.fixture
def cuda_native():
    import torch

    if torch.__version__ != "2.4.1+cu121":
        pytest.skip("Run CUDA integration tests using .tools/b0_env/Scripts/python.exe")
    from src.gaussian_splatting.native import initialize

    environment = initialize()
    return environment


def test_real_cuda_rasterizer_and_all_parameter_gradients(cuda_native):
    from src.gaussian_splatting.validation import cuda_unit_fixture

    result = cuda_unit_fixture(deepcopy(DEFAULTS))
    assert result["optimizer_update_verified"]
    assert all(v > 0 for v in result["gradient_norms"].values())


def test_native_adam_density_surgery_and_full_checkpoint(tmp_path, cuda_native):
    import torch
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import BasicPointCloud
    from src.gaussian_splatting import checkpoint
    from src.gaussian_splatting.runner import density_event
    from src.gaussian_splatting.validation import validate_model

    xyz = np.array(
        [[-0.3, 0, 3], [0.3, 0, 3], [0, 0.4, 3], [0, -0.4, 3.2], [0.1, 0.1, 3.1]],
        np.float32,
    )
    model = GaussianModel(2)
    model.create_from_pcd(
        BasicPointCloud(xyz, np.ones_like(xyz) * 0.5, np.zeros_like(xyz)), 1.0
    )
    options = SimpleNamespace(
        iterations=1200, **DEFAULTS["optimizer"], **DEFAULTS["density"]
    )
    model.training_setup(options)
    sum(
        p.square().mean()
        for group in model.optimizer.param_groups
        for p in group["params"]
    ).backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.xyz_gradient_accum[:] = 0.001
        model.denom[:] = 1.0
        model._scaling[0] = np.log(0.005)
        model._scaling[1:] = np.log(0.05)
        event = density_event(model, options, 1.0, 100, None)
    assert (
        event["cloned"] == 1
        and event["split_parents"] == 4
        and event["split_children"] == 8
    )
    assert len(model.get_xyz) == 10
    validate_model(model)
    with torch.no_grad():
        capped = density_event(model, options, 1.0, 4, None)
    assert capped["status"] == "SKIPPED_RESOURCE_CAP"
    identity = {"test_fixture": True, "split": "train-only"}
    checkpoint.save(tmp_path, model, 200, [11, 13], identity, 200)
    expected = model._xyz.detach().clone()
    next_random = random.random()
    next_numpy = np.random.rand()
    next_cuda = torch.rand(3, device="cuda")
    restored = GaussianModel(2)
    result = checkpoint.load_latest(tmp_path, restored, options, identity)
    assert torch.equal(restored._xyz, expected)
    assert result["iteration"] == 200 and result["sampler_remaining_ids"] == [11, 13]
    assert random.random() == next_random and np.random.rand() == next_numpy
    assert torch.equal(torch.rand(3, device="cuda"), next_cuda)
    validate_model(restored)
    with pytest.raises(FileExistsError):
        checkpoint.save(tmp_path, restored, 200, [], identity, 200)
    with pytest.raises(ValueError):
        checkpoint.load_latest(tmp_path, restored, options, {"wrong": True})
    pointer = read_json(tmp_path / "latest.json")
    (tmp_path / pointer["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        checkpoint.load_latest(tmp_path, restored, options, identity)


def test_official_ssim_identity_and_photometric_gradients(cuda_native):
    import torch
    from utils.loss_utils import l1_loss, ssim

    image = torch.full((3, 32, 32), 0.4, device="cuda", requires_grad=True)
    same = image.detach().clone()
    assert abs(float(ssim(image, same)) - 1) < 1e-6
    assert float(l1_loss(image, same)) == 0
    loss = 0.8 * l1_loss(image, same + 0.1) + 0.2 * (1 - ssim(image, same + 0.1))
    loss.backward()
    assert torch.isfinite(image.grad).all() and image.grad.norm() > 0
