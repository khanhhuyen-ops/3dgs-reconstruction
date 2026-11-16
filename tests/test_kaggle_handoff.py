"""Serialization/import contract fixtures only; no claimed Kaggle training evidence."""

from copy import deepcopy
import ast
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/kaggle"))
from workflow import fingerprint, inspect_checkpoint, seal, sha256, verify, write_json  # noqa: E402
from import_kaggle_results import import_results  # noqa: E402
from src.gaussian_splatting.config import DEFAULTS  # noqa: E402


def fixture_checkpoint(path, config):
    from workflow import torch_module

    torch = torch_module()
    n = 4
    sh = (config["sh_degree"] + 1) ** 2 - 1
    tensors = [
        torch.zeros(s)
        for s in [
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
    ]
    names = ["xyz", "f_dc", "f_rest", "scaling", "rotation", "opacity"]
    optimizer = {
        "param_groups": [{"name": name, "params": [i]} for i, name in enumerate(names)],
        "state": {
            i: {
                "step": torch.tensor(4.0),
                "exp_avg": torch.zeros_like(tensors[i]),
                "exp_avg_sq": torch.zeros_like(tensors[i]),
            }
            for i in range(6)
        },
    }
    payload = {
        "schema": "b0-checkpoint-v1",
        "identity": {
            "config": fingerprint(config),
            "metadata": "meta",
            "split": "split",
        },
        "iteration": 4,
        "optimizer_steps": 4,
        "native_capture": (0, *tensors, optimizer, 1.0),
        "sampler_remaining_ids": [1, 2],
        "rng": {
            "python": (),
            "numpy": (),
            "torch_cpu": torch.zeros(1, dtype=torch.uint8),
            "torch_cuda": [torch.zeros(1, dtype=torch.uint8)],
        },
    }
    torch.save(payload, path)
    return payload


def test_checkpoint_rejects_missing_optimizer_and_wrong_config(tmp_path):
    from workflow import torch_module

    config = deepcopy(DEFAULTS)
    path = tmp_path / "fixture.pth"
    payload = fixture_checkpoint(path, config)
    assert inspect_checkpoint(path, config)["iteration"] == 4
    wrong = deepcopy(config)
    wrong["resolution"] = 2
    with pytest.raises(ValueError, match="config"):
        inspect_checkpoint(path, wrong)
    payload["native_capture"][10]["state"].clear()
    torch_module().save(payload, path)
    with pytest.raises(ValueError, match="Adam"):
        inspect_checkpoint(path, config)


def test_bundle_integrity_and_traversal(tmp_path):
    (tmp_path / "data").write_text("trusted")
    seal(tmp_path, {"schema": "phase02-kaggle-input-v1"})
    assert verify(tmp_path)["files"]
    (tmp_path / "data").write_text("tampered")
    with pytest.raises(ValueError, match="Checksum"):
        verify(tmp_path)
    m = json.loads((tmp_path / "manifest.json").read_text())
    m["files"] = {"../outside": "hash"}
    write_json(tmp_path / "manifest.json", m)
    with pytest.raises(ValueError, match="Unsafe"):
        verify(tmp_path)


def test_partial_import_is_idempotent_and_rejects_dataset_change(tmp_path):
    root = tmp_path / "local"
    bundle = tmp_path / "bundle"
    run = bundle / "run"
    cp = run / "checkpoints"
    cp.mkdir(parents=True)
    config = deepcopy(DEFAULTS)
    config.update(
        dataset="datasets/fixture",
        output_root="outputs/test",
        experiment="kaggle-fixture",
    )
    write_json(root / "outputs/test/local-parent/resolved_config.json", config)
    dataset = root / config["dataset"]
    dataset.mkdir(parents=True)
    write_json(dataset / "metadata.json", {"fixture_only": True})
    write_json(dataset / "splits.json", {"hash": "split"})
    checkpoint = cp / "iteration_000004.pth"
    payload = fixture_checkpoint(checkpoint, config)
    payload["identity"]["metadata"] = sha256(dataset / "metadata.json")
    from workflow import torch_module

    torch_module().save(payload, checkpoint)
    write_json(
        cp / "latest.json",
        {"path": checkpoint.name, "sha256": sha256(checkpoint), "iteration": 4},
    )
    write_json(run / "resolved_config.json", config)
    seal(
        bundle,
        {
            "schema": "phase02-kaggle-output-v1",
            "dataset_metadata_sha256": sha256(dataset / "metadata.json"),
            "split_hash": "split",
            "iteration": 4,
            "complete": False,
            "run_relative": "outputs/test/kaggle-fixture",
            "parent_run": "local-parent",
        },
    )
    target = import_results(bundle, root)
    assert import_results(bundle, root) == target
    assert (
        json.loads((target / "blender_handoff.json").read_text())["gaussian_ply"]
        is None
    )
    (target / "resolved_config.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        import_results(bundle, root)
    write_json(dataset / "metadata.json", {"fixture_only": "changed"})
    with pytest.raises(ValueError, match="dataset"):
        import_results(bundle, root)


def test_notebook_cells_parse_and_no_embedded_outputs():
    notebook = json.loads(
        (ROOT / "notebooks/kaggle/phase02_resume_3dgs.ipynb").read_text()
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert not cell["outputs"] and cell["execution_count"] is None


def test_final_import_checks_metrics_and_gaussian_ply(tmp_path):
    import numpy as np
    from plyfile import PlyData, PlyElement
    from workflow import torch_module

    root = tmp_path / "local"
    bundle = tmp_path / "bundle"
    run = bundle / "run"
    cp = run / "checkpoints"
    cp.mkdir(parents=True)
    config = deepcopy(DEFAULTS)
    config.update(
        dataset="datasets/fixture",
        output_root="outputs/test",
        experiment="kaggle-final",
        iterations=30000,
    )
    dataset = root / config["dataset"]
    dataset.mkdir(parents=True)
    write_json(dataset / "metadata.json", {"fixture_only": True})
    write_json(
        dataset / "splits.json", {"hash": "split", "image_ids": {"test": [9, 10]}}
    )
    write_json(root / "outputs/test/local-parent/resolved_config.json", config)
    checkpoint = cp / "iteration_030000.pth"
    payload = fixture_checkpoint(checkpoint, config)
    payload.update(iteration=30000, optimizer_steps=29999)
    payload["identity"]["metadata"] = sha256(dataset / "metadata.json")
    torch_module().save(payload, checkpoint)
    write_json(
        cp / "latest.json",
        {"path": checkpoint.name, "sha256": sha256(checkpoint), "iteration": 30000},
    )
    write_json(run / "resolved_config.json", config)
    ply = run / "point_cloud/iteration_030000/point_cloud.ply"
    ply.parent.mkdir(parents=True)
    fields = (
        ["x", "y", "z", "nx", "ny", "nz", "opacity"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range(3 * ((config["sh_degree"] + 1) ** 2 - 1))]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    vertices = np.zeros(4, dtype=[(k, "f4") for k in fields])
    PlyData([PlyElement.describe(vertices, "vertex")]).write(ply)
    write_json(
        run / "training_result.json",
        {
            "status": "SUCCESS",
            "iterations": 30000,
            "export": {
                "path": "outputs/test/kaggle-final/point_cloud/iteration_030000/point_cloud.ply",
                "sha256": sha256(ply),
                "render_reload_max_abs_difference": 0.0,
            },
        },
    )
    values = {"psnr": 1.0, "ssim": 0.5, "lpips": 0.5}
    metrics = {
        "status": "SUCCESS",
        "split_hash": "split",
        "view_count": 2,
        "per_view": [{"image_id": i, **values} for i in [9, 10]],
        "aggregate_arithmetic_mean_per_view": values,
        "checkpoint_export_sha256": sha256(ply),
    }
    write_json(run / "metrics/test_030000.json", metrics)
    metadata = {
        "schema": "phase02-kaggle-output-v1",
        "dataset_metadata_sha256": sha256(dataset / "metadata.json"),
        "split_hash": "split",
        "iteration": 30000,
        "complete": True,
        "run_relative": "outputs/test/kaggle-final",
        "parent_run": "local-parent",
    }
    seal(bundle, metadata)
    target = import_results(bundle, root)
    assert (
        json.loads((target / "kaggle_import.json").read_text())["status"]
        == "IMPORTED_FINAL"
    )
    assert json.loads((target / "blender_handoff.json").read_text())[
        "gaussian_ply"
    ].endswith("point_cloud.ply")
    metrics["per_view"][0]["image_id"] = 99
    write_json(run / "metrics/test_030000.json", metrics)
    seal(bundle, metadata)
    with pytest.raises(ValueError, match="evaluation"):
        import_results(bundle, root)
