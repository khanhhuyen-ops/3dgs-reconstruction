"""Policy tests verify that blocked BA can never construct a mapper command."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.common.config import ROOT, load_config
from src.common.io import PipelineError
from src.geometry.sfm.colmap_runner import feature_commands, mapper_command, require_runtime_log


def test_hidden_cuda_fails_fast() -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    result = subprocess.run([sys.executable, "-c", "from src.common.gpu import require_cuda; require_cuda()"], env=env, capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode != 0
    assert "STOP_GPU_UNAVAILABLE" in result.stderr or "CUDA required" in result.stderr


@pytest.mark.parametrize("report", [{"verified": False}, {"verified": True, "backend": "CERES"}, {"verified": True, "backend": "CASPAR", "colmap_binary_sha256": "old"}])
def test_unverified_ba_forbids_mapper(report: dict) -> None:
    config = load_config(ROOT / "configs/sfm_ignatius.yaml")
    with pytest.raises(PipelineError, match="Mapper forbidden") as caught:
        mapper_command(config, {"binary_sha256": "new"}, Path("db"), Path("images"), Path("sparse"), report)
    assert caught.value.status == "BLOCKED_GPU_BA"


@pytest.mark.parametrize("log", ["SIFT CPU feature extractor", "Creating SIFT GPU extractor; fallback to CPU", "finished images"])
def test_gpu_log_requires_positive_evidence(log: str) -> None:
    with pytest.raises(PipelineError):
        require_runtime_log(log, "features")


def test_version_specific_old_flags() -> None:
    config = load_config(ROOT / "configs/sfm_ignatius.yaml")
    caps = {"binary": "colmap", "options": {
        "feature_extractor": ["SiftExtraction.use_gpu", "SiftExtraction.gpu_index", "SiftExtraction.max_image_size", "SiftExtraction.max_num_features"],
        "sequential_matcher": ["SiftMatching.use_gpu", "SiftMatching.gpu_index", "SiftMatching.max_num_matches", "SequentialMatching.overlap"]}}
    extraction, matching = feature_commands(config, caps, Path("with space/db.db"), Path("with space/images"))
    assert "--SiftExtraction.use_gpu" in extraction
    assert "--FeatureExtraction.use_gpu" not in extraction
    assert str(Path("with space/images")) in extraction
    assert matching[matching.index("--SiftMatching.gpu_index") + 1] == "0"
    caps["options"]["feature_extractor"].remove("SiftExtraction.use_gpu")
    with pytest.raises(PipelineError):
        feature_commands(config, caps, Path("db"), Path("images"))


@pytest.mark.parametrize("path", ["datasets/x/raw/result", "../outside", "datasets/tanks_and_temples/Ignatius"])
def test_protect_raw_and_input_ancestors(tmp_path: Path, path: str) -> None:
    import yaml
    config = load_config(ROOT / "configs/sfm_ignatius.yaml")
    config["paths"]["output"] = path
    source = tmp_path / "unsafe.yaml"
    source.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(source)


def test_blocked_ba_preserves_database_and_never_calls_mapper(tmp_path: Path, monkeypatch) -> None:
    """Control-flow fixture: subprocess output is mocked, not GPU benchmark evidence."""
    import sqlite3
    from PIL import Image
    from src.common.io import code_hash, sha256, write_json
    from src.geometry.sfm import colmap_runner

    config = load_config(ROOT / "configs/sfm_ignatius.yaml")
    video = tmp_path / config["input"]["video"]
    video.parent.mkdir(parents=True)
    video.write_bytes(b"explicit test fixture; no real video reconstruction")
    selected, output = tmp_path / "selected", tmp_path / "output"
    selected.mkdir()
    output.mkdir()
    frames = []
    for index in (1, 2):
        path = selected / f"frame_{index:06d}.jpg"
        Image.new("RGB", (8, 8), (index * 60, 0, 0)).save(path)
        frames.append({"filename": path.name, "sha256": sha256(path), "width": 8, "height": 8, "timestamp_sec": float(index)})
    write_json(selected / "manifest.json", {"status": "SUCCESS", "cache_key": "test-selection", "code_hash": code_hash(),
                                             "config": config["preprocessing"], "seed": config["project"]["seed"],
                                             "source_sha256": sha256(video), "frames": frames})
    write_json(output / "gpu_report.json", {"test_fixture": True})
    options = {"feature_extractor": ["FeatureExtraction.use_gpu", "FeatureExtraction.gpu_index", "FeatureExtraction.max_image_size", "SiftExtraction.max_num_features"],
               "sequential_matcher": ["FeatureMatching.use_gpu", "FeatureMatching.gpu_index", "FeatureMatching.max_num_matches", "SequentialMatching.overlap"]}
    environment = {"cuda": {"status": "SUCCESS"}, "colmap": {"binary": "test-colmap", "version": "test build with CUDA", "binary_sha256": "fixture", "options": options},
                   "gpu_ba": {"verified": False, "reason": "explicit blocked BA fixture"}}
    calls = []

    def fake_process(command, log, timeout):
        calls.append(command[1])
        database = output / "database.db"
        with sqlite3.connect(database) as connection:
            if command[1] == "feature_extractor":
                connection.executescript("CREATE TABLE images(name TEXT); CREATE TABLE keypoints(image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB); CREATE TABLE descriptors(image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB); CREATE TABLE two_view_geometries(pair_id INTEGER, rows INTEGER, cols INTEGER, config INTEGER, data BLOB);")
                connection.executemany("INSERT INTO images VALUES (?)", [(r["filename"],) for r in frames])
                for table in ("keypoints", "descriptors"):
                    connection.executemany(f"INSERT INTO {table} VALUES (?, 1, 1, ?)", [(1, b"a"), (2, b"b")])
            elif command[1] == "sequential_matcher":
                connection.execute("INSERT INTO two_view_geometries VALUES (1, 1, 2, 2, ?)", (b"test",))
            else:
                pytest.fail("Mapper must not execute for a blocked backend")
        return {"output": "Creating SIFT GPU feature extractor/matcher (mock fixture)", "returncode": 0}

    monkeypatch.setattr(colmap_runner, "run_process", fake_process)
    result = colmap_runner.run_colmap(config, tmp_path, output, selected, environment, resume=False)
    assert result["status"] == "BLOCKED_GPU_BA"
    assert result["registered_images"] is None
    assert calls == ["feature_extractor", "sequential_matcher"]
    before = sha256(output / "database.db")
    resumed = colmap_runner.run_colmap(config, tmp_path, output, selected, environment, resume=True)
    assert resumed["status"] == "BLOCKED_GPU_BA"
    assert calls == ["feature_extractor", "sequential_matcher"]
    assert sha256(output / "database.db") == before
    assert not list(output.glob("sparse_*"))

