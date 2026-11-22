"""Lightweight evidence-recovery and deletion-boundary regression tests; no training."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research"))
import analyze
import cleanup
from discover import digest,write_json
from assets import ply_header


def test_loss_recovery_preserves_raw_zero_and_missing_components(tmp_path,monkeypatch):
    monkeypatch.setattr(analyze,"ROOT",tmp_path)
    source=tmp_path/"training.jsonl"
    source.write_text('{"iteration":10,"loss":1.2,"raw":{"rgb":0.2,"flatten":0.01,"single":0,"multi_geo":0,"multi_ncc":0}}\ntruncated\n')
    rows,errors=analyze.load_history(tmp_path,{"method":"B2_PGSR"})
    assert len(rows)==1 and len(errors)==1
    assert rows[0]["loss_normal"]==0
    assert "loss_l1" not in rows[0]
    assert rows[0]["source_line"]==1


def test_console_does_not_replace_full_precision_structured_row(tmp_path,monkeypatch):
    monkeypatch.setattr(analyze,"ROOT",tmp_path)
    (tmp_path/"training.jsonl").write_text('{"iteration":100,"loss":0.123456789,"l1":0.1,"dssim":0.2}\n')
    (tmp_path/"run.log").write_text('INFO: iteration 100/1000 loss 0.123457\nINFO: iteration 200/1000 loss 0.100000\n')
    rows,_=analyze.load_history(tmp_path,{"method":"B0_Original3DGS"})
    assert len(rows)==2 and rows[0]["loss_total"]==0.123456789
    assert rows[1]["source_kind"].startswith("recorded_console")


def test_summary_never_inserts_missing_zero():
    rows=[{"iteration":1,"loss_total":1.,"source_file":"fixture"},{"iteration":10,"loss_total":.5,"source_file":"fixture"}]
    summary=analyze.summarize(rows,{"components":{"loss_l1":{"status":"NOT RECORDED"}}},100)
    assert summary["components"]["loss_l1"]["mean"] is None
    assert summary["components"]["loss_l1"]["missing_value_count"]==2
    assert summary["components"]["loss_total"]["iteration_of_minimum"]==10
    assert summary["distinct_recorded_iterations"]==2


@pytest.mark.parametrize("relative",["datasets/raw/__pycache__/x.pyc","external/library/__pycache__/x.pyc",".tools/pgsr_env/Lib/site-packages/x.pyc",".git/__pycache__/x.pyc","outputs/scene/checkpoint.pth"])
def test_protected_files_are_not_disposable(tmp_path,monkeypatch,relative):
    monkeypatch.setattr(cleanup,"ROOT",tmp_path)
    assert not cleanup.disposable(tmp_path/relative)


def test_changed_cache_is_preserved_after_planning(tmp_path,monkeypatch):
    monkeypatch.setattr(cleanup,"ROOT",tmp_path)
    monkeypatch.setattr(cleanup,"RESEARCH",tmp_path/"research")
    cache=tmp_path/"src/__pycache__/fixture.pyc";cache.parent.mkdir(parents=True)
    cache.write_bytes(b"old")
    row={"original_path":cache.relative_to(tmp_path).as_posix(),"bytes":3,"sha256":digest(cache),"mtime_ns":cache.stat().st_mtime_ns,"proposed_action":"AUTO_DELETE_DISPOSABLE_FILE"}
    write_json(tmp_path/"research/cleanup/cleanup_manifest.json",{"status":"PLAN_READY","candidates":[row]})
    cache.write_bytes(b"new content")
    result=cleanup.apply()
    assert result["removed"]==0 and cache.read_bytes()==b"new content"


def test_ply_point_cloud_is_not_reported_as_mesh(tmp_path):
    path=tmp_path/"fixture.ply"
    path.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nelement face 0\nend_header\n")
    assert ply_header(path)["kind"]=="point_cloud_not_mesh"
