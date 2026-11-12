"""Phase 0 environment audit, with independent per-capability blockers."""
from __future__ import annotations

import platform
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import repo_path, resolve_tool
from src.common.gpu import audit_cuda
from src.common.io import PipelineError, code_hash, run_process, write_json
from src.geometry.sfm.gpu_probe import inspect_colmap, probe_gpu_ba
from src.preprocessing.video_probe import probe_video


def audit_environment(config: dict, root: Path, output: Path) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    audit = output / "audits" / stamp
    audit.mkdir(parents=True)
    report = {"timestamp_utc": stamp, "platform": platform.platform(), "python": sys.version,
              "python_executable": sys.executable, "code_hash": code_hash(), "tools": {}, "blockers": [], "audit_path": str(audit)}
    for name in ("ffmpeg", "ffprobe", "colmap"):
        binary = resolve_tool(name, config)
        report["tools"][name] = binary
        if not binary:
            report["blockers"].append({"stage": "preprocess" if name != "colmap" else "colmap", "reason": f"Missing {name}; set {name.upper()}_BIN or tools.{name}"})
        elif name != "colmap":
            try:
                report[f"{name}_version"] = run_process([binary, "-version"], audit / f"{name}_version.log", timeout=15)["output"].splitlines()[0]
            except PipelineError as exc:
                report["blockers"].append({"stage": "preprocess", "reason": str(exc)})
                report["tools"][name] = None
    cuda = audit_cuda()
    report["cuda"] = cuda
    if cuda["status"] != "SUCCESS":
        report["blockers"].append({"stage": "select", "reason": cuda["error"]})
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            report["nvidia_smi"] = run_process([smi, "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"], audit / "nvidia_smi.log", timeout=15)["output"].strip()
        except PipelineError as exc:
            report["nvidia_smi_error"] = str(exc)
    if report["tools"]["ffprobe"]:
        try:
            report["video"] = probe_video(repo_path(root, config["input"]["video"]), report["tools"]["ffprobe"])
        except PipelineError as exc:
            report["blockers"].append({"stage": "extract", "reason": str(exc)})
    capabilities = None
    ba = {"status": "BLOCKED_GPU_BA", "verified": False, "backend": "UNAVAILABLE", "reason": "COLMAP/CUDA not available"}
    if report["tools"]["colmap"]:
        try:
            capabilities = inspect_colmap(report["tools"]["colmap"], audit, config["sfm"]["matching"])
            if cuda["status"] == "SUCCESS":
                ba = probe_gpu_ba(capabilities, audit)
        except PipelineError as exc:
            report["blockers"].append({"stage": "colmap", "reason": str(exc)})
    if not ba["verified"]:
        report["blockers"].append({"stage": "mapper", "status": "BLOCKED_GPU_BA", "reason": ba["reason"]})
    report["colmap"] = capabilities
    report["gpu_ba"] = ba
    report["status"] = "SUCCESS" if not report["blockers"] else "BLOCKED"
    gpu = {"cuda_tensor": cuda, "ba": ba, "feature_runtime": None, "matching_runtime": None,
           "cpu_tasks": ["FFmpeg decode/resize", "image decode", "I/O", "SQLite", "orchestration", "COLMAP geometric verification / pose estimation", "diagnostic metrics"]}
    write_json(audit / "environment.json", report)
    write_json(audit / "gpu_report.json", gpu)
    write_json(output / "environment.json", report)
    write_json(output / "gpu_report.json", gpu)
    return report
