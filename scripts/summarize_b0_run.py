"""Consolidate actual B0 evidence, preserving interrupted-run timing limitations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import ROOT, repo_path
from src.common.io import read_json, write_json


def summarize(run: Path) -> dict:
    result = read_json(run / "training_result.json")
    if result["status"] != "SUCCESS":
        raise ValueError("Require completed training before final summary")
    rows = [
        json.loads(line)
        for line in (run / "training.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    segments = []
    for row in rows:
        if (
            not segments
            or row["elapsed_segment_seconds"]
            < segments[-1][-1]["elapsed_segment_seconds"]
            or row["iteration"] <= segments[-1][-1]["iteration"]
        ):
            segments.append([])
        segments[-1].append(row)
    missing = []
    segment_info = []
    for index, segment in enumerate(segments):
        last = segment[-1]
        export = (
            run
            / "point_cloud"
            / f"iteration_{last['iteration']:06d}"
            / "validation.json"
        )
        clean_end = export.is_file() and read_json(export)["status"] == "SUCCESS"
        segment_info.append(
            {
                "first_logged_iteration": segment[0]["iteration"],
                "last_logged_iteration": last["iteration"],
                "last_observed_elapsed_seconds": last["elapsed_segment_seconds"],
                "completed_export_at_segment_end": clean_end,
            }
        )
        if index < len(segments) - 1 and not clean_end:
            missing.append(last["elapsed_segment_seconds"])
    # Pinned runner carries timing across clean stops; an abrupt stop has no final
    # timing record. Its last observed elapsed value is a truthful lower bound.
    duration = result["training_seconds_total"] + sum(missing)
    density = [
        json.loads(line)
        for line in (run / "density.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    metrics = {}
    for path in sorted((run / "metrics").glob("*.json")):
        payload = read_json(path)
        metrics[path.name] = {
            "status": payload["status"],
            "view_count": payload["view_count"],
            "aggregate": payload["aggregate_arithmetic_mean_per_view"],
            "path": path.relative_to(ROOT).as_posix(),
        }
    return {
        "status": "SUCCESS",
        "run": run.relative_to(ROOT).as_posix(),
        "iterations": result["iterations"],
        "optimizer_step_invocations": result["optimizer_steps"],
        "initial_gaussians": result["initial_gaussians"],
        "final_gaussians": result["final_gaussians"],
        "final_sampled_training_loss": result["final_sampled_training_loss"],
        "recorded_runner_training_seconds": result["training_seconds_total"],
        "observed_training_wall_seconds": duration,
        "duration_is_lower_bound": bool(missing),
        "timing_note": "Includes loop/validation/checkpoint/export, excludes dataset qualification and offline time between processes; interrupted segments include only last observed elapsed time",
        "segments": segment_info,
        "max_recorded_torch_peak_allocated_bytes": max(
            [result["memory"]["peak_allocated_bytes"]]
            + [r["peak_allocated_bytes"] for r in rows]
        ),
        "max_recorded_torch_peak_reserved_bytes": max(
            [result["memory"]["peak_reserved_bytes"]]
            + [r["peak_reserved_bytes"] for r in rows]
        ),
        "memory_scope": "Maximum of recorded process PyTorch peaks across segments; not exact total CUDA/process/device peak",
        "current_segment_device_wide_sampling": result["device_wide_memory"],
        "density": {
            "completed_events": sum(r.get("status") == "SUCCESS" for r in density),
            "deferred_resource_events": sum(
                r.get("status") == "SKIPPED_RESOURCE_CAP" for r in density
            ),
            "cloned": sum(r.get("cloned", 0) for r in density),
            "split_parents": sum(r.get("split_parents", 0) for r in density),
            "split_children": sum(r.get("split_children", 0) for r in density),
            "pruned": sum(r.get("pruned", 0) for r in density),
            "opacity_resets": sum(r.get("event") == "opacity_reset" for r in density),
        },
        "metrics": metrics,
        "export": result["export"],
        "dataset": result["dataset"],
        "resume_checks": [read_json(p) for p in sorted(run.glob("resume_*.json"))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = repo_path(ROOT, args.run)
    if not run.is_relative_to(ROOT / "outputs"):
        raise ValueError("Expected output run")
    report = summarize(run)
    destination = run / "execution_summary.json"
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                in (
                    "status",
                    "run",
                    "iterations",
                    "observed_training_wall_seconds",
                    "duration_is_lower_bound",
                    "metrics",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
