"""Stage orchestration; all image/geometry work lives in separate modules."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

from src.common.config import ROOT, load_config, repo_path, resolve_tool
from src.common.environment import audit_environment
from src.common.io import PipelineError, fingerprint, read_json, write_json

LOGGER = logging.getLogger(__name__)


def prepare_output(output: Path, config: dict, resume: bool) -> None:
    marker = output / "run.json"
    config_key = fingerprint(config)
    if output.exists() and any(output.iterdir()):
        if not marker.is_file():
            raise PipelineError("FAILED", f"Existing unowned output preserved; choose a new paths.output: {output}")
        if read_json(marker)["config_key"] != config_key:
            raise PipelineError("FAILED", "Configuration changed; use a new experiment output directory")
        if not resume:
            raise PipelineError("FAILED", "Experiment exists; pass --resume or choose new output paths")
    output.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        write_json(marker, {"config_key": config_key, "config": config})


def run_pipeline(args: argparse.Namespace) -> dict:
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    output = repo_path(ROOT, config["paths"]["output"])
    if args.dry_run:
        return {"status": "DRY_RUN", "stage": args.stage, "input": config["input"]["video"], "paths": config["paths"],
                "tools": {name: resolve_tool(name, config) for name in config["tools"]},
                "gpu_policy": config["gpu_policy"], "actions": ["environment audit and real CUDA/BA probes", "requested stage only; prerequisites must validate"],
                "mapper": "blocked unless live CASPAR probe succeeds", "note": "No commands or writes executed"}
    prepare_output(output, config, args.resume)
    environment = audit_environment(config, ROOT, output)
    LOGGER.info("Environment audit %s; GPU BA %s", environment["status"], environment["gpu_ba"]["status"])
    if args.stage == "preflight":
        return {"status": environment["status"], "report": str(output / "environment.json"), "blockers": environment["blockers"]}
    if args.stage == "custom":
        raise PipelineError("BLOCKED_V0_PREREQUISITE", "Custom V1 is deferred until video V0 is actually validated under GPU-strict; see docs/video_sfm.md")
    tools = environment["tools"]
    extracted = repo_path(ROOT, config["paths"]["extracted_frames"])
    selected = repo_path(ROOT, config["paths"]["selected_frames"])
    if args.stage in ("extract", "preprocess", "all"):
        if not tools["ffmpeg"] or not tools["ffprobe"]:
            raise PipelineError("STOP_INPUT_INVALID", "FFmpeg/ffprobe required; see environment.json")
        from src.preprocessing.extract_frames import extract_frames
        result = extract_frames(repo_path(ROOT, config["input"]["video"]), extracted, config["preprocessing"], tools, ROOT, args.resume)
        LOGGER.info("Extraction SUCCESS: %d frames", len(result["frames"]))
        if args.stage == "extract":
            return {"status": "SUCCESS", "frames": len(result["frames"]), "manifest": str(extracted / "manifest.json")}
    if args.stage in ("select", "preprocess", "all"):
        from src.preprocessing.select_keyframes import select_keyframes
        result = select_keyframes(extracted, selected, config["preprocessing"], ROOT, config["project"]["seed"], args.resume)
        LOGGER.info("Selection SUCCESS: %d frames on %s", len(result["frames"]), result["gpu"]["device"])
        if args.stage != "all":
            return {"status": "SUCCESS", "frames": len(result["frames"]), "manifest": str(selected / "manifest.json"), "gpu": result["gpu"]}
    from src.geometry.sfm.colmap_runner import run_colmap
    return run_colmap(config, ROOT, output, selected, environment, args.resume)


def main() -> int:
    parser = argparse.ArgumentParser(description="Video -> COLMAP SfM V0, GPU-strict (no CPU feature/matching/BA fallback)")
    parser.add_argument("--config", type=Path, default=Path("configs/sfm_ignatius.yaml"))
    parser.add_argument("--stage", choices=["preflight", "extract", "select", "preprocess", "colmap", "all", "custom"], default="preflight")
    parser.add_argument("--dry-run", action="store_true", help="Describe actions; execute no subprocess and write no files")
    parser.add_argument("--resume", action="store_true", help="Validate config, hashes, manifests and recorded stage provenance before reuse")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        result = run_pipeline(args)
    except (PipelineError, ValueError, KeyError, OSError, RuntimeError) as exc:
        result = {"status": getattr(exc, "status", "FAILED"), "error": str(exc)}
        LOGGER.error("%s: %s", result["status"], exc)
        # Preserve previous valid artifacts and retain a separate failure record.
        try:
            config = load_config(args.config if args.config.is_absolute() else ROOT / args.config)
            output = repo_path(ROOT, config["paths"]["output"])
            if (output / "run.json").is_file():
                write_json(output / "failures" / f"{uuid.uuid4().hex}.json", {"stage": args.stage, **result, "argv": sys.argv})
        except (ValueError, KeyError, OSError):
            pass
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
    return 0 if result["status"] in ("SUCCESS", "DRY_RUN") else 2


if __name__ == "__main__":
    raise SystemExit(main())

