"""Small Phase01 CLIs sharing configuration, diagnostics and exit semantics."""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import ROOT, repo_path
from src.common.io import PipelineError, read_json, write_json
from src.data.compatibility import check_compatibility
from src.data.config import load_config
from src.data.loader import PreparedDataset, verify_dataset
from src.data.preparation import audit_sources, prepare, qualify


def main(operation: str) -> int:
    parser = argparse.ArgumentParser(
        description=f"Phase01 {operation}: protected source artifacts are read-only"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Prepared dataset directory; required if more than one exists",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Inventory configured model discovery roots",
    )
    parser.add_argument(
        "--references",
        type=Path,
        help="Pinned native reference cache with registry.json; no automatic downloads",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    report_root = repo_path(ROOT, config["output"]["report_root"])
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    run = report_root / (operation + "_" + run_id)
    run.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run / "run.log", encoding="utf-8"),
        ],
    )
    write_json(
        run / "invocation.json",
        {
            "operation": operation,
            "argv": sys.argv,
            "cwd": ".",
            "resolved_config": config,
        },
    )
    try:
        if operation == "validate":
            _, report = qualify(config)
            if args.discover:
                report["discovery"] = audit_sources(config)
        elif operation == "prepare":
            report = prepare(config, resume=args.resume, dry_run=args.dry_run)
        else:
            if args.dataset:
                directory = repo_path(ROOT, str(args.dataset))
            else:
                candidates = sorted(
                    p.parent
                    for p in repo_path(ROOT, config["output"]["dataset_root"]).glob(
                        "*/metadata.json"
                    )
                )
                if len(candidates) != 1:
                    raise ValueError(
                        f"Expected one prepared dataset, found {len(candidates)}; supply --dataset explicitly"
                    )
                directory = candidates[0]
            report = verify_dataset(directory)
            if report["status"] == "SUCCESS":
                if operation == "compatibility":
                    report["compatibility"] = check_compatibility(
                        directory, args.references
                    )
                    report["status"] = report["compatibility"]["status"]
                else:
                    sample = PreparedDataset(directory, verify=False)[0]
                    report["sample"] = {
                        "image_id": sample["image_id"],
                        "rgb_shape": list(sample["rgb"].shape),
                        "rgb_dtype": str(sample["rgb"].dtype),
                        "rgb_range": [
                            float(sample["rgb"].min()),
                            float(sample["rgb"].max()),
                        ],
                    }
                    meta = read_json(directory / "metadata.json")
                    report["statistics"] = meta["statistics"]
                report["dataset_path"] = directory.relative_to(ROOT).as_posix()
    except (OSError, ValueError, KeyError, TypeError, PipelineError) as exc:
        report = {"status": getattr(exc, "status", "FAILED"), "reason": str(exc)}
        logging.error("%s: %s", report["status"], exc)
    write_json(run / "report.json", report)
    print(f"{report['status']} | report={run.relative_to(ROOT).as_posix()}/report.json")
    if "dataset_path" in report:
        print("dataset=" + report["dataset_path"])
    return 0 if report["status"] in ("SUCCESS", "DRY_RUN", "PARTIAL") else 2
