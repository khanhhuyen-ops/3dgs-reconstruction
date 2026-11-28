"""Print recorded per-experiment losses for reports without running training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research"

PRIMARY_COMPONENTS = (
    "loss_total",
    "loss_l1",
    "loss_ssim",
    "loss_depth",
    "loss_normal",
    "loss_geometry",
    "loss_multiview",
    "loss_regularization",
)


def display(value: object) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def selected_runs(registry: list[dict], experiment_ids: list[str]) -> list[dict]:
    if not experiment_ids:
        return registry
    by_id = {run["experiment_id"]: run for run in registry}
    missing = sorted(set(experiment_ids) - set(by_id))
    if missing:
        raise ValueError(f"Unknown experiment IDs: {', '.join(missing)}")
    return [by_id[identifier] for identifier in experiment_ids]


def component_order(components: dict) -> list[str]:
    extras = sorted(
        key
        for key in components
        if key not in PRIMARY_COMPONENTS
        and key.startswith(("raw_", "weighted_"))
    )
    return [key for key in PRIMARY_COMPONENTS if key in components] + extras


def summary_lines(run: dict, summary: dict) -> list[str]:
    lines = [
        f"[{run['experiment_id']}]",
        (
            f"model={run['model']} status={run['training_status']} "
            f"iteration={run['actual_observed_completed_iteration']}/"
            f"{run['target_iterations']} recorded_rows={run['loss_observations']}"
        ),
        "component | status | initial | final | minimum | min_iteration | mean | std | valid | missing",
    ]
    for name in component_order(summary["components"]):
        item = summary["components"][name]
        lines.append(
            " | ".join(
                (
                    name,
                    item.get("status", "NOT RECORDED"),
                    display(item.get("initial_recorded_value")),
                    display(item.get("final_recorded_value")),
                    display(item.get("minimum")),
                    display(item.get("iteration_of_minimum")),
                    display(item.get("mean")),
                    display(item.get("standard_deviation_population")),
                    display(item.get("valid_observations", 0)),
                    display(item.get("missing_value_count", 0)),
                )
            )
        )
    lines.append(
        "note=Totals are comparable only within an identical objective/configuration; "
        "N/A is never treated as zero."
    )
    return lines


def history_lines(run: dict, history_path: Path) -> list[str]:
    with history_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    preferred = list(PRIMARY_COMPONENTS) + [
        "raw_rgb",
        "weighted_rgb",
        "raw_flatten",
        "weighted_flatten",
        "raw_single",
        "weighted_single",
        "raw_multi_geo",
        "weighted_multi_geo",
        "raw_multi_ncc",
        "weighted_multi_ncc",
        "geometry_valid_pixels",
        "multi_valid_pixels",
        "ncc_valid_patches",
        "exposure_gate_used",
    ]
    available = [
        key for key in preferred if any(row.get(key, "") != "" for row in rows)
    ]
    lines = [f"history[{run['experiment_id']}] rows={len(rows)}"]
    lines.append("iteration | " + " | ".join(available))
    for row in rows:
        lines.append(
            " | ".join([row["iteration"]] + [display(row.get(key)) for key in available])
        )
    return lines


def render(experiment_ids: list[str], include_history: bool) -> str:
    registry = json.loads(
        (RESEARCH / "metadata/experiment_registry.json").read_text(encoding="utf-8")
    )
    blocks = []
    for run in selected_runs(registry, experiment_ids):
        directory = RESEARCH / "metrics" / run["experiment_id"]
        summary_path = directory / "loss_summary.json"
        history_path = directory / "loss_history.csv"
        if not summary_path.is_file() or not history_path.is_file():
            raise FileNotFoundError(f"Missing recovered loss artifacts for {run['experiment_id']}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        lines = summary_lines(run, summary)
        if include_history:
            lines += [""] + history_lines(run, history_path)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print evidence-backed loss summaries or recorded iteration rows"
    )
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--history", action="store_true", help="Print every recovered row")
    parser.add_argument("--output", type=Path, help="Also write the exact console text")
    args = parser.parse_args()
    text = render(args.experiment, args.history)
    print(text, end="")
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
