"""Export sampled Gaussian loss records without filling missing observations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from src.common.io import staging_directory, write_json


def flatten(record: dict, prefix: str = "") -> dict:
    result = {}
    for key, value in record.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def export(log: Path, destination: Path) -> dict:
    rows = []
    previous = 0
    maximum_error = 0.0
    with log.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            iteration = record["iteration"]
            if iteration <= previous:
                raise ValueError(f"Non-increasing loss iteration {iteration}")
            previous = iteration
            loss = record["losses"]
            terms = loss["weighted"]
            if not terms or set(terms) != set(loss["raw"]) or set(terms) != set(loss["coefficients"]):
                raise ValueError(f"Incomplete additive loss decomposition at iteration {iteration}")
            reconstructed = math.fsum(terms.values())
            if not math.isclose(loss["total"], reconstructed, rel_tol=5e-6, abs_tol=1e-8):
                raise ValueError(f"Recorded weighted terms do not sum to loss at iteration {iteration}")
            reconstruction_error = abs(loss["total"] - reconstructed)
            maximum_error = max(maximum_error, reconstruction_error)
            rows.append({**flatten(record), "reconstructed_total": reconstructed,
                         "reconstruction_abs_error": reconstruction_error})
    if not rows:
        raise ValueError("No sampled training losses recorded")
    staging = staging_directory(destination)
    fields = ["iteration", *sorted(set().union(*(row.keys() for row in rows)) - {"iteration"})]
    with (staging / "loss_history.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    x = [row["iteration"] for row in rows]
    axes[0].plot(x, [row["losses.total"] for row in rows], label="Total", marker=".", markersize=3)
    for name in (field for field in fields if field.startswith("losses.weighted.")):
        axes[1].plot(x, [row.get(name, math.nan) for row in rows],
                     label=name.removeprefix("losses.weighted."), marker=".", markersize=3)
    for axis in axes:
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1].set_xlabel("Training iteration (sampled)")
    figure.savefig(staging / "loss_curves.png", dpi=160)
    plt.close(figure)
    report = {"status": "SUCCESS", "observations": len(rows), "first_iteration": x[0],
              "last_iteration": x[-1], "maximum_total_reconstruction_abs_error": maximum_error}
    write_json(staging / "loss_summary.json", report)
    staging.rename(destination)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(export(arguments.log, arguments.destination)))
