"""Export scientific plots from actual B0 JSONL logs; no model execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import ROOT, repo_path


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = repo_path(ROOT, args.run)
    if not run.is_relative_to(ROOT / "outputs"):
        raise ValueError("Expected a B0 output run")
    rows = [
        json.loads(line)
        for line in (run / "training.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    destination = run / "training_curves.png"
    if destination.exists():
        raise FileExistsError(destination)
    x = [r["iteration"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    axes[0, 0].plot(x, [r["loss"] for r in rows], lw=1, label="Sampled training loss")
    axes[0, 0].plot(x, [r["l1"] for r in rows], lw=0.8, alpha=0.7, label="L1")
    axes[0, 0].set_ylabel("Photometric loss")
    axes[0, 0].legend()
    axes[0, 1].plot(
        x, [r["train_psnr"] for r in rows], lw=1, label="Sampled training view"
    )
    vals = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(run.glob("validation_*.json"))
    ]
    if vals:
        vx = [int(p.stem.split("_")[-1]) for p in sorted(run.glob("validation_*.json"))]
        axes[0, 1].plot(
            vx,
            [v["mean_psnr"] for v in vals],
            marker="o",
            ms=3,
            label=f"Fixed validation subset ({len(vals[0]['image_ids'])} views)",
        )
    axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 1].legend()
    axes[1, 0].plot(x, [r["gaussians"] for r in rows], lw=1)
    if config["max_gaussians"]:
        axes[1, 0].axhline(
            config["max_gaussians"],
            ls="--",
            color="darkred",
            label="Declared profile cap",
        )
        axes[1, 0].legend()
    axes[1, 0].set_ylabel("Gaussian count")
    for key, label in [
        ("allocated_bytes", "Allocated"),
        ("reserved_bytes", "Reserved"),
        ("peak_allocated_bytes", "Peak allocated"),
    ]:
        axes[1, 1].plot(x, [r[key] / 1024**3 for r in rows], lw=1, label=label)
    axes[1, 1].set_ylabel("PyTorch process memory (GiB)")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel("Completed iteration")
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"B0 {config['profile']}: actual logged samples; not paper benchmark scores"
    )
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    print(destination.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
