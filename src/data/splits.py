"""Frozen, disjoint video-aware splits; no implicit native-loader holdout."""

from __future__ import annotations

import re

import numpy as np

from src.common.io import fingerprint
from src.data.colmap import Model


def natural_key(name: str) -> list:
    return [int(x) if x.isdigit() else x.casefold() for x in re.split(r"(\d+)", name)]


def make_split(
    model: Model, config: dict, source_mode: str, frames: dict | None = None
) -> dict:
    frames = frames or {}
    if source_mode == "video":
        if any(
            im.name not in frames or frames[im.name].get("timestamp_sec") is None
            for im in model.images.values()
        ):
            raise ValueError(
                "Video split requires real per-image timestamp manifest; filename/FPS inference forbidden"
            )
        order = sorted(
            model.images,
            key=lambda iid: (
                frames[model.images[iid].name]["timestamp_sec"],
                frames[model.images[iid].name]["frame_id"],
                iid,
            ),
        )
    else:
        order = sorted(
            model.images, key=lambda iid: (natural_key(model.images[iid].name), iid)
        )
    subsets = {"train": [], "val": [], "test": []}
    if config["strategy"] == "interleaved":
        for index, iid in enumerate(order):
            offset = index % config["period"]
            split = (
                "test"
                if offset == config["test_offset"]
                else "val"
                if offset == config["val_offset"]
                else "train"
            )
            subsets[split].append(iid)
    else:
        sequence = np.array(order)
        if config["strategy"] == "random":
            sequence = np.random.default_rng(config["seed"]).permutation(sequence)
        nval = max(1, int(len(order) * config["ratios"][1]))
        ntest = max(1, int(len(order) * config["ratios"][2]))
        ntrain = len(order) - nval - ntest
        if ntrain < 2:
            raise ValueError("Too few images for requested train/val/test ratios")
        for key, subset in zip(
            subsets,
            (
                sequence[:ntrain],
                sequence[ntrain : ntrain + nval],
                sequence[ntrain + nval :],
            ),
        ):
            subsets[key] = sorted([int(v) for v in subset], key=order.index)
    if any(not v for v in subsets.values()) or len(subsets["train"]) < 2:
        raise ValueError("All splits must be nonempty; train needs at least two views")
    content = {
        "image_ids": subsets,
        "ordered_image_ids": order,
        "policy": config,
        "effective_ratios": {k: len(v) / len(order) for k, v in subsets.items()},
        "disclosure": "COLMAP poses/geometry may use held-out RGB; fixed-pose transductive evaluation, not strict unseen-view reconstruction",
        "overlap_warning": "Adjacent video views may overlap; temporal_block is an explicit stress-test variant, not proof of independent camera estimation",
    }
    if config["protocol"] == "strict_train_only":
        # A declaration alone cannot prove training-only geometry or test registration.
        raise ValueError(
            "BLOCKED_INPUT: strict_train_only requires independently qualified train-only SfM and test registration evidence; SfM is never rerun by Phase01"
        )
    content["hash"] = fingerprint(content)
    return content
