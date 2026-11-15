"""COLMAP little-endian/TXT model I/O, following colmap.github.io/format.html.

This downstream reader does not modify or replace the protected SfM reader.
IDs remain dictionary keys; point2D indices remain zero-based array indices.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from src.common.io import fingerprint

# Official model IDs/parameter counts. Parsing does not imply projection support.
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
MODEL_IDS = {name: (idx, count) for idx, (name, count) in CAMERA_MODELS.items()}
STEMS = ("cameras", "images", "points3D")


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class Image:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray


@dataclass
class Point:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    track: np.ndarray


@dataclass
class Model:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    points: dict[int, Point]


def _unpack(stream: BinaryIO, fmt: str) -> tuple:
    codec = struct.Struct("<" + fmt)
    data = stream.read(codec.size)
    if len(data) != codec.size:
        raise ValueError(f"Truncated COLMAP file at byte {stream.tell()}")
    return codec.unpack(data)


def _array(stream: BinaryIO, count: int, dtype: np.dtype) -> np.ndarray:
    size = count * dtype.itemsize
    if size > Path(stream.name).stat().st_size - stream.tell():
        raise ValueError(f"Invalid count/truncated COLMAP file: {stream.name}")
    data = stream.read(size)
    return np.frombuffer(data, dtype=dtype).copy()


def _count(stream: BinaryIO, minimum_bytes: int) -> int:
    count = _unpack(stream, "Q")[0]
    if count * minimum_bytes > Path(stream.name).stat().st_size - stream.tell():
        raise ValueError(f"Invalid count/truncated COLMAP file: {stream.name}")
    return count


def _insert(table: dict, item: Camera | Image | Point) -> None:
    if item.id <= 0 or item.id in table:
        raise ValueError(f"Invalid/duplicate {type(item).__name__} ID: {item.id}")
    table[item.id] = item


def _name(stream: BinaryIO) -> str:
    result = bytearray()
    for _ in range(65536):
        char = stream.read(1)
        if not char:
            raise ValueError("Unterminated COLMAP image name")
        if char == b"\0":
            return result.decode("utf-8")
        result.extend(char)
    raise ValueError("COLMAP image name exceeds 65535 bytes")


def read_binary(directory: Path) -> Model:
    """Read legacy-compatible camera/image/point records without renumbering."""
    cameras, images, points = {}, {}, {}
    for stem in STEMS:
        with (directory / f"{stem}.bin").open("rb") as stream:
            count = _count(stream, {"cameras": 24, "images": 74, "points3D": 51}[stem])
            for _ in range(count):
                if stem == "cameras":
                    idx, model_id, width, height = _unpack(stream, "iiQQ")
                    if model_id not in CAMERA_MODELS:
                        raise ValueError(
                            f"Unsupported COLMAP camera model ID {model_id}"
                        )
                    name, nparams = CAMERA_MODELS[model_id]
                    _insert(
                        cameras,
                        Camera(
                            idx,
                            name,
                            width,
                            height,
                            np.array(_unpack(stream, "d" * nparams)),
                        ),
                    )
                elif stem == "images":
                    record = _unpack(stream, "i7di")
                    name = _name(stream)
                    nobs = _count(stream, 24)
                    obs = _array(
                        stream, nobs, np.dtype([("xy", "<f8", (2,)), ("id", "<i8")])
                    )
                    _insert(
                        images,
                        Image(
                            record[0],
                            np.array(record[1:5]),
                            np.array(record[5:8]),
                            record[8],
                            name,
                            obs["xy"].copy(),
                            obs["id"].copy(),
                        ),
                    )
                else:
                    record = _unpack(stream, "Q3d3Bd")
                    ntrack = _count(stream, 8)
                    track = (
                        _array(stream, ntrack * 2, np.dtype("<i4"))
                        .reshape(-1, 2)
                        .astype(np.int64)
                    )
                    _insert(
                        points,
                        Point(
                            record[0],
                            np.array(record[1:4]),
                            np.array(record[4:7]),
                            record[7],
                            track,
                        ),
                    )
            if stream.read(1):
                raise ValueError(f"Unexpected trailing data: {stream.name}")
    return Model(cameras, images, points)


def read_text(directory: Path) -> Model:
    """Stream TXT rows, preserving empty observation rows and names with spaces."""
    cameras, images, points = {}, {}, {}
    with (directory / "cameras.txt").open(encoding="utf-8-sig") as stream:
        for row in stream:
            if not row.strip() or row.lstrip().startswith("#"):
                continue
            t = row.split()
            if len(t) < 4 or t[1] not in MODEL_IDS or len(t) != 4 + MODEL_IDS[t[1]][1]:
                raise ValueError(f"Invalid/unsupported camera row: {row[:100]}")
            _insert(
                cameras,
                Camera(
                    int(t[0]),
                    t[1],
                    int(t[2]),
                    int(t[3]),
                    np.array(t[4:], dtype=np.float64),
                ),
            )
    with (directory / "images.txt").open(encoding="utf-8-sig") as stream:
        for row in stream:
            if not row.strip() or row.lstrip().startswith("#"):
                continue
            t = row.split(maxsplit=9)
            if len(t) != 10:
                raise ValueError("Invalid image pose row")
            obs = stream.readline()
            if obs == "":
                raise ValueError("Missing image observation row")
            tokens = obs.split()
            if len(tokens) % 3:
                raise ValueError("Invalid image observation triples")
            _insert(
                images,
                Image(
                    int(t[0]),
                    np.array(t[1:5], dtype=np.float64),
                    np.array(t[5:8], dtype=np.float64),
                    int(t[8]),
                    t[9].strip(),
                    np.array(tokens, dtype=object)
                    .reshape(-1, 3)[:, :2]
                    .astype(np.float64),
                    np.array([int(v) for v in tokens[2::3]], dtype=np.int64),
                ),
            )
    with (directory / "points3D.txt").open(encoding="utf-8-sig") as stream:
        for row in stream:
            if not row.strip() or row.lstrip().startswith("#"):
                continue
            t = row.split()
            if len(t) < 8 or (len(t) - 8) % 2:
                raise ValueError("Invalid point track row")
            _insert(
                points,
                Point(
                    int(t[0]),
                    np.array(t[1:4], dtype=np.float64),
                    np.array(t[4:7], dtype=np.int64),
                    float(t[7]),
                    np.array(t[8:], dtype=np.int64).reshape(-1, 2),
                ),
            )
    return Model(cameras, images, points)


def read_model(directory: Path, format: str = "auto") -> Model:
    if format not in ("auto", "bin", "txt"):
        raise ValueError("model format must be auto/bin/txt")
    if format == "auto":
        # Never fall back from a corrupt or partial BIN set to a seemingly valid TXT.
        format = (
            "bin"
            if any((directory / f"{stem}.bin").exists() for stem in STEMS)
            else "txt"
        )
    missing = [
        f"{stem}.{format}"
        for stem in STEMS
        if not (directory / f"{stem}.{format}").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing COLMAP model files at {directory}: {', '.join(missing)}"
        )
    return read_binary(directory) if format == "bin" else read_text(directory)


def discover_models(root: Path) -> list[dict]:
    """Inventory candidates, including incomplete exports; no automatic selection."""
    dirs = {
        p.parent
        for stem in STEMS
        for ext in ("bin", "txt")
        for p in root.rglob(f"{stem}.{ext}")
    }
    return [
        {
            "path": p.as_posix(),
            "formats": {
                ext: [s for s in STEMS if (p / f"{s}.{ext}").is_file()]
                for ext in ("bin", "txt")
            },
        }
        for p in sorted(dirs)
    ]


def semantic_hashes(model: Model) -> dict[str, str]:
    """Float64 exact canonical hashes for official round-trippable TXT/BIN."""
    import hashlib

    digests = {name: hashlib.sha256() for name in STEMS}
    for c in sorted(model.cameras.values(), key=lambda x: x.id):
        digests["cameras"].update(
            fingerprint([c.id, c.model, c.width, c.height, c.params.tolist()]).encode()
        )
    for im in sorted(model.images.values(), key=lambda x: x.id):
        d = digests["images"]
        d.update(
            fingerprint(
                [im.id, im.name, im.camera_id, im.qvec.tolist(), im.tvec.tolist()]
            ).encode()
        )
        d.update(im.xys.astype("<f8").tobytes())
        d.update(im.point3D_ids.astype("<i8").tobytes())
    for p in sorted(model.points.values(), key=lambda x: x.id):
        d = digests["points3D"]
        d.update(fingerprint([p.id, p.xyz.tolist(), p.rgb.tolist(), p.error]).encode())
        d.update(p.track.astype("<i8").tobytes())
    return {name: d.hexdigest() for name, d in digests.items()}


def write_model(model: Model, directory: Path, format: str = "bin") -> None:
    """Write a new derived export; existing files are never overwritten."""
    directory.mkdir(parents=True, exist_ok=True)
    if format not in ("bin", "txt"):
        raise ValueError("Output format must be bin/txt")
    for stem in STEMS:
        if (directory / f"{stem}.{format}").exists():
            raise FileExistsError(f"Refusing to overwrite {directory / stem}")
    if format == "txt":
        with (directory / "cameras.txt").open("x", encoding="utf-8") as f:
            for c in sorted(model.cameras.values(), key=lambda x: x.id):
                f.write(
                    f"{c.id} {c.model} {c.width} {c.height} "
                    + " ".join(format_float(v) for v in c.params)
                    + "\n"
                )
        with (directory / "images.txt").open("x", encoding="utf-8") as f:
            for im in sorted(model.images.values(), key=lambda x: x.id):
                f.write(
                    f"{im.id} "
                    + " ".join(format_float(v) for v in [*im.qvec, *im.tvec])
                    + f" {im.camera_id} {im.name}\n"
                )
                f.write(
                    " ".join(
                        f"{format_float(x)} {format_float(y)} {pid}"
                        for (x, y), pid in zip(im.xys, im.point3D_ids)
                    )
                    + "\n"
                )
        with (directory / "points3D.txt").open("x", encoding="utf-8") as f:
            for p in sorted(model.points.values(), key=lambda x: x.id):
                f.write(
                    f"{p.id} "
                    + " ".join(format_float(v) for v in p.xyz)
                    + " "
                    + " ".join(str(int(v)) for v in p.rgb)
                    + f" {format_float(p.error)} "
                    + " ".join(str(int(v)) for v in p.track.ravel())
                    + "\n"
                )
        return
    with (directory / "cameras.bin").open("xb") as f:
        f.write(struct.pack("<Q", len(model.cameras)))
        for c in sorted(model.cameras.values(), key=lambda x: x.id):
            f.write(
                struct.pack("<iiQQ", c.id, MODEL_IDS[c.model][0], c.width, c.height)
            )
            f.write(c.params.astype("<f8").tobytes())
    with (directory / "images.bin").open("xb") as f:
        f.write(struct.pack("<Q", len(model.images)))
        for im in sorted(model.images.values(), key=lambda x: x.id):
            f.write(struct.pack("<i7di", im.id, *im.qvec, *im.tvec, im.camera_id))
            f.write(im.name.encode("utf-8") + b"\0")
            f.write(struct.pack("<Q", len(im.xys)))
            obs = np.empty(len(im.xys), dtype=[("xy", "<f8", (2,)), ("id", "<i8")])
            obs["xy"], obs["id"] = im.xys, im.point3D_ids
            f.write(obs.tobytes())
    with (directory / "points3D.bin").open("xb") as f:
        f.write(struct.pack("<Q", len(model.points)))
        for p in sorted(model.points.values(), key=lambda x: x.id):
            f.write(
                struct.pack("<Q3d3BdQ", p.id, *p.xyz, *p.rgb, p.error, len(p.track))
            )
            f.write(p.track.astype("<i4").tobytes())


def format_float(value: float) -> str:
    return f"{value:.17g}"


def validate_trivial_rigs(directory: Path, model: Model) -> dict:
    """Accept only verified one-camera reference rigs; reject nontrivial sensors."""
    from src.data.geometry import pose

    rigs, frames = {}, []
    if (directory / "rigs.bin").exists():
        with (directory / "rigs.bin").open("rb") as f:
            for _ in range(_count(f, 16)):
                idx, ns, kind, sensor = _unpack(f, "IIiI")
                if ns != 1 or kind != 0 or idx in rigs:
                    raise ValueError(
                        "Nontrivial/invalid rigs are unsupported; explicit rig adapter required"
                    )
                rigs[idx] = sensor
            if f.read(1):
                raise ValueError("Unexpected rigs.bin data")
        if not (directory / "frames.bin").exists():
            raise ValueError("rigs.bin requires frames.bin for verified equivalence")
        with (directory / "frames.bin").open("rb") as f:
            for _ in range(_count(f, 84)):
                rec = _unpack(f, "II7dI")
                if rec[-1] != 1:
                    raise ValueError("Nontrivial frame sensors unsupported")
                kind, sensor, iid = _unpack(f, "iIQ")
                frames.append(
                    (
                        rec[0],
                        rec[1],
                        np.array(rec[2:6]),
                        np.array(rec[6:9]),
                        kind,
                        sensor,
                        iid,
                    )
                )
            if f.read(1):
                raise ValueError("Unexpected frames.bin data")
    elif (directory / "rigs.txt").exists():
        for row in (directory / "rigs.txt").read_text(encoding="utf-8").splitlines():
            if not row.strip() or row.lstrip().startswith("#"):
                continue
            t = row.split()
            if len(t) != 4 or t[1:3] != ["1", "CAMERA"] or int(t[0]) in rigs:
                raise ValueError("Nontrivial/invalid rig requires an explicit adapter")
            rigs[int(t[0])] = int(t[3])
        for row in (directory / "frames.txt").read_text(encoding="utf-8").splitlines():
            if not row.strip() or row.lstrip().startswith("#"):
                continue
            t = row.split()
            if len(t) != 13 or t[9:11] != ["1", "CAMERA"]:
                raise ValueError("Nontrivial frame requires an explicit adapter")
            frames.append(
                (
                    int(t[0]),
                    int(t[1]),
                    np.array(t[2:6], float),
                    np.array(t[6:9], float),
                    0,
                    int(t[11]),
                    int(t[12]),
                )
            )
    elif any((directory / f"frames.{ext}").exists() for ext in ("txt", "bin")):
        raise ValueError("Frames without rigs cannot be qualified")
    else:
        return {"status": "legacy_no_rigs"}
    seen, frame_ids = set(), set()
    for fid, rid, q, t, kind, sensor, iid in frames:
        if (
            fid in frame_ids
            or iid in seen
            or iid not in model.images
            or kind != 0
            or rigs.get(rid) != sensor
        ):
            raise ValueError("Invalid rig/frame references")
        im = model.images[iid]
        if im.camera_id != sensor or not np.allclose(
            pose(q, t), pose(im.qvec, im.tvec), atol=1e-9, rtol=0
        ):
            raise ValueError("Frame/image pose mismatch")
        seen.add(iid)
        frame_ids.add(fid)
    if seen != set(model.images):
        raise ValueError("Rig frames do not cover all registered images")
    return {"status": "verified_trivial", "rigs": len(rigs), "frames": len(frames)}
