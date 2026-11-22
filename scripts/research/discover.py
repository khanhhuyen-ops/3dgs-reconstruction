"""Read-only repository inventory with reusable size/mtime cache and artifact indexes."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research"
FIELDS = ["path", "bytes", "mtime_ns", "category", "sha256", "hash_status"]


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    temporary=path.with_name("."+path.name+".writing")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k,v in row.items()})
    temporary.replace(path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def category(rel: str) -> str:
    p = Path(rel)
    if rel.startswith(".git/"):
        return "git_internal"
    if rel.startswith(("research/", ".tools/research_")):
        return "audit_generated"
    if rel.startswith((".tools/b0_env/", ".tools/pgsr_env/", ".tools/kaggle_control/", "external/")):
        return "installed_environment_or_external_source"
    if "__pycache__" in p.parts or p.suffix.lower() in (".pyc", ".pyo") or p.parts[0] in (".pytest_cache", ".ruff_cache"):
        return "bytecode_or_tool_cache"
    if rel.startswith("datasets/"):
        return "raw_dataset" if "raw" in p.parts else "derived_dataset"
    if p.suffix.lower() in (".pth", ".pt", ".ckpt", ".safetensors"):
        return "checkpoint_or_pretrained_weight"
    if p.suffix.lower() in (".ply", ".obj", ".stl", ".off", ".glb"):
        return "point_cloud_or_mesh"
    if p.name.startswith("events.out.tfevents"):
        return "tensorboard_events"
    if p.suffix.lower() in (".log", ".jsonl", ".csv"):
        return "log_or_metrics"
    if p.suffix.lower() in (".zip", ".bundle", ".tar", ".gz", ".7z"):
        return "archive"
    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".npz", ".npy", ".blend", ".blend1"):
        return "generated_visual_or_tensor"
    if p.suffix.lower() in (".json", ".yaml", ".yml", ".ini", ".txt"):
        return "metadata_or_config"
    return "source_or_other"


def scan(hash_artifacts: bool = False, hash_all_project: bool = False) -> list[dict]:
    inventory = RESEARCH / "cleanup/file_inventory.csv"
    cache = {}
    if inventory.exists():
        with inventory.open(encoding="utf-8") as stream:
            cache = {r["path"]:r for r in csv.DictReader(stream)}
    rows, errors, links = [], [], []
    def onerror(error):
        errors.append(str(error))
    for folder, directories, filenames in os.walk(ROOT, followlinks=False, onerror=onerror):
        parent = Path(folder)
        for directory in directories[:]:
            path = parent/directory
            if path.is_symlink():
                links.append(path.relative_to(ROOT).as_posix())
                directories.remove(directory)
        for filename in filenames:
            path = parent/filename
            rel = path.relative_to(ROOT).as_posix()
            if path.is_symlink():
                links.append(rel)
                continue
            try:
                stat = path.stat()
                old = cache.get(rel, {})
                valid = str(stat.st_size)==str(old.get("bytes")) and str(stat.st_mtime_ns)==str(old.get("mtime_ns"))
                rows.append({"path":rel,"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns,
                             "category":category(rel),"sha256":old.get("sha256", "") if valid else "",
                             "hash_status":"CACHED_SIZE_MTIME" if valid and old.get("sha256") else "NOT_HASHED"})
            except OSError as exc:
                errors.append(f"{rel}: {exc}")
    rows.sort(key=lambda r:r["path"])
    if hash_artifacts or hash_all_project:
        selected = [r for r in rows if r["category"] in ("checkpoint_or_pretrained_weight", "point_cloud_or_mesh", "archive", "log_or_metrics", "tensorboard_events")
                    and not r["path"].startswith("datasets/")]
        selected += [r for r in rows if r["category"]=="metadata_or_config" and not r["path"].startswith("datasets/") and r["bytes"]<4*1024*1024]
        if hash_all_project:
            selected=[r for r in rows if r["category"] not in ("git_internal","installed_environment_or_external_source","audit_generated","bytecode_or_tool_cache")]
        for index,row in enumerate(selected):
            if not row["sha256"]:
                try:
                    row["sha256"] = digest(ROOT/row["path"])
                    row["hash_status"] = "SHA256_VERIFIED"
                except OSError as exc:
                    errors.append(f"Hash {row['path']}: {exc}")
            if index%40==0:
                print(f"Hash progress {index}/{len(selected)}: {row['path']}",flush=True)
                write_csv(inventory,rows,FIELDS)
    write_csv(inventory,rows,FIELDS)
    sizes=Counter();counts=Counter()
    for row in rows:
        sizes[row["category"]]+=row["bytes"]
        counts[row["category"]]+=1
    write_json(RESEARCH/"metadata/discovery.json",{
        "audit_date":"2026-11-22", "inventory_executed_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"All regular files under repository root including Git, environments and external sources; directory symlinks not followed",
        "total_bytes":sum(r["bytes"] for r in rows),"file_count":len(rows),"bytes_by_category":dict(sizes),
        "counts_by_category":dict(counts),"errors":errors,"symlinks_not_followed":links,
        "size_semantics":"Logical file lengths; not compressed/sparse physical allocation; generated audit files can change after this snapshot"})
    return rows


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--hash-artifacts", action="store_true")
    parser.add_argument("--hash-all-project-artifacts", action="store_true")
    args=parser.parse_args()
    result=scan(args.hash_artifacts,args.hash_all_project_artifacts)
    print(json.dumps({"files":len(result),"bytes":sum(r["bytes"] for r in result)}))
