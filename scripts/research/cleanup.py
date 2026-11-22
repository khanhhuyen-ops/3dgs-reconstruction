"""Dependency-aware cleanup plan; only enumerated bytecode/tool-cache files can be removed."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import shutil

from discover import ROOT, RESEARCH, FIELDS, digest, read_json, write_csv, write_json, category


def inventory() -> list[dict]:
    with (RESEARCH/"cleanup/file_inventory.csv").open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def disposable(path: Path) -> bool:
    resolved=path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()) or path.is_symlink():return False
    relative=resolved.relative_to(ROOT.resolve())
    if relative.parts[0] in ("datasets","external",".git","research",".venv") or "site-packages" in relative.parts:return False
    if relative.as_posix().startswith((".tools/b0_env/",".tools/pgsr_env/",".tools/kaggle_control/")):return False
    return ("__pycache__" in relative.parts and relative.suffix in (".pyc",".pyo")) or relative.parts[0] in (".pytest_cache",".ruff_cache")


def plan() -> list[dict]:
    rows=inventory()
    references=[]
    for row in rows:
        p=ROOT/row["path"]
        if p.suffix.lower() not in (".py",".ps1",".sh",".md",".json",".yaml",".yml",".ipynb"):continue
        if int(row["bytes"])>4*1024*1024 or row["category"] in ("git_internal","installed_environment_or_external_source","raw_dataset","derived_dataset","audit_generated"):continue
        try:references.append((row["path"],p.read_text(encoding="utf-8-sig",errors="replace").replace("\\","/")))
        except OSError:pass
    hashes=defaultdict(list)
    for row in rows:
        if row["sha256"]:hashes[row["sha256"]].append(row)
    duplicate_sets=[]
    candidates=[];dependencies=[]
    for sha,copies in hashes.items():
        if len(copies)<2:continue
        copies.sort(key=lambda r:(not r["path"].startswith("datasets/"),not r["path"].startswith("outputs/tanks_and_temples/"),len(r["path"]),r["path"]))
        canonical=copies[0]
        duplicate_sets.append({"sha256":sha,"bytes_per_copy":int(canonical["bytes"]),"canonical":canonical["path"],"copies":[r["path"] for r in copies],"action":"PRESERVE; protected artifact copies require approval"})
        for row in copies[1:]:
            if row["path"].startswith(("datasets/",".git/","external/")) or row["category"]=="audit_generated":continue
            hits=[name for name,text in references if row["path"] in text]
            basename_hits=[name for name,text in references if Path(row["path"]).name in text and name not in hits]
            sealed=[]
            for ancestor in (ROOT/row["path"]).parents:
                if ancestor==ROOT:break
                manifest=read_json(ancestor/"manifest.json",{})
                if (ROOT/row["path"]).relative_to(ancestor).as_posix() in manifest.get("files",{}):
                    sealed.append((ancestor/"manifest.json").relative_to(ROOT).as_posix())
            decision={"path":row["path"],"canonical_copy":canonical["path"],"sha256":sha,
                      "exact_path_reference_files":hits,"basename_reference_files":basename_hits,
                      "sealed_manifest_memberships":sealed,"decision":"APPROVAL REQUIRED; canonical equality does not retire bundle/path dependencies"}
            dependencies.append(decision)
            required_recovery=row["path"].startswith("phase03_results/recovery/")
            if required_recovery:
                decision["decision"]="PRESERVE REQUIRED ORIGINAL RECOVERY: canonical imported layout is not directly resume-compatible"
            candidates.append({"original_path":row["path"],"bytes":int(row["bytes"]),"category":row["category"],
                               "reason":"SHA256-exact duplicate of retained canonical copy", "dependency_verification":decision,
                               "canonical_copy":canonical["path"],"sha256":sha,
                               "proposed_action":"PRESERVE_REQUIRED_ORIGINAL_RECOVERY" if required_recovery else "REQUIRE_USER_APPROVAL",
                               "estimated_storage_savings":0 if required_recovery else int(row["bytes"]),"risk_level":"HIGH" if sealed or hits or required_recovery else "MEDIUM"})
    for row in rows:
        path=ROOT/row["path"]
        if disposable(path):
            candidates.append({"original_path":row["path"],"bytes":int(row["bytes"]),"category":row["category"],
                               "reason":"Regenerable Python bytecode or pytest/ruff cache outside installed packages and protected artifacts",
                               "dependency_verification":"Interpreter/tool cache only; source, logs, datasets, environments and model artifacts retained",
                               "canonical_copy":None,"sha256":row["sha256"] or digest(path),"mtime_ns":int(row["mtime_ns"]),
                               "proposed_action":"AUTO_DELETE_DISPOSABLE_FILE","estimated_storage_savings":int(row["bytes"]),"risk_level":"LOW"})
    write_json(RESEARCH/"cleanup/duplicate_groups.json",duplicate_sets)
    write_csv(RESEARCH/"cleanup/dependency_audit.csv",dependencies,["path","canonical_copy","sha256","exact_path_reference_files","basename_reference_files","sealed_manifest_memberships","decision"])
    write_csv(RESEARCH/"cleanup/deletion_plan.csv",candidates)
    write_csv(RESEARCH/"cleanup/approval_required.csv",[r for r in candidates if r["proposed_action"]=="REQUIRE_USER_APPROVAL"])
    write_json(RESEARCH/"cleanup/cleanup_manifest.json",{"status":"PLAN_READY","audit_date":"2026-11-22",
               "baseline":read_json(RESEARCH/"metadata/discovery.json"),"candidates":candidates,
               "policy":"Only precisely enumerated bytecode/tool cache files may be auto-deleted; all potentially valuable artifacts preserved"})
    return candidates


def apply() -> dict:
    manifest=read_json(RESEARCH/"cleanup/cleanup_manifest.json")
    if manifest["status"]!="PLAN_READY":raise ValueError("Generate a new plan before applying cleanup")
    removed=[];skipped=[]
    free_before=shutil.disk_usage(ROOT).free
    for row in manifest["candidates"]:
        if row["proposed_action"]!="AUTO_DELETE_DISPOSABLE_FILE":continue
        path=ROOT/row["original_path"]
        if not disposable(path):raise ValueError("Cleanup path violates the explicit disposable-file policy")
        if not path.exists():skipped.append({"path":row["original_path"],"reason":"Already absent"});continue
        stat=path.stat()
        if stat.st_size!=row["bytes"] or stat.st_mtime_ns!=row["mtime_ns"] or digest(path)!=row["sha256"]:
            skipped.append({"path":row["original_path"],"reason":"Changed since planning; preserved"});continue
        # Unlink one fully verified file at a time. Never recurse or remove directories.
        path.unlink()
        removed.append({"path":row["original_path"],"bytes":row["bytes"],"sha256":row["sha256"]})
    free_after=shutil.disk_usage(ROOT).free
    manifest.update(status="SAFE_CLEANUP_APPLIED",removed=removed,skipped=skipped,
                    removed_count=len(removed),logical_bytes_removed=sum(r["bytes"] for r in removed),
                    volume_free_bytes_before=free_before,volume_free_bytes_after=free_after,
                    observed_free_space_delta=free_after-free_before,
                    storage_measurement_note="Logical bytes are exact file lengths removed; volume free-space delta may include concurrent writes and cluster allocation",
                    protected_artifacts_deleted=0)
    write_json(RESEARCH/"cleanup/cleanup_manifest.json",manifest)
    return {"removed":len(removed),"logical_bytes_removed":manifest["logical_bytes_removed"],"protected_artifacts_deleted":0}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--apply-disposable",action="store_true")
    p.add_argument("--refresh-plan-preserve-receipt",action="store_true")
    args=p.parse_args()
    if args.refresh_plan_preserve_receipt:
        previous=read_json(RESEARCH/"cleanup/cleanup_manifest.json")
        if previous["status"]!="SAFE_CLEANUP_APPLIED":raise ValueError("An applied receipt is required")
        candidates=plan()
        current=read_json(RESEARCH/"cleanup/cleanup_manifest.json")
        current["supplementary_inventory"]=current["baseline"]
        for key,value in previous.items():
            if key!="candidates":current[key]=value
        current["candidates"]=candidates
        current["plan_refresh_note"]="Additional read-only full project-artifact duplicate audit; original cleanup receipt retained; no further deletions"
        write_json(RESEARCH/"cleanup/cleanup_manifest.json",current)
        print({"updated_candidates":len(candidates),"previous_removed_count":current["removed_count"]})
    else:
        print(apply() if args.apply_disposable else {"planned_files":len(plan())})
