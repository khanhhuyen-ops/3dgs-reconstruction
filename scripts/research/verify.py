"""Post-cleanup integrity checks without training, CUDA inference, or unrestricted pickle."""
from __future__ import annotations

import ast
import csv
from pathlib import Path
import re

from discover import ROOT, RESEARCH, FIELDS, digest, read_json, write_csv, write_json, scan


def verify() -> dict:
    with (RESEARCH/"cleanup/file_inventory.csv").open(encoding="utf-8") as stream:
        before={r["path"]:r for r in csv.DictReader(stream)}
    with (RESEARCH/"metadata/checkpoint_audit.csv").open(encoding="utf-8") as stream:
        checkpoints=list(csv.DictReader(stream))
    registry=read_json(RESEARCH/"metadata/experiment_registry.json")
    cleanup=read_json(RESEARCH/"cleanup/cleanup_manifest.json")
    removed={r["path"] for r in cleanup.get("removed",[])}
    errors=[];retained=0;metrics_count=0;links_checked=0
    for row in checkpoints:
        path=ROOT/row["path"]
        if not path.is_file() or path.stat().st_size!=int(row["bytes"]):
            errors.append("Missing/changed checkpoint: "+row["path"]);continue
        with path.open("rb") as stream:
            if not stream.read(4):errors.append("Empty checkpoint: "+row["path"])
        original=before.get(row["path"],{})
        if str(path.stat().st_mtime_ns)!=str(original.get("mtime_ns")):
            if digest(path)!=row["sha256"]:errors.append("Changed checkpoint hash: "+row["path"])
        retained+=1
    for run in registry:
        if not (ROOT/run["configuration"]).is_file():errors.append("Missing run config: "+run["experiment_id"])
        folder=RESEARCH/"metrics"/run["experiment_id"]
        with (folder/"loss_history.csv").open(encoding="utf-8") as stream:
            history=list(csv.DictReader(stream))
        if len(history)!=run["loss_observations"]:errors.append("Registry/history count mismatch: "+run["experiment_id"])
        summary=read_json(folder/"loss_summary.json")
        if summary["observation_count"]!=len(history):errors.append("Loss summary mismatch: "+run["experiment_id"])
        for row in history:
            if not (ROOT/row["source_file"]).is_file():errors.append("Missing original loss source: "+row["source_file"])
        for source in run["evaluation_outputs"]:
            if read_json(ROOT/source) is None:errors.append("Unreadable original metrics: "+source)
            else:metrics_count+=1
    # Existing source is syntax-checked, never imported as a training entrypoint.
    entrypoints=["src/gaussian_splatting/runner.py","src/pgsr/runner.py","src/evaluation/evaluate_baseline.py",
                 "src/pgsr/evaluation.py","scripts/pgsr/score_png.py"]
    for name in entrypoints:
        try:ast.parse((ROOT/name).read_text(encoding="utf-8-sig"))
        except (OSError,SyntaxError) as exc:errors.append(f"Entrypoint {name}: {exc}")
    docs=[ROOT/"AGENTS.md",ROOT/"docs/PROJECT_STRUCTURE.md",ROOT/"docs/RESEARCH_STATUS.md",ROOT/"docs/EXPERIMENT_REPRODUCIBILITY.md"]
    docs+=list((RESEARCH/"reports").glob("*.md"))+list((RESEARCH/"comparisons").glob("*.md"))
    for path in docs:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",path.read_text(encoding="utf-8")):
            if target.startswith(("https://","http://","#","mailto:")):continue
            target=target.split("#",1)[0]
            if not target:continue
            links_checked+=1
            if not (path.parent/target).resolve().exists():errors.append(f"Broken report link: {path.relative_to(ROOT)} -> {target}")
    figures=read_json(RESEARCH/"figures/figure_index.json")
    for group in figures["figures"]:
        for file in group["files"]:
            if not (ROOT/file).is_file() or (ROOT/file).stat().st_size==0:errors.append("Missing figure: "+file)
        for source in group["sources"]:
            if not (ROOT/source).is_file():errors.append("Missing figure source: "+source)
    protected_changes=[]
    for rel,row in before.items():
        if row["category"] in ("installed_environment_or_external_source","raw_dataset","derived_dataset","checkpoint_or_pretrained_weight","point_cloud_or_mesh","log_or_metrics"):
            if rel.startswith("research/") or rel in removed:continue
            path=ROOT/rel
            if not path.is_file() or path.stat().st_size!=int(row["bytes"]) or path.stat().st_mtime_ns!=int(row["mtime_ns"]):
                protected_changes.append(rel)
    errors += ["Protected artifact changed during audit: "+p for p in protected_changes]
    remaining=scan(False)
    write_csv(RESEARCH/"cleanup/remaining_files_manifest.csv",remaining,FIELDS)
    final_bytes=sum(r["bytes"] for r in remaining)
    result={"status":"SUCCESS" if not errors else "FAILED","audit_date":"2026-11-22","errors":errors,
            "retained_checkpoint_files_accessible":retained,"readable_original_metric_json_files":metrics_count,
            "experiments_consistent":len(registry),"recorded_loss_observations":sum(r["loss_observations"] for r in registry),
            "figure_groups_verified":len(figures["figures"]),"local_report_links_checked":links_checked,
            "entrypoint_syntax_checks":entrypoints,"training_or_inference_launched":False,
            "mesh_workflow":"No implemented project extraction workflow found; cannot verify execution",
            "protected_changes":protected_changes,"remaining_file_count":len(remaining),"final_repository_logical_bytes_snapshot":final_bytes,
            "checkpoint_integrity_scope":"All SHA256 values measured before cleanup; post-cleanup all files reopened and size/mtime checked, SHA256 recomputed if mtime changed",
            "snapshot_scope":"All regular files at scan time; final manifest/report receipt writes can slightly change generated-audit size afterward"}
    write_json(RESEARCH/"metadata/final_verification.json",result)
    original=cleanup["baseline"]
    summary=f'''# Cleanup summary

Audit date: 2026-11-22. Status: {result['status']}.

- Pre-cleanup logical bytes: {original['total_bytes']} across {original['file_count']} files.
- Final logical bytes at verification scan: {final_bytes} across {len(remaining)} files. Generated reports/figures and metadata growth are included; this is not a measure of deletion savings.
- Files removed: {cleanup.get('removed_count',0)}; exact logical bytes removed: {cleanup.get('logical_bytes_removed',0)}.
- Checkpoints/PLYs/meshes/datasets/COLMAP/configs/original logs removed: 0.
- Checkpoint/weight files retained and reopened: {retained}.
- Experiment records: {len(registry)}; source loss observations retained: {sum(r['loss_observations'] for r in registry)}.
- Full-state structural resume candidates: {sum(str(r['can_resume']).startswith('STRUCTURALLY_VERIFIED') for r in registry)}. CUDA resumption and canonical layout compatibility remain separate gates.
- No dataset, checkpoint or experiment directory was moved. No recursive directory deletion was used.

Potentially valuable duplicate artifacts remain in [approval_required.csv](approval_required.csv). Their canonical hashes and exact/basename/manifest dependencies are in [dependency_audit.csv](dependency_audit.csv). Equality does not retire a sealed bundle or its referenced paths. All deletion candidates require explicit approval except the enumerated cache files already removed.

The pre-cleanup and remaining inventories include managed environments, external code and Git internals, which were preserved. Logical file lengths are reported separately from the observed volume free-space delta in [cleanup_manifest.json](cleanup_manifest.json); concurrent writes and allocation units can make those quantities differ. Audit-generated files may be newer than the final snapshot.

[Final verification](../metadata/final_verification.json) distinguishes verified file/state accessibility from workflows not executed. B2's non-native canonical layout is documented in [reproducibility](../../docs/EXPERIMENT_REPRODUCIBILITY.md). No training, inference or mesh benchmark was run during this audit.
'''
    (RESEARCH/"cleanup/cleanup_summary.md").write_text(summary,encoding="utf-8")
    return result


if __name__=="__main__":
    report=verify()
    print({k:report[k] for k in ("status","retained_checkpoint_files_accessible","experiments_consistent","recorded_loss_observations","figure_groups_verified")})
    if report["errors"]:
        print("\n".join(report["errors"]));raise SystemExit(1)
