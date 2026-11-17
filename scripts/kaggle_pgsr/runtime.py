"""PGSR Kaggle orchestration. Training and evaluation always use isolated CUDA processes."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from handoff import ROOT, verify, read_json, write_json, sha256, export, extract_scene, seal


def invoke(*arguments):
    subprocess.run([sys.executable,"-m","src.pgsr.runner",*map(str,arguments)],cwd=ROOT,check=True)


def prepare(source,archive,previous=None):
    import yaml
    from src.pgsr.config import load
    from src.pgsr.native import code_identity
    manifest=verify(source)
    if manifest["schema"]!="phase03-pgsr-source-v1" or manifest["source_code_hash"]!=code_identity():
        raise ValueError("PGSR source identity mismatch")
    config=load(ROOT/"configs/training/pgsr_research_quality.yaml")
    dataset=ROOT/config["dataset"]
    if not dataset.exists():
        extract_scene(archive,dataset)
    run=ROOT/config["output_root"]/("pgsr-kaggle-"+code_identity()[:10])
    if previous:
        old=verify(previous)
        if old["schema"]!="phase03-pgsr-output-v1" or old["source_manifest_sha256"]!=sha256(source/"manifest.json"):
            raise ValueError("Previous session source differs; cannot resume another method/source")
        if read_json(previous/"run/resolved_config.json")!=config:
            raise ValueError("Previous PGSR optimization config changed")
        shutil.copytree(previous/"run",run)
    elif run.exists():
        raise FileExistsError(run)
    write_json(ROOT/"pgsr_active_run.json",{"run":run.relative_to(ROOT).as_posix(),
               "source_manifest_sha256":sha256(source/"manifest.json"),"resume":bool(previous)})
    return str(run)


def smoke():
    subprocess.run([sys.executable,"scripts/pgsr/check_geometry_cuda.py","--output",
                    "/kaggle/working/phase03_cuda_numerics"],cwd=ROOT,check=True)
    run=ROOT/"outputs/tanks_and_temples/Ignatius/research/B2_PGSR/kaggle-geometry-smoke"
    config=ROOT/"configs/training/pgsr_smoke.yaml"
    invoke("--config",config,"--run",run,"--stop-after",80)
    invoke("--config",config,"--run",run,"--resume")
    invoke("--config",config,"--run",run,"--stage","evaluate")
    result=read_json(run/"training_result.json")
    if result["iterations"]!=300 or not all(result["geometry_coverage_totals"].values()):
        raise ValueError("Real CUDA geometry/resume smoke incomplete")
    return str(run)


def session(output):
    active=read_json(ROOT/"pgsr_active_run.json")
    run=ROOT/active["run"]
    arguments=["--config",ROOT/"configs/training/pgsr_research_quality.yaml","--run",run]
    if active["resume"]:
        arguments.append("--resume")
    try:
        invoke(*arguments)
    finally:
        if (run/"checkpoints/latest.json").is_file():
            export(run,output/"recovery",active["source_manifest_sha256"])
            smoke_run=ROOT/"outputs/tanks_and_temples/Ignatius/research/B2_PGSR/kaggle-geometry-smoke"
            if smoke_run.exists():
                # Preserve small actual CUDA evidence, not development training checkpoints.
                target=output/"smoke_evidence"
                shutil.copytree(smoke_run,target,ignore=shutil.ignore_patterns("*.pth","*.ply","*.npz"))
            write_json(output/"index.json",{"recovery":"recovery","evaluation":None,
                       "status":"RECOVERY_SAVED","iteration":read_json(run/"checkpoints/latest.json")["iteration"]})
    return read_json(run/"training_result.json")


def evaluate(output):
    active=read_json(ROOT/"pgsr_active_run.json")
    run=ROOT/active["run"]
    result=read_json(run/"training_result.json")
    if result["iterations"]!=30000:
        return {"status":"PARTIAL_RESUME_REQUIRED","iterations":result["iterations"]}
    invoke("--config",ROOT/"configs/training/pgsr_research_quality.yaml","--run",run,"--stage","evaluate")
    target=output/"evaluation"
    shutil.copytree(run/"evaluation",target/"evaluation")
    seal(target,{"schema":"phase03-pgsr-evaluation-v1","complete":True,"iteration":30000,
                 "recovery_manifest_sha256":sha256(output/"recovery/manifest.json"),
                 "source_manifest_sha256":active["source_manifest_sha256"]})
    write_json(output/"index.json",{"recovery":"recovery","evaluation":"evaluation",
               "status":"SUCCESS","iteration":30000})
    return read_json(run/"evaluation/result.json")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["prepare","smoke","train","evaluate"])
    p.add_argument("--source",type=Path)
    p.add_argument("--archive",type=Path)
    p.add_argument("--previous",type=Path)
    p.add_argument("--output",type=Path,default=Path("/kaggle/working/phase03_results"))
    args=p.parse_args()
    if args.stage=="prepare":
        print(prepare(args.source,args.archive,args.previous))
    elif args.stage=="smoke":
        print(smoke())
    elif args.stage=="train":
        print(session(args.output))
    else:
        print(evaluate(args.output))
