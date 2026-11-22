"""Index dataset/SfM/model artifacts using file metadata and lightweight PLY headers."""
from __future__ import annotations

import csv
from pathlib import Path
import re
import subprocess

from discover import ROOT, RESEARCH, read_json, write_csv, write_json


def ply_header(path: Path) -> dict:
    with path.open("rb") as stream:
        header=bytearray()
        for _ in range(4096):
            line=stream.readline(1024)
            header.extend(line)
            if line.strip()==b"end_header":break
        else:raise ValueError("PLY header exceeded inspection limit")
    text=header.decode("ascii",errors="replace")
    if not text.startswith("ply"):raise ValueError("Missing PLY signature")
    elements={m[1]:int(m[2]) for m in re.finditer(r"^element\s+(\w+)\s+(\d+)\s*$",text,re.M)}
    properties=re.findall(r"^property\s+\S+\s+(\S+)\s*$",text,re.M)
    return {"vertices":elements.get("vertex"),"faces":elements.get("face",0),
            "kind":"triangle_or_polygon_mesh" if elements.get("face",0)>0 else "point_cloud_not_mesh",
            "gaussian_attributes":all(p in properties for p in ("f_dc_0","opacity","scale_0","rot_0"))}


def main() -> None:
    with (RESEARCH/"cleanup/file_inventory.csv").open(encoding="utf-8") as stream:
        inventory=list(csv.DictReader(stream))
    assets=[]
    for row in inventory:
        rel=row["path"];path=ROOT/rel
        if row["category"] in ("installed_environment_or_external_source","git_internal","audit_generated"):continue
        p=Path(rel)
        role=None
        if p.suffix.lower() in (".ply",".obj",".stl",".glb",".off"):
            role="ground_truth_geometry" if "ground_truth" in p.parts else "generated_geometry"
        elif p.name in ("cameras.bin","images.bin","points3D.bin","cameras.txt","images.txt","points3D.txt","project.ini") or p.suffix.lower()==".db":role="COLMAP_or_SfM"
        elif p.suffix.lower()==".mp4":role="source_video"
        elif p.suffix.lower() in (".json",".csv") and any(k in p.name.lower() for k in ("metric","manifest","split","metadata","camera","pose")):role="experiment_metadata_or_metrics"
        elif p.suffix.lower() in (".md",".pdf") and rel.startswith(("docs/","paper/")):role="existing_report_or_reference"
        elif p.name.startswith("events.out.tfevents"):role="tensorboard"
        if role:
            record={"path":rel,"role":role,"bytes":int(row["bytes"]),"sha256":row["sha256"] or None}
            if p.suffix.lower()==".ply":
                try:record.update(ply_header(path))
                except (OSError,ValueError) as exc:record["parse_error"]=str(exc)
            assets.append(record)
    write_csv(RESEARCH/"metadata/artifact_index.csv",assets)
    trained={"B0_Original3DGS":"src/gaussian_splatting","B2_PGSR":"src/pgsr"}
    revisions={}
    for directory in ("gaussian-splatting","PGSR","pytorch3d"):
        try:revisions[directory]=subprocess.check_output(["git","-C",str(ROOT/"external"/directory),"rev-parse","HEAD"],text=True).strip()
        except subprocess.CalledProcessError:revisions[directory]="UNKNOWN"
    write_json(RESEARCH/"metadata/implementation_inventory.json",{
        "implemented_project_methods":trained,"unimplemented_baselines":["B1_2DGS","B3_GOF","Ours"],
        "source_revisions":revisions,"training_entrypoints":["src/gaussian_splatting/runner.py","src/pgsr/runner.py"],
        "evaluation_entrypoints":["src/evaluation/evaluate_baseline.py","src/pgsr/evaluation.py","scripts/pgsr/score_png.py"],
        "mesh_entrypoint":"NOT IMPLEMENTED IN PROJECT",
        "configuration_files":[r["path"] for r in inventory if r["path"].startswith("configs/")],
        "notebooks":[r["path"] for r in inventory if r["path"].startswith("notebooks/")],
        "dataset_reference_files":[a["path"] for a in assets if a["role"] in ("source_video","COLMAP_or_SfM")],
        "predicted_mesh_count":sum(a.get("faces",0)>0 and a["role"]=="generated_geometry" for a in assets),
        "geometry_validation_scope":"PLY header element counts only; full vertex finiteness/inference not rerun by this audit"})
    print(f"Indexed scientific artifacts: {len(assets)}; predicted PLY meshes: {sum(a.get('faces',0)>0 and a['role']=='generated_geometry' for a in assets)}")


if __name__=="__main__":main()
