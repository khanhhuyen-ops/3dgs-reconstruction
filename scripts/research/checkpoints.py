"""CPU-only restricted checkpoint inspection; never fall back to unrestricted pickle."""
from __future__ import annotations

import _codecs
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

from discover import ROOT, RESEARCH, read_json, write_csv, write_json


def load_torch():
    handles=[]
    if os.name=="nt":
        for path in (Path(sys.base_prefix)/"Library/bin", Path(sys.base_prefix)/"DLLs"):
            if path.is_dir():handles.append(os.add_dll_directory(str(path)))
    import torch
    torch.set_num_threads(2)
    return torch,handles


def inspect(path: Path, torch) -> dict:
    import numpy as np
    from torch.torch_version import TorchVersion
    # Legacy project RNG captures need these fixed, imported numeric constructors.
    # No function/class is dynamically imported from checkpoint-controlled names.
    safe=[np.core.multiarray._reconstruct,np.core.multiarray.scalar,np.ndarray,np.dtype,TorchVersion,_codecs.encode]
    safe += [type(np.dtype(t)) for t in (np.uint32,np.int32,np.int64,np.float32,np.float64)]
    torch.serialization.add_safe_globals(safe)
    try:
        modern=zipfile.is_zipfile(path)
        storages=None
        if modern:
            with zipfile.ZipFile(path) as archive:
                if not any(n.endswith("/data.pkl") for n in archive.namelist()):
                    return {"readability":"UNKNOWN_NON_TORCH_ZIP","schema":None}
                storages=sum("/data/" in n for n in archive.namelist())
        payload=torch.load(path,map_location="cpu",weights_only=True,mmap=modern)
        if not isinstance(payload,dict):
            return {"readability":"SAFE_READ_OTHER_LAYOUT","schema":None,"storage_entries":storages}
        schema=payload.get("schema")
        if schema not in ("b0-checkpoint-v1","pgsr-full-checkpoint-v1"):
            return {"readability":"SAFE_READ_PRETRAINED_OR_OTHER","schema":schema,"storage_entries":storages}
        cap=payload["native_capture"]
        pgsr=schema=="pgsr-full-checkpoint-v1"
        if len(cap)!=(16 if pgsr else 12):raise ValueError("Native capture length mismatch")
        n=len(cap[1])
        if not n or tuple(cap[1].shape)!=(n,3):raise ValueError("Empty/malformed Gaussian positions")
        indices=range(1,14) if pgsr else range(1,10)
        # Full shape checks include parameters and density-control state.
        if pgsr:
            sh=payload["sh_degree"]
            shapes=[(n,3),(n,6),(n,1,3),(n,(sh+1)**2-1,3),(n,3),(n,4),(n,1),(n,),(n,),(n,1),(n,1),(n,1),(n,1)]
            optimizer=cap[14]
            expected={"xyz":cap[1],"knn_f":cap[2],"f_dc":cap[3],"f_rest":cap[4],"scaling":cap[5],"rotation":cap[6],"opacity":cap[7]}
        else:
            sh=round((cap[3].shape[1]+1)**.5)-1
            shapes=[(n,3),(n,1,3),(n,(sh+1)**2-1,3),(n,3),(n,4),(n,1),(n,),(n,1),(n,1)]
            optimizer=cap[10]
            expected={"xyz":cap[1],"f_dc":cap[2],"f_rest":cap[3],"scaling":cap[4],"rotation":cap[5],"opacity":cap[6]}
        for i,shape in zip(indices,shapes):
            if tuple(cap[i].shape)!=shape or cap[i].dtype!=torch.float32:
                raise ValueError("Native tensor layout/dtype mismatch")
        checked=0
        def finite(value):
            nonlocal checked
            if isinstance(value,torch.Tensor) and value.is_floating_point():
                if value.device.type!="cpu":raise ValueError("Audit unexpectedly moved a tensor to GPU")
                for chunk in value.reshape(-1).split(262144):
                    if not torch.isfinite(chunk).all():raise ValueError("Nonfinite checkpoint tensor")
                checked+=value.numel()
            elif isinstance(value,dict):
                for item in value.values():finite(item)
            elif isinstance(value,(list,tuple)):
                for item in value:finite(item)
        finite(cap)
        if {g["name"] for g in optimizer["param_groups"]}!=set(expected):
            raise ValueError("Missing/duplicate native optimizer groups")
        steps=payload.get("optimizer_steps")
        for group in optimizer["param_groups"]:
            state=optimizer["state"].get(group["params"][0],{})
            if steps and group["name"]!="knn_f" and not state:
                raise ValueError("Missing used-parameter Adam state")
            for key in ("exp_avg","exp_avg_sq"):
                if key in state and state[key].shape!=expected[group["name"]].shape:
                    raise ValueError("Optimizer row state mismatch")
        if set(payload["rng"])!={"python","numpy","torch_cpu","torch_cuda"}:
            raise ValueError("Incomplete RNG state")
        sampler=payload["sampler_remaining_ids"]
        if len(set(sampler))!=len(sampler):raise ValueError("Duplicate sampler entries")
        if pgsr:
            finite(payload["app_state"]);finite(payload["app_optimizer"])
            if payload["app_state"]["appear_ab"].shape[1]!=2:
                raise ValueError("Invalid exposure table")
        return {"readability":"VERIFIED_RESTRICTED_CPU_LOAD","schema":schema,"iteration":payload["iteration"],
                "optimizer_steps":steps,"gaussians":n,"active_sh_degree":cap[0],"identity":payload["identity"],
                "exposure_rows":len(payload["app_state"]["appear_ab"]) if pgsr else None,
                "full_state":True,"finite_float_elements_checked":checked,"storage_entries":storages,
                "scope":"CPU serialization, finite/layout/Adam/RNG checks; CUDA restore not rerun during audit"}
    except (OSError,ValueError,RuntimeError,KeyError,TypeError,zipfile.BadZipFile,Exception) as exc:
        # Unknown serialized globals stay unknown and preserved; never use weights_only=False.
        return {"readability":"UNKNOWN_OR_RESTRICTED_LOAD_FAILED","reason":type(exc).__name__+": "+str(exc)[:600]}
    finally:
        torch.serialization.clear_safe_globals()
        gc.collect()


def main() -> None:
    with (RESEARCH/"cleanup/file_inventory.csv").open(encoding="utf-8") as stream:
        inventory=list(csv.DictReader(stream))
    runs=read_json(RESEARCH/"metadata/experiment_registry.json",[])
    sources=[r for r in inventory if r["category"]=="checkpoint_or_pretrained_weight"]
    # Prefer canonical experiment copies; identical hashes are inspected only once.
    sources.sort(key=lambda r:(not r["path"].startswith("outputs/tanks_and_temples/"),r["path"]))
    cache=read_json(RESEARCH/"metadata/checkpoint_structures.json",{})
    torch,handles=load_torch()
    rows=[];canonicals={}
    for i,row in enumerate(sources):
        path=ROOT/row["path"]
        sha=row["sha256"]
        if not sha:raise ValueError("Hash inventory must complete before checkpoint validation")
        if sha not in cache or cache[sha].get("reason","").startswith("BadZipFile:"):
            print(f"Inspect checkpoint {i+1}/{len(sources)}: {row['path']}",flush=True)
            cache[sha]=inspect(path,torch)
            write_json(RESEARCH/"metadata/checkpoint_structures.json",cache)
        check=cache[sha]
        owners=[r for r in runs if any(row["path"].startswith(prefix+"/") for prefix in [r["root"],*r["aliases"]])]
        owner=max(owners,key=lambda r:len(r["root"]),default=None)
        synthetic="preflight" in Path(row["path"]).parts or "unit_fixture" in check.get("identity",{}) or "unit" in check.get("identity",{})
        if synthetic:owner=None
        compatibility={}
        if owner and check.get("full_state"):
            config=owner["config_values"]
            expected_config=hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest()
            identity=check.get("identity",{})
            compatibility["config_identity_matches"]=identity.get("config")==expected_config
            compatibility["model_schema_matches"]=(owner["model"]=="B2_PGSR")== (check["schema"]=="pgsr-full-checkpoint-v1")
            dataset=ROOT/config["dataset"]
            split=read_json(dataset/"splits.json",{})
            compatibility["split_identity_matches"]=identity.get("split")==split.get("hash")
            compatibility["iteration_within_declared_schedule"]=0<check["iteration"]<=config["iterations"]
            expected_steps=check["iteration"]-int(check["iteration"]==config["iterations"])
            compatibility["native_optimizer_step_accounting_matches"]=check["optimizer_steps"]==expected_steps
        canonical=canonicals.setdefault(sha,row["path"])
        pointer=read_json(path.parent/"latest.json",{})
        published=pointer.get("path")==path.name
        expected=pointer.get("sha256") if published else None
        match=sha==expected if expected else None
        if match is False:
            classification="CORRUPTED"
        elif compatibility and not all(compatibility.values()):
            classification="UNKNOWN"
        elif row["path"].startswith("phase03_results/recovery/") and check.get("full_state"):
            classification="ESSENTIAL"
        elif canonical!=row["path"]:
            classification="REDUNDANT"
        elif check.get("full_state") and owner and (published or check.get("iteration")==owner["target_iterations"]):
            classification="ESSENTIAL"
        elif check.get("full_state"):
            classification="USEFUL"
        elif "PRETRAINED" in check.get("readability",""):
            classification="ESSENTIAL"
        else:classification="UNKNOWN"
        record={"path":row["path"],"bytes":int(row["bytes"]),"sha256":sha,"classification":classification,
                "canonical_copy":canonical,"duplicate":canonical!=row["path"],"model":owner["model"] if owner else "pretrained/unit/unknown",
                "experiment_id":owner["experiment_id"] if owner else None,"dataset":owner["dataset"] if owner else None,
                "configuration":owner["configuration"] if owner else None,"schema":check.get("schema"),
                "compatibility":compatibility,"scope":"synthetic/unit fixture" if synthetic or not owner else "project scene experiment",
                "iteration":check.get("iteration"),"gaussians":check.get("gaussians"),"readability":check["readability"],
                "published_pointer_sha_match":match,"is_declared_final":bool(owner and check.get("iteration")==owner["target_iterations"]),
                "resume_required":bool(check.get("full_state") and published),"inference_required":"PLY exists separately where exported; checkpoint preserves native inference state",
                "mesh_required":"No implemented compatible extractor verified","reason":check.get("reason"),
                "deletion_policy":"REQUIRE USER APPROVAL; preserve full state and reported-result evidence"}
        if row["path"].startswith("phase03_results/recovery/"):
            record["deletion_policy"]="PRESERVE REQUIRED ORIGINAL FLAT RECOVERY; canonical imported folder is not directly resume-compatible"
        rows.append(record)
    write_csv(RESEARCH/"metadata/checkpoint_audit.csv",rows)
    for run in runs:
        cp=next((r for r in rows if r["path"]==run["final_checkpoint"]),None)
        if cp:
            run["can_resume"]="STRUCTURALLY_VERIFIED_FULL_STATE_CUDA_RESTORE_NOT_RERUN" if cp["readability"]=="VERIFIED_RESTRICTED_CPU_LOAD" and cp["published_pointer_sha_match"] is not False and all(cp["compatibility"].values()) else "UNKNOWN_OR_BLOCKED"
            run["checkpoint_validation"]=cp["readability"]
    write_json(RESEARCH/"metadata/experiment_registry.json",runs)
    write_csv(RESEARCH/"metadata/experiment_registry.csv",runs,[k for k in runs[0] if k not in ("config_values","training_result","memory","device_wide_memory")])
    print(f"Checkpoint files {len(rows)}, unique content hashes {len(cache)}, restricted CPU inspection only",flush=True)


if __name__=="__main__":main()
