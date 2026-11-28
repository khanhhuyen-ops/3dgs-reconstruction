"""Recover recorded experiment evidence without training, rendering, or changing inputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from discover import ROOT, RESEARCH, digest, read_json, write_csv, write_json

BASE_COLUMNS = ["iteration","epoch","wall_time","loss_total","loss_l1","loss_ssim","loss_depth",
                "loss_normal","loss_geometry","loss_multiview","loss_regularization"]


def existing(root: Path, name: str) -> Path | None:
    for parent in (root, root/"logs", root/"configs"):
        candidate=parent/name
        if candidate.is_file():
            return candidate
    return None


def json_at(root: Path, name: str) -> dict:
    path=existing(root,name)
    return read_json(path,{}) if path else {}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def text_lines(path: Path) -> list[str]:
    data=path.read_bytes()
    return data.decode("utf-16" if data.startswith((b'\xff\xfe',b'\xfe\xff')) else "utf-8-sig",errors="replace").splitlines()


def load_history(root: Path, config: dict) -> tuple[list[dict], list[dict]]:
    observations=[]; failures=[]
    path=existing(root,"training.jsonl")
    pgsr=config.get("method")=="B2_PGSR"
    if path:
        segment=0; previous_elapsed=None
        for line_number,line in enumerate(text_lines(path),1):
            try:
                raw=json.loads(line)
                if "iteration" not in raw or "loss" not in raw: continue
                elapsed=raw.get("elapsed_segment_seconds")
                if previous_elapsed is not None and elapsed is not None and elapsed<previous_elapsed:
                    segment+=1
                previous_elapsed=elapsed
                row={"iteration":raw["iteration"],"epoch":None,"wall_time":None,"loss_total":raw["loss"],
                     "source_file":relative(path),"source_line":line_number,"source_kind":"recorded_jsonl",
                     "elapsed_segment_seconds":elapsed,"segment_index":segment,"observation_index":len(observations),
                     "gaussians":raw.get("gaussians"),"train_psnr":raw.get("train_psnr"),
                     "peak_allocated_bytes":raw.get("peak_allocated_bytes"),"peak_reserved_bytes":raw.get("peak_reserved_bytes")}
                if pgsr:
                    normalized=raw.get("losses",{})
                    components=normalized.get("raw",raw.get("raw",{}))
                    photometric=normalized.get("photometric",{})
                    row.update(loss_normal=components.get("single"),loss_geometry=components.get("multi_geo"),
                               loss_multiview=components.get("multi_ncc"),loss_regularization=components.get("flatten"))
                    if "l1_selected" in photometric:
                        row["loss_l1"]=photometric["l1_selected"]
                    if "ssim_dissimilarity" in photometric:
                        row["loss_ssim"]=photometric["ssim_dissimilarity"]
                    for name,value in components.items():
                        row["raw_"+name]=value
                        row["weighted_"+name]=normalized.get("weighted",raw.get("weighted",{})).get(name)
                        row["coefficient_"+name]=normalized.get("coefficients",raw.get("coefficients",{})).get(name)
                    for name,value in photometric.items():
                        row["photometric_"+name]=value
                    for name in ("geometry_valid_pixels","multi_valid_pixels","ncc_candidate_patches","ncc_valid_patches","exposure_gate_used"):
                        row[name]=raw.get(name)
                else:
                    row.update(loss_l1=raw.get("l1"),loss_ssim=raw.get("dssim"))
                observations.append(row)
            except (ValueError,TypeError) as exc:
                failures.append({"source":relative(path),"line":line_number,"reason":str(exc)})
    # Console precision is lower; use it only when no structured observation exists.
    seen={r["iteration"] for r in observations}
    log=existing(root,"run.log")
    if log:
        for line_number,line in enumerate(text_lines(log),1):
            match=re.search(r"iteration\s+(\d+)/(\d+)\s+loss\s+([-+\d.eE]+)",line)
            if match and int(match[1]) not in seen:
                value=float(match[3]);iteration=int(match[1])
                if math.isfinite(value):
                    observations.append({"iteration":iteration,"loss_total":value,"source_file":relative(log),
                                         "source_line":line_number,"source_kind":"recorded_console_rounded_6dp",
                                         "observation_index":len(observations)})
                    seen.add(iteration)
    return observations,failures


def definitions(config: dict) -> dict:
    pgsr=config.get("method")=="B2_PGSR"
    opt=config.get("optimizer",{})
    if not pgsr:
        weight=opt.get("lambda_dssim",.2)
        components={
            "loss_total":{"variable":"loss","definition":"(1-lambda_dssim)*l1 + lambda_dssim*structural","weighted":True,"activation":"iteration 1 onward"},
            "loss_l1":{"variable":"l1","definition":"mean absolute RGB error; native utils.loss_utils.l1_loss","weighted":False,"weight":1-weight},
            "loss_ssim":{"variable":"structural","definition":"1-SSIM (not SSIM similarity and not half-DSSIM)","weighted":False,"weight":weight},
        }
        source="src/gaussian_splatting/runner.py"
    else:
        components={
            "loss_total":{"variable":"total","definition":"rgb + scale_weight*flat + single_weight*single + geo_weight*geometric + ncc_weight*photometric","weighted":True},
            "raw_rgb":{"variable":"rgb","definition":"(1-lambda_dssim)*L1(gated app or canonical RGB,target) + lambda_dssim*(1-SSIM(canonical,target))","weighted":False,"weight":1},
            "loss_l1":{"variable":"losses.photometric.l1_selected","definition":"mean absolute RGB error from the branch selected by the native exposure gate","weighted":False,"weight":1-opt.get("lambda_dssim",.2),"availability":"Recorded by gaussian-loss-v1 rows; historical rows remain unavailable"},
            "loss_ssim":{"variable":"losses.photometric.ssim_dissimilarity","definition":"1-SSIM of canonical rendered RGB and target","weighted":False,"weight":opt.get("lambda_dssim",.2),"availability":"Recorded by gaussian-loss-v1 rows; historical rows remain unavailable"},
            "loss_regularization":{"variable":"flat","source_name":"raw.flatten","definition":"mean visible minimum activated Gaussian scale","weighted":False,"weight":opt.get("scale_loss_weight"),"activation":"iteration 1 onward"},
            "loss_normal":{"variable":"single","source_name":"raw.single","definition":"full H*W mean of masked-zero edge_weight * L1(native alpha-weighted depth normal - accumulated plane normal)","weighted":False,"weight":opt.get("single_view_weight"),"activation":f"iteration > {opt.get('single_view_weight_from_iter')}"},
            "loss_geometry":{"variable":"geometric","source_name":"raw.multi_geo","definition":"accepted-pixel mean detached exp(-roundtrip_error) * roundtrip_error in pixels","weighted":False,"weight":opt.get("multi_view_geo_weight"),"activation":f"iteration > {opt.get('multi_view_weight_from_iter')} and eligible train neighbor"},
            "loss_multiview":{"variable":"photometric","source_name":"raw.multi_ncc","definition":"accepted-patch mean of detached roundtrip weight * native squared-correlation LNCC loss; not a sum of geometry losses","weighted":False,"weight":opt.get("multi_view_ncc_weight"),"activation":f"iteration > {opt.get('multi_view_weight_from_iter')} and valid nonconstant patches"},
        }
        source="src/pgsr/geometry.py"
    for key in BASE_COLUMNS[3:]:
        if key not in components:
            components[key]={"status":"NOT IMPLEMENTED AS A SEPARATE TERM",
                             "value":None,"weight":None}
    return {"method":config.get("method"),"source":source,"source_sha256":digest(ROOT/source),"components":components,
            "exposure":"PGSR appearance parameters change the gated L1 branch; gaussian-loss-v1 records both L1 branches and the selected branch, but exposure is not a separate additive loss" if pgsr else "Not used",
            "source_mapping":{"loss_l1":"losses.photometric.l1_selected","loss_ssim":"losses.photometric.ssim_dissimilarity","loss_regularization":"losses.raw.flatten","loss_normal":"losses.raw.single","loss_geometry":"losses.raw.multi_geo","loss_multiview":"losses.raw.multi_ncc"} if pgsr else {"loss_l1":"losses.raw.l1","loss_ssim":"losses.raw.ssim_dissimilarity"},
            "missing_policy":"Empty CSV cell/null means unavailable; never substitute zero. Logged scheduled-inactive zeros are retained as observations.",
            "comparison_warning":"Total objectives differ across models; lower total loss is not evidence of better reconstruction"}


def summarize(rows: list[dict], defs: dict, target: int) -> dict:
    keys=list(dict.fromkeys(BASE_COLUMNS[3:]+[k for r in rows for k in r if k.startswith(("raw_","weighted_"))]))
    result={}
    for key in keys:
        valid=[r for r in rows if isinstance(r.get(key),(float,int)) and math.isfinite(r[key])]
        values=[r[key] for r in valid]
        result[key]={"initial_recorded_value":values[0] if values else None,"final_recorded_value":values[-1] if values else None,
                     "minimum":min(values) if values else None,"maximum":max(values) if values else None,
                     "mean":statistics.fmean(values) if values else None,"standard_deviation_population":statistics.pstdev(values) if values else None,
                     "iteration_of_minimum":min(valid,key=lambda r:r[key])["iteration"] if values else None,
                     "valid_observations":len(valid),"missing_value_count":len(rows)-len(valid),
                     "status":"RECORDED" if values else defs["components"].get(key,{}).get("status","NOT RECORDED"),
                     "source_evidence":sorted({r["source_file"] for r in valid})}
        alias={"raw_single":"loss_normal","raw_multi_geo":"loss_geometry","raw_multi_ncc":"loss_multiview",
               "weighted_single":"loss_normal","weighted_multi_geo":"loss_geometry","weighted_multi_ncc":"loss_multiview"}.get(key,key)
        activation=defs["components"].get(alias,{}).get("activation","")
        threshold=re.search(r"iteration > (\d+)",activation)
        if threshold:
            active=[r for r in valid if r["iteration"]>int(threshold[1])]
            result[key]["activation_threshold"]=int(threshold[1])
            result[key]["active_range_observations"]=len(active)
            result[key]["active_range_minimum"]=min((r[key] for r in active),default=None)
            result[key]["statistics_scope"]="Overall statistics include recorded scheduled-inactive zeros; active-range statistics are separate"
    observed={r["iteration"] for r in rows}
    return {"components":result,"observation_count":len(rows),"distinct_recorded_iterations":len(observed),
            "target_iterations":target,"unobserved_target_iterations":target-len({i for i in observed if 1<=i<=target}),
            "missing_count_scope":"Missing-value count counts empty cells among recovered rows, not all unlogged iterations",
            "best_training_is_not_best_validation":True,"smoothing":"NONE","interpolation":"NONE"}


def collect_runs(inventory: list[dict]) -> list[dict]:
    candidates=[]
    for row in inventory:
        rel=row["path"]
        if not rel.endswith("resolved_config.json") or not rel.startswith(("outputs/","phase03_results/")):
            continue
        path=ROOT/rel
        root=path.parent.parent if path.parent.name=="configs" else path.parent
        config=read_json(path,{})
        if config.get("method") not in ("B0_Original3DGS","B2_PGSR"):
            continue
        history=existing(root,"training.jsonl")
        result=existing(root,"training_result.json")
        key=(config["method"],digest(history) if history else relative(root))
        candidates.append((key,root,config))
    groups=defaultdict(list)
    for key,root,config in candidates:
        groups[key].append((root,config))
    runs=[]
    for aliases in groups.values():
        aliases.sort(key=lambda a:(not relative(a[0]).startswith("outputs/tanks_and_temples/"),len(relative(a[0]))))
        root,config=aliases[0]
        identity=json_at(root,"identity.json")
        result=json_at(root,"training_result.json")
        failure=json_at(root,"failure.json")
        manifest=json_at(root,"run_manifest.json")
        pointer=read_json(root/"checkpoints/latest.json",{})
        history,parse_failures=load_history(root,config)
        last_recorded=max((r["iteration"] for r in history),default=0)
        completed=max(result.get("iterations",0),pointer.get("iteration",0),last_recorded)
        expected=config["iterations"]
        if result.get("status")=="SUCCESS" and result.get("iterations")==expected and not failure:
            status="COMPLETED"
        elif failure:
            status="INTERRUPTED" if "RESOURCE" in failure.get("status","") or "INTERRUPT" in failure.get("status","") else "FAILED"
        elif result.get("status","").startswith(("PARTIAL","INTERRUPTED")):
            status="INTERRUPTED"
        elif pointer or history:
            status="PARTIALLY RECOVERABLE"
        else:
            status="UNKNOWN"
        experiment_id=root.name if root.name!="run" else "kaggle-"+config["method"]+"-"+str(completed)
        if root.name=="smoke_evidence":experiment_id="kaggle-pgsr-geometry-smoke"
        directory=RESEARCH/"metrics"/experiment_id
        defs=definitions(config)
        write_json(directory/"loss_definitions.json",defs)
        write_csv(directory/"loss_history.csv",history,BASE_COLUMNS+list(dict.fromkeys(k for r in history for k in r if k not in BASE_COLUMNS)))
        summary=summarize(history,defs,expected)
        write_json(directory/"loss_summary.json",summary)
        initializer=json_at(root,"initialize_result.json")
        if initializer:
            source=existing(root,"initialize_result.json")
            write_json(directory/"initialization_diagnostics.json",{
                "scope":"Initialization/render/backward diagnostic; not a training iteration or convergence history",
                "source":relative(source),"source_sha256":digest(source),"recorded_loss":initializer.get("loss"),
                "reported_scope":initializer.get("scope"),"reported_status":initializer.get("status")})
        validation=[]
        for path in sorted(root.glob("validation_*.json")):
            v=read_json(path,{})
            if v.get("split")!="val":continue
            iteration=int(re.search(r"(\d+)\.json",path.name)[1])
            per=v.get("per_view",[])
            validation.append({"iteration":iteration,"psnr":v.get("mean_psnr"),
                               "ssim":statistics.fmean(r["ssim"] for r in per) if per and all("ssim" in r for r in per) else None,
                               "lpips":None,"views":len(v.get("image_ids",[])),"source":relative(path),"scope":"periodic validation subset, not final test"})
        write_csv(directory/"validation_history.csv",validation,["iteration","psnr","ssim","lpips","views","source","scope"])
        best=max((v for v in validation if v.get("psnr") is not None),key=lambda v:v["psnr"],default=None)
        best_path=root/"checkpoints"/f"iteration_{best['iteration']:06d}.pth" if best else None
        cp=root/"checkpoints"/pointer.get("path", "MISSING")
        environment=json_at(root,"environment.json")
        migration=json_at(root,"migration.json")
        lineage={"relation":"independent run record or no migration evidence"}
        if migration:
            lineage={"relation":"continuation from published parent checkpoint; not an independent replication",
                     "parent_checkpoint_iteration":migration.get("parent_iteration"),"parent_checkpoint_sha256":migration.get("parent_sha256"),
                     "migration_source":relative(existing(root,"migration.json")),"input_manifest_sha256":migration.get("input_manifest_sha256")}
            for candidate in (ROOT/"outputs/kaggle_handoff").glob("*/manifest.json"):
                if digest(candidate)==migration.get("input_manifest_sha256"):
                    lineage["parent_run"]=read_json(candidate,{}).get("parent_run")
                    lineage["input_manifest"]=relative(candidate)
        elapsed_by_segment={}
        for observation in history:
            elapsed=observation.get("elapsed_segment_seconds")
            if elapsed is not None:
                segment=observation.get("segment_index",0)
                elapsed_by_segment[segment]=max(elapsed_by_segment.get(segment,0),elapsed)
        observed_seconds=sum(elapsed_by_segment.values()) if elapsed_by_segment else None
        measured_memory=dict(result.get("memory",{}))
        for name in ("peak_allocated_bytes","peak_reserved_bytes"):
            values=[o[name] for o in history if isinstance(o.get(name),(int,float))]
            values += [v[name] for v in (result.get("memory",{}),failure.get("memory",{})) if isinstance(v.get(name),(int,float))]
            if values:measured_memory[name]=max(values)
        logged_gaussians=next((o["gaussians"] for o in reversed(history) if o.get("gaussians") is not None),None)
        result_current=result.get("iterations",0)>=completed
        timestamps=[]
        for match in re.finditer(r"\d{8}T\d{6}Z",root.name):timestamps.append(match[0])
        metrics=[]
        for folder in (root/"metrics",root/"evaluation"):
            if folder.exists():
                metrics += [relative(p) for p in folder.glob("*.json")]
        dataset_parts=Path(config.get("dataset") or ".").parts
        scene=dataset_parts[2].title() if dataset_parts[:2]==("datasets","tanks_and_temples") else "Unknown"
        record={"experiment_id":experiment_id,"model":config["method"],"dataset":config.get("dataset"),"scene":scene,
                "root":relative(root),"aliases":[relative(a[0]) for a in aliases[1:]],"configuration":relative(existing(root,"resolved_config.json")),
                "lineage":lineage,
                "profile":config.get("profile",config.get("variant")),"seed":config.get("seed"),"resolution_divisor":config.get("resolution"),
                "target_iterations":expected,"actual_observed_completed_iteration":completed,"last_loss_iteration":last_recorded,
                "published_checkpoint_iteration":pointer.get("iteration"),"training_status":status,
                "reported_status":failure.get("status",result.get("status",manifest.get("status","UNKNOWN"))),
                "start_timestamp":timestamps[0] if timestamps else None,"start_timestamp_scope":"UTC encoded run name when present; not a measured loop start",
                "end_timestamp":None,"training_duration_seconds":observed_seconds,
                "reported_training_seconds":result.get("training_seconds_total",result.get("training_seconds")),
                "reported_segment_start_iteration":result.get("current_segment_start_iteration"),
                "recorded_elapsed_maxima_by_segment":elapsed_by_segment,
                "duration_scope":"Sum of maximum recorded elapsed time in each detected log segment; observed lower bound, excludes unlogged tails/setup and does not repair missing earlier sessions",
                "gpu":environment.get("gpu"),"final_checkpoint":relative(cp) if cp.is_file() else None,
                "checkpoint_role":"final declared schedule" if pointer.get("iteration")==expected else "latest resume checkpoint",
                "best_validation":best,"best_checkpoint":relative(best_path) if best_path and best_path.is_file() else None,
                "best_checkpoint_note":"Highest recorded validation PSNR is diagnostic; not preregistered selection and checkpoint availability not assumed",
                "source_logs":sorted({r["source_file"] for r in history}),"loss_observations":len(history),
                "evaluation_outputs":metrics,"can_resume":"PENDING_CHECKPOINT_VALIDATION" if cp.is_file() else "NO_LOCAL_FULL_STATE_FOUND",
                "inference_reproducibility":"EXPORTED_PLY_PRESENT" if list((root/"point_cloud").glob("*/point_cloud.ply")) else "NO_EXPORTED_PLY_FOUND",
                "resume_entrypoint_layout":"FLAT_LAYOUT_PRESENT" if (root/"resolved_config.json").is_file() and (root/"identity.json").is_file() else "ROOT_LAYOUT_MISMATCH_OR_METHOD_SPECIFIC_IDENTITY",
                "flat_recovery_aliases":[relative(a[0]) for a in aliases if (a[0]/"resolved_config.json").is_file() and (a[0]/"identity.json").is_file()],
                "mesh_extraction_reproducibility":"NOT IMPLEMENTED/NOT VERIFIED",
                "final_gaussians":result.get("final_gaussians") if result_current else logged_gaussians,
                "gaussian_count_scope":"reported completed segment" if result_current else "last recorded loss row, not a validated final model",
                "memory":measured_memory,"memory_scope":"Maximum of available log/report/failure peaks; earlier unreported peaks remain unknown",
                "device_wide_memory":result.get("device_wide_memory",{}),"identity":identity,"parse_failures":parse_failures,
                "geometry_coverage_totals":result.get("geometry_coverage_totals"),
                "config_values":config,"training_result":result}
        runs.append(record)
    return sorted(runs,key=lambda r:r["experiment_id"])


def comparisons(runs: list[dict]) -> list[dict]:
    rows=[]
    for run in runs:
        root=ROOT/run["root"]
        candidates=list((root/"metrics").glob("test_*.json")) if (root/"metrics").exists() else []
        candidates += [p for p in (root/"evaluation/rendering_metrics_png.json",root/"evaluation/rendering_metrics.json") if p.exists()]
        metric=None;path=None
        for candidate in candidates:
            value=read_json(candidate,{})
            if value.get("status")=="SUCCESS":metric=value;path=candidate;break
        means=(metric.get("aggregate_arithmetic_mean_per_view",metric.get("mean",{})) if metric else {})
        cfg=run["config_values"]
        protocol=("rounded_uint8_png" if metric and ("metrics/test_" in relative(path) or path.name.endswith("_png.json")) else "float_render_diagnostic" if metric else "NOT RECORDED")
        split=(metric.get("split_hash") or metric.get("dataset_identity",{}).get("split")) if metric else run.get("identity",{}).get("split")
        rows.append({"experiment_id":run["experiment_id"],"model":run["model"],"status":run["training_status"],
                     "dataset":"Ignatius legacy_v1","resolution_divisor":run["resolution_divisor"],"seed":run["seed"],
                     "completed_iterations":run["actual_observed_completed_iteration"],"target_iterations":run["target_iterations"],
                     "psnr":means.get("psnr"),"ssim":means.get("ssim"),"lpips":means.get("lpips"),
                     "evaluation_split":"test" if metric else None,"evaluation_views":metric.get("view_count",metric.get("count")) if metric else None,
                     "metric_protocol":protocol,"split_hash":split,"test_exposure_fit":metric.get("test_exposure_fit",False) if metric else None,
                     "train_view_limit":cfg.get("train_view_limit"),"gpu":run["gpu"],"training_seconds":run["training_duration_seconds"],
                     "timing_scope":run["duration_scope"],"peak_torch_reserved_gib":run["memory"].get("peak_reserved_bytes",0)/1024**3 if run["memory"].get("peak_reserved_bytes") else None,
                     "sampled_device_peak_mib":run["device_wide_memory"].get("peak_device_memory_used_mib_sampled"),
                     "final_gaussians":run["final_gaussians"],"final_checkpoint_bytes":(ROOT/run["final_checkpoint"]).stat().st_size if run["final_checkpoint"] else None,
                     "chamfer_distance":None,"f_score":None,"mesh_vertices":None,"mesh_faces":None,"mesh_status":"NOT PRODUCED/NOT MEASURED",
                     "source":relative(path) if path else None,"quality_scope":"development diagnostic" if run["target_iterations"]<30000 else "resource/profile-specific single-scene experiment"})
    rows.append({"experiment_id":"B1_2DGS_NOT_TRAINED","model":"B1_2DGS","status":"UNKNOWN","quality_scope":"No project trainer/configured run/checkpoint found; upstream names in papers are not execution evidence","psnr":None,"ssim":None,"lpips":None})
    pairs=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            reasons=[]
            for key in ("resolution_divisor","split_hash","metric_protocol","evaluation_views","completed_iterations","train_view_limit"):
                if a.get(key)!=b.get(key):reasons.append(f"Different or unavailable {key}")
            if any(a.get(k) is None or b.get(k) is None for k in ("psnr","ssim","lpips")):
                reasons.append("Missing complete rendering metrics")
            if a.get("target_iterations",0)<30000 or b.get("target_iterations",0)<30000:
                reasons.append("Development schedule is not a final baseline")
            pairs.append({"a":a["experiment_id"],"b":b["experiment_id"],"rendering_comparable":not reasons,"reasons":reasons,
                          "performance_comparable":False,"performance_reason":"GPU/segment/timing scope and workload differ; no common controlled timing benchmark"})
    write_csv(RESEARCH/"comparisons/model_comparison.csv",rows)
    write_json(RESEARCH/"comparisons/comparability_audit.json",{"pairs":pairs,"geometry":"No verified predicted mesh/GT alignment/evaluation protocol; do not invent geometric accuracy","missing_policy":"null/blank, never zero"})
    return rows


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--museum", action="store_true", help="Register current Museum runs without reprocessing Ignatius")
    args=parser.parse_args()
    if args.museum:
        output=ROOT/"outputs/tanks_and_temples/Museum/research"
        inventory=[{"path":relative(path)} for path in sorted(output.glob("*/*/resolved_config.json"))]
        prior=read_json(RESEARCH/"metadata/experiment_registry.json",[])
        runs=[row for row in prior if row["scene"]!="Museum"] + collect_runs(inventory)
    else:
        with (RESEARCH/"cleanup/file_inventory.csv").open(encoding="utf-8") as stream:
            inventory=list(csv.DictReader(stream))
        runs=collect_runs(inventory)
    write_json(RESEARCH/"metadata/experiment_registry.json",runs)
    fields=[k for k in runs[0] if k not in ("config_values","training_result","memory","device_wide_memory")]
    write_csv(RESEARCH/"metadata/experiment_registry.csv",runs,fields)
    comparisons(runs)
    print(json.dumps({"experiments":len(runs),"models":sorted({r['model'] for r in runs}),"loss_observations":sum(r['loss_observations'] for r in runs)}))
    for run in runs:
        print(f"{run['model']:18} {run['actual_observed_completed_iteration']:6}/{run['target_iterations']:<6} {run['training_status']:22} {run['loss_observations']:5} observations | {run['experiment_id']}")


if __name__=="__main__":
    main()
