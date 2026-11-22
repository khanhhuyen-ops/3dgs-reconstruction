"""Standalone scientific figures from observed CSV rows, with no invented samples."""
from __future__ import annotations

import csv
from pathlib import Path
import statistics

from discover import ROOT, RESEARCH, read_json, write_json


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":300})
    runs=read_json(RESEARCH/"metadata/experiment_registry.json")
    index=[];omissions=[]
    def save(fig,folder: str,name: str,sources: list[str],description: str):
        target=RESEARCH/"figures"/folder/name
        target.parent.mkdir(parents=True,exist_ok=True)
        fig.text(.01,.008,"Source: "+sources[0]+(" (+ recorded sources in figure_index.json)" if len(sources)>1 else ""),fontsize=6)
        fig.tight_layout(rect=(0,.045,1,1))
        files=[]
        for extension in ("png","svg","pdf"):
            path=target.with_suffix("."+extension)
            fig.savefig(path,bbox_inches="tight")
            files.append(path.relative_to(ROOT).as_posix())
        plt.close(fig)
        index.append({"files":files,"sources":sources,"description":description,"smoothing":"NONE","interpolation":"NONE; markers are observed samples"})
    for run in runs:
        identifier=run["experiment_id"]
        path=RESEARCH/"metrics"/identifier/"loss_history.csv"
        rows=list(csv.DictReader(path.open(encoding="utf-8")))
        if not rows:
            omissions.append({"experiment":identifier,"plot":"training curves","reason":"NOT RECORDED"});continue
        title=f"{run['model']} / Ignatius / r={run['resolution_divisor']} / {run['training_status']}"
        x=[int(r["iteration"]) for r in rows]
        y=[float(r["loss_total"]) for r in rows]
        fig,ax=plt.subplots(figsize=(9,4.5))
        ax.scatter(x,y,s=10,alpha=.7,label="Recorded total objective")
        ax.set(xlabel="Training iteration",ylabel="Objective value (method-specific units)",title=title)
        ax.grid(alpha=.2);ax.legend()
        if run["actual_observed_completed_iteration"]<run["target_iterations"]:
            ax.axvline(run["actual_observed_completed_iteration"],ls="--",color="black",lw=.8,label="Interrupted")
            ax.annotate("Incomplete declared schedule",(x[-1],y[-1]),xytext=(.6,.86),textcoords="axes fraction",arrowprops={"arrowstyle":"->"})
        save(fig,"training",identifier+"_total",run["source_logs"],"Raw recorded loss; gaps are not filled and totals are not comparable across objectives")
        keys=(["loss_l1","loss_ssim"] if run["model"]=="B0_Original3DGS" else ["raw_rgb","raw_flatten","raw_single","raw_multi_geo","raw_multi_ncc"])
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for key in keys:
            valid=[r for r in rows if r.get(key)]
            if valid:axes[0].scatter([int(r["iteration"]) for r in valid],[float(r[key]) for r in valid],s=8,label=key,alpha=.65)
            if run["model"]=="B2_PGSR":
                weighted=key.replace("raw_","weighted_")
                valid=[r for r in rows if r.get(weighted)]
                if valid:axes[1].scatter([int(r["iteration"]) for r in valid],[float(r[weighted]) for r in valid],s=8,label=weighted,alpha=.65)
            else:
                weight=run["config_values"].get("optimizer",{}).get("lambda_dssim",.2)
                factor=1-weight if key=="loss_l1" else weight
                if valid:axes[1].scatter([int(r["iteration"]) for r in valid],[float(r[key])*factor for r in valid],s=8,label=f"{factor:g} x {key}",alpha=.65)
        for ax,subtitle in zip(axes,("Raw components","Weighted objective contributions")):
            ax.set(xlabel="Training iteration",ylabel="Loss value (symlog, linear region 1e-4)",title=subtitle)
            ax.set_yscale("symlog",linthresh=1e-4);ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle(title)
        save(fig,"training",identifier+"_components",run["source_logs"],"Distinct raw and weighted components; scheduled-inactive recorded zeros preserved")
        fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
        differences=[(y[i]-y[i-1])/max(abs(y[i-1]),1e-12) for i in range(1,len(y))]
        axes[0].scatter(x[1:],differences,s=9);axes[0].axhline(0,color="grey",lw=.8)
        axes[0].set(ylabel="Relative change between logged samples",title=title+" / stability diagnostics")
        count_rows=[r for r in rows if r.get("gaussians")]
        axes[1].scatter([int(r["iteration"]) for r in count_rows],[int(r["gaussians"]) for r in count_rows],s=9)
        axes[1].set(xlabel="Training iteration",ylabel="Recorded Gaussian count")
        for ax in axes:ax.grid(alpha=.2)
        save(fig,"training",identifier+"_stability",run["source_logs"],"Stochastic sample-to-sample change and Gaussian count; not a deterministic convergence proof")
        validation=list(csv.DictReader((path.parent/"validation_history.csv").open(encoding="utf-8")))
        for metric in ("psnr","ssim","lpips"):
            valid=[r for r in validation if r.get(metric)]
            if not valid:
                omissions.append({"experiment":identifier,"plot":"validation_"+metric,"reason":"NOT RECORDED"});continue
            fig,ax=plt.subplots(figsize=(9,4.5))
            ax.scatter([int(r["iteration"]) for r in valid],[float(r[metric]) for r in valid],s=15,label=f"Recorded validation subset ({valid[-1]['views']} views)")
            ax.set(xlabel="Iteration",ylabel=metric.upper()+(" (dB)" if metric=="psnr" else ""),title=title+" / validation")
            ax.grid(alpha=.2);ax.legend()
            save(fig,"training",identifier+"_validation_"+metric,[r["source"] for r in valid],"Periodic validation subset; not final held-out test")
    comparisons=list(csv.DictReader((RESEARCH/"comparisons/model_comparison.csv").open(encoding="utf-8")))
    valid=[r for r in comparisons if r.get("psnr")]
    labels=[r["model"].replace("_Original3DGS","")+f" r={r['resolution_divisor']}\n{r['completed_iterations']} it / "+("Kaggle" if r["gpu"]=="Tesla T4" else "Local") for r in valid]
    if valid:
        fig,axes=plt.subplots(1,3,figsize=(12,4.5))
        for ax,key in zip(axes,("psnr","ssim","lpips")):
            ax.bar(range(len(valid)),[float(r[key]) for r in valid],color="#47789c")
            ax.set_xticks(range(len(valid)),labels,rotation=35,ha="right",fontsize=7)
            ax.set(ylabel=key.upper()+(" (dB)" if key=="psnr" else ""),title=key.upper())
        fig.suptitle("Ignatius recorded rendering diagnostics — unequal resolutions/budgets/protocols; not a ranking")
        save(fig,"comparisons","rendering_quality",[r["source"] for r in valid],"Reported test scores grouped by explicit resolution/budget; float and PNG regimes cannot be ranked together")
    resource=[r for r in comparisons if r.get("training_seconds") and r.get("peak_torch_reserved_gib")]
    if resource:
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        labels=[r["model"].replace("_Original3DGS","")+f" r={r['resolution_divisor']}\n{r['completed_iterations']} it; {r['gpu'].replace('NVIDIA GeForce ','').replace(' Laptop GPU','')}" for r in resource]
        for ax,key,title in zip(axes,("training_seconds","peak_torch_reserved_gib"),("Logged-segment duration lower bound (seconds)","Available peak Torch reserved memory (GiB)")):
            ax.bar(range(len(resource)),[float(r[key]) for r in resource],color="#657c69")
            ax.set_xticks(range(len(resource)),labels,rotation=45,ha="right",fontsize=6);ax.set_title(title)
        fig.suptitle("Ignatius resource records — different hardware/workloads and incomplete timing scopes")
        save(fig,"comparisons","resources",["research/comparisons/model_comparison.csv"],"Descriptive measurements only; no controlled performance ranking")
    omissions.append({"plot":"mesh_quality_comparison","reason":"No verified predicted triangle mesh or measured geometric accuracy in the discovered experiments"})
    write_json(RESEARCH/"figures/figure_index.json",{"figures":index,"omissions":omissions})
    print(f"Scientific figure groups: {len(index)}; three formats each. Explicit missing-data omissions: {len(omissions)}")


if __name__=="__main__":main()
