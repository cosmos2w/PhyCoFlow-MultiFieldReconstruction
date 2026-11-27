#!/usr/bin/env python3
"""Build a Round 8 cached-evidence report; no model inference or TEST.

The output directory is a new DATA2 report attempt. A coordinator-supplied
Markdown section can add later engineering/position decisions without changing
the cached scientific measurements or their figures.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import shutil
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10})

from scripts.analysis.car_round8_cached_summary import checked_ref, ref
from scripts.analysis.car_round8_relation_intervals import relation_intervals
from scripts.analysis.car_round7_render import render
from scripts.analysis.report_car_round5 import save_figure
from scripts.posttraining.assess_car_round2_checkpoints import matched_rows
from scripts.posttraining.assess_car_round7 import read_row


def load(path):
    return json.loads(Path(path).read_text())


def paired(values, reference):
    values, reference = np.asarray(values, float), np.asarray(reference, float)
    if values.shape != reference.shape or values.ndim != 1 or not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise ValueError("matched finite condition arrays required")
    return float(values.mean() - reference.mean())


def aggregate(readout, model, *keys):
    values = []
    for record in readout["records"]:
        value = record["models"][model]
        for key in keys:
            value = value[key]
        values.append(value)
    data = np.asarray(values, dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError("nonfinite cached values")
    return data.mean(0)


def validate(summary_path, train_path, earlier_path, support_path, strata_path=None):
    validation, train, earlier, support = map(load, (summary_path, train_path, earlier_path, support_path))
    if (validation.get("schema") != "car_round8_cached_summary_v1" or validation.get("split") != "validation"
            or train.get("split") != "train" or earlier.get("split") != "validation"
            or len(validation["sample_ids"]) != 100 or len(train["sample_ids"]) != 800
            or earlier["sample_ids"] != validation["sample_ids"] or support.get("status") != "completed"
            or support.get("sample_ids") != validation["sample_ids"] or support.get("members") != 4
            or support["inputs"]["summary"]["sha256"] != ref(summary_path)["sha256"]
            or support["inputs"]["joint500_assessment"] != validation["inputs"]["assessment"]
            or support["inputs"]["joint100_assessment"] != earlier["inputs"]["assessment"]):
        raise ValueError("completed matched cached summaries and support readout are required")
    assessment = load(checked_ref(validation["inputs"]["assessment"]))
    train_assessment = load(checked_ref(train["inputs"]["assessment"]))
    task_path = checked_ref(validation["inputs"]["tasks"]); tasks = load(task_path)
    if (tasks.get("status") != "completed" or tasks.get("sample_ids") != validation["sample_ids"]
            or tasks["source_native"] != assessment["source_native"] or tasks["child_native"] != assessment["child_native"]):
        raise ValueError("task and assessment bank identities differ")
    for path_ref in (assessment["spec"], assessment["source_native"], assessment["child_native"],
                     tasks["event"], tasks["regional_task"], *support["figures"]):
        checked_ref(path_ref)
    spec = load(checked_ref(assessment["spec"]))
    contract = load(checked_ref(spec["assessment_contract"]))
    for side in ("SOURCE", "JointAnchor500"):
        state = aggregate(support, side, "state")
        saved = np.mean([r["state"] for r in assessment["per_condition_scores"]["SOURCE" if side == "SOURCE" else "candidate"]])
        if not math.isclose(state[0]-state[1], state[2], rel_tol=2e-5, abs_tol=1e-7) or not math.isclose(state[2], saved, rel_tol=2e-5, abs_tol=1e-7):
            raise ValueError("State direct components differ from sealed fair score")
    strata = None
    if strata_path is not None:
        strata = load(strata_path)
        if (strata.get("status") != "completed" or strata["split"]["train"]["sample_ids"] != train["sample_ids"]
                or strata["split"]["validation"]["sample_ids"] != validation["sample_ids"]
                or strata["inputs"]["train_assessment"] != train["inputs"]["assessment"]
                or strata["inputs"]["validation_assessment"] != validation["inputs"]["assessment"]):
            raise ValueError("reference-energy strata do not match both cached panels")
    return {"validation": validation, "train": train, "earlier": earlier, "support": support,
            "assessment": assessment, "train_assessment": train_assessment, "tasks": tasks,
            "spec": spec, "contract": contract, "strata": strata}


def field_ratio_figure(data, figures):
    fig, ax = plt.subplots(figsize=(9.5, 3.8), constrained_layout=True)
    names = [r["field"] for r in data["validation"]["fields"]]; x = np.arange(len(names))
    for offset, key, label, color in ((-.10, "earlier", "Joint100", "#397aa6"), (.10, "validation", "Joint500", "#a04446")):
        rows = data[key]["fields"]; mean = np.asarray([r["member_ratio"] for r in rows]); ci = np.asarray([r["member_ratio_interval_95"] for r in rows])
        ax.errorbar(x+offset, mean, yerr=np.stack((mean-ci[:,0], ci[:,1]-mean)), fmt="o", capsize=3, label=label, color=color)
    ax.axhline(1, color="0.5", lw=1); ax.axhline(1.1, color="#a04446", ls="--", lw=1.4, label="Prospective point limit 1.10")
    ax.set_xticks(x, names); ax.set_ylabel("Pooled member MSE / matched SOURCE")
    ax.set_title("VALIDATION100: fieldwise member-MSE ratios"); ax.legend(frameon=False, ncol=3, fontsize=10)
    save_figure(fig, figures, "field_member_ratios")


def scientific_score_figure(data, figures):
    support, summary = data["support"], data["validation"]
    source = [aggregate(support, "SOURCE", "state"), np.asarray(summary["scale"]["SOURCE"]["Complete_Expanded_Prefit"]), np.asarray(summary["shape"]["SOURCE"]["total"])]
    child = [aggregate(support, "JointAnchor500", "state"), np.asarray(summary["scale"]["candidate"]["Complete_Expanded_Prefit"]), np.asarray(summary["shape"]["candidate"]["total"])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, title, a, b in zip(axes, ("State", "Scale: complete expanded", "Shape: PH"), source, child, strict=True):
        x = np.arange(3); ax.bar(x-.18, a, .36, label="SOURCE", color="#397aa6"); ax.bar(x+.18, b, .36, label="Joint500", color="#a04446")
        ax.set_xticks(x, ("A", "R", "S = A − R")); ax.set_title(title); ax.set_ylabel("Raw frozen-observer units")
        for values, alignment, offset in ((a, "right", -2), (b, "left", 2)):
            for xx, vv in zip(x, values, strict=True):
                ax.annotate(f"{vv:.3g}", (xx,vv), xytext=(offset,3), textcoords="offset points", ha=alignment, fontsize=11)
        ax.set_ylim(top=max(a.max(),b.max())*1.2)
    axes[0].legend(frameon=False, fontsize=10)
    save_figure(fig, figures, "direct_fair_scores")


def scale_figure(data, figures):
    support, fields = data["support"], data["support"]["field_order"]
    a = aggregate(support,"SOURCE","scale","rho_signed_error_mean_member")
    b = aggregate(support,"JointAnchor500","scale","rho_signed_error_mean_member")
    aa = aggregate(support,"SOURCE","scale","rho_absolute_error_mean_member")
    bb = aggregate(support,"JointAnchor500","scale","rho_absolute_error_mean_member")
    p0 = aggregate(support,"SOURCE","scale","log1p_power_over_floor_signed_error_mean_member")
    p1 = aggregate(support,"JointAnchor500","scale","log1p_power_over_floor_signed_error_mean_member")
    low = aggregate(support,"SOURCE","scale","reference_low_energy")
    pairs = [f"{fields[i]} / {fields[j]}" for i,j in itertools.combinations(range(len(fields)),2)]
    signed_limit = max(float(np.max(np.abs(a))),float(np.max(np.abs(b))),1e-8)
    groups=[("signed_rho_pair_band",[(a.T,"SOURCE signed rho error",pairs,signed_limit,False), (b.T,"Joint500 signed rho error",pairs,signed_limit,False),
              ((bb-aa).T,"Change in mean absolute rho error",pairs,max(float(np.max(np.abs(bb-aa))),1e-8),False)]),
            ("floor_aware_power",[(p0.T,"SOURCE signed log-power error",fields,max(float(np.max(np.abs(p0))),float(np.max(np.abs(p1))),1e-8),False),
              (p1.T,"Joint500 signed log-power error",fields,max(float(np.max(np.abs(p0))),float(np.max(np.abs(p1))),1e-8),False),
              (low.T,"Reference: fraction power ≤ floor",fields,1.,True)])]
    for name,panels in groups:
        fig,axes=plt.subplots(1,3,figsize=(12,5),constrained_layout=True)
        for ax,(arr,title,labels,limit,nonnegative) in zip(axes,panels,strict=True):
            image=ax.imshow(arr, aspect="auto", cmap="viridis" if nonnegative else "coolwarm", vmin=0 if nonnegative else -limit, vmax=limit)
            ax.set_xticks(range(3), ("band 1","band 2","band 3")); ax.set_yticks(range(len(labels)),labels,fontsize=10); ax.set_title(title,fontsize=12)
            fig.colorbar(image,ax=ax,shrink=.8)
        save_figure(fig,figures,name)


def support_figure(data, figures):
    r = data["support"]; fields = r["field_order"]; x = np.arange(len(fields))
    fig, axes = plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for offset,name,color in ((-.19,"SOURCE","#397aa6"),(.19,"JointAnchor500","#a04446")):
        native=aggregate(r,name,"support_risks","native")
        fixed=aggregate(r,name,"support_risks","fixed_contribution_to_native")/native
        complement=aggregate(r,name,"support_risks","complement_contribution_to_native")/native
        if not np.allclose(fixed+complement,1): raise ValueError("native support contributions do not sum")
        axes[0].bar(x+offset,fixed,.32,label=name,color=color)
        axes[0].bar(x+offset,complement,.32,bottom=fixed,color=color,alpha=.25)
        selected=aggregate(r,name,"support_risks","fixed_self_normalized")/native
        sample=aggregate(r,name,"support_risks","fixed_plus_uniform_complement512_estimate")/native
        axes[1].plot(x+offset,selected,"o-",color=color,label=name+" fixed only")
        axes[1].plot(x+offset,sample,"x--",color=color,label=name+" fixed + C512")
    for key,label,marker in (("native_reconstruction","Native raster","o"),("sparse_slice_reconstruction","Sparse slice raster","s"),("separately_queried_reconstruction","Queried sparse raster","x")):
        source=aggregate(r,"SOURCE","raster",key,"member_mse");child=aggregate(r,"JointAnchor500","raster",key,"member_mse")
        axes[2].plot(x,child/source,marker=marker,label=label)
    axes[0].set_title("Native risk: fixed + complement");axes[0].set_ylabel("Fraction of pooled native risk")
    axes[1].set_title("Field-risk approximations");axes[1].set_ylabel("MSE / native MSE");axes[1].axhline(1,color="0.5",lw=.7)
    axes[2].set_title("Raster reconstruction ranking");axes[2].set_ylabel("Joint500 / SOURCE member MSE");axes[2].axhline(1,color="0.5",lw=.7)
    for ax in axes: ax.set_xticks(x,fields);ax.legend(fontsize=8,frameon=False)
    save_figure(fig,figures,"support_and_raster")


def risk_gap_figure(data, figures):
    fig, ax = plt.subplots(figsize=(10,4),constrained_layout=True)
    train = data["train"]["fields"]; val = data["validation"]["fields"]; fields=[r["field"] for r in train]
    x=np.arange(len(fields))
    for offset, rows, label, color in ((-.12,train,"TRAIN800","#397aa6"),(.12,val,"VALIDATION100","#a04446")):
        divisor=np.asarray([r["source"]["member_mse"] for r in train])
        q=np.asarray([r["source_condition_member_mse_quantiles_10_50_90"] for r in rows])/divisor[:,None]
        ax.vlines(x+offset,q[:,0],q[:,2],color=color,lw=2);ax.scatter(x+offset,q[:,1],color=color,label=label)
        means=np.asarray([r["source"]["member_mse"] for r in rows])/divisor
        ax.scatter(x+offset,means,marker="x",color=color)
    ax.set_yscale("log");ax.set_xticks(x,fields);ax.set_ylabel("SOURCE member MSE /\nTRAIN pooled SOURCE MSE")
    ax.set_title("SOURCE error distributions across the saved condition panels");ax.legend(frameon=False)
    save_figure(fig,figures,"source_error_strata")


def matched_native(data, sample_id):
    a=data["assessment"];i=a["sample_ids"].index(sample_id)
    paths={k:checked_ref(a[k+"_native"]) for k in ("source","child")}
    receipts={k:load(v) for k,v in paths.items()}
    rows={k:read_row(paths[k],receipts[k],i) for k in paths}
    matched_rows(rows["source"],rows["child"])
    return rows


def physical_native(row):
    offset,scale=row["normalizer_offset"],row["normalizer_scale"]
    truth=(row["truth_model"][0]*scale+offset).numpy()
    members=(row["members_model"][:,0]*scale+offset).numpy()
    return truth,members


def native_grid(row):
    coordinates=row["coordinates_raw"].numpy();x=np.unique(coordinates[:,0]);y=np.unique(coordinates[:,1])
    if len(x)*len(y)!=len(coordinates): raise ValueError("native grid is not complete rectilinear support")
    order=np.lexsort((coordinates[:,0],coordinates[:,1]));expected=np.stack(np.meshgrid(x,y,indexing="xy"),-1).reshape(-1,2)
    if not np.array_equal(coordinates[order,:2],expected):raise ValueError("native coordinates are not a unique complete grid")
    return order,(len(y),len(x)),[float(x.min()),float(x.max()),float(y.min()),float(y.max())]


def physical_relative_l2(prediction, reference, mass):
    prediction,reference,mass=(np.asarray(value,dtype=float) for value in (prediction,reference,mass))
    if (prediction.shape != reference.shape or prediction.shape != mass.shape or not np.isfinite(prediction).all()
            or not np.isfinite(reference).all() or not np.isfinite(mass).all() or (mass<0).any() or mass.sum()<=0):
        raise ValueError("finite matching fields and positive physical measure required")
    reference_energy=float(np.sum(mass*reference**2))
    if reference_energy<=0: raise ValueError("relative L2 unavailable for a zero reference")
    return math.sqrt(float(np.sum(mass*(prediction-reference)**2))/reference_energy)


def physical_rmse(prediction, reference, mass):
    prediction,reference,mass=(np.asarray(value,dtype=float) for value in (prediction,reference,mass))
    if (prediction.shape != reference.shape or prediction.shape != mass.shape or not np.isfinite(prediction).all()
            or not np.isfinite(reference).all() or not np.isfinite(mass).all() or (mass<0).any() or mass.sum()<=0):
        raise ValueError("finite matching fields and positive physical measure required")
    return math.sqrt(float(np.sum(mass*(prediction-reference)**2)/mass.sum()))


def reconstruction_figures(data, figures):
    fields=data["assessment"]["field_order"]; captions={}; selected=data["validation"]["example_selection"]
    for name,title in (("representative_median","Representative"),("difficult_maximum","Difficult")):
        sid=selected[name]["sample_id"];rows=matched_native(data,sid);truth,members=physical_native(rows["source"]);_,child_members=physical_native(rows["child"])
        src,child=members.mean(0),child_members.mean(0);order,shape,extent=native_grid(rows["source"])
        mass=rows["source"]["native_masses"].double().numpy();mask=mass>0
        for start in range(0,len(fields),2):
            count=min(2,len(fields)-start);fig,axes=plt.subplots(count,5,figsize=(12,2.8*count),squeeze=False,constrained_layout=True)
            for j in range(count):
                i=start+j;t=truth[:,i];s=src[:,i];c=child[:,i];lo=min(t[mask].min(),s[mask].min(),c[mask].min());hi=max(t[mask].max(),s[mask].max(),c[mask].max())
                v=max(float(np.abs(s[mask]-t[mask]).max()),float(np.abs(c[mask]-t[mask]).max()),1e-20)
                errors=[physical_relative_l2(value,t,mass) for value in (s,c)]
                labels=("Reference",f"SOURCE mean\nrelative L2 {errors[0]:.3f}",f"Joint500 mean\nrelative L2 {errors[1]:.3f}",f"SOURCE error\nrelative L2 {errors[0]:.3f}",f"Joint500 error\nrelative L2 {errors[1]:.3f}")
                arrays=(t,s,c,s-t,c-t)
                for k,(ax,array,label) in enumerate(zip(axes[j],arrays,labels,strict=True)):
                    image=np.ma.array(array[order].reshape(shape),mask=(~mask[order]).reshape(shape))
                    im=ax.imshow(image,origin="lower",extent=extent,aspect="auto",cmap="viridis" if k<3 else "coolwarm",vmin=lo if k<3 else -v,vmax=hi if k<3 else v)
                    ax.set_title(label,fontsize=14);ax.tick_params(labelsize=10)
                    ticks=[extent[0]+.25*(extent[1]-extent[0]),extent[0]+.75*(extent[1]-extent[0])]
                    ax.set_xticks(ticks,[f"{value:.2f}" for value in ticks])
                    if k==0:ax.set_ylabel(fields[i]+"\nraw coordinate 2",fontsize=12)
                    ax.set_xlabel("raw coordinate 1",fontsize=11)
                    colorbar=fig.colorbar(im,ax=ax,fraction=.046,pad=.02);colorbar.ax.tick_params(labelsize=9)
            fig.suptitle(f"{title}: {sid} | native fields, dataset units",fontsize=15)
            key=f"{name}_fields_{start//2+1}";save_figure(fig,figures,key)
            captions[key]=(f"{title} SOURCE-selected condition {sid}, fields {', '.join(fields[start:start+count])}. Reference, SOURCE and Joint500 share fieldwise color limits; errors share symmetric fieldwise limits. Each prediction and error panel shows relative L2 computed on the displayed positive-mass native mask; coordinate and field units are dataset units. Axes are stretched independently for readability.")
    return captions


def event_mask(values, definition, names):
    d=definition
    return ((values[...,names.index("T")]>d["T"]["threshold"]) & (values[...,names.index("CO")]>d["CO"]["threshold"])
            & (values[...,names.index("CH4")]<d["CH4"]["threshold"]))


def event_figure(data, figures):
    receipt=load(checked_ref(data["tasks"]["event"]));names=receipt["field_names"];definition=receipt["definition"]
    if [definition[k]["operator"] for k in ("T","CO","CH4")] != [">",">","<"]:raise ValueError("unexpected frozen event")
    for key in ("representative_median","difficult_maximum"):
        fig,axes=plt.subplots(1,4,figsize=(12,3.8),constrained_layout=True)
        sid=data["validation"]["example_selection"][key]["sample_id"];rows=matched_native(data,sid);truth,src=physical_native(rows["source"]);_,child=physical_native(rows["child"])
        order,shape,extent=native_grid(rows["source"]);mass=rows["source"]["native_masses"].double().numpy();mask=mass>0
        reference=event_mask(truth,definition,names);source=event_mask(src,definition,names);candidate=event_mask(child,definition,names)
        areas={"reference": float((reference*mass).sum()/mass.sum()),"SOURCE": (source*mass).sum(1)/mass.sum(),"Joint500": (candidate*mass).sum(1)/mass.sum()}
        index=data["assessment"]["sample_ids"].index(sid)
        for side, model in (("SOURCE","SOURCE"),("Joint500","candidate")):
            task=data["tasks"]["rows"][model][index]
            if task["sample_id"]!=sid or not np.allclose(areas[side],task["event_member_area"],rtol=3e-5,atol=1e-8) or not np.isclose(areas["reference"],task["event_target_area"],rtol=3e-5,atol=1e-8):
                raise ValueError("native event areas differ from the sealed task")
        maps=(reference.astype(float),source.mean(0),candidate.mean(0))
        for col,(array,label) in enumerate(zip(maps,("Reference event","SOURCE J4 event frequency","Joint500 J4 event frequency"),strict=True)):
            image=np.ma.array(array[order].reshape(shape),mask=(~mask[order]).reshape(shape))
            im=axes[col].imshow(image,origin="lower",extent=extent,aspect="auto",cmap="magma",vmin=0,vmax=1)
            axes[col].set_title(label,fontsize=12);axes[col].set_xlabel("raw coordinate 1",fontsize=11)
            if col:
                error=physical_rmse(array,reference.astype(float),mass)
                axes[col].text(.02,.03,f"Probability RMSE {error:.3f}",transform=axes[col].transAxes,
                               color="white",fontsize=11,bbox={"facecolor":"black","alpha":.75,"pad":2})
            if col==0:axes[col].set_ylabel("raw coordinate 2",fontsize=11)
            fig.colorbar(im,ax=axes[col],fraction=.05,pad=.02)
        ax=axes[3];jitter=np.linspace(-.09,.09,4)
        ax.scatter(jitter,areas["SOURCE"],color="#397aa6",label="SOURCE members");ax.scatter(1+jitter,areas["Joint500"],color="#a04446",label="Joint500 members")
        ax.axhline(areas["reference"],color="black",ls="--",label="Reference");ax.set_xticks([0,1],["SOURCE","Joint500"]);ax.set_xlim(-.25,1.25);ax.set_ylabel("Native mass fraction");ax.set_title("Four event areas",fontsize=12)
        ax.legend(fontsize=8,frameon=False)
        fig.suptitle(f"Frozen joint excursion: {key.replace('_',' ')} | {sid}",fontsize=15)
        save_figure(fig,figures,"joint_event_"+key)


def regional_figure(data, figures):
    tasks=data["tasks"];fig,axes=plt.subplots(1,2,figsize=(9,3.9),sharex=True,sharey=True,constrained_layout=True)
    all_values=[]
    for ax,side,title,color in ((axes[0],"SOURCE","SOURCE","#397aa6"),(axes[1],"candidate","Joint500","#a04446")):
        rows=tasks["rows"][side];truth=np.asarray([r["regional_truth"] for r in rows]);members=np.asarray([r["regional_members"] for r in rows]);mean=members.mean(1)
        ax.scatter(np.repeat(truth,4),members.reshape(-1),s=6,alpha=.09,color=color);ax.scatter(truth,mean,s=12,alpha=.65,color=color)
        all_values.extend((truth,members.ravel()));ax.set_title(title);ax.set_xlabel("Reference regional covariance")
    lo=min(float(x.min()) for x in all_values);hi=max(float(x.max()) for x in all_values);pad=(hi-lo)*.04 or .1
    for ax in axes:ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],"--",color="0.4",lw=1);ax.set_xlim(lo-pad,hi+pad);ax.set_ylim(lo-pad,hi+pad)
    axes[0].set_ylabel("Member and member-mean covariance");fig.suptitle("Frozen regional CO–U₁ covariance, TRAIN-scaled",fontsize=13)
    save_figure(fig,figures,"regional_relation")


def fmt(value): return f"{float(value):.5g}"
def interval(value): return f"[{fmt(value[0])}, {fmt(value[1])}]"


def rebase_section_figure_links(section, captions):
    """Point inline registered figures at the lowercase exported pairs."""
    known={name.casefold():name for name in captions}
    def replace(match):
        path=Path(match.group("path"));name=known.get(path.stem.casefold())
        if name is None:
            raise ValueError(f"inline figure is not in the figure manifest: {path.name}")
        return match.group("prefix")+f"figures/{name}{path.suffix.lower()})"
    return re.sub(r"(?P<prefix>!?\[[^\]\n]*\]\()(?P<path>figures/[A-Za-z0-9_-]+\.(?:png|pdf))\)",
                  replace,section,flags=re.IGNORECASE)


def figure_sections(captions):
    body=[]
    for name,caption in captions.items():
        label=name.replace('_',' ')
        body += [f"## Figure {label}",'',f"![{caption}](figures/{name}.png)",'',f"{caption} [PDF master](figures/{name}.pdf)",'']
    return body+["## Figure index",""]+[f"- [Figure {name.replace('_',' ')}](#figure-{name.replace('_','-')})" for name in captions]+[""]


def make_markdown(data, captions, extra_sections=None):
    v,t,r=data["validation"],data["train"],data["support"];fields=[row["field"] for row in v["fields"]]
    source_state=aggregate(r,"SOURCE","state");child_state=aggregate(r,"JointAnchor500","state")
    source_scale=np.asarray(v["scale"]["SOURCE"]["Complete_Expanded_Prefit"]);child_scale=np.asarray(v["scale"]["candidate"]["Complete_Expanded_Prefit"])
    source_shape=np.asarray(v["shape"]["SOURCE"]["total"]);child_shape=np.asarray(v["shape"]["candidate"]["total"])
    score=data["assessment"]["paired_core"];task=data["tasks"]["paired"]
    crossing=[row["field"] for row in v["fields"] if row["member_ratio_interval_95"][0] <= 1.1 <= row["member_ratio_interval_95"][1]]
    point_count=sum(row["member_point_pass"] for row in v["fields"])
    old_count=sum(not row["old_signed_point_pass"] for row in v["fields"])
    native_sign=np.sign(aggregate(r,"JointAnchor500","raster","native_reconstruction","member_mse")-aggregate(r,"SOURCE","raster","native_reconstruction","member_mse"))
    changed=[]
    for definition in ("sparse_slice_reconstruction","separately_queried_reconstruction"):
        sign=np.sign(aggregate(r,"JointAnchor500","raster",definition,"member_mse")-aggregate(r,"SOURCE","raster",definition,"member_mse"))
        changed.extend(fields[i] for i in range(len(fields)) if sign[i] != native_sign[i])
    support_result=("the SOURCE–Joint500 member-MSE ordering is the same for all five fields on this panel" if not changed else
                    "the SOURCE–Joint500 member-MSE ordering changes for " + ", ".join(sorted(set(changed))) + " on this panel")
    body=["# Coherence Adaptive Repair Round 8: member fidelity and spatial Shape","",
      "The historical reassessment uses the sealed Round 7 Joint100 and nominated raw Joint500 endpoints on cached matched SOURCE panels. Joint500 is a continuation endpoint with greater exposure than the separate Composite100 arm. This cached analysis adds no model inference, training, TEST access or full-population assessment. Round 8 engineering and any new learning are identified separately below.","",
      "The held-out development panel contains 100 ordered frames and TRAIN contains 800 selected frames, both from one trajectory. All uncertainty intervals are paired frame bootstraps, not independent trajectory-transfer evidence. VALIDATION contributed to development and is not independent confirmation. Field and coordinate units in the frozen metadata are unspecified dataset units.","",
      f"The SOURCE checkpoint SHA256 is `{data['spec']['source_checkpoint']['sha256']}`; the nominated Joint500 checkpoint SHA256 is `{data['spec']['checkpoint']['sha256']}`. Native assessment uses all 40,300 positive-mass sites, the fixed 256 temperature observations and J4 independent member streams. State/Scale/Shape use the separately reconstructed, fixed Q4096 support and the same matched condition/member identities.","",
      "## Field accuracy and the two contracts","",
      "Native physical-area member MSE is averaged over J4 independent members and then equally over conditions. The prospective member margin is child MSE − 1.10 × the same-split SOURCE MSE, in dataset units squared; point passes are descriptive and intervals spanning zero do not certify population noninferiority. The old signed verdict uses the separate frozen TRAIN-law absolute signed-risk allowance and is not changed retroactively.","",
      "| Field | SOURCE MSE | Joint500 MSE | Ratio [95% CI] | Member margin [95% CI] | Old signed verdict |","| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in v["fields"]:
        body.append(f"| {row['field']} | {fmt(row['source']['member_mse'])} | {fmt(row['candidate']['member_mse'])} | {fmt(row['member_ratio'])} {interval(row['member_ratio_interval_95'])} | {fmt(row['member_margin'])} {interval(row['member_margin_interval_95'])} | {'pass' if row['old_signed_point_pass'] else 'fail'} |")
    body+=['',"Mean-output risk and unbiased member spread are distinct quantities. Member MSE equals finite-J mean-output MSE plus 3/4 of unbiased member variance; the historical signed estimator equals mean-output MSE minus one quarter of that variance.",'',
           "| Field | SOURCE mean MSE | Joint500 mean MSE | SOURCE spread | Joint500 spread | Old signed Δ / allowance |","| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in v["fields"]:
        body.append(f"| {row['field']} | {fmt(row['source']['ensemble_mean_mse'])} | {fmt(row['candidate']['ensemble_mean_mse'])} | {fmt(row['source']['unbiased_member_variance'])} | {fmt(row['candidate']['unbiased_member_variance'])} | {fmt(row['old_signed_difference']/row['old_signed_allowance'])} |")
    body+=['',f"Joint500 satisfies the prospective point limit in {point_count} of {len(fields)} VALIDATION fields; the intervals crossing 1.10 are {', '.join(crossing) if crossing else 'none'}. The old signed criterion fails {old_count} fields, while the recorded sensor safeguard {'passes' if data['assessment']['native_safeguards']['sensor_pass'] else 'fails'}. These are different scientific contracts, and neither point check is a posterior-calibration claim.",'',
      "## Direct discrepancy, member separation and fair score",'',
      "For each fixed feature distance, A is mean member-to-reference discrepancy, R is the positive unordered member-pair sum divided by J(J−1), and S = A − R. R is half the mean unordered-pair distance. A, R and S answer different questions; a lower A alone does not establish calibrated uncertainty.",'',
      "| Family | SOURCE A / R / S | Joint500 A / R / S | Definition |","| --- | --- | --- | --- |"]
    for name,ss,cc,definition in (("State",source_state,child_state,"Exact weighted-CDF norm"),("Scale",source_scale,child_scale,"Complete expanded prefit"),("Shape",source_shape,child_shape,"Existing PH, self + mutual")):
        body.append(f"| {name} | {' / '.join(map(fmt,ss))} | {' / '.join(map(fmt,cc))} | {definition} |")
    source_joint=np.asarray(v['scale']['SOURCE']['joint']);child_joint=np.asarray(v['scale']['candidate']['joint'])
    differences=relation_intervals(v,data["assessment"],r)
    body+=['',"The following paired changes and frame-bootstrap intervals use raw fixed-observer units; each family has its own scale.",'',
           "| Family | ΔA [95% CI] | ΔR [95% CI] | ΔS [95% CI] |","| --- | ---: | ---: | ---: |"]
    for name in ("State","Scale","Shape"):
        value=differences[name]
        cells=[f"{fmt(value['mean'][i])} {interval([value['interval_95'][0][i],value['interval_95'][1][i]])}" for i in range(3)]
        body.append(f"| {name} | {' | '.join(cells)} |")
    body+=['',
      f"Raw joint Scale A / R / S is {' / '.join(map(fmt,source_joint))} for SOURCE and {' / '.join(map(fmt,child_joint))} for Joint500. For historical comparison only, the saved Published Scale and State paired score deltas are {fmt(score['scale']['mean'])} and {fmt(score['state']['mean'])}, respectively, each divided by its frozen TRAIN normalizer ({fmt(data['contract']['normalizers']['scale'])} and {fmt(data['contract']['normalizers']['state'])}). These values are not percentages of pooled VALIDATION SOURCE scores. Complete expanded, Published and raw joint Scale are distinct definitions.",'',
      "Each Scale row below is an unweighted represented-distance block. Its fixed multiplier is the saved prefit coefficient, including the existing 0.25 joint coefficient. Weighted changes sum to Complete Expanded Prefit Scale; raw block magnitudes have different units and are not additive without those multipliers. Intervals are paired frame bootstraps, not differences of separate interval endpoints.",'',
      "| Scale block | SOURCE raw A / R / S | Joint500 raw A / R / S | Raw ΔS [95% CI] | Fixed multiplier | Weighted ΔS [95% CI] |","| --- | --- | --- | ---: | ---: | ---: |"]
    for block,value in differences["Scale_blocks"].items():
        raw,weighted=value["raw"],value["weighted"]
        raw_cell=f"{fmt(raw['mean'][2])} {interval([raw['interval_95'][0][2],raw['interval_95'][1][2]])}"
        weighted_cell=f"{fmt(weighted['mean'][2])} {interval([weighted['interval_95'][0][2],weighted['interval_95'][1][2]])}"
        source_cell=' / '.join(map(fmt,v['scale']['SOURCE'][block]))
        child_cell=' / '.join(map(fmt,v['scale']['candidate'][block]))
        body.append(f"| {block} | {source_cell} | {child_cell} | {raw_cell} | {fmt(value['weight'])} | {weighted_cell} |")
    joint=differences["Scale_blocks"]["joint"]
    other_change=sum(value["weighted"]["mean"][2] for block,value in differences["Scale_blocks"].items() if block!="joint")
    body+=['',f"The joint block contributes {fmt(joint['weighted']['mean'][2])} to the complete Scale score change; the other four blocks together contribute {fmt(other_change)}, giving {fmt(differences['Scale']['mean'][2])} overall. In raw joint units, ΔA is {fmt(joint['raw']['mean'][0])} {interval([joint['raw']['interval_95'][0][0],joint['raw']['interval_95'][1][0]])} and ΔR is {fmt(joint['raw']['mean'][1])} {interval([joint['raw']['interval_95'][0][1],joint['raw']['interval_95'][1][1]])}. Since S = A − R, the reduced joint separation contributes to the worse joint fair score. This is a decomposition of the saved score, not a calibration claim."]
    body+=['',"The four PH leaves below retain their frozen query and component weights and sum to the PH Shape total; they are distinct from the optional location observation.",'',
      "| PH Shape component | SOURCE A / R / S | Joint500 A / R / S | ΔS [95% CI] |","| --- | --- | --- | ---: |"]
    for component in ("self.finite","self.essential","mutual.finite","mutual.essential"):
        value=differences["Shape_components"][component]
        delta=f"{fmt(value['mean'][2])} {interval([value['interval_95'][0][2],value['interval_95'][1][2]])}"
        body.append(f"| {component} | {' / '.join(map(fmt,v['shape']['SOURCE'][component]))} | {' / '.join(map(fmt,v['shape']['candidate'][component]))} | {delta} |")
    rho_source=float(aggregate(r,"SOURCE","scale","rho_absolute_error_mean_member").mean())
    rho_child=float(aggregate(r,"JointAnchor500","scale","rho_absolute_error_mean_member").mean())
    low_reference=float(aggregate(r,"SOURCE","scale","reference_low_energy").mean())
    low_source=float(aggregate(r,"SOURCE","scale","member_low_energy_fraction").mean())
    low_child=float(aggregate(r,"JointAnchor500","scale","member_low_energy_fraction").mean())
    body+=['',f"The complete fixed pair/band mean absolute rho error is {fmt(rho_source)} for SOURCE and {fmt(rho_child)} for Joint500. Averaged over fields, bands and conditions, the fractions of power at or below the frozen floor are {fmt(low_reference)} for references, {fmt(low_source)} for SOURCE members and {fmt(low_child)} for Joint500 members. The floor and band definitions are the frozen Scale definitions, not tuning choices.",'',
      "The signed-rho panels show member-minus-reference error on the fixed Q4096 spectral basis; the absolute-rho panel displays the change in direct magnitude error. Log-power panels show mean member error in log1p(power/floor). The floor-coverage panel prevents a normalized rho from being read without its amplitude context. These are selected-band within-snapshot relationships, not posterior covariance or full-native spectra.",'',
      "## Spatial support and error structure",'',
      "The field-risk plot decomposes each native MSE into the fixed Q4096 contribution and its exact native complement using native total area. The C512 estimator samples the complement uniformly and is a sampling diagnostic of saved native fields. A fixed-only self-normalized MSE is a different quantity. The same-grid raster comparison applies either native physical-cell overlap or sparse inverse-distance reconstruction followed by overlap to both prediction and reference; separately queried endpoints are also displayed. Slicing a native prediction need not reproduce another model's separately queried endpoint.",'',
      f"The selected support's outside-convex-hull native area is {100*float(r['selected_geometry']['sparse_outside_convex_hull_area_fraction']):.3g}% of native area. The compared support/raster definitions change numerical errors; {support_result}. This finite comparison does not establish an unbiased nonlinear Shape score or a general query-invariance theorem.",'',
      "SOURCE risk differs strongly between panels: the figure reports condition-level 10th–90th percentile ranges, medians and pooled means relative to each field's TRAIN SOURCE mean. This describes the saved panel gap rather than its cause.",'',
      "| Field | TRAIN SOURCE member MSE | VALIDATION SOURCE member MSE | VALIDATION / TRAIN |","| --- | ---: | ---: | ---: |"]
    for tr,vr in zip(t["fields"],v["fields"],strict=True):
        body.append(f"| {tr['field']} | {fmt(tr['source']['member_mse'])} | {fmt(vr['source']['member_mse'])} | {fmt(vr['source']['member_mse']/tr['source']['member_mse'])} |")
    if data["strata"] is not None:
        body += ['',"The cached reference-energy strata use native physical masses, compute centered per-frame spatial variance separately for each field, and divide conditions within each split at that field's own median. Different split medians are descriptive, and the resulting groups are not matched treatments.",'',
                 "| Field | TRAIN member risk / reference variance | VALIDATION member risk / reference variance | TRAIN lower / upper risk | VALIDATION lower / upper risk |","| --- | ---: | ---: | ---: | ---: |"]
        for a,b in zip(data['strata']['split']['train']['fields'],data['strata']['split']['validation']['fields'],strict=True):
            body.append(f"| {a['field']} | {fmt(a['source_member_mse_over_reference_spatial_variance'])} | {fmt(b['source_member_mse_over_reference_spatial_variance'])} | {fmt(a['within_split_strata'][0]['mean_member_risk'])} / {fmt(a['within_split_strata'][1]['mean_member_risk'])} | {fmt(b['within_split_strata'][0]['mean_member_risk'])} / {fmt(b['within_split_strata'][1]['mean_member_risk'])} |")
        if all(b['source_member_mse_over_reference_spatial_variance'] > a['source_member_mse_over_reference_spatial_variance'] for a,b in zip(data['strata']['split']['train']['fields'],data['strata']['split']['validation']['fields'],strict=True)):
            body += ['',"The VALIDATION-to-TRAIN SOURCE risk gap persists in every field after this descriptive normalization by pooled per-frame reference spatial variance. This observation does not identify why the panels differ."]
    body+=['',"## Unoptimized scientific tasks and selected fields",'',
      f"The frozen joint event is T > {data['event']['definition']['T']['threshold']!r}, CO > {data['event']['definition']['CO']['threshold']!r}, and CH4 < {data['event']['definition']['CH4']['threshold']!r} in the declared dataset units. Maps show the native binary reference and the fraction of four independent members satisfying the event at each site; the adjacent dots are physically mass-weighted member event areas. Four-member event frequencies are not a calibration test.",'',
      f"Joint500 minus SOURCE event-area fair score is {fmt(task['event']['mean'])} {interval(task['event']['interval_95'])} in units of the frozen TRAIN task normalizer {fmt(data['spec']['task_normalizers']['event'])}. The reserved regional CO–U₁ covariance fair-score change is {fmt(task['regional']['mean'])} {interval(task['regional']['interval_95'])} in units of its frozen TRAIN normalizer {fmt(data['spec']['task_normalizers']['regional'])}. These are not percentages of the VALIDATION SOURCE scores. The covariance is a dimensionless TRAIN-scaled association, not a conservation or density-weighted flux estimate.",'',
      f"The representative {v['example_selection']['representative_median']['sample_id']} and difficult {v['example_selection']['difficult_maximum']['sample_id']} frames were selected only from SOURCE native member errors, using equal field weighting after normalization by each SOURCE pooled field risk. The member-mean reconstruction plots preserve that selection, common color scales and physical-area relative L2 labels.",'',
      "## Scope and remaining evidence",'',
      f"The cached VALIDATION support/State/rho readout used {fmt(r['timing']['wall_seconds'])} wall seconds and {fmt(r['timing']['cpu_seconds'])} CPU seconds with {r['timing']['threads']} CPU threads and no GPU. This diagnostic cost excludes earlier prediction generation and is not a training-speed improvement. " + ("The following sections identify the completed engineering and positional measurements and the learning decision separately." if extra_sections else "The positional Shape fit, affected-action qualification, engineering disposition and conditional learning decision remain to be reconciled from their own completed receipts."),'']
    if extra_sections:
        body += [rebase_section_figure_links(extra_sections,captions).strip(),""]
    body += figure_sections(captions)
    return "\n".join(body)


def relink_pdf(output,captions):
    worker="""
import fitz,json,pathlib,sys
output=pathlib.Path(sys.argv[1]);expected=set(json.loads(sys.argv[2]));document=fitz.open(output/'report.pdf');found=set()
for page in document:
    for link in page.get_links():
        target=link.get('file','')
        if '/figures/' not in target:continue
        name=target.rsplit('/figures/',1)[1]
        if name not in expected or '/' in name:raise ValueError('unexpected master link')
        document.xref_set_key(link['xref'],'A','<</S /URI /URI (figures/'+name+')>>');found.add(name)
if found!=expected:raise ValueError('missing master links')
document.set_metadata({**document.metadata,'title':'Coherence Adaptive Repair: Round 8'})
tmp=output/'report_relinked.pdf';document.save(tmp,garbage=3,deflate=True);document.close();tmp.replace(output/'report.pdf')
"""
    interpreter=os.environ.get("CAR_REPORT_PDF_PYTHON","/home/wanglz/miniconda3/envs/fig/bin/python")
    subprocess.run([interpreter,"-c",worker,str(output),json.dumps([k+".pdf" for k in captions])],check=True,capture_output=True,text=True)
    urls=subprocess.run(["pdfinfo","-url",str(output/"report.pdf")],check=True,capture_output=True,text=True).stdout.splitlines()[1:]
    actual=[row.split(None,2)[2] for row in urls if row.strip()]
    if set(actual)!={"figures/"+name+".pdf" for name in captions}:
        raise ValueError("rendered PDF master links are not portable")


def add_extra_figures(manifest_path, source_dir, figures, captions):
    """Export lowercase names and retain references to the original pairs."""
    manifest=load(manifest_path);source_dir=Path(source_dir).resolve(strict=True);figures=Path(figures)
    if not isinstance(manifest,dict) or not source_dir.is_dir() or not figures.is_dir():
        raise ValueError("extra figure manifest must be an object and source must be a directory")
    used={name.casefold() for name in captions}
    used.update(path.stem.casefold() for path in figures.iterdir() if path.suffix.casefold() in (".png",".pdf"))
    pending=[]
    for filename,caption in manifest.items():
        if not isinstance(filename,str) or Path(filename).name!=filename:
            raise ValueError("extra figure names must be plain filenames")
        path=Path(filename);stem=path.stem if path.suffix.casefold() in (".png",".pdf") else filename
        normalized=stem.casefold()
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*",stem) or normalized in used
                or not isinstance(caption,str) or not caption.strip() or "\n" in caption
                or "[" in caption or "]" in caption):
            raise ValueError("extra figures require unique safe names and single-line plain captions")
        pair={}
        for ext in ("png","pdf"):
            source=source_dir/f"{stem}.{ext}"
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"ordinary extra figure {ext} is required: {source}")
            pair[ext]=ref(source)
        used.add(normalized);pending.append((normalized,caption.strip(),pair))
    sources={}
    for stem,caption,pair in pending:
        for ext,source in pair.items():
            shutil.copyfile(source["path"],figures/f"{stem}.{ext}")
        captions[stem]=caption;sources[stem]=pair
    return sources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",type=Path,required=True);p.add_argument("--train-summary",type=Path,required=True)
    p.add_argument("--earlier-summary",type=Path,required=True);p.add_argument("--support",type=Path,required=True)
    p.add_argument("--strata",type=Path);p.add_argument("--extra-sections",type=Path)
    p.add_argument("--extra-figure-manifest",type=Path,help="JSON object mapping a figure stem or filename to a single-line caption")
    p.add_argument("--extra-figure-source",type=Path,help="directory containing each named PNG/PDF pair")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();started=time.time();output=args.output.resolve()
    if bool(args.extra_figure_manifest)!=bool(args.extra_figure_source):
        raise ValueError("extra figure manifest and source directory must be supplied together")
    generator_sources={name:ref(ROOT/'scripts/analysis'/name) for name in ("report_car_round8.py","car_round8_relation_intervals.py","car_round7_render.py")}
    data=validate(args.summary,args.train_summary,args.earlier_summary,args.support,args.strata)
    data["event"]=load(checked_ref(data["tasks"]["event"]))
    output.mkdir(parents=True,exist_ok=False);figures=output/"figures";figures.mkdir()
    field_ratio_figure(data,figures);scientific_score_figure(data,figures);scale_figure(data,figures);support_figure(data,figures)
    risk_gap_figure(data,figures);event_figure(data,figures);regional_figure(data,figures)
    captions={
      "field_member_ratios":"Joint100 and Joint500 native member-MSE ratios versus matched SOURCE with paired frame-bootstrap 95% intervals. The dashed line is the prospective 1.10 point limit, not the historical signed-risk allowance.",
      "direct_fair_scores":"Direct member-to-reference discrepancy A, positive member separation R and fair score S = A − R in raw frozen-observer units for State, complete-expanded Scale and existing PH Shape.",
      "signed_rho_pair_band":"Fixed-Q4096 pair/band signed rho errors and the change in mean absolute rho error. Bands are fixed basis bands, not physical wavelengths.",
      "floor_aware_power":"Fixed-Q4096 floor-aware signed log-power errors and the fraction of reference band powers at or below their frozen floors. The retained bands are not physical wavelengths.",
      "support_and_raster":"Exact native-risk contributions from fixed sites and their complement; fixed-only and selected-plus-C512 estimates relative to native risk; and the Joint500/SOURCE member-MSE ratio after native or sparse reconstruction to the same physical raster.",
      "source_error_strata":"For each field, SOURCE condition-level native member-MSE 10th–90th percentile whiskers and medians (dots), with pooled means (crosses), all divided by that field's TRAIN SOURCE pooled mean. The panels share one trajectory and have different condition-selection laws.",
      "joint_event_representative_median":"Native joint-event reference mask and four-member event frequencies for the SOURCE-selected representative frame. Each frequency panel displays its physical-mass-weighted probability RMSE versus the binary event reference. Dots show each member's native-area event fraction; the dashed line shows its matched reference. Axes are stretched independently for readability.",
      "joint_event_difficult_maximum":"Native joint-event reference mask and four-member event frequencies for the SOURCE-selected difficult frame. Each frequency panel displays its physical-mass-weighted probability RMSE versus the binary event reference. Dots show each member's native-area event fraction; the dashed line shows its matched reference. Axes are stretched independently for readability.",
      "regional_relation":"Reference versus SOURCE and Joint500 member and member-mean regional CO–U₁ signed covariance over VALIDATION100. Covariance is dimensionless after frozen TRAIN-truth scaling; faint points are individual members and solid points their means.",
    }
    captions.update(reconstruction_figures(data,figures))
    extra_sources=add_extra_figures(args.extra_figure_manifest,args.extra_figure_source,figures,captions) if args.extra_figure_manifest else {}
    text=make_markdown(data,captions,args.extra_sections.read_text() if args.extra_sections else None)
    (output/"report.md").write_text(text.rstrip()+"\n");pages=render(output,captions);relink_pdf(output,captions)
    evidence={"schema":"car_round8_report_evidence_v1","status":"rendered_pending_visual_inspection","test_locked":True,
              "inputs":{key:ref(value) for key,value in (("summary",args.summary),("train_summary",args.train_summary),("earlier_summary",args.earlier_summary),("support",args.support))},
              "strata":ref(args.strata) if args.strata else None,"extra_sections":ref(args.extra_sections) if args.extra_sections else None,"generator_sources":generator_sources,
              "extra_figure_manifest":ref(args.extra_figure_manifest) if args.extra_figure_manifest else None,"extra_figure_sources":extra_sources,
              "report_md":ref(output/"report.md"),"report_pdf":ref(output/"report.pdf"),
              "figures":{name:{ext:ref(figures/f"{name}.{ext}") for ext in ("png","pdf")} for name in captions},
              "pages":pages,"elapsed_seconds":time.time()-started}
    (output/"evidence_index.json").write_text(json.dumps(evidence,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":evidence["status"],"pages":pages,"output":str(output),"figures":len(captions),"elapsed_seconds":evidence["elapsed_seconds"]}))


if __name__=="__main__":main()
