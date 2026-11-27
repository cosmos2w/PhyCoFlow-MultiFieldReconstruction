"""Paired frame intervals for cached families and their fixed components."""
from __future__ import annotations

import json
import numpy as np
from scripts.analysis.car_round8_cached_summary import checked_ref


def paired_triplet(source, candidate, *, seed=6042026, replicates=2000):
    """Condition-paired A/R/S deltas in each family's raw fixed-observer units."""
    source,candidate=np.asarray(source,dtype=float),np.asarray(candidate,dtype=float)
    if (source.ndim!=2 or source.shape[1]!=3 or candidate.shape!=source.shape or not len(source)
            or not np.isfinite(source).all() or not np.isfinite(candidate).all() or replicates<1):
        raise ValueError("finite paired condition-by-triplet records required")
    if not np.allclose(source[:,0]-source[:,1],source[:,2],rtol=2e-5,atol=1e-7) or not np.allclose(candidate[:,0]-candidate[:,1],candidate[:,2],rtol=2e-5,atol=1e-7):
        raise ValueError("each triplet must satisfy S = A - R")
    delta=candidate-source
    indices=np.random.default_rng(seed).integers(0,len(delta),size=(replicates,len(delta)))
    samples=delta[indices].mean(axis=1)
    return {"mean":delta.mean(0).tolist(),"interval_95":np.quantile(samples,[.025,.975],axis=0).tolist(),
            "seed":seed,"replicates":replicates,"unit":"paired condition; frames share one trajectory"}


def relation_intervals(summary, assessment, support):
    if assessment["sample_ids"]!=summary["sample_ids"] or support["sample_ids"]!=summary["sample_ids"]:
        raise ValueError("condition identities differ")
    decomposition=json.loads(checked_ref(summary["inputs"]["decomposition"]).read_text())
    if decomposition["sample_ids"]!=summary["sample_ids"]:
        raise ValueError("Scale decomposition condition identities differ")
    terms=("attraction","separation","score");weights=summary["scale_weights"]
    result={}
    for family in ("State","Scale","Shape"):
        values={}
        for name,side in (("SOURCE","SOURCE"),("JointAnchor500","candidate")):
            if family=="State":
                values[side]=[row["models"][name]["state"] for row in support["records"]]
            elif family=="Scale":
                values[side]=[[sum(float(weights[key])*row["models"][side]["blocks"][key][term] for key in weights)
                               for term in terms] for row in decomposition["records"]]
            else:
                values[side]=[]
                for row in assessment["per_condition_scores"][side]:
                    attraction=sum(row[f"shape.{kind}.{category}.attraction"] for kind in ("self","mutual") for category in ("finite","essential"))
                    separation=sum(row[f"shape.{kind}.{category}.separation"] for kind in ("self","mutual") for category in ("finite","essential"))
                    values[side].append([attraction,separation,row["shape"]])
        result[family]=paired_triplet(values["SOURCE"],values["candidate"])
    result["Scale_blocks"]={}
    for block,weight in weights.items():
        values={side:np.asarray([[row["models"][side]["blocks"][block][term] for term in terms]
                                 for row in decomposition["records"]],dtype=float)
                for side in ("SOURCE","candidate")}
        result["Scale_blocks"][block]={
            "weight":float(weight),
            "raw":paired_triplet(values["SOURCE"],values["candidate"]),
            "weighted":paired_triplet(float(weight)*values["SOURCE"],float(weight)*values["candidate"])}
    result["Shape_components"]={}
    for kind in ("self","mutual"):
        for category in ("finite","essential"):
            key=f"{kind}.{category}"
            values={}
            for side in ("SOURCE","candidate"):
                attraction=np.asarray([row[f"shape.{key}.attraction"] for row in assessment["per_condition_scores"][side]])
                separation=np.asarray([row[f"shape.{key}.separation"] for row in assessment["per_condition_scores"][side]])
                values[side]=np.column_stack((attraction,separation,attraction-separation))
            result["Shape_components"][key]=paired_triplet(values["SOURCE"],values["candidate"])
    return result
