"""Common-observer State, Scale and Shape descriptors from matched cached fields.

This produces an illustrative, single-condition descriptor figure. It loads
existing frozen observer artifacts and performs no model inference or fitting.
"""
from __future__ import annotations
from copy import deepcopy
import json
import math
from pathlib import Path
import numpy as np
import torch

from .round8_campaign import canonical_hash, file_hash
from .round8_inference import validate_matched_panels


def _quantile_step(values, weights):
    values=np.asarray(values,dtype=float);weights=np.asarray(weights,dtype=float)
    if values.ndim!=1 or weights.shape!=values.shape or not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
        raise ValueError('invalid weighted quantile representation')
    valid=weights>0; order=np.argsort(values[valid],kind='stable'); v=values[valid][order];w=weights[valid][order]; c=np.cumsum(w/w.sum());c[-1]=1.
    return np.r_[0.,c],np.r_[v[0],v]


def _render(data, png, pdf):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    styles={'Reference':('black','-'),'SOURCE member 1':('#2974a8','-'),'SOURCE member 2':('#2974a8','--'),'Blend member 1':('#cc6425','-'),'Blend member 2':('#cc6425','--')}
    for name,(color,style) in styles.items():
        descriptor=data['endpoints'][name]
        for axis,key,title in ((axes[0,0],'marginal','State: marginal quantiles'),(axes[0,1],'mixed','State: frozen projected quantiles')):
            if descriptor[key] is not None:
                q,v=descriptor[key];axis.step(q,v,where='pre',color=color,ls=style,lw=1,label=name)
            else: axis.text(.5,.5,'Mixed channel inactive',ha='center',va='center',transform=axis.transAxes)
            axis.set(xlabel='Cumulative probability',ylabel=data['state_units'],title=title);axis.grid(alpha=.2)
        axes[1,0].plot(range(len(data['band_names'])),descriptor['scale'],marker='o',color=color,ls=style,lw=1,label=name)
        finite=np.asarray(descriptor['finite'],dtype=float).reshape(-1,2)
        if len(finite): axes[1,1].scatter(finite[:,0],finite[:,1],s=9,alpha=.55,color=color,marker='o' if name=='Reference' else ('x' if 'member 2' in name else '+'),label=f"{name}; essential={descriptor['essential_count']}")
        else: axes[1,1].scatter([],[],color=color,label=f"{name}; no finite bars; essential={descriptor['essential_count']}")
    axes[0,0].legend(fontsize=6,ncol=2)
    axes[1,0].set_xticks(range(len(data['band_names'])),data['band_names'],rotation=30,ha='right')
    axes[1,0].set(ylabel='TRAIN-normalized log band power',title='Scale: frozen spectral descriptor');axes[1,0].grid(alpha=.2)
    lo,hi=axes[1,1].get_xlim();lo2,hi2=axes[1,1].get_ylim(); lower=min(lo,lo2);upper=max(hi,hi2);axes[1,1].plot([lower,upper],[lower,upper],color='0.6',lw=.7)
    axes[1,1].set(xlabel='Birth (frozen transform units)',ylabel='Death (frozen transform units)',title=f"Shape: {data['shape_unit']} H{data['dimension']} finite bars");axes[1,1].legend(fontsize=6)
    fig.suptitle(f"Common SSS descriptors: {data['sample_id']}, {data['field']}, {data['support']}, alpha={data['alpha']:g}")
    for path in (png,pdf):fig.savefig(path,dpi=170,bbox_inches='tight')
    plt.close(fig)


def build_sss_figure(spec, *, png_path, pdf_path):
    """Render a matched condition from a previously scored cached fixed endpoint.

    spec keys: config (resolved mapping), source_panel, endpoint_panel,
    score_report, fair_shape_artifact, sample_id, field, alpha; ASSESS also
    requires native_bundle. Optional direction_index and homology_dimension.
    Scores and frozen artifacts are identity-checked before recomputing the
    displayed descriptors; no alternative query bank or score is installed.
    """
    from ..coherence import build_enabled_families
    from ..data.factory import open_field_dataset
    from ..training.coherence_sampling import _fit_training_references
    from ..training.round8_query_runtime import _scale_statistics
    from ..training.post_training import _configure_persistence_workers
    config=spec['config']; source_path=Path(spec['source_panel']);endpoint_path=Path(spec['endpoint_panel'])
    source,endpoint=validate_matched_panels(source_path,endpoint_path)
    with Path(spec['score_report']).open() as f: report=json.load(f)
    alpha=float(spec['alpha']);field=spec['field'];sample_id=str(spec['sample_id'])
    if (report.get('schema')!='round8_common_observers_v1' or report.get('population')!=source['population'] or report.get('source_panel_sha256')!=file_hash(source_path) or report.get('endpoint_panel_sha256')!=file_hash(endpoint_path) or not math.isclose(float(report.get('alpha',-1)),alpha,abs_tol=1e-12) or not 0<=alpha<=1):
        raise ValueError('cached fields and fixed common score report differ')
    if sum(row.get('sample_id')==sample_id for row in report['rows'])!=1: raise ValueError('condition is not present in the scored panel')
    expected_config=canonical_hash({key:config[key] for key in ('dataset','coherence','rollout','observation_consistency')})
    artifacts={name:file_hash(item['path']) for name,item in config['coherence'].get('frozen_family_artifacts',{}).items()}
    if report.get('config_sha256')!=expected_config or report.get('frozen_family_artifacts_sha256')!=artifacts or report.get('fair_shape_artifact_sha256')!=file_hash(spec['fair_shape_artifact']):
        raise ValueError('frozen common observer artifacts or config differ')
    if field not in source['field_names']: raise ValueError('field not in panel')
    pairs=[(a,b) for a,b in zip(source['conditions'],endpoint['conditions']) if a['sample_id']==sample_id]
    if len(pairs)!=1: raise ValueError('cached condition missing or ambiguous')
    a,b=pairs[0]
    with np.load(source_path.parent/a['path'],allow_pickle=False) as x, np.load(endpoint_path.parent/b['path'],allow_pickle=False) as y:
        for key in ('target_model','model_coordinates','physical_mask','physical_weights','native_ids'):
            if not np.array_equal(x[key],y[key]): raise ValueError('cached condition support differs')
        values=np.asarray(x['draws_model']); repaired=np.asarray(y['draws_model']); target=np.asarray(x['target_model']);coords=np.asarray(x['model_coordinates']);weights=np.asarray(x['physical_weights']);mask=np.asarray(x['physical_mask'])
        if values.shape!=repaired.shape or values.shape[0]!=2 or values.shape[1:]!=target.shape or not np.all(mask): raise ValueError('valid compact J2 fields required')
        members=[torch.from_numpy(target.copy())[None],*[torch.from_numpy(v.copy())[None] for v in values],*[torch.from_numpy((v+alpha*(w-v)).copy())[None] for v,w in zip(values,repaired)]]
        coords_t=torch.from_numpy(coords.copy())[None]; mass=weights.astype(float)
    _configure_persistence_workers(config)
    dataset=open_field_dataset(config['dataset'],split='train')
    try:
        families=build_enabled_families(config['coherence'],dataset.data_spec,dataset.normalizer)
        _fit_training_references(families,dataset,config,device=torch.device('cpu'),structured_grid_required=False)
        fair_settings=deepcopy(config['coherence']);fair_settings['families']['topology']['persistence']['definition']='conditional_fixed_transform_fair_v1'
        fair=build_enabled_families(fair_settings,dataset.data_spec,dataset.normalizer)['topology']
        fair.load_state_artifact(torch.load(spec['fair_shape_artifact'],map_location='cpu',weights_only=True));families['topology']=fair
        if source['population']=='ASSESS':
            if not spec.get('native_bundle'): raise ValueError('native observer bundle required for ASSESS')
            from .round8_native_observers import native_common_families
            families,native=native_common_families(config,dataset,families,fair,spec['native_bundle'],device='cpu')
            if canonical_hash(native)!=canonical_hash(report.get('native_observer_identity')): raise ValueError('native common observer identity differs')
        elif report.get('native_observer_identity') is not None: raise ValueError('TRAIN observer has unexpected native binding')
        state=families['global_distribution'];scale=families['multiscale_spectrum'];shape=families['topology'];objective=shape.spatial_objective
        if not bool(objective.transform_frozen) or not bool(objective.calibrated): raise ValueError('frozen fair Shape transform and calibration required')
        field_id=source['field_names'].index(field);local=tuple(state.field_ids).index(field_id)
        mixed=state.components_by_key['cross_joint_topk_swd'].directions if 'cross_joint_topk_swd' in state.components_by_key else None
        direction_index=int(spec.get('direction_index',0))
        if mixed is not None and not 0<=direction_index<len(mixed):raise ValueError('direction index outside frozen common bank')
        shape_field=tuple(objective.fields).index(field)
        matches=[i for i,u in enumerate(objective.units) if u[0]=='self.persistence' and u[2]==(shape_field,) and u[3]==(1,)]
        if not matches:raise ValueError('positive self filtration absent from frozen Shape observer')
        unit_index=matches[0];dimension=int(spec.get('homology_dimension',0))
        if dimension not in objective.dimensions:raise ValueError('homology dimension absent from Shape observer')
        bands=list(scale.settings.bands.names);scale_local=tuple(scale.field_ids).index(field_id)
        names=('Reference','SOURCE member 1','SOURCE member 2','Blend member 1','Blend member 2');results={}
        with torch.no_grad():
            for name,member in zip(names,members):
                transformed=state._in_declared_units(member,members[0])[0][0,:,list(state.field_ids)]
                marginal=_quantile_step(transformed[:,local].cpu().numpy(),mass)
                projected=None if mixed is None else _quantile_step((transformed @ mixed[direction_index].to(transformed)).cpu().numpy(),mass)
                _,_,descriptor=_scale_statistics(scale,member)
                band_values=descriptor.reshape(1,len(bands),len(scale.field_names))[0,:,scale_local].cpu().tolist()
                grid=shape.fair_prepare_grid(member,coordinates=coords_t)
                diagrams,_=objective._diagrams(grid,(unit_index,),shape.domain_mask if shape.native_masked else None)
                diagram=diagrams[tuple(objective.dimensions).index(dimension)]
                results[name]={'marginal':marginal,'mixed':projected,'scale':band_values,'finite':diagram.finite.detach().cpu().tolist(),'essential_count':int(diagram.essential.numel())}
        data={'schema':'round8_common_sss_descriptors_v1','sample_id':sample_id,'field':field,'alpha':alpha,'support':'Q4096' if source['population']=='TRAIN' else 'native physical mask','state_units':state.units,'band_names':bands,'shape_unit':objective.units[unit_index][1],'dimension':dimension,'direction_index':None if mixed is None else direction_index,'direction':None if mixed is None else mixed[direction_index].detach().cpu().tolist(),'endpoints':results}
        png_path,pdf_path=Path(png_path),Path(pdf_path);png_path.parent.mkdir(parents=True,exist_ok=True);pdf_path.parent.mkdir(parents=True,exist_ok=True)
        _render(data,png_path,pdf_path)
        return {k:v for k,v in data.items() if k!='endpoints'} | {'score_report_sha256':file_hash(spec['score_report']),'fair_shape_artifact_sha256':file_hash(spec['fair_shape_artifact'])}
    finally:dataset.close()
