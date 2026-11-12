"""Resource-only timing of six existing SOURCE conditions per Round 8 panel."""
from __future__ import annotations
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

from .round8_campaign import canonical_hash, file_hash
from .round8_inference import validate_panel


def _setup(config, population, fair_artifact, native_bundle, device):
    from ..coherence.compose import build_enabled_families
    from ..data.factory import open_field_dataset
    from ..training.coherence_sampling import _fit_training_references
    train=open_field_dataset(config['dataset'],split='train')
    try:
        families={k:v.to(device) for k,v in build_enabled_families(config['coherence'],train.data_spec,train.normalizer).items()}
        _fit_training_references(families,train,config,device=torch.device(device),structured_grid_required=False)
        settings=deepcopy(config['coherence']);settings['families']['topology']['persistence']['definition']='conditional_fixed_transform_fair_v1'
        fair=build_enabled_families(settings,train.data_spec,train.normalizer)['topology'].to(device)
        fair.load_state_artifact(torch.load(fair_artifact,map_location=device,weights_only=True))
        families['topology']=fair
        native_identity=None
        if population=='ASSESS':
            from .round8_native_observers import native_common_families
            families,native_identity=native_common_families(config,train,families,fair,native_bundle,device=device)
        if set(families)!={'global_distribution','multiscale_spectrum','topology'}: raise ValueError('three frozen common observers required')
        return train,families,native_identity
    except BaseException:
        train.close()
        raise


def profile_observers(config, *, train_panel, assess_panel, manifest_path, fair_shape_artifact, native_bundle, output_path, device='cuda:0'):
    """Profile existing SOURCE J2 fields without inference or retaining scores.

    Exactly the first six IDs of each already-complete panel are used. Recorded
    condition time includes NPZ loading, transfers, complete common family
    forwards, diagnostic materialization and device synchronization. Observer
    setup is timed separately so the caller can amortize it per full callback.
    """
    from ..training.post_training import _configure_persistence_workers
    from ..training.coherence_sampling import _family_context
    from .loss_aligned import _collect_family_result, _diagnostic_value
    manifest_path=Path(manifest_path); fair_shape_artifact=Path(fair_shape_artifact); native_bundle=Path(native_bundle);output_path=Path(output_path)
    if output_path.exists(): raise FileExistsError(output_path)
    manifest_sha=file_hash(manifest_path)
    with manifest_path.open() as stream: manifest=json.load(stream)
    if manifest.get('schema')!='round8_manifest_v1': raise ValueError('frozen Round 8 manifest required')
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA profiling requested but unavailable')
    def sync():
        if device.type=='cuda': torch.cuda.synchronize(device)
    original=os.environ.get('PHYCOFLOW_TOPOLOGY_WORKERS')
    os.environ['PHYCOFLOW_TOPOLOGY_WORKERS']='4'
    try:
        _configure_persistence_workers(config)
        output={}
        for population,panel_path in (('TRAIN',Path(train_panel)),('ASSESS',Path(assess_panel))):
            panel=validate_panel(panel_path)
            if panel.get('endpoint')!='SOURCE' or panel.get('population')!=population or panel.get('manifest_sha256')!=manifest_sha or panel.get('checkpoint_sha256')!=panel.get('source_checkpoint_sha256') or panel.get('source_checkpoint_sha256')!=manifest['inputs']['source_checkpoint']['sha256']:
                raise ValueError('completed matching SOURCE panel required')
            rows=panel['conditions'][:6]
            if len(rows)!=6 or len({r['sample_id'] for r in rows})!=6 or [str(r['sample_id']) for r in rows] != [str(r['id']) for r in manifest[population][:6]]: raise ValueError('first six frozen manifest conditions required')
            setup_started=time.perf_counter()
            train,families,native_identity=_setup(config,population,fair_shape_artifact,native_bundle,device)
            sync();setup_seconds=time.perf_counter()-setup_started
            if train.normalizer.digest()!=panel['normalizer_sha256']:
                train.close()
                raise ValueError('SOURCE panel normalizer differs from common observers')
            times=[]
            try:
                with torch.no_grad():
                    for row in rows:
                        sync();started=time.perf_counter()
                        with np.load(panel_path.parent/row['path'],allow_pickle=False) as cache, np.load(panel_path.parent/row['path'],allow_pickle=False) as paired:
                            if str(cache['sample_id'])!=str(row['sample_id']) or tuple(cache['field_names'].tolist())!=tuple(panel['field_names']): raise ValueError('SOURCE condition identity differs')
                            for key in ('target_model','native_ids','model_coordinates','raw_coordinates','physical_mask','physical_weights'):
                                if not np.array_equal(cache[key],paired[key]): raise ValueError('SOURCE paired support differs')
                            physical=cache['draws_physical'] + 0.0*(paired['draws_physical']-cache['draws_physical'])
                            transient_risk=(((physical-cache['target_physical'][None])**2) * (cache['physical_weights']/float(cache['inclusion_probability']))[None,:,None]).sum(axis=1)
                            if not np.isfinite(transient_risk).all(): raise ValueError('nonfinite source physical data')
                            predictions=torch.from_numpy(cache['draws_model'] + 0.0*(paired['draws_model']-cache['draws_model'])).to(device)
                            if predictions.ndim!=3 or len(predictions)!=2: raise ValueError('two SOURCE members required')
                            target=torch.from_numpy(cache['target_model']).unsqueeze(0).to(device)
                            coords=torch.from_numpy(cache['model_coordinates']).unsqueeze(0).to(device)
                            weights=torch.from_numpy(cache['physical_weights']).unsqueeze(0).to(device)
                            mask=torch.from_numpy(cache['physical_mask']).unsqueeze(0).to(device)
                            context={'sample_ids':(row['sample_id'],),'reference_ids':(row['sample_id'],),'generated_weights':weights,'reference_weights':weights,'generated_mask':mask,'reference_mask':mask,'measure_metadata':{'native_measure_sha256':train.physical_measure_receipt['sha256']}}
                            losses,components,leaves={}, {}, {}
                            for name,family in families.items():
                                local=_family_context(family,context,[predictions[1:2]],report_detail=True)
                                result=family(predictions[0:1],target,coordinates=coords,context=local)
                                _collect_family_result(name,result,losses,components,leaves)
                                for value in result.diagnostics.values():
                                    try: _diagnostic_value(value)
                                    except (TypeError,ValueError): pass
                            del losses,components,leaves,result
                        sync();times.append(time.perf_counter()-started)
            finally:train.close()
            output[population]={'source_panel_sha256':file_hash(panel_path),'sample_ids':[r['sample_id'] for r in rows],'condition_sha256':[r['sha256'] for r in rows],'support':'Q4096' if population=='TRAIN' else 'native_40300','setup_seconds':setup_seconds,'per_condition_seconds':times,'mean_seconds':sum(times)/len(times),'max_seconds':max(times),'native_observer_identity':native_identity}
        result={'schema':'round8_common_observer_resource_profile_v1','purpose':'resource measurement only; no quality scores or policy feedback','manifest_sha256':manifest_sha,'config_sha256':canonical_hash({k:config[k] for k in ('dataset','coherence','rollout','observation_consistency')}),'frozen_family_artifacts_sha256':{name:file_hash(x['path']) for name,x in config['coherence'].get('frozen_family_artifacts',{}).items()},'fair_shape_artifact_sha256':file_hash(fair_shape_artifact),'native_bundle_sha256':file_hash(native_bundle),'device':str(device),'physical_gpu_uuid':os.environ.get('ROUND8_PHYSICAL_GPU_UUID'),'persistence_workers':4,'members':2,'condition_count_per_population':6,'populations':output}
        output_path.parent.mkdir(parents=True,exist_ok=True)
        with output_path.open('x') as f: json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
        return output_path
    finally:
        if original is None:os.environ.pop('PHYCOFLOW_TOPOLOGY_WORKERS',None)
        else:os.environ['PHYCOFLOW_TOPOLOGY_WORKERS']=original
