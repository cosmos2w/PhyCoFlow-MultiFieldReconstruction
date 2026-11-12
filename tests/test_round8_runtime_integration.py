import copy
from types import SimpleNamespace
import pytest
import torch

from phycoflow_reconstruction.training.round8_calibration import BLOCKS
from phycoflow_reconstruction.training.round8_runtime import Round8Runtime
from phycoflow_reconstruction.contracts import FamilyResult, TermResult, FieldSample


class FakeAdapter:
    def __init__(self, families, shape): pass
    def score(self, name, prediction, reference, *, coordinates, context, batch, coefficients):
        prefix = 'scale' if name == 'multiscale_spectrum' else 'shape'
        names = BLOCKS[prefix]
        blocks = {f'{prefix}.{n}': prediction.mean() * (i + 1) for i, n in enumerate(names)}
        weighted = sum(v * coefficients.get(k, 1.) for k,v in blocks.items())
        units = coefficients.get('_difficulty_units', {})
        leaves = {'self': {d.index: 4. / units.get(f'{prefix}.self' if prefix == 'shape' else 'scale.signed', 1.) for d in batch.self_queries},
                  'pair': {d.index: 6. / units.get(f'{prefix}.mutual' if prefix == 'shape' else 'scale.rho', 1.) for d in batch.pair_queries}}
        return FamilyResult({}, weighted.reshape(1), weighted, {}), blocks, leaves


def fixture(tmp_path, monkeypatch):
    import phycoflow_reconstruction.training.round8_query_runtime as adapter
    monkeypatch.setattr(adapter, 'Round8QueryAdapter', FakeAdapter)
    keys = [f'{family}.{name}' for family, names in BLOCKS.items() for name in names]
    artifacts = {'priorities':{k: 3. for k in keys}, 'units':{k: 2. for k in keys}, 'active':{k: True for k in keys},
      'probe_sample_ids':['probe:0','other:0'], 'probe_dataset_indices':[0,1], 'probe_query_indices':torch.tensor([0,1]), 'state_bank':{'field_order':['a','b'],'directions':torch.tensor([[1., 0.]])},
      'shape_artifact':{'field_mean':[0.,0.], 'provenance':{'split':'train','source_checkpoint_sha256':'a'*64}},
      'provenance':{'split':'train','manifest_sha256':'b'*64,'field_order':['a','b'],'source_checkpoint_sha256':'a'*64}}
    model = torch.nn.Linear(1,1,bias=False)
    families = {k:SimpleNamespace(field_names=('a','b')) for k in ('global_distribution','multiscale_spectrum','topology')}
    class TinyDataset:
        field_names = ('a', 'b')
        physical_weights = torch.ones(3)
        physical_measure_receipt = {'sha256':'c'*64}
        def __getitem__(self, i):
            coords = torch.tensor([[0.,0.], [1.,0.], [0.,1.]])
            return FieldSample(torch.tensor([[1.,2.],[3.,4.],[5.,6.]]) + i,
                               coords, coords, torch.tensor(0.), ('probe','other')[i], 0,
                               torch.empty(0), self.field_names, (3,))
    dataset = TinyDataset()
    (tmp_path/'logs').mkdir()
    config = {'optimization':{'steps_per_epoch':1,'batch_size':2}, 'observations':{'protocol':'random_uniform','fields':{'a':{'count':1},'b':{'count':1}}}}
    def state(family,prediction,reference,context,coeff):
        a=prediction.mean(); b=2*a
        v=a*coeff['state.marginal']+b*coeff['state.mixed']
        return FamilyResult({},v.reshape(1),v,{}),{'state.marginal':a,'state.mixed':b}
    def make(mode, artifacts=artifacts, saved=None, step=0):
        obj = Round8Runtime(artifacts,mode=mode)
        obj._state = state
        obj.initialize(model=model,dataset=dataset,families=families,config=config,run_dir=tmp_path,start_step=step,saved_state=saved)
        return obj
    return artifacts,model,families,make


def test_warmup_probes_freeze_resume_and_candidate(tmp_path,monkeypatch):
    artifacts,model,families,make = fixture(tmp_path,monkeypatch)
    warm=make('warmup')
    for step in range(10):
        warm.begin_step(step=step,epoch=step+1)
        x=model(torch.ones(1,1)); endpoint={'members':(x,x*2),'target':torch.zeros_like(x),'measure':{},'batch':warm.probe_batch}
        warm.after_objective(model=model,endpoint=endpoint,result=None)
        warm.after_step({'step':step+1})
    assert [r['epoch'] for r in warm.calibration.records] == [1,3,6,8,10]
    # Gradient is physical block / TRAIN unit; retained priority must not be applied twice.
    assert warm.calibration.records[0]['norms']['scale.signed'] == pytest.approx(.5)
    assert warm.calibration.coefficients is not None
    state=warm.state_dict()
    records=[__import__('json').loads(line) for line in (tmp_path/'logs/round8_calibration.jsonl').read_text().splitlines()]
    probes=[record for record in records if 'norms' in record]
    assert all(record['query_protocol']=='round8_implicit_queries_v2' and record['query_batch']['self'][0]['inner_sign'] for record in probes)
    bad=copy.deepcopy(state); bad['sampler']['version']='round8_implicit_queries_v1'
    with pytest.raises(ValueError,match='fresh warmup'):
        make('candidate',saved=bad,step=10)
    candidate=make('candidate', saved=state,step=10)
    assert candidate.coefficients == warm.calibration.coefficients
    candidate.begin_step(step=10,epoch=11)
    x=model(torch.ones(1,1)); context={'conditional_draws':(x,)}
    for name in ('global_distribution','multiscale_spectrum','topology'):
        candidate.score_family(name,families[name],x,torch.zeros_like(x),coordinates=None,context=context)
    row={'step':11}; candidate.after_step(row)
    assert row['round8']['scores'].keys() == {'scale','shape','state'}
    assert row['round8']['blocks_raw']['scale.signed'] == pytest.approx(float(x.detach().mean()))
    assert all(v == pytest.approx(2.) for v in candidate.sampler.cache['self'].values())
    resumed=make('candidate',saved=candidate.state_dict(),step=11)
    resumed.begin_step(step=11,epoch=12); candidate.begin_step(step=11,epoch=12)
    assert resumed.batch == candidate.batch
    with pytest.raises(ValueError,match='cursor'):
        make('candidate',saved=state,step=11)


def test_artifact_identity_content_and_fork_validation(tmp_path,monkeypatch):
    artifacts,model,families,make=fixture(tmp_path,monkeypatch)
    a=make('warmup')
    copied=copy.deepcopy(artifacts)
    b=make('warmup',artifacts=copied)
    assert a._artifact_identity() == b._artifact_identity()
    torch.save(artifacts, tmp_path / 'artifact.pt')
    reloaded = torch.load(tmp_path / 'artifact.pt', weights_only=True)
    assert a._artifact_identity() == make('warmup', artifacts=reloaded)._artifact_identity()
    copied['state_bank']['directions'][0,0] = 2
    assert a._artifact_identity() != b._artifact_identity()
    with pytest.raises(ValueError,match='fork'):
        make('candidate',saved=a.state_dict(),step=0)


def test_fork_copies_exact_complete_checkpoint_and_rejects_bad_boundary(tmp_path):
    from phycoflow_reconstruction.training.round8_campaign import fork_warmup, equal_state
    from phycoflow_reconstruction.training.round8_calibration import InternalCalibration, PROBES
    from phycoflow_reconstruction.coherence.round8_queries import SharedQuerySampler
    import json
    warm = tmp_path/'warmup'
    for d in ('checkpoints','artifacts','logs','evaluation'): (warm/d).mkdir(parents=True)
    (warm/'run_manifest.json').write_text(json.dumps({'experiment_name':'warmup','source':'source-id'}))
    (warm/'logs'/'history.jsonl').write_text('{"step":130}\n')
    keys=[f'{family}.{name}' for family,names in BLOCKS.items() for name in names]
    cal=InternalCalibration({k:1. for k in keys},{k:1. for k in keys},{k:True for k in keys})
    for epoch in PROBES: cal.observe(epoch,{k:float(i+1) for i,k in enumerate(keys)})
    cal.freeze(10)
    from phycoflow_reconstruction.training.round8_runtime import VERSION
    runtime={'version':VERSION,'mode':'warmup','step':130,'artifact_identity':'a'*64,
             'calibration':cal.state_dict(),'sampler':SharedQuerySampler(['a','b']).state_dict(),
             'stream_protocol':{'version':'round8_common_warmup_stream_v1'}}
    payload={'model':{'w':torch.tensor([1.])},'optimizer':{'state':{0:{'step':torch.tensor(130.),'exp_avg':torch.tensor([.1])}},'param_groups':[]},
             'external_policy':{'dual':torch.tensor([.2,.3])}, 'rng_state':{'torch_cpu':torch.get_rng_state(), 'index_generator':torch.Generator().get_state()},
             'global_step':130,'config_sha256':'old', 'round8_runtime':runtime,
             'query_coverage':{'query_rng':torch.tensor([7])}, 'family_states':{'scale':{'frozen':torch.tensor([2.])}},
             'source_hashes':{'checkpoint':'b'*64}}
    torch.save(payload,warm/'checkpoints'/'last.pt')
    config={'stage':'post_training','case':'synthetic','optimization':{'steps_per_epoch':13,'lr_warmup':{'updates':76},'multitask':{'fidelity_budget':{},'external_policy':{}}},
            'runtime':{},'output':{},'checkpointing':{},'coherence':{'compute_budget':{'point_count':4096}}}
    payload['round8_runtime']['version'] = 'round8_training_runtime_v1'
    torch.save(payload,warm/'checkpoints'/'last.pt')
    with pytest.raises(ValueError,match='fresh warmup'):
        fork_warmup(warm,case_dir=tmp_path,retained_config=config)
    payload['round8_runtime']['version'] = VERSION
    torch.save(payload,warm/'checkpoints'/'last.pt')
    result=fork_warmup(warm,case_dir=tmp_path,retained_config=config)
    assert result['shared_updates'] == 130
    for arm in ('B8','C8'):
        run=__import__('pathlib').Path(result['arms'][arm]['run'])
        child=torch.load(run/'checkpoints'/'last.pt',weights_only=True)
        assert child['config_sha256'] != payload['config_sha256']
        for key in payload.keys() - {'config_sha256'}:
            assert equal_state(payload[key],child[key]), key
        assert (run/'logs'/'history.jsonl').read_text() == (warm/'logs'/'history.jsonl').read_text()
    assert equal_state(torch.load(warm/'checkpoints'/'last.pt',weights_only=True),payload)
    payload['round8_runtime']['step']=129
    torch.save(payload,warm/'checkpoints'/'last.pt')
    with pytest.raises(ValueError,match='five probes|ten completed'):
        fork_warmup(warm,case_dir=tmp_path,retained_config=config)


def test_post_training_objective_dispatches_runtime_without_extra_rollout():
    from test_execution_strategy_integration import _EndpointModel, _RecordingFamily, _batch, FAMILY_WEIGHTS
    from phycoflow_reconstruction.training.post_training import _coherence_objective
    class Runtime:
        def __init__(self): self.names=[]
        def score_family(self, name, family, prediction, reference, *, coordinates, context):
            self.names.append(name)
            return family(prediction,reference,coordinates=coordinates,context=context)
    model=_EndpointModel(); batch=_batch(); runtime=Runtime(); endpoint={}
    families={name:_RecordingFamily(name) for name in FAMILY_WEIGHTS}
    _coherence_objective(model,batch,families,None,
        {'coherence':{'compute_budget':{'batch_size':2,'point_count':3}},
         'rollout':{'steps':1,'solver':'euler'},
         'observation_consistency':{'mode':'none','final_clamp':False}},
        step=0,generator=torch.Generator().manual_seed(4),endpoint_context=endpoint,
        family_losses={},round8_runtime=runtime)
    assert set(runtime.names)==set(families)
    assert model.calls == 1
    assert endpoint['members'][0] is endpoint['prediction']
    assert endpoint['batch'].sample_ids == ('case-0','case-1')


def test_candidate_raw_state_telemetry_is_no_grad_without_detaching_objective():
    runtime=Round8Runtime.__new__(Round8Runtime)
    runtime.mode='candidate';runtime.manages_queries=True;runtime.epoch=11;runtime.step=10;runtime.steps_per_epoch=1
    runtime.coefficients={'state.marginal':1.,'state.mixed':1.};runtime.blocks={};runtime.raw_blocks={};runtime.family_scores={}
    calls=[]
    def score(family,prediction,reference,context,coefficients):
        calls.append(torch.is_grad_enabled())
        value=prediction.square().mean()
        return FamilyResult({},value.reshape(1),value,{}),{'state.marginal':value,'state.mixed':None}
    runtime._state=score
    prediction=torch.tensor([2.],requires_grad=True)
    result=runtime.score_family('global_distribution',None,prediction,torch.zeros(1),coordinates=None,context={})
    assert calls==[True,False]
    assert result.scalar_loss.requires_grad
    assert runtime.raw_blocks['state.marginal'].requires_grad is False
    assert torch.autograd.grad(result.scalar_loss,prediction)[0].item()==4.


def test_shared_graph_probe_preserves_common_stream_update_and_rng(tmp_path, monkeypatch):
    import random
    import numpy as np
    artifacts, model, families, make = fixture(tmp_path, monkeypatch)
    initial = copy.deepcopy(model.state_dict())
    def run(enabled):
        random.seed(17); np.random.seed(17); torch.manual_seed(17)
        model.load_state_dict(initial)
        runtime = make("warmup")
        runtime.begin_step(step=0, epoch=1)
        ordinary_batch = object()
        selected = runtime.training_batch(ordinary_batch)
        assert selected is runtime.probe_batch
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        dual = torch.tensor(.3)
        calls = []
        hook = model.register_forward_hook(lambda *args: calls.append(1))
        # One ordinary model forward provides both training and calibration.
        x = model(torch.ones(1, 1)) * (torch.rand(()) + random.random() + float(np.random.rand()))
        endpoint = {"members": (x, x * 2), "target": torch.zeros_like(x),
                    "measure": {}, "batch": selected}
        loss = x.square().sum() + dual * x.sum()
        if enabled:
            runtime.after_objective(model=model, endpoint=endpoint, result=None)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        dual = dual + .1 * x.detach().sum()
        hook.remove()
        assert len(calls) == 1
        adam = optimizer.state[model.weight]
        return (model.weight.detach().clone(), dual, adam["step"].clone(),
                adam["exp_avg"].clone(), adam["exp_avg_sq"].clone(),
                torch.rand(()), random.random(), np.random.rand(), runtime.calibration.records)
    baseline, probed = run(False), run(True)
    for a, b in zip(baseline[:-1], probed[:-1]):
        assert a == b
    assert baseline[-1] == [] and len(probed[-1]) == 1


@pytest.mark.parametrize("mode", ["warmup", "control"])
def test_retained_family_scores_are_recorded(mode):
    runtime = Round8Runtime.__new__(Round8Runtime)
    runtime.mode = mode; runtime.manages_queries = False
    runtime.epoch = 1; runtime.step = 0; runtime.steps_per_epoch = 13
    runtime.family_scores = {}
    value = torch.tensor(2., requires_grad=True)
    result = FamilyResult({}, value.reshape(1), value, {})
    runtime._state_control = lambda *args: (result, {})
    family = lambda *args, **kwargs: result
    for name in ("global_distribution", "multiscale_spectrum", "topology"):
        assert runtime.score_family(name, family, value, value, coordinates=None, context={}) is result
    assert runtime.family_scores == {name: {"raw": 2., "calibrated": 2.}
                                     for name in ("state", "scale", "shape")}
    assert result.scalar_loss.requires_grad
