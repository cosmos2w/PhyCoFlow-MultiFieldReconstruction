"""Fresh-process checkpoint/reload contract for endpoint visits and accumulation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_teacher_stream_fresh_process_resume_matches_uninterrupted(tmp_path):
    fixture = tmp_path / 'fixture.py'
    fixture.write_text(textwrap.dedent('''
        import json, sys, torch
        from pathlib import Path
        from phycoflow_reconstruction.coherence.teacher_stream import TeacherStream
        from phycoflow_reconstruction.coherence.empirical_teacher import teacher_step
        from run_car_round4_teacher import append_history, configure_numerical_execution
        from types import SimpleNamespace
        import phycoflow_reconstruction.training.scalable_runtime as runtime_module
        from phycoflow_reconstruction.training.internal_calibration import BLOCKS, fit_common_prefit
        mode, directory = sys.argv[1], Path(sys.argv[2])
        numerical = configure_numerical_execution({'runtime': {'seed':47, 'deterministic':True},
                        'coherence': {'scalable_anchored_coherence': {'common_prefit': {'frozen':True}}}})
        assert numerical['strict_algorithms'] is True and numerical['cudnn_deterministic'] is True
        assert numerical['cudnn_benchmark'] is False
        definition = {'population': 5, 'seed': 931, 'numerical_execution': numerical, 'columns': ['epoch', 'visits', 'cumulative_visits', 'dataset_pass_equivalents', 'batch_stream', 'loss'],
                      'accumulation_order': ['state', 'scale', 'shape']}
        config = directory / 'definitions.json'
        if mode == 'initial': config.write_text(json.dumps(definition))
        else: assert json.loads(config.read_text()) == definition
        stream = TeacherStream(5, 931)
        runtime_module.ScalableQueryAdapter = lambda *args: None
        runtime_module.validate_scalable_artifacts = lambda *args: None
        fields = ('u', 'v')
        families = {name:SimpleNamespace(field_names=fields) for name in ('global_distribution','multiscale_spectrum','topology')}
        keys = [f'{name}.{block}' for name, blocks in BLOCKS.items() for block in blocks]
        provenance = {'split':'train', 'field_order':fields, 'sample_ids':('id',),
                      'source_checkpoint_sha256':'a'*64, 'normalizer_sha256':'b'*64,
                      'support_sha256':'c'*64, 'config_sha256':'d'*64, 'definition_version':'v1'}
        artifacts = {'sha256':'f'*64, 'provenance':provenance, 'state_bank':{'field_order':fields},
                     'shape_artifact':{'provenance':{'split':'train','source_checkpoint_sha256':'a'*64}},
                     'priorities':dict.fromkeys(keys,1.), 'units':dict.fromkeys(keys,1.), 'active':dict.fromkeys(keys,True)}
        runtime = runtime_module.ScalableRuntime(artifacts, families, steps_per_epoch=1,
                                               expected_provenance=provenance, seed=122)
        param = torch.nn.Parameter(torch.tensor(1.))
        fit_common_prefit(runtime.calibration, [(['id'], {key:param for key in keys})], (param,), binding={'split':'train'})
        source = torch.zeros(4, 5, 3, 1, dtype=torch.float64)
        truth = torch.ones(5, 3, 1, dtype=torch.float64)
        current = source.clone(); history = []; start = 0
        if mode == 'resume':
            saved = torch.load(directory/'checkpoint.pt', weights_only=True)
            assert saved['definition'] == definition
            stream.load_state_dict(saved['stream'])
            runtime.load_state_dict(saved['runtime'])
            current = saved['current']; history = saved['history']; start = saved['epoch']
            # Simulate recovery after the first checkpoint was saved but before
            # the logs directory or history file existed.
            assert not (directory/'logs').exists()
            append_history(directory/'logs/resumed.jsonl', history)
        for epoch in range(start + 1, 5 if mode != 'initial' else 3):
            indices = stream.next_batch(3)
            runtime.begin_step(runtime.step)
            queries = [[q.index for q in runtime.batch.self_queries], [q.index for q in runtime.batch.pair_queries]]
            difficulties = {kind:{q.index:1. for q in getattr(runtime.batch, f'{kind}_queries')} for kind in ('self','pair')}
            runtime.difficulties = {name:difficulties for name in ('multiscale_spectrum','topology')}
            runtime.finish_step(True)
            x, y, z = source[:, indices], current[:, indices], truth[indices]
            loss = lambda val: -(val * 100).sum((0, 2, 3))
            updated, info = teacher_step(x, y, z, torch.tensor([1., 2., 1.]), torch.tensor([.01]),
                                         torch.zeros_like(z, dtype=torch.bool), loss)
            current[:, indices] = updated
            history.append({'epoch': epoch, 'visits': len(indices), 'cumulative_visits': int(stream.visits.sum()),
                            'dataset_pass_equivalents': int(stream.visits.sum()) / 5,
                            'batch_stream': [{'condition_indices': indices.tolist(), 'batch_size':len(indices),
                                              'accepted_alpha':info['alpha'].tolist()}],
                            'loss': float(loss(updated).sum()), 'queries':queries})
        torch.save({'definition':definition, 'stream':stream.state_dict(), 'current':current,
                    'history':history, 'runtime':runtime.state_dict(), 'epoch':history[-1]['epoch']}, directory / ('resumed.pt' if mode == 'resume' else ('checkpoint.pt' if mode == 'initial' else 'continuous.pt')))
        if mode != 'initial':
            log = directory / 'logs' / ('continuous.jsonl' if mode == 'continuous' else 'resumed.jsonl')
            append_history(log, history)
    '''))
    env = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1] / 'src')
    env['PYTHONPATH'] = source + os.pathsep + str(Path(__file__).resolve().parents[1] / 'scripts/posttraining') + os.pathsep + env.get('PYTHONPATH', '')
    for mode in ('initial', 'resume', 'continuous'):
        subprocess.run([sys.executable, str(fixture), mode, str(tmp_path)], env=env, check=True)
    import torch
    resumed = torch.load(tmp_path / 'resumed.pt', weights_only=True)
    continuous = torch.load(tmp_path / 'continuous.pt', weights_only=True)
    assert resumed['history'] == continuous['history']
    assert torch.equal(resumed['current'], continuous['current'])
    for key in ('generator', 'permutation', 'visits'):
        assert torch.equal(resumed['stream'][key], continuous['stream'][key])
    assert resumed['stream']['cursor'] == continuous['stream']['cursor']
    assert resumed['runtime']['sampler'] == continuous['runtime']['sampler']
    assert resumed['runtime']['step'] == continuous['runtime']['step']
    assert (tmp_path/'logs/resumed.jsonl').read_text() == (tmp_path/'logs/continuous.jsonl').read_text()


def test_teacher_curves_save_pdf_and_decodable_png(tmp_path):
    import importlib.util
    from PIL import Image
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('teacher_plot_driver', root/'scripts/posttraining/run_car_round4_teacher.py')
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    row = {'epoch':1, 'family_v':dict.fromkeys(('global_distribution','multiscale_spectrum','topology'), 1.),
           'mean_alpha':.5, 'field_displacement':[.1], 'field_member_spread':[.2], 'min_field_slack':[.3],
           'elapsed_seconds':1., 'cpu_seconds':2.}
    pdf=tmp_path/'logs/teacher_v_scores.pdf'
    driver.plot_history(pdf, [row], ['T'])
    assert pdf.read_bytes().startswith(b'%PDF')
    with Image.open(pdf.with_suffix('.png')) as image:
        image.verify()
    # The same output paths are refreshed on subsequent calls.
    row['epoch']=10
    driver.plot_history(pdf, [row], ['T'])
    with Image.open(pdf.with_suffix('.png')) as image:
        image.verify()


def test_resume_diagnostics_and_rejection_do_not_write(tmp_path):
    import copy
    import importlib.util
    import pytest
    import torch
    import hashlib
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'scripts/posttraining'))
    spec = importlib.util.spec_from_file_location('teacher_resume_driver', root/'scripts/posttraining/run_car_round4_teacher.py')
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    columns = [{'name':'epoch', 'type':'int', 'units':'epoch', 'meaning':'completed epoch'},
               {'name':'state_v', 'type':'float', 'units':'source units', 'meaning':'state score'},
               {'name':'shape_v', 'type':'float', 'units':'source units', 'meaning':'shape score'}]
    identity = {'recording_columns':columns, 'accumulation_order':['state','scale','shape'],
                'observer':{'family':'frozen', 'seed':2}, 'epsilon':[1., 2.]}
    normalizers={'state':1.}
    definitions=tmp_path/'config.json'; checkpoint=tmp_path/'checkpoint.pt';history=tmp_path/'history.jsonl'
    definitions.write_text(json.dumps({'identity':identity,'normalizers':normalizers}, indent=2))
    torch.save({**identity, 'normalizers':normalizers, 'history':[{'epoch':1}]}, checkpoint)
    history.write_text('{"epoch":1}\n')
    before={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in (definitions,checkpoint,history)}
    cases=[]
    changed=copy.deepcopy(identity);changed['recording_columns'].append({'name':'new_loss','type':'float','units':'1','meaning':'new'});cases.append((changed,'"added": ["new_loss"]'))
    changed=copy.deepcopy(identity);changed['recording_columns'].pop(1);cases.append((changed,'"missing": ["state_v"]'))
    changed=copy.deepcopy(identity);changed['recording_columns'][0],changed['recording_columns'][1]=changed['recording_columns'][1],changed['recording_columns'][0];cases.append((changed,'"reordered": {"incoming": ["state_v", "epoch", "shape_v"]'))
    changed=copy.deepcopy(identity);changed['recording_columns'][1]['units']='other units';cases.append((changed,'"semantically_changed": {"state_v": {"units"'))
    changed=copy.deepcopy(identity);changed['accumulation_order']=['shape','state','new_family'];cases.append((changed,'"added": ["new_family"]'))
    changed=copy.deepcopy(identity);changed['observer']['seed']=3;cases.append((changed,'"scientific_identity_changed": ["observer.seed"]'))
    for changed, expected in cases:
        with pytest.raises(ValueError) as error:
            driver.load_resume_checkpoint(definitions,checkpoint,changed)
        assert expected in str(error.value)
        assert {path:hashlib.sha256(path.read_bytes()).hexdigest() for path in before} == before
    order_change=copy.deepcopy(identity); order_change['accumulation_order']=['shape','state','new_family']
    with pytest.raises(ValueError) as error:
        driver.load_resume_checkpoint(definitions,checkpoint,order_change)
    detail=json.loads(str(error.value).split('incompatible: ',1)[1])
    assert detail['accumulation_order'] == {'added':['new_family'], 'missing':['scale'],
            'reordered':{'saved':['state','shape'], 'incoming':['shape','state']}}
    assert {path:hashlib.sha256(path.read_bytes()).hexdigest() for path in before} == before
    # Key order and equivalent numeric spelling are not changes to scientific content.
    reordered={'epsilon':[1,2], 'observer':{'seed':2.0,'family':'frozen'},
               'accumulation_order':['state','scale','shape'],
               'recording_columns':[dict(reversed(list(item.items()))) for item in columns]}
    driver.load_resume_checkpoint(definitions,checkpoint,reordered)
    assert {path:hashlib.sha256(path.read_bytes()).hexdigest() for path in before} == before
    # A checkpoint mismatch reports the same concrete incompatibility before append.
    damaged={**copy.deepcopy(identity),'normalizers':normalizers};damaged['recording_columns'][1]['meaning']='different'
    torch.save(damaged,checkpoint)
    checkpoint_before=checkpoint.read_bytes()
    with pytest.raises(ValueError,match='saved checkpoint.*semantically_changed'):
        driver.load_resume_checkpoint(definitions,checkpoint,identity)
    assert checkpoint.read_bytes()==checkpoint_before and history.read_text()=='{"epoch":1}\n'


def test_resource_admission_checks_reserve_and_bound_receipts(tmp_path, monkeypatch):
    import importlib.util
    import pytest
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'scripts/posttraining'))
    spec=importlib.util.spec_from_file_location('teacher_admission_test',root/'scripts/posttraining/run_car_round4_teacher.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    monkeypatch.setattr(driver,'validate_driver_amendment',lambda old,new: None)
    campaign=tmp_path/'campaign';campaign.mkdir();(campaign/'operations').mkdir()
    origin=tmp_path/'origin';origin.mkdir(); execution=tmp_path/'execution';execution.mkdir()
    def bind(name,data):
        path=campaign/name;path.write_text(json.dumps(data));return {'path':str(path),'sha256':driver.sha(path)}
    checkpoint=bind('checkpoint.json',{'epoch':20});config=bind('config.json',{'definition':1});history=bind('history.json',{'epochs':20})
    ledger=bind('operations/gpu_ledger.json',{'active':{},'closed':[]})
    envelope={'base_teacher_gpu_allowance_seconds':14400,'variation_reserve_gpu_seconds':18000,
              'reserve_committed_elsewhere_gpu_seconds':1000,'candidate_other_gpu_seconds':10000,
              'whole_other_gpu_seconds':20000,'candidate_gpu_cap_seconds':35317,'whole_gpu_cap_seconds':86400}
    projections={'projected_teacher_gpu_seconds':16000.,'projected_teacher_cpu_worker_seconds':20000.,
                 'projected_whole_gpu_seconds':36000.,'projected_candidate_gpu_seconds':26000.}
    forecast=bind('forecast.json',{'schema':'car_round4_teacher_warm_cache_forecast_v1','train_manifest_sha256':'hash',
        'completed_epochs':20,'teacher_checkpoint_sha256':checkpoint['sha256'],'teacher_history_sha256':history['sha256'],
        'teacher_cpu_cap_seconds':21600,'safety_factor':1.2,'remaining_epochs':80,**envelope,
        'teacher_gpu_cap_seconds':25317,**projections})
    receipt={'schema':'car_round4_teacher_resource_admission_v1','admitted':True,'attempt':'teacher_attempt',
        'train_manifest_sha256':'hash','origin_code_root':str(origin),'execution_code_root':str(execution),
        'max_epochs':100,'completed_epochs':20,'teacher_cpu_cap_seconds':21600,'whole_gpu_cap_seconds':86400,
        'candidate_gpu_cap_seconds':35317,'teacher_gpu_cap_seconds':17000,'reserve_allocated_seconds':2600,
        'reserve_available_seconds':17000,'checkpoint':checkpoint,'scientific_config':config,'history':history,'forecast':forecast,'ledger':ledger,**projections}
    admission=bind('admission.json',receipt)
    def check():
        return driver.validate_resource_admission(admission['path'],campaign,'hash','teacher_attempt',origin,execution,
                                                  Path(checkpoint['path']),Path(config['path']),Path(history['path']))
    assert check()['teacher_gpu_cap_seconds']==17000
    for key,value in (('teacher_gpu_cap_seconds',26000),('reserve_allocated_seconds',18001),
                      ('projected_teacher_cpu_worker_seconds',21601),('max_epochs',101)):
        bad=dict(receipt);bad[key]=value;Path(admission['path']).write_text(json.dumps(bad))
        with pytest.raises(ValueError): check()
    Path(admission['path']).write_text(json.dumps(receipt));Path(ledger['path']).write_text('{}')
    with pytest.raises(ValueError,match='ledger binding'): check()


def test_driver_amendment_allows_only_operational_edits(tmp_path):
    import hashlib
    import importlib.util
    import pytest
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts/posttraining'))
    spec=importlib.util.spec_from_file_location('teacher_capsule_test',root/'scripts/posttraining/run_car_round4_teacher.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    source='''\ndef score():\n    return 1\ndef plot_history():\n    return 1\ndef formal():\n    def rows(): return 1\n    def score_batch(): return 1\n    def panel_scores(): return 1\n    for epoch in range(3):\n        epoch_start = 1\n        update = score()\n        persist(epoch)\n    else:\n        replay()\ndef main():\n    pass\n'''
    def capsule(text,src='unchanged'):
        files={'scripts/posttraining/run_car_round4_teacher.py':hashlib.sha256(text.encode()).hexdigest(),
               'src/science.py':hashlib.sha256(src.encode()).hexdigest()}
        digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
        path=tmp_path/digest;(path/'scripts/posttraining').mkdir(parents=True);(path/'src').mkdir()
        (path/'scripts/posttraining/run_car_round4_teacher.py').write_text(text);(path/'src/science.py').write_text(src)
        (path/'execution_receipt.json').write_text(json.dumps({'schema':'car_execution_code_v1','sha256':digest,'files':files}))
        return path
    original=capsule(source)
    amended=capsule(source.replace('def plot_history():\n    return 1','def plot_history():\n    return 2'))
    driver.validate_driver_amendment(original,amended)
    changed_update=capsule(source.replace('update = score()','update = score() + 1'))
    with pytest.raises(ValueError,match='scoring, update'):
        driver.validate_driver_amendment(original,changed_update)
    changed_source=capsule(source.replace('def plot_history():\n    return 1','def plot_history():\n    return 3'),'modified')
    with pytest.raises(ValueError,match='only the teacher driver'):
        driver.validate_driver_amendment(original,changed_source)
