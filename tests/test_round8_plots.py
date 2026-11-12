import json
import shutil
import subprocess
import pytest
from PIL import Image
from phycoflow_reconstruction.training.round8_plots import _collect, refresh


def row(step, mode='candidate'):
    return {'step':step,
        'coherence_family/global_distribution/raw': 1.,
        'coherence_family/global_distribution/weighted_contribution': 2.,
        'coherence_family/multiscale_spectrum/raw': .4,
        'coherence_family/multiscale_spectrum/weighted_contribution': .8,
        'coherence_family/multiscale_spectrum/outer_weight':2.,
        'coherence_family/multiscale_spectrum/calibration_scale':.5,
        'coherence_family/topology/raw':-.2,
        'coherence_family/topology/weighted_contribution':-.4,
        'coherence_family/topology/outer_weight':2.,
        'coherence_family/topology/calibration_scale':.5,
        'round8':{'mode':mode, 'scores':{'state':{'raw':.9,'calibrated':1.1}},
            'blocks_raw':{'scale.signed':.1,'shape.self':-.3,'state.marginal':.2},
            'blocks_calibrated':{'scale.signed':.15,'shape.self':-.4,'state.marginal':.3}}}


def append(path, data):
    with path.open('a') as stream: stream.write(json.dumps(data)+'\n')


def test_history_series_are_labeled_and_preserve_signed_fair_values():
    x=_collect([row(1),row(2)],2)
    assert x['family']['state raw'][1] == [.9,.9]
    assert x['family']['shape calibrated'][1] == [-.4,-.4]
    assert x['shape']['self raw weighted'][1] == [-.3,-.3]
    assert x['scale']['signed calibrated weighted'][1] == [.15,.15]
    assert 'marginal raw chain-rule surrogate' in x['state_channels']
    assert all('state.' not in key for key in x['scale'])
    baseline={'step':3,'coherence_component/multiscale_spectrum/relation_rho_crps.conditional/weighted_contribution':-.7,
              'coherence_family/multiscale_spectrum/raw':.5}
    assert _collect([baseline],2)['scale']['rho retained weighted'][1] == [-.7]
    with pytest.raises(ValueError,match='strictly increasing'):
        _collect([row(1),row(1)],2)


def test_render_refresh_cadence_resume_and_valid_files(tmp_path):
    logs=tmp_path/'logs'; logs.mkdir(); history=logs/'full_updates.jsonl'
    assert refresh(tmp_path,2)['reason']=='no_history'
    append(history,row(1))
    first=refresh(tmp_path,2); assert first['rendered'] and first['step']==1
    with Image.open(first['png']) as image: image.verify()
    assert (logs/'round8_history.pdf').read_bytes().startswith(b'%PDF-')
    if shutil.which('pdftoppm'):
        subprocess.run(['pdftoppm','-f','1','-l','1','-singlefile','-png','-r','30',first['pdf'],str(tmp_path/'pdf_preview')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        with Image.open(tmp_path/'pdf_preview.png') as image: image.verify()
    for step in range(2,20): append(history,row(step))
    assert refresh(tmp_path,2)['reason']=='not_due'
    append(history,row(20))
    assert refresh(tmp_path,2)['rendered']
    assert refresh(tmp_path,2)['reason']=='not_due'
    append(history,row(21))
    assert refresh(tmp_path,2,force=True)['rendered']
    child=tmp_path/'child'; child.mkdir(); shutil.copytree(logs,child/'logs')
    assert refresh(child,2)['reason']=='not_due'
    append(child/'logs/full_updates.jsonl',row(22))
    assert refresh(child,2,force=True)['history_rows']==22
    with pytest.raises(ValueError,match='epoch size differs'):
        refresh(child,3)


def test_resume_archives_exact_tail_and_restores_curve_continuity(tmp_path):
    from phycoflow_reconstruction.training.round8_plots import restore_history
    logs=tmp_path/'logs'; logs.mkdir(); history=logs/'full_updates.jsonl'
    for step in range(1,7): append(history,row(step))
    original=history.read_bytes()
    refresh(tmp_path,2,force=True)
    restored=restore_history(tmp_path,4)
    assert restored['archived_rows']==2
    assert history.read_bytes()==b''.join(original.splitlines(keepends=True)[:4])
    assert __import__('pathlib').Path(restored['archive']).read_bytes()==b''.join(original.splitlines(keepends=True)[4:])
    assert not (logs/'round8_history_state.json').exists()
    rollback=refresh(tmp_path,2,force=True)
    assert rollback['step']==4 and rollback['history_rows']==4
    append(history,row(5)); append(history,row(6))
    final=refresh(tmp_path,2,force=True)
    assert final['step']==6 and final['history_rows']==6
    with Image.open(final['png']) as image: image.verify()
    assert restore_history(tmp_path,4)['archive']==restored['archive']
    assert restore_history(tmp_path,4)['archived_rows']==0


@pytest.mark.parametrize('content,start', [
    (b'{"step":1}\n{"step":2',1),
    (b'{"step":1}\nnot json\n',1),
    (b'{"step":1}\n{"step":3}\n',1),
    (b'{"step":1}\n{"step":1}\n',1),
    (b'{"step":1,"bad":NaN}\n',1),
    (b'{"step":1}\n',2),
])
def test_resume_rejects_bad_history_without_changing_any_bytes(tmp_path,content,start):
    from phycoflow_reconstruction.training.round8_plots import restore_history
    logs=tmp_path/'logs'; logs.mkdir(); path=logs/'full_updates.jsonl'; path.write_bytes(content)
    with pytest.raises(ValueError): restore_history(tmp_path,start)
    assert path.read_bytes()==content
    assert sorted(p.name for p in logs.iterdir()) == ['full_updates.jsonl']
