import json
import pytest
from phycoflow_reconstruction.evaluation.round7_final_test import validate_completed_test
from phycoflow_reconstruction.training.run_store import file_sha256


def test_complete_test_receipt_reuse_validates_without_inference(tmp_path, monkeypatch):
    from phycoflow_reconstruction.evaluation import round7_native
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'status': 'frozen_before_test', 'source_checkpoint_sha256': 'a'*64,
                                     'endpoint': None, 'B1000': None}))
    q = tmp_path / 'q.json'
    q.write_text(json.dumps({'split': 'test', 'manifest_sha256': 'b'*64,
                             'observer_identities': {'legacy': 'c'*64}, 'rows': [{}] * 100}))
    native = tmp_path / 'native.json'
    native.write_text('{}')
    receipt = tmp_path / 'result.json'
    receipt.write_text(json.dumps({'schema': 'round7_fixed_test_v1', 'selection_sha256': file_sha256(selection),
        'manifest_sha256': 'b'*64, 'observer_identities': {'legacy': 'c'*64}, 'sensor_manifest_sha256': 'd'*64,
        'endpoints': {'SOURCE': {'checkpoint_sha256': 'a'*64, 'reductions': str(q), 'reductions_sha256': file_sha256(q),
                                  'native_receipt': str(native), 'native_receipt_sha256': file_sha256(native)}}}))
    monkeypatch.setattr(round7_native, '_load_receipt', lambda path: ({'checkpoint_sha256': 'a'*64,
        'sensor_manifest_sha256': 'd'*64}, {}))
    assert validate_completed_test(receipt, selection)['schema'] == 'round7_fixed_test_v1'
    q.write_text('changed')
    with pytest.raises(ValueError, match='artifact changed'):
        validate_completed_test(receipt, selection)
