import pytest
import torch
from scripts.posttraining.prepare_car_round7_selected500_banks import canonical_checkpoint, ref


def test_canonical_checkpoint_compares_complete_payload_not_zip_bytes(tmp_path):
    checkpoints=tmp_path/'checkpoints';checkpoints.mkdir()
    last=checkpoints/'last.pt';canonical=checkpoints/'epoch_500.pt'
    state={'global_step':6000,'model':{'weight':torch.tensor([1.,2.])},'optimizer':{'step':5}}
    torch.save(state,last);torch.save(state,canonical)
    assert ref(last)['sha256'] != ref(canonical)['sha256']
    result={'run_dir':str(tmp_path),'checkpoint':ref(last)}
    binding,parity=canonical_checkpoint(result)
    assert binding==ref(canonical) and parity['differences']==[] and parity['last']==ref(last)
    state['optimizer']['step']=6;torch.save(state,canonical)
    with pytest.raises(ValueError,match='full payload differs'):canonical_checkpoint(result)
