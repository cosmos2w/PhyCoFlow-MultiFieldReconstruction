"""Parity of the exact native endpoint cache with the historical v3 observer."""
import pytest
import torch

from phycoflow_reconstruction.evaluation.conditional_state import CommonStateObserver
from phycoflow_reconstruction.evaluation.quantile_state_cache import cached_state_matrices
from test_state_weighted_measure import family


@pytest.mark.parametrize('j', [2,4])
@pytest.mark.parametrize('mode', ['uniform','weighted','masked','ties'])
def test_cached_pair_values_leaves_and_all_endpoint_gradients(j,mode):
    torch.manual_seed(17)
    f=family(fields=('a','b'))
    target=torch.randn(2,8,2,dtype=torch.double,requires_grad=True)
    members=torch.randn(j,2,8,2,dtype=torch.double,requires_grad=True)
    if mode=='ties':
        target=torch.round(target.detach()).requires_grad_(); members=torch.round(members.detach()).requires_grad_()
    weights=torch.ones(2,8,dtype=torch.double)
    if mode in ('weighted','masked','ties'):
        weights=torch.tensor([[1.,2.,3.,4.,5.,6.,7.,8.],[4.,3.,2.,1.,7.,8.,5.,6.]],dtype=torch.double)
    mask=torch.ones(2,8,dtype=torch.bool)
    if mode=='masked': mask[:,[2,6]]=False
    observer=CommonStateObserver(f)
    for bank in (observer.main,observer.heldout):
        context={'generated_weights':weights,'reference_weights':weights,'generated_mask':mask,'reference_mask':mask}
        expected=observer._bank(bank,members,target,context)
        actual=cached_state_matrices(bank,members,target,weights=weights,mask=mask)
        torch.testing.assert_close(actual['squared_distances'],expected['squared_distances'],rtol=1e-10,atol=1e-12)
        assert actual['leaf_squared_distances'].keys()==expected['leaf_squared_distances'].keys()
        for key in actual['leaf_squared_distances']:
            torch.testing.assert_close(actual['leaf_squared_distances'][key],expected['leaf_squared_distances'][key],rtol=1e-10,atol=1e-12)
        old=torch.autograd.grad(expected['squared_distances'].sum(),(target,members),retain_graph=True)
        new=torch.autograd.grad(actual['squared_distances'].sum(),(target,members),retain_graph=True)
        for x,y in zip(old,new): torch.testing.assert_close(x,y,rtol=1e-9,atol=1e-11)
        def fair(matrix):
            positive=matrix > 0
            d=torch.where(positive,torch.sqrt(torch.where(positive,matrix,torch.ones_like(matrix))),torch.zeros_like(matrix))
            indices=torch.triu_indices(j,j,1)
            return d[0,1:].mean(0)-d[1:,1:][indices[0],indices[1]].sum(0)/(j*(j-1))
        old=torch.autograd.grad(fair(expected['squared_distances']).sum(),(target,members),retain_graph=True)
        new=torch.autograd.grad(fair(actual['squared_distances']).sum(),(target,members),retain_graph=True)
        for x,y in zip(old,new): torch.testing.assert_close(x,y,rtol=1e-9,atol=1e-11)


def test_zero_and_permutation_and_one_field():
    f=family(fields=('a',)); target=torch.zeros(1,8,1,dtype=torch.double,requires_grad=True)
    members=torch.zeros(4,1,8,1,dtype=torch.double,requires_grad=True)
    weights=torch.tensor([[1.,2.,3.,4.,5.,6.,7.,8.]],dtype=torch.double)
    result=cached_state_matrices(f,members,target,weights=weights)
    assert torch.count_nonzero(result['squared_distances'])==0
    matrix=result['squared_distances']; positive=matrix>0
    distance=torch.where(positive,torch.sqrt(torch.where(positive,matrix,torch.ones_like(matrix))),torch.zeros_like(matrix))
    indices=torch.triu_indices(4,4,1)
    fair=distance[0,1:].mean(0)-distance[1:,1:][indices[0],indices[1]].sum(0)/12
    grads=torch.autograd.grad(fair.sum(),(target,members))
    assert all(torch.isfinite(x).all() and torch.count_nonzero(x)==0 for x in grads)
    target=torch.arange(8,dtype=torch.double).reshape(1,8,1); members=torch.stack([target+v for v in [1.,3.,5.,7.]])
    perm=torch.tensor([6,2,7,4,0,1,3,5])
    a=cached_state_matrices(f,members,target,weights=weights)['squared_distances']
    b=cached_state_matrices(f,members[:,:,perm],target[:,perm],weights=weights[:,perm])['squared_distances']
    torch.testing.assert_close(a,b)
