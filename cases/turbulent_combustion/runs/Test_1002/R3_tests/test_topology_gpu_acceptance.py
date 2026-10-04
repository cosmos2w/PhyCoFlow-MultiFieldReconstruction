"""Opt-in physical-GPU-0 parity; never schedules a scientific training epoch."""
import os

import pytest
import torch

from phycoflow_reconstruction.coherence.families.topology.persistence import (
    cubical_diagrams, sliced_diagram_distance, sliced_diagram_distances,
)


@pytest.mark.skipif(os.environ.get("PHYCOFLOW_R3_GPU_ACCEPTANCE") != "1",
                    reason="main must explicitly schedule physical GPU 0 acceptance")
@pytest.mark.parametrize("dtype", [torch.float32,torch.float64])
def test_cpu_cuda_gudhi_component_values_and_gradients(dtype):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    assert torch.cuda.is_available()
    generator = torch.Generator().manual_seed(873)
    target = torch.randn(3,8,8,generator=generator,dtype=dtype)
    prediction = target+0.17*torch.randn(3,8,8,generator=generator,dtype=dtype)
    def evaluate(device):
        x = prediction.to(device).detach().requires_grad_()
        refs = cubical_diagrams(target.to(device),periodic=False)
        diagrams = cubical_diagrams(x,periodic=False)
        left = [d[degree] for d in diagrams for degree in (0,1)]
        right = [d[degree] for d in refs for degree in (0,1)]
        f,e = sliced_diagram_distances(left,right,normalization=64,return_components=True)
        scalar = torch.stack([sliced_diagram_distance(a,b,normalization=64) for a,b in zip(left,right)])
        torch.testing.assert_close(scalar,f+0.1*e,atol=1e-7 if dtype==torch.float32 else 1e-15,rtol=2e-6)
        gradient = torch.autograd.grad((0.9*f+0.1*e).sum(),x)[0]
        return f.detach().cpu(),e.detach().cpu(),gradient.detach().cpu()
    cpu,cuda = evaluate("cpu"),evaluate("cuda:0")
    tolerance = 2e-6 if dtype==torch.float32 else 2e-13
    for left,right in zip(cpu,cuda):
        torch.testing.assert_close(left,right,atol=tolerance,rtol=tolerance)
