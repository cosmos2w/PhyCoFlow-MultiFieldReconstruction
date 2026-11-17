import pytest
import torch

from phycoflow_reconstruction.coherence.state_compute import exact_common_embeddings
from phycoflow_reconstruction.utils.reproducibility import seed_everything


def test_strict_runtime_sets_workspace_before_cuda_probe(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    seen = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: seen.append(__import__('os').environ.get('CUBLAS_WORKSPACE_CONFIG')) or False)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        seed_everything(42, True, strict_algorithms=True)
        assert seen and seen[0] == ':4096:8'
        assert torch.are_deterministic_algorithms_enabled()
        monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', 'invalid')
        with pytest.raises(ValueError, match='CUBLAS_WORKSPACE_CONFIG'):
            seed_everything(42, True, strict_algorithms=True)
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA needed for deterministic CDF fallback')
def test_detached_mass_cdf_preserves_weighted_values_and_field_gradients():
    generator = torch.Generator().manual_seed(71)
    fields = [torch.randn(2, n, 2, generator=generator, dtype=torch.float64) for n in (7, 9)]
    weights = [torch.rand(2, len(x[0]), generator=generator, dtype=torch.float64) for x in fields]
    directions = torch.tensor([[2**-0.5, 2**-0.5]], dtype=torch.float64)
    def evaluate(device):
        x = [item.to(device).requires_grad_() for item in fields]
        m = [item.to(device) for item in weights]
        embeddings = exact_common_embeddings(x, m, directions.to(device))
        loss = sum((embeddings[0][k] - embeddings[1][k]).square().sum() for k in (0, 1))
        return loss.detach().cpu(), [g.cpu() for g in torch.autograd.grad(loss, x)]
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        cpu_value, cpu_grads = evaluate('cpu')
        cuda_value, cuda_grads = evaluate('cuda:0')
        torch.testing.assert_close(cpu_value, cuda_value, rtol=1e-10, atol=1e-11)
        for cpu, cuda in zip(cpu_grads, cuda_grads):
            torch.testing.assert_close(cpu, cuda, rtol=1e-9, atol=1e-10)
    finally:
        torch.use_deterministic_algorithms(previous)
