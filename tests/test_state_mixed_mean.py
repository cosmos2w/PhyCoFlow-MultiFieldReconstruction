"""Information and derivative contracts of the opt-in mean projection bank."""
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

from phycoflow_reconstruction.coherence import build_coherence_family
from phycoflow_reconstruction.coherence.base import projection_bank
from phycoflow_reconstruction.config import load_config
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer
from phycoflow_reconstruction.evaluation.coherence_set import GlobalDistributionAccumulator


def _family(fields=("a", "b", "c")):
    config = load_config("configs/coherence/state_mixed_mean.yaml")["coherence"]["families"]["global_distribution"]
    spec = DataSpec(fields, ("1",) * len(fields), 2, (8,))
    return build_coherence_family("global_distribution", config, spec, FieldNormalizer.identity(len(fields)))


def test_mean_bank_is_exact_sobol_gaussian_without_axes_or_pair_training():
    f = _family()
    assert f.score_definition == "marginal_mixed_mean_v2"
    assert "mutual_pairwise_swd" not in f.components_by_key
    bank = f.components_by_key["cross_joint_topk_swd"].directions
    torch.testing.assert_close(bank, projection_bank(3, 32, seed=1234, qmc=True, include_axes=False), rtol=0, atol=0)
    assert not (bank.abs().max(dim=1).values == 1).any()


def test_joint_permutation_invariance_independent_permutation_and_parity_witness():
    f = _family()
    states = torch.tensor(list(product((-1., 1.), repeat=3)))
    plus = states[states.prod(dim=1) > 0].repeat(2, 1)[None]
    minus = states[states.prod(dim=1) < 0].repeat(2, 1)[None]
    assert f(plus, plus[:, torch.tensor([4, 0, 7, 1, 6, 2, 5, 3])]).scalar_loss == 0
    result = f(plus, minus)
    assert result.component_results["global_distribution.self.marginal_w2"].scalar_loss == 0
    assert result.component_results["global_distribution.cross.joint_topk_swd"].scalar_loss > 0
    # A separate independent-channel rearrangement preserves marginals and changes occupation.
    base = torch.arange(24.).reshape(1, 8, 3)
    shuffled = base.clone(); shuffled[..., 1] = shuffled[..., 1].flip(1)
    assert f(base, shuffled).scalar_loss > 0


def test_fixed_normalized_units_ties_and_gradients():
    torch.manual_seed(91)
    f = _family()
    x = torch.randn(1, 8, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn_like(x)
    assert torch.autograd.gradcheck(lambda z: f(z, y).scalar_loss, (x,), eps=1e-6, atol=2e-5)
    tied = torch.zeros_like(x, requires_grad=True)
    loss = f(tied, torch.ones_like(tied)).scalar_loss
    assert torch.isfinite(torch.autograd.grad(loss, tied)[0]).all()
    offset = torch.tensor([20., -3., 101.], dtype=torch.float64)
    scale = torch.tensor([.01, 70., 2.], dtype=torch.float64)
    physical_x, physical_y = x * scale + offset, y * scale + offset
    torch.testing.assert_close(f((physical_x-offset)/scale, (physical_y-offset)/scale).scalar_loss, f(x,y).scalar_loss)
    one = _family(("a",))
    assert set(one.components_by_key) == {"self_marginal_w2"}
    assert one(x[..., :1], x[..., :1]).scalar_loss == 0


def test_disabled_state_component_roundtrip_is_unavailable(tmp_path):
    f = _family()
    other = _family(); other.load_state_artifact(f.state_artifact())
    x, y = torch.randn(1, 8, 3), torch.randn(1, 8, 3)
    torch.testing.assert_close(other(x,y).scalar_loss, f(x,y).scalar_loss, rtol=0, atol=0)
    spec = DataSpec(("a", "b", "c"), ("1",)*3, 2, (8,))
    acc = GlobalDistributionAccumulator(f, spec, f.config)
    acc.update(x, y, torch.rand(1, 8, 2), "train:0")
    assert np.isnan(acc.pairwise_total[0])
    out = acc.finalize(tmp_path, split="train", checkpoint_label="SOURCE", run_label="fixture", scale="linear")
    report = json.loads(Path(out["report"]).read_text())
    assert report["component_availability"]["pairwise"] is False
    assert report["statistics"]["weighted_component_totals"]["pairwise"]["mean"] is None
