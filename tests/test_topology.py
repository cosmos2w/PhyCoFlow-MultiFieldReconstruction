"""Contract and old-vs-new consistency gates for the topology family."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from phycoflow_reconstruction.coherence.registry import build_coherence_family
from phycoflow_reconstruction.contracts import DataSpec
from phycoflow_reconstruction.data.normalization import FieldNormalizer

pytest.importorskip("gudhi")

FAMILY = "topology"
OLDTERM = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "phycoflow_reconstruction"
    / "coherence"
    / "families"
    / "topology"
    / "oldterm"
)


def _lattice(height: int, width: int) -> torch.Tensor:
    """`[H*W, 2]` coordinates `(x, y)` with x along columns and y along rows."""
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def _config(
    height: int,
    width: int,
    *,
    fields: tuple[str, ...] = ("u", "v", "w"),
    self_enabled: bool = True,
    mutual_enabled: bool = False,
    dimensions: tuple[int, ...] = (0, 1),
    directions: tuple[str, ...] = ("sublevel", "superlevel"),
    order: float = 1.0,
    lambda_spatial: float = 1.0,
    spatial_mode: str = "multiplicative",
    min_persistence: float = 0.0,
    lines: int = 3,
    seed: int = 0,
    line_sampling: str = "uniform",
) -> dict:
    return {
        "fields": list(fields),
        "geometry": {"grid_shape": [height, width], "neighbors": 1},
        "filtration": {
            "dimensions": list(dimensions),
            "directions": list(directions),
            "smoothing_sigma": 0.0,
        },
        "matching": {
            "order": order,
            "lambda_spatial": lambda_spatial,
            "spatial_mode": spatial_mode,
            "min_persistence": min_persistence,
        },
        "components": {
            "self": {"enabled": self_enabled, "weight": 1.0 if self_enabled else 0.0},
            "mutual": {
                "enabled": mutual_enabled,
                "weight": 1.0 if mutual_enabled else 0.0,
                "lines": lines,
                "seed": seed,
                "line_sampling": line_sampling,
            },
        },
    }


def _family(config: dict, fields: tuple[str, ...] = ("u", "v", "w")):
    spec = DataSpec(fields, ("1",) * len(fields), 2, (4, 4))
    return build_coherence_family(FAMILY, config, spec, FieldNormalizer.identity(len(fields)))


def _batch(height: int, width: int, channels: int, seed: int, count: int = 2, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    coordinates = _lattice(height, width).to(dtype).unsqueeze(0).expand(count, -1, -1).contiguous()
    generated = torch.randn(count, height * width, channels, generator=generator, dtype=dtype)
    reference = torch.randn(count, height * width, channels, generator=generator, dtype=dtype)
    return coordinates, generated, reference


# ----------------------------------------------------------------------------- contract


def test_family_registers_both_components_with_cubical_metadata() -> None:
    family = _family(_config(4, 4, mutual_enabled=True))
    names = [component.name for component in family.spec.components]
    assert names == ["self.persistence_matching", "mutual.fibered_matching"]
    for component in family.spec.components:
        assert component.required_geometry == "fixed_2d_raster"
        assert component.metadata["complex"] == "cubical"
        assert component.metadata["min_persistence"] == 0.0
    assert family.mutual_pairs == ((0, 1), (0, 2), (1, 2))


def test_defaults_are_a_floor_of_one_percent_and_eight_stratified_lines() -> None:
    config = _config(4, 4, mutual_enabled=True)
    del config["matching"]["min_persistence"]
    del config["components"]["mutual"]["lines"]
    del config["components"]["mutual"]["line_sampling"]
    family = _family(config)
    assert family.min_persistence == 0.01
    assert family.fibered_lines == 8 and family.line_sampling == "stratified"


@pytest.mark.parametrize("weight", [0.0, -1.0])
def test_outer_weight_must_be_positive(weight: float) -> None:
    config = _config(4, 4)
    config["weight"] = weight
    with pytest.raises(ValueError, match=r"topology\.weight must be positive"):
        _family(config)


def test_periodic_geometry_and_bad_matching_settings_are_rejected() -> None:
    config = _config(4, 4)
    config["geometry"]["periodic"] = True
    with pytest.raises(ValueError, match="nonperiodic"):
        _family(config)
    with pytest.raises(ValueError, match="spatial_mode"):
        _family(_config(4, 4, spatial_mode="geometric"))
    with pytest.raises(ValueError, match="lambda_spatial"):
        _family(_config(4, 4, lambda_spatial=-1.0))
    with pytest.raises(ValueError, match="min_persistence"):
        _family(_config(4, 4, min_persistence=1.5))
    with pytest.raises(ValueError, match="line_sampling"):
        _family(_config(4, 4, mutual_enabled=True, line_sampling="jittered"))
    config = _config(4, 4)
    config["matching"]["solver"] = "auction"
    with pytest.raises(ValueError, match="removed"):
        _family(config)


def test_identical_fields_cost_nothing_and_random_fields_carry_gradient() -> None:
    family = _family(_config(5, 6, mutual_enabled=True, lines=2))
    coordinates, generated, reference = _batch(5, 6, 3, seed=1)
    zero = family(reference.clone(), reference, coordinates=coordinates)
    assert torch.equal(zero.per_sample_cost, torch.zeros_like(zero.per_sample_cost))
    tracked = generated.clone().requires_grad_(True)
    result = family(tracked, reference, coordinates=coordinates)
    result.scalar_loss.backward()
    assert result.per_sample_cost.shape == (2,)
    assert (result.per_sample_cost > 0).all()
    assert tracked.grad is not None and tracked.grad.abs().sum() > 0
    term = result.component_results["topology.self.persistence_matching"]
    assert term.diagnostics["matchings"] == 2 * 2 * 3 * 2  # samples x directions x fields x dims
    assert term.diagnostics["generated_finite_bars"] > 0
    assert term.diagnostics["generated_essential_bars"] == 2 * 2 * 3  # one H0 per field/direction
    assert result.diagnostics["solver"] == "exact_hungarian"


def test_multiplicative_spatial_term_never_lowers_the_cost() -> None:
    coordinates, generated, reference = _batch(6, 6, 2, seed=2)
    plain = _family(_config(6, 6, fields=("u", "v"), lambda_spatial=0.0), ("u", "v"))
    spatial = _family(_config(6, 6, fields=("u", "v"), lambda_spatial=1.0), ("u", "v"))
    plain_cost = plain(generated, reference, coordinates=coordinates).per_sample_cost
    spatial_cost = spatial(generated, reference, coordinates=coordinates).per_sample_cost
    assert (spatial_cost >= plain_cost).all()
    assert (spatial_cost > plain_cost).any()


def test_additive_spatial_term_penalizes_a_translated_feature() -> None:
    height = width = 8
    coordinates = _lattice(height, width).unsqueeze(0)
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    def bump(center_y: int, center_x: int) -> torch.Tensor:
        return torch.exp(-(((y - center_y) ** 2 + (x - center_x) ** 2) / 2.0)).to(torch.float64)

    reference = bump(2, 2).reshape(1, -1, 1)
    generated = bump(5, 5).reshape(1, -1, 1)
    common = {"fields": ("u",), "dimensions": (0,), "directions": ("superlevel",)}
    plain = _family(_config(height, width, lambda_spatial=0.0, **common), ("u",))
    additive = _family(
        _config(height, width, lambda_spatial=1.0, spatial_mode="additive", **common), ("u",)
    )
    # The diagrams are identical, so only creator separation can make this positive.
    assert plain(generated, reference, coordinates=coordinates).scalar_loss == 0
    assert additive(generated, reference, coordinates=coordinates).scalar_loss > 0


@pytest.mark.parametrize("sampling", ["uniform", "stratified"])
def test_slice_lines_are_seeded_redrawn_per_call_and_continued_by_artifacts(sampling) -> None:
    config = _config(
        5, 5, fields=("u", "v"), self_enabled=False, mutual_enabled=True, seed=7, line_sampling=sampling
    )
    coordinates, generated, reference = _batch(5, 5, 2, seed=4)
    first = _family(config, ("u", "v"))
    second = _family(config, ("u", "v"))
    call_1 = first(generated, reference, coordinates=coordinates).per_sample_cost
    # Same seed, same stream: a fresh family reproduces the first call exactly.
    assert torch.equal(second(generated, reference, coordinates=coordinates).per_sample_cost, call_1)
    # A reload continues the stream, so it matches the original family's *next* call.
    restored = _family(config, ("u", "v"))
    restored.load_state_artifact(first.state_artifact())
    call_2 = first(generated, reference, coordinates=coordinates).per_sample_cost
    assert torch.equal(restored(generated, reference, coordinates=coordinates).per_sample_cost, call_2)
    # Lines are re-drawn every call, so consecutive calls are stochastic estimates.
    assert not torch.equal(call_1, call_2)
    term = first(generated, reference, coordinates=coordinates).component_results[
        "topology.mutual.fibered_matching"
    ]
    assert term.diagnostics["line_sampling"] == f"{sampling}_random_per_call"
    assert term.diagnostics["line_seed"] == 7


def test_stratified_lines_hit_every_angle_bin_and_uniform_lines_are_sorted() -> None:
    from phycoflow_reconstruction.coherence.families.topology.persistence import (
        slice_lines,
    )

    lo_left = np.array([0.1, -2.0, 3.0])
    lo_right = np.array([-1.0, 0.5, 2.5])
    margin, count = 0.12, 6
    _, stratified = slice_lines(lo_left, lo_right, count, margin, np.random.default_rng(3))
    angles = np.arctan2(stratified[..., 1], stratified[..., 0]) / (np.pi / 2.0)
    edges = margin + (1.0 - 2.0 * margin) * np.arange(count + 1) / count
    assert angles.shape == (3, count)
    assert (angles >= edges[None, :-1]).all() and (angles <= edges[None, 1:]).all()
    base, uniform = slice_lines(lo_left, lo_right, count, margin, np.random.default_rng(3), "uniform")
    angles = np.arctan2(uniform[..., 1], uniform[..., 0])
    assert (np.diff(angles, axis=1) >= 0).all()
    assert np.allclose(base[:, :, 0], lo_left[:, None]) and np.allclose(base[:, :, 1], lo_right[:, None])
    # Both forms consume the same number of draws, so the stream position is sampling-independent.
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    slice_lines(lo_left, lo_right, count, margin, a, "stratified")
    slice_lines(lo_left, lo_right, count, margin, b, "uniform")
    assert a.bit_generator.state == b.bit_generator.state


def test_reference_generator_cache_is_bitwise_transparent_and_cleared_on_reload() -> None:
    config = _config(6, 7, mutual_enabled=True, lines=2, seed=3)
    coordinates, generated, reference = _batch(6, 7, 3, seed=8)
    reference[1] = reference[0]  # a repeat inside one batch is served after its twin
    family = _family(config)
    tracked = generated.clone().requires_grad_(True)
    cold = family(tracked, reference, coordinates=coordinates)
    cold.scalar_loss.backward()
    cold_misses = family._reference_cache_misses
    assert cold_misses > 0 and cold.diagnostics["reference_cache"]["entries"] > 0

    tracked = generated.clone().requires_grad_(True)
    warm = family(tracked, reference, coordinates=coordinates)
    warm.scalar_loss.backward()
    # Every self-term reference in the second call was already computed in the first.
    assert family._reference_cache_misses == cold_misses
    assert warm.diagnostics["reference_cache"]["hits"] > cold.diagnostics["reference_cache"]["hits"]
    self_path = "topology.self.persistence_matching"
    assert torch.equal(
        cold.component_results[self_path].per_sample_cost,
        warm.component_results[self_path].per_sample_cost,
    )
    assert cold.component_results[self_path].diagnostics == warm.component_results[self_path].diagnostics
    self_only = _family(_config(6, 7, seed=3))
    a = generated.clone().requires_grad_(True)
    self_only(a, reference, coordinates=coordinates).scalar_loss.backward()
    b = generated.clone().requires_grad_(True)
    self_only(b, reference, coordinates=coordinates).scalar_loss.backward()
    assert torch.equal(a.grad, b.grad)

    family.load_state_artifact(family.state_artifact())
    assert not family._reference_cache


def test_pooled_and_inline_solves_agree_bitwise(monkeypatch) -> None:
    from phycoflow_reconstruction.coherence.families.topology import (
        execution as pool_module,
    )

    config = _config(6, 6, mutual_enabled=True, lines=2, seed=5)
    coordinates, generated, reference = _batch(6, 6, 3, seed=9)
    outputs = []
    for workers in (4, 1):
        monkeypatch.setattr(pool_module, "_WORKER_COUNT", workers)
        family = _family(config)
        tracked = generated.clone().requires_grad_(True)
        result = family(tracked, reference, coordinates=coordinates)
        result.scalar_loss.backward()
        assert result.diagnostics["workers"] == workers
        outputs.append((result.per_sample_cost.detach(), tracked.grad))
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], outputs[1][1])


def test_state_artifact_roundtrip_restores_geometry() -> None:
    family = _family(_config(4, 5))
    coordinates, generated, reference = _batch(4, 5, 3, seed=3)
    before = family(generated, reference, coordinates=coordinates).per_sample_cost
    restored = _family(_config(4, 5))
    restored.load_state_artifact(family.state_artifact())
    assert restored.geometry_sha256 == family.geometry_sha256
    assert torch.equal(restored.grid_coordinates, family.grid_coordinates)
    after = restored(generated, reference, coordinates=coordinates).per_sample_cost
    assert torch.equal(before, after)


# ---------------------------------------------------------- consistency with oldterm/


@pytest.fixture(scope="module")
def oldterm():
    """The historical modules, imported from the untracked `oldterm/` snapshot."""
    if not OLDTERM.is_dir():
        pytest.skip("historical oldterm/ snapshot is not present")
    inserted = str(OLDTERM) not in sys.path
    if inserted:
        sys.path.insert(0, str(OLDTERM))
    try:
        import fibered_satloss_cubical
        import persistence_satloss_cubical
    finally:
        if inserted:
            sys.path.remove(str(OLDTERM))
    return persistence_satloss_cubical, fibered_satloss_cubical


@pytest.mark.parametrize(
    ("spatial_mode", "lambda_spatial", "order"),
    [
        ("multiplicative", 1.0, 1.0),
        ("multiplicative", 0.0, 1.0),
        ("additive", 0.7, 1.0),
        ("multiplicative", 1.0, 2.0),
    ],
)
def test_self_term_matches_historical_persistence_satloss_cubical(
    oldterm, spatial_mode: str, lambda_spatial: float, order: float
) -> None:
    """With no floor, the value and gradient reproduce the historical mode to 1e-9."""
    persistence_satloss_cubical, _ = oldterm
    height, width, channels = 7, 9, 3
    coordinates, generated, reference = _batch(height, width, channels, seed=11)
    directions = ("sublevel", "superlevel")
    dims = (0, 1)
    family = _family(
        _config(
            height,
            width,
            order=order,
            lambda_spatial=lambda_spatial,
            spatial_mode=spatial_mode,
            dimensions=dims,
            directions=directions,
        )
    )
    tracked = generated.clone().requires_grad_(True)
    result = family(tracked, reference, coordinates=coordinates)
    result.per_sample_cost.sum().backward()

    for sample in range(generated.shape[0]):
        old_gen = generated[sample].clone().requires_grad_(True)
        pieces = []
        for direction in directions:
            sign = 1.0 if direction == "sublevel" else -1.0
            distances = persistence_satloss_cubical.differentiable_persistence_satloss_cubical_distances(
                sign * old_gen,
                sign * reference[sample],
                coordinates[sample],
                dims,
                order=order,
                lambda_spatial=lambda_spatial,
                spatial_mode=spatial_mode,
            )
            pieces.extend(distances[dim] for dim in dims)
        old_cost = torch.cat(pieces).mean()
        old_cost.backward()
        assert torch.allclose(result.per_sample_cost[sample], old_cost, rtol=1e-9, atol=1e-9)
        assert torch.allclose(tracked.grad[sample], old_gen.grad, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(
    ("spatial_mode", "lambda_spatial"),
    [("multiplicative", 1.0), ("multiplicative", 0.0), ("additive", 0.5)],
)
def test_mutual_term_matches_historical_fibered_satloss_cubical(
    oldterm, spatial_mode: str, lambda_spatial: float
) -> None:
    """With uniform lines and no floor, the mutual term reproduces the historical mode."""
    _, fibered_satloss_cubical = oldterm
    height, width, channels, lines, seed = 6, 8, 3, 4, 19
    coordinates, generated, reference = _batch(height, width, channels, seed=12)
    directions = ("sublevel", "superlevel")
    degrees = (0, 1)
    family = _family(
        _config(
            height,
            width,
            self_enabled=False,
            mutual_enabled=True,
            lines=lines,
            seed=seed,
            lambda_spatial=lambda_spatial,
            spatial_mode=spatial_mode,
            dimensions=degrees,
            directions=directions,
        )
    )
    tracked = generated.clone().requires_grad_(True)
    result = family(tracked, reference, coordinates=coordinates)
    result.per_sample_cost.sum().backward()
    assert "topology.self.persistence_matching" not in result.component_results

    # The family draws lines per (sample, direction, pair) in that nesting order; the
    # historical driver draws per pair inside one (sample, direction) call, so one
    # shared generator walked in the same order yields identical slice lines.
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(channels) for j in range(i + 1, channels)]
    for sample in range(generated.shape[0]):
        old_gen = generated[sample].clone().requires_grad_(True)
        pieces = []
        for direction in directions:
            matrices = fibered_satloss_cubical._fibered_satloss_matrices(
                old_gen,
                reference[sample],
                coordinates[sample],
                lines,
                degrees,
                direction == "superlevel",
                lambda_spatial,
                spatial_mode,
                rng=rng,
            )
            pieces.extend(matrices[degree][i, j] for i, j in pairs for degree in degrees)
        old_cost = torch.stack(pieces).mean()
        old_cost.backward()
        assert torch.allclose(result.per_sample_cost[sample], old_cost, rtol=1e-9, atol=1e-9)
        assert torch.allclose(tracked.grad[sample], old_gen.grad, rtol=1e-9, atol=1e-9)


# ------------------------------------------------------------------- degenerate cases


def test_empty_diagrams_and_zero_cost_matches_have_finite_gradients() -> None:
    """Planes have no H1 at all, and their shared minimum vertex matches at zero cost."""
    height, width = 6, 8
    coordinates = _lattice(height, width).unsqueeze(0).expand(2, -1, -1).contiguous()
    x, y = coordinates[0, :, 0], coordinates[0, :, 1]
    generated = torch.stack([x + 0.5 * y, 2.0 * x - 0.25 * y], dim=1)[None].repeat(2, 1, 1)
    reference = torch.stack([x + 0.3 * y, 1.5 * x - 0.5 * y], dim=1)[None].repeat(2, 1, 1)
    generated[1] = generated[1] * 1.7
    family = _family(_config(height, width, fields=("u", "v"), mutual_enabled=True, lines=2, order=2.0), ("u", "v"))
    tracked = generated.clone().requires_grad_(True)
    result = family(tracked, reference, coordinates=coordinates)
    result.per_sample_cost.sum().backward()
    assert torch.isfinite(tracked.grad).all()


@pytest.mark.parametrize("order", [2.0, 1.5, 0.5])
def test_zero_distance_has_a_zero_gradient_at_every_order(order: float) -> None:
    """Identical fields are the minimum of the distance, so zero is its subgradient there."""
    config = _config(6, 7, mutual_enabled=True, lines=2, order=order)
    coordinates, generated, _ = _batch(6, 7, 3, seed=4)
    family = _family(config)
    tracked = generated.clone().requires_grad_(True)
    result = family(tracked, generated.clone(), coordinates=coordinates)
    result.per_sample_cost.sum().backward()
    assert torch.equal(result.per_sample_cost.detach(), torch.zeros(2, dtype=torch.float64))
    assert torch.equal(tracked.grad, torch.zeros_like(tracked))


def test_pack_host_rows_pads_with_usable_indices() -> None:
    from phycoflow_reconstruction.coherence.families.topology.persistence import (
        pack_host_rows,
    )

    rows = [np.array([4, 2, 7]), np.zeros(0, dtype=np.int64), np.array([1])]
    packed, mask = pack_host_rows(rows, torch.device("cpu"))
    assert packed.shape == mask.shape == (3, 3)
    assert packed[0].tolist() == [4, 2, 7] and mask[0].all()
    assert not mask[1].any() and (packed[1] >= 0).all()
    assert packed[2].tolist() == [1, 1, 1] and mask[2].tolist() == [True, False, False]
    empty, empty_mask = pack_host_rows([np.zeros(0), np.zeros(0)], torch.device("cpu"))
    assert empty.shape == empty_mask.shape == (2, 0)


# ---------------------------------------------------------------- device reductions


def _structured_fields() -> list[tuple[str, int, int, np.ndarray]]:
    """Random, heavily tied, degenerate and hole-bearing fields for the reduction gates."""
    rng = np.random.default_rng(0)
    cases = []
    for height, width in [(7, 9), (2, 2), (1, 6), (5, 1), (3, 3), (12, 16)]:
        cases.append((f"random {height}x{width}", height, width, rng.standard_normal((height, width))))
        cases.append(
            (f"plateau {height}x{width}", height, width, rng.integers(0, 3, size=(height, width)) * 1.0)
        )
    yy, xx = np.mgrid[0:12, 0:14]
    ring = ((xx - 6) ** 2 + (yy - 5) ** 2 - 12.0) ** 2 / 100 + 0.01 * rng.standard_normal((12, 14))
    cases.append(("ring", 12, 14, ring))
    cases.append(("saddle", 12, 14, ((xx - 6) ** 2 - (yy - 5) ** 2) * 1.0))
    cases.append(("constant", 6, 8, np.zeros((6, 8))))
    return cases


@pytest.mark.parametrize("floor", [0.0, 0.3])
@pytest.mark.parametrize("case", _structured_fields(), ids=lambda case: case[0])
def test_device_reduction_reproduces_gudhi_bar_for_bar(case, floor: float) -> None:
    from phycoflow_reconstruction.coherence.families.topology.persistence import (
        cubical_generators,
        cubical_generators_on_device,
    )

    _, height, width, field = case
    host = cubical_generators(field.reshape(-1), height, width, (0, 1), floor)
    block = torch.as_tensor(np.stack([field.reshape(-1)] * 3))
    device, _ = cubical_generators_on_device(block, height, width, (0, 1), floor)
    for unit in range(3):
        for (finite_host, essential_host), (finite_dev, essential_dev) in zip(host, device[unit]):
            assert set(map(tuple, finite_host.tolist())) == set(map(tuple, finite_dev.tolist()))
            assert finite_host.shape == finite_dev.shape
            assert essential_host.tolist() == essential_dev.tolist()


def test_persistence_floor_drops_short_bars_and_only_slightly_moves_the_value() -> None:
    coordinates, generated, reference = _batch(9, 11, 2, seed=17)
    reference = torch.nn.functional.avg_pool1d(reference.transpose(1, 2), 3, 1, 1).transpose(1, 2)
    generated = reference + 0.05 * generated
    results = {}
    for floor in (0.0, 0.02, 0.1):
        family = _family(_config(9, 11, fields=("u", "v"), min_persistence=floor), ("u", "v"))
        results[floor] = family(generated, reference, coordinates=coordinates)
    bars = [
        results[f].component_results["topology.self.persistence_matching"].diagnostics[
            "generated_finite_bars"
        ]
        for f in (0.0, 0.02, 0.1)
    ]
    assert bars[0] > bars[1] > bars[2] >= 0
    for floor in (0.02, 0.1):
        term = results[floor].component_results["topology.self.persistence_matching"]
        assert term.diagnostics["min_persistence"] == floor
    # The floor removes noise-level bars whose cost was their persistence, so the
    # value can only decrease, and by a bounded amount.
    assert (results[0.02].per_sample_cost <= results[0.0].per_sample_cost).all()
    assert (results[0.02].per_sample_cost > 0.5 * results[0.0].per_sample_cost).all()


@pytest.mark.parametrize("floor", [0.0, 0.05])
def test_family_is_unchanged_by_which_reduction_backend_runs(monkeypatch, floor: float) -> None:
    config = _config(7, 9, mutual_enabled=True, lines=2, seed=8, min_persistence=floor)
    coordinates, generated, reference = _batch(7, 9, 3, seed=31)
    outputs = {}
    for mode in ("host", "tensor"):
        monkeypatch.setenv("PHYCOFLOW_TOPOLOGY_PAIRING", mode)
        family = _family(config)
        tracked = generated.clone().requires_grad_(True)
        result = family(tracked, reference, coordinates=coordinates)
        result.per_sample_cost.sum().backward()
        expected = "tensor" if mode == "tensor" else "gudhi"
        assert result.diagnostics["reduction_backend"] == expected
        self_term = result.component_results["topology.self.persistence_matching"]
        assert (self_term.diagnostics["merge_rounds"] > 0) == (mode == "tensor")
        outputs[mode] = (result.per_sample_cost.detach(), tracked.grad, self_term.diagnostics)
    assert torch.allclose(outputs["tensor"][0], outputs["host"][0], rtol=1e-12, atol=1e-12)
    assert torch.allclose(outputs["tensor"][1], outputs["host"][1], rtol=1e-12, atol=1e-12)
    for key in ("generated_finite_bars", "reference_finite_bars", "generated_essential_bars"):
        assert outputs["tensor"][2][key] == outputs["host"][2][key]


def test_a_declined_reduction_falls_back_to_gudhi_and_says_so(monkeypatch) -> None:
    from phycoflow_reconstruction.coherence.families.topology import persistence

    def decline(*args, **kwargs):
        raise persistence.MergeTreeUnavailable("declined for the test")

    config = _config(6, 7, mutual_enabled=True, lines=2, seed=2)
    coordinates, generated, reference = _batch(6, 7, 3, seed=5)
    monkeypatch.setenv("PHYCOFLOW_TOPOLOGY_PAIRING", "host")
    host = _family(config)(generated, reference, coordinates=coordinates)
    monkeypatch.setenv("PHYCOFLOW_TOPOLOGY_PAIRING", "tensor")
    monkeypatch.setattr(persistence, "sublevel_generators_on_device", decline)
    fallen = _family(config)(generated, reference, coordinates=coordinates)
    assert fallen.diagnostics["reduction_backend"] == "gudhi"
    assert torch.equal(fallen.per_sample_cost, host.per_sample_cost)


def test_reference_cache_serves_the_device_backend_too(monkeypatch) -> None:
    monkeypatch.setenv("PHYCOFLOW_TOPOLOGY_PAIRING", "tensor")
    config = _config(6, 6, mutual_enabled=False)
    coordinates, generated, reference = _batch(6, 6, 3, seed=13)
    family = _family(config)
    cold = family(generated, reference, coordinates=coordinates)
    warm = family(generated, reference, coordinates=coordinates)
    cache = warm.diagnostics["reference_cache"]
    assert cache["misses"] == cold.diagnostics["reference_cache"]["misses"] and cache["hits"] > 0
    assert torch.equal(cold.per_sample_cost, warm.per_sample_cost)


def test_fused_degrees_match_the_single_degree_pairings() -> None:
    from phycoflow_reconstruction.coherence.families.topology.merge_tree import (
        sublevel_generators_on_device,
        sublevel_pairs_on_device,
    )

    rng = np.random.default_rng(5)
    for height, width in [(9, 11), (2, 7), (1, 5)]:
        block = torch.as_tensor(np.round(rng.standard_normal((5, height * width)), 1))
        fused, _ = sublevel_generators_on_device(block, height, width, (0, 1))
        for degree in (0, 1):
            row, birth, death, essential, _ = sublevel_pairs_on_device(block, height, width, degree)
            single = set(zip(row.tolist(), birth.tolist(), death.tolist()))
            got = set(zip(*(t.tolist() for t in fused[degree][:3])))
            assert got == single
            assert torch.equal(fused[degree][3], essential)


def test_stacked_and_separate_degree_merges_agree(monkeypatch) -> None:
    from phycoflow_reconstruction.coherence.families.topology import merge_tree

    rng = np.random.default_rng(6)
    block = torch.as_tensor(np.round(rng.standard_normal((6, 9 * 11)), 1))
    outputs = []
    for limit in (1 << 30, 0):
        monkeypatch.setattr(merge_tree, "FUSED_ELEMENT_LIMIT", limit)
        bars, _ = merge_tree.sublevel_generators_on_device(block, 9, 11, (0, 1))
        outputs.append(
            {
                degree: (set(zip(*(t.tolist() for t in bars[degree][:3]))), bars[degree][3].tolist())
                for degree in (0, 1)
            }
        )
    assert outputs[0] == outputs[1]
