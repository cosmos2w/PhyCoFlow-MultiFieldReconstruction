"""Cubical persistence primitives for spatially-aware diagram matching.

The construction is the standard "detach the combinatorics, keep the geometry"
recipe for persistence losses. Every finite bar of a lower-star filtration is born
and dies at the field value of one specific vertex (its generator), and both the
generator assignment and the optimal diagram matching are piecewise constant in the
field. The generators are read off the detached values, and the distance is
re-expressed as differentiable gathers from the live tensor, which yields the exact
detached value together with a valid subgradient.

The matching is spatially aware: each bar-to-bar cost is modified by the domain
separation of the two bars' creators (their birth vertices), so producing a
correct-looking feature in the wrong place is not free. This follows Wen et al.,
"Topology-Preserving Image Segmentation with Spatial-Aware Persistent Feature
Matching" (arXiv:2412.02076) and Soler et al., "Lifted Wasserstein Matcher"
(arXiv:1808.05870).

One pipeline serves every device. Reductions run as merge trees on the fields'
device (:func:`cubical_generators_on_device`, through :mod:`..topology.merge_tree`,
H0 directly and H1 through the planar dual) or through GUDHI on the host
(:func:`cubical_generators`); the two agree bar for bar. Bars shorter than a
persistence floor are dropped at the reduction, since a diagram of a smoothed field
is mostly noise-level bars that only ever match the diagonal. The cost matrices of
every unit are then built as padded batched tensors (:func:`plan_units`), each
size-sorted chunk crosses to the host once, and SciPy's Hungarian solve runs per
unit on the shared thread pool. The differentiable half
(:func:`apply_units_batched`) contracts the solved plans against the live diagrams of
the whole call in one set of kernels.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Any, NamedTuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .execution import parallel_map
from .merge_tree import MergeTreeUnavailable, sublevel_generators_on_device

#: Sentinel distinguishing "import not attempted yet" from "GUDHI is absent".
_GUDHI_UNRESOLVED = object()
_GUDHI: Any = _GUDHI_UNRESOLVED

SPATIAL_MODES = ("multiplicative", "additive")
LINE_SAMPLINGS = ("stratified", "uniform")

__all__ = ["MergeTreeUnavailable"]

#: Detached generators of one homological degree: finite `(birth, death)` vertex
#: ids as an `(n, 2)` array, and essential birth vertex ids as a `(k,)` array.
Generators = tuple[np.ndarray, np.ndarray]



def gudhi_module() -> ModuleType:
    """Return GUDHI, or raise an ImportError naming the extra that provides it."""
    global _GUDHI
    if _GUDHI is _GUDHI_UNRESOLVED:
        try:
            import gudhi
        except ImportError:  # pragma: no cover - exercised only without the extra
            _GUDHI = None
        else:
            _GUDHI = gudhi
    if _GUDHI is None:
        raise ImportError(
            "topology requires GUDHI; install the optional `topology` extra"
        )
    return _GUDHI


def gudhi_available() -> bool:
    try:
        gudhi_module()
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------- reductions


def cubical_generators(
    values: np.ndarray,
    height: int,
    width: int,
    dimensions: tuple[int, ...],
    min_persistence: float = 0.0,
) -> list[Generators]:
    """Vertex generators of the cubical lower-star persistence of a `[H,W]` field.

    `values` is the detached field in C order (row-major over `[H,W]`). GUDHI reads a
    flat vertex array in Fortran order, so the grid is declared as `(width, height)`;
    the vertex ids it returns then index the C-order array directly. Persistence is
    computed once and every requested degree is read off that single reduction.

    The filtration handed to GUDHI is each vertex's rank in `(value ascending, id
    ascending)` order rather than its value, as the Betti family does: ranks are a
    strict total order, so which vertex generates a bar is fixed by that order rather
    than by GUDHI's own tie convention, and the device backend reproduces it exactly.
    Finite bars whose persistence in value is zero, or below `min_persistence`, are
    then dropped; essential bars are always kept.

    Returns one `(finite, essential)` pair per entry of `dimensions`, in order.
    """
    gudhi = gudhi_module()
    cells = np.ascontiguousarray(values.reshape(-1), dtype=np.float64)
    if cells.size != height * width:
        raise ValueError("cubical_generators expects height*width values")
    order = np.lexsort((np.arange(cells.size), cells))
    rank = np.empty(cells.size, dtype=np.float64)
    rank[order] = np.arange(cells.size, dtype=np.float64)
    complex_ = gudhi.CubicalComplex(vertices=rank, dimensions=(width, height))
    complex_.compute_persistence(homology_coeff_field=2, min_persistence=0)
    finite_by_dim, essential_by_dim = complex_.vertices_of_persistence_pairs()
    out: list[Generators] = []
    for dim in dimensions:
        if 0 <= dim < len(finite_by_dim) and len(finite_by_dim[dim]):
            finite = np.asarray(finite_by_dim[dim], dtype=np.int64).reshape(-1, 2)
            persistence = np.abs(cells[finite[:, 1]] - cells[finite[:, 0]])
            finite = finite[(persistence > 0.0) & (persistence >= float(min_persistence))]
        else:
            finite = np.zeros((0, 2), dtype=np.int64)
        if 0 <= dim < len(essential_by_dim) and len(essential_by_dim[dim]):
            essential = np.asarray(essential_by_dim[dim], dtype=np.int64).reshape(-1)
        else:
            essential = np.zeros((0,), dtype=np.int64)
        out.append((finite, essential))
    return out


def cubical_generators_on_device(
    block: torch.Tensor,
    height: int,
    width: int,
    dimensions: tuple[int, ...],
    min_persistence: float = 0.0,
) -> tuple[list[list[Generators]], int]:
    """:func:`cubical_generators` for a whole `[K, N]` detached block on its device.

    All requested degrees are merged in one device pass over the block, the floor is
    applied there, and each degree's bars come back to the host in one copy however
    many units the block holds; the per-unit host arrays come out in the layout
    :func:`cubical_generators` produces. Returns `(generators per unit, merge rounds)`.
    Raises :class:`MergeTreeUnavailable` when the pipeline declines the block, in
    which case the caller runs the host reduction instead.
    """
    if block.ndim != 2:
        raise ValueError("cubical_generators_on_device expects [K,N]")
    count = block.shape[0]
    per_degree: list[list[Generators]] = []
    bars, total_rounds = sublevel_generators_on_device(
        block, height, width, dimensions, min_persistence=min_persistence
    )
    for dim in dimensions:
        row, birth, death, essential = bars[dim]
        # Bars arrive grouped by unit in unit order. Rows, births and deaths travel as
        # one `[M, 3]` copy, the essentials as one `[K]` copy.
        packed = torch.stack((row, birth, death), dim=1).cpu().numpy().astype(np.int64, copy=False)
        essential_host = essential.cpu().numpy()
        splits = np.cumsum(np.bincount(packed[:, 0], minlength=count))[:-1]
        finite_rows = np.split(packed[:, 1:], splits)
        per_degree.append(
            [
                (
                    np.ascontiguousarray(finite_rows[unit]),
                    essential_host[unit : unit + 1]
                    if essential_host[unit] >= 0
                    else np.zeros((0,), dtype=np.int64),
                )
                for unit in range(count)
            ]
        )
    return [[degree[unit] for degree in per_degree] for unit in range(count)], total_rounds


# ------------------------------------------------------------------ geometry


def domain_scale(coordinates: np.ndarray) -> float:
    """Bounding-box diagonal, so a creator separation of 1 means opposite corners."""
    if coordinates.size == 0:
        return 1.0
    extent = coordinates.max(axis=0) - coordinates.min(axis=0)
    scale = float(np.linalg.norm(extent))
    return scale if scale > 0.0 else 1.0


def validate_spatial_mode(spatial_mode: str) -> str:
    if spatial_mode not in SPATIAL_MODES:
        raise ValueError(
            f"unknown spatial_mode={spatial_mode!r}; expected one of {SPATIAL_MODES}"
        )
    return spatial_mode


def _power(values, order: float):
    """`values ** order`, with a zero subgradient at an exact zero for `order < 1`.

    The order-`p` distance is a root of a sum of `p`-th powers, and a root has no
    derivative at zero: autograd would multiply an infinite slope by a zero cost and
    return NaN wherever two diagrams match at exactly zero cost. Zero distance is the
    minimum of the distance, so zero is a valid subgradient there, and that is the
    convention taken: the forward value is unchanged (`0 ** order == 0`), and only the
    gradient at that single point is defined rather than left undefined. Powers of at
    least one have a finite derivative at zero already, and detached NumPy inputs need
    no guard at all.
    """
    if order == 1.0:
        return values
    if order >= 1.0 or not isinstance(values, torch.Tensor):
        return values**order
    positive = values > 0
    return torch.where(positive, values, torch.ones_like(values)) ** order * positive


# ---------------------------------------------------------- padded generators


class PaddedGenerators(NamedTuple):
    """Per-unit generators padded to `[K, M]` on a device: ids, essential and valid masks.

    Built once per degree and side for a whole call by :func:`pad_generators`, then
    shared by the cost build and the live half. Padded slots hold index 0 so they
    stay usable gather indices; `valid` is what removes their contribution.
    """

    births: torch.Tensor
    deaths: torch.Tensor
    essential: torch.Tensor
    valid: torch.Tensor

    @property
    def sizes(self) -> torch.Tensor:
        return self.valid.sum(dim=1)

    def rows(self, index: torch.Tensor, width: int) -> PaddedGenerators:
        """The units `index`, trimmed to `width` columns."""
        return PaddedGenerators(
            *(field.index_select(0, index)[:, :width] for field in self)
        )


def pad_generators(generators: Sequence[Generators], device: torch.device) -> PaddedGenerators:
    """Pad per-unit generators into `[K, M]` device tensors with one upload per field."""
    sizes = [int(finite.shape[0] + essential.shape[0]) for finite, essential in generators]
    width = max(max(sizes, default=0), 1)
    births = np.zeros((len(generators), width), dtype=np.int64)
    deaths = np.zeros((len(generators), width), dtype=np.int64)
    essential_mask = np.zeros((len(generators), width), dtype=bool)
    valid = np.zeros((len(generators), width), dtype=bool)
    for k, (finite, essential) in enumerate(generators):
        m = int(finite.shape[0])
        births[k, :m] = finite[:, 0]
        deaths[k, :m] = finite[:, 1]
        e = int(essential.shape[0])
        births[k, m : m + e] = essential
        essential_mask[k, m : m + e] = True
        valid[k, : m + e] = True
    return PaddedGenerators(
        *(torch.as_tensor(array, device=device) for array in (births, deaths, essential_mask, valid))
    )


def live_diagrams(
    block: torch.Tensor, padded: PaddedGenerators, cap: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`[K, M]` birth and death values gathered from a `[K, N]` block; essentials die at `cap`."""
    birth = block.gather(1, padded.births)
    death = torch.where(padded.essential, cap.expand_as(birth), block.gather(1, padded.deaths))
    return birth, death


# ------------------------------------------------------------------ matching


class MatchingPlan(NamedTuple):
    """The combinatorial half of a diagram matching: who pairs with whom.

    `pair_gen` / `pair_ref` are row-aligned index arrays of the bar-to-bar matches;
    `gen_diagonal` and `ref_diagonal` index the bars sent to the diagonal on each
    side. The spatial factor of a matched pair is a function of the two creators'
    positions, so the live half recomputes it on the device rather than carrying it.
    """

    pair_gen: np.ndarray
    pair_ref: np.ndarray
    gen_diagonal: np.ndarray
    ref_diagonal: np.ndarray


def _plan_from_reduced(reduced: np.ndarray) -> MatchingPlan:
    """Exact host solve of an `m x n` clipped reduced-cost matrix into a plan.

    The partial matching with diagonal is solved in its reduced `m x n` form rather
    than the classical `(m + n)`-square augmentation. Writing `p` for a bar's
    diagonal cost, the total cost of any partial matching is
    `sum(p) + sum over matched pairs of (c_ij - p_i - p_j)`, so a rectangular
    assignment on the reduced costs `min(c_ij - p_i - p_j, 0)` finds the same
    optimum: a pair assigned at reduced cost zero is one both sides leave on the
    diagonal. Only strictly negative reduced costs count as matched, so ties resolve
    as "both to the diagonal", which has the same value either way.
    """
    m, n = reduced.shape
    if m == 0 or n == 0:
        return MatchingPlan(
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.arange(m, dtype=np.int64),
            np.arange(n, dtype=np.int64),
        )
    rows, columns = linear_sum_assignment(reduced)
    matched = reduced[rows, columns] < 0.0
    pair_gen = rows[matched]
    pair_ref = columns[matched]
    # The diagonal sets are the complements of the matched ones: a mask each, not a
    # sorted set difference, since this runs once per unit and degree.
    on_diagonal_gen = np.ones(m, dtype=bool)
    on_diagonal_gen[pair_gen] = False
    on_diagonal_ref = np.ones(n, dtype=bool)
    on_diagonal_ref[pair_ref] = False
    return MatchingPlan(
        pair_gen.astype(np.int64, copy=False),
        pair_ref.astype(np.int64, copy=False),
        np.flatnonzero(on_diagonal_gen),
        np.flatnonzero(on_diagonal_ref),
    )


def spatial_factor(pos_x, pos_y, scale, *, lambda_spatial, spatial_mode):
    """The creator-separation term for positions broadcast against each other.

    `multiplicative` returns the factor `1 + lambda * s`, `additive` the summand
    `lambda * s`, where `s` is the L1 creator separation over `scale`.
    """
    separation = (pos_x - pos_y).abs().sum(dim=-1) / scale
    if spatial_mode == "multiplicative":
        return 1.0 + float(lambda_spatial) * separation
    return float(lambda_spatial) * separation


def _weighted(pair, weight, spatial_mode):
    if weight is None:
        return pair
    return pair * weight if spatial_mode == "multiplicative" else pair + weight


def _pair_costs(
    birth_x, death_x, pos_x, birth_y, death_y, pos_y, scale, *, order, lambda_spatial, spatial_mode
):
    """Powered, spatially weighted bar-to-bar costs `[k, M, N]`."""
    pair = (birth_x[:, :, None] - birth_y[:, None, :]).abs() + (
        death_x[:, :, None] - death_y[:, None, :]
    ).abs()
    weight = None
    if float(lambda_spatial) != 0.0:
        weight = spatial_factor(
            pos_x[:, :, None, :], pos_y[:, None, :, :], scale,
            lambda_spatial=lambda_spatial, spatial_mode=spatial_mode,
        )
    return _power(_weighted(pair, weight, spatial_mode), order)


#: Largest `K * M * N` a single chunk of cost matrices may occupy on the device. A
#: call normally fits one chunk; splitting by problem size to trim padding was
#: measured to cost more in launches and copies than the padding it saved.
DEVICE_ELEMENT_BUDGET = 2**26


def plan_units(
    gen_block: torch.Tensor,
    ref_block: torch.Tensor,
    padded_gen: PaddedGenerators,
    padded_ref: PaddedGenerators,
    coordinates: torch.Tensor,
    scale: float,
    cap: torch.Tensor,
    *,
    order: float,
    lambda_spatial: float,
    spatial_mode: str,
    element_budget: int = DEVICE_ELEMENT_BUDGET,
) -> list[MatchingPlan]:
    """Solve one degree's matchings for every unit of a call.

    Diagrams, creator positions and the reduced cost matrices are built as padded
    batched tensors on the blocks' device; each chunk then crosses to the host in one
    copy, and every unit is solved exactly by SciPy on the pool, which releases the
    GIL. Units are sorted by problem size and chunked under `element_budget`, so
    device memory stays bounded. `cap` is the `[K]` value at which essential bars die.
    """
    validate_spatial_mode(spatial_mode)
    device = gen_block.device
    sizes_gen = padded_gen.sizes
    sizes_ref = padded_ref.sizes
    problem = (sizes_gen.clamp_min(1) * sizes_ref.clamp_min(1)).cpu().numpy()
    m_host = sizes_gen.cpu().numpy()
    n_host = sizes_ref.cpu().numpy()
    order_of_units = np.argsort(problem, kind="stable")
    count = int(problem.size)
    plans: list[MatchingPlan | None] = [None] * count
    costs = {"order": order, "lambda_spatial": lambda_spatial, "spatial_mode": spatial_mode}
    start = 0
    while start < count:
        # Greedy chunk: the largest problem in the chunk sets its padded shape.
        stop = start
        while stop < count and (stop - start + 1) * problem[order_of_units[stop]] <= element_budget:
            stop += 1
        stop = max(stop, start + 1)
        chunk = order_of_units[start:stop]
        start = stop
        index = torch.as_tensor(chunk, device=device)
        width_gen = max(int(m_host[chunk].max()), 1)
        width_ref = max(int(n_host[chunk].max()), 1)
        gen = padded_gen.rows(index, width_gen)
        ref = padded_ref.rows(index, width_ref)
        chunk_cap = cap.index_select(0, index)[:, None]
        birth_g, death_g = live_diagrams(gen_block.index_select(0, index), gen, chunk_cap)
        birth_r, death_r = live_diagrams(ref_block.index_select(0, index), ref, chunk_cap)
        powered_pair = _pair_costs(
            birth_g, death_g, coordinates[gen.births],
            birth_r, death_r, coordinates[ref.births],
            scale, **costs,
        )
        persistence_g = _power((death_g - birth_g).abs(), order)
        persistence_r = _power((death_r - birth_r).abs(), order)
        reduced = (powered_pair - persistence_g[:, :, None] - persistence_r[:, None, :]).clamp_max(0.0)
        # One host copy per chunk; the solves overlap on the pool.
        reduced_host = reduced.cpu().numpy()

        def solve_local(local: int, reduced_host=reduced_host, chunk=chunk) -> MatchingPlan:
            m, n = int(m_host[chunk[local]]), int(n_host[chunk[local]])
            return _plan_from_reduced(reduced_host[local, :m, :n])

        for local, plan in enumerate(parallel_map(solve_local, range(len(chunk)))):
            plans[int(chunk[local])] = plan
    return [plan for plan in plans if plan is not None]


class SolvedBatch(NamedTuple):
    """Everything the live half needs for one block pair: plans, padding, caps, geometry."""

    plans: list[list[MatchingPlan]]  # per degree, per unit
    padded: list[tuple[PaddedGenerators, PaddedGenerators]]  # per degree: (generated, reference)
    cap: torch.Tensor  # [K]
    coordinates: torch.Tensor  # [N, 2] grid positions on the device
    scale: float


# ------------------------------------------------------------ batched live half


def pack_host_rows(
    rows: Sequence[np.ndarray], device: torch.device, dtype: torch.dtype = torch.long
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad ragged host rows into one `[K, width]` device tensor plus its validity mask.

    The rows are joined on the host and cross to the device in a single copy,
    however many there are. Padding repeats each row's last entry, or borrows a
    neighbour's for an empty row, so a padded slot is always a usable gather index;
    the mask is what removes its contribution.
    """
    lengths = np.fromiter((int(row.shape[0]) for row in rows), dtype=np.int64, count=len(rows))
    width = int(lengths.max()) if lengths.size else 0
    count = len(rows)
    if width == 0:
        return (
            torch.zeros((count, 0), dtype=dtype, device=device),
            torch.zeros((count, 0), dtype=torch.bool, device=device),
        )
    flat = torch.as_tensor(
        np.concatenate([np.asarray(row).reshape(-1) for row in rows]), device=device, dtype=dtype
    )
    lengths_t = torch.as_tensor(lengths, device=device)
    offsets = torch.cumsum(lengths_t, 0) - lengths_t
    column = torch.arange(width, device=device)[None]
    index = offsets[:, None] + torch.minimum(column, (lengths_t - 1).clamp_min(0)[:, None])
    return flat[index.clamp_(max=flat.numel() - 1)], column < lengths_t[:, None]


def apply_units_batched(
    live_gen: torch.Tensor,
    live_ref: torch.Tensor,
    solved: SolvedBatch,
    *,
    order: float,
    lambda_spatial: float,
    spatial_mode: str,
) -> torch.Tensor:
    """The differentiable half for a whole call: `[K, N]` live blocks to `[K, degrees]`.

    The three ragged axes -- bars per diagram, matched pairs, and bars sent to the
    diagonal -- are padded and masked rather than looped, so the differentiable half
    of a call is a fixed handful of kernels however many units it holds, and so is its
    backward. A unit with nothing to match contributes an exact zero, and
    :func:`_power` gives that zero a zero gradient at `order > 1`.
    """
    validate_spatial_mode(spatial_mode)
    device = live_gen.device
    count = live_gen.shape[0]
    cap = solved.cap[:, None]
    columns: list[torch.Tensor] = []
    for plans, (padded_gen, padded_ref) in zip(solved.plans, solved.padded):
        birth_g, death_g = live_diagrams(live_gen, padded_gen, cap)
        birth_r, death_r = live_diagrams(live_ref, padded_ref, cap)
        pair_gen, pair_mask = pack_host_rows([plan.pair_gen for plan in plans], device)
        pair_ref, _ = pack_host_rows([plan.pair_ref for plan in plans], device)
        gen_diagonal, gen_mask = pack_host_rows([plan.gen_diagonal for plan in plans], device)
        ref_diagonal, ref_mask = pack_host_rows([plan.ref_diagonal for plan in plans], device)
        total = live_gen.new_zeros((count,))
        if pair_gen.shape[1]:
            cost = (birth_g.gather(1, pair_gen) - birth_r.gather(1, pair_ref)).abs() + (
                death_g.gather(1, pair_gen) - death_r.gather(1, pair_ref)
            ).abs()
            if float(lambda_spatial) != 0.0:
                # The same factor the assignment was solved with, from the same tensors.
                weight = spatial_factor(
                    solved.coordinates[padded_gen.births.gather(1, pair_gen)],
                    solved.coordinates[padded_ref.births.gather(1, pair_ref)],
                    solved.scale,
                    lambda_spatial=lambda_spatial,
                    spatial_mode=spatial_mode,
                )
                cost = _weighted(cost, weight, spatial_mode)
            total = total + (_power(cost, order) * pair_mask).sum(dim=1)
        if gen_diagonal.shape[1]:
            persistence = (death_g - birth_g).abs().gather(1, gen_diagonal)
            total = total + (_power(persistence, order) * gen_mask).sum(dim=1)
        if ref_diagonal.shape[1]:
            persistence = (death_r - birth_r).abs().gather(1, ref_diagonal)
            total = total + (_power(persistence, order) * ref_mask).sum(dim=1)
        columns.append(_power(total, 1.0 / order))
    return torch.stack(columns, dim=1)


# ------------------------------------------------------------- normalization


def reference_minmax(reference: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row `(lo, span)` over the last axis of a reference for min-max normalization.

    Both fields are mapped by the same affine map derived from the reference alone.
    These are detached constants: they set the value scale of the distance and the
    spatial term relative to it, but contribute no gradient of their own.
    """
    stats = reference.detach()
    lo = stats.amin(dim=-1, keepdim=True)
    return lo, (stats.amax(dim=-1, keepdim=True) - lo).clamp_min(eps)


def slice_lines(
    lo_left: np.ndarray,
    lo_right: np.ndarray,
    count: int,
    angle_margin: float,
    rng: np.random.Generator,
    sampling: str = "stratified",
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly drawn monotone slice lines through `K` (left, right) value boxes.

    Every line starts at the lower-left corner of its combined value box, so each
    vertex enters at a non-negative line parameter, and its angle lies in
    `(margin, 1 - margin) * pi/2`, so both direction components stay positive. Fresh
    draws on every call make the mutual term a stochastic estimate of the sliced
    matching distance over the whole monotone family. `stratified` draws one angle
    uniformly inside each of `count` equal bins, which keeps the estimate unbiased
    while removing most of the variance of independent draws; `uniform` draws all
    `count` angles independently, as the historical modes did. Both consume `K *
    count` numbers from the stream in row-major order, so the seeded stream advances
    identically however the call is batched. Returns `(basepoints, directions)`, each
    `(K, count, 2)`.
    """
    if sampling not in LINE_SAMPLINGS:
        raise ValueError(f"unknown line sampling {sampling!r}; expected one of {LINE_SAMPLINGS}")
    lo_left = np.asarray(lo_left, dtype=np.float64).reshape(-1)
    lo_right = np.asarray(lo_right, dtype=np.float64).reshape(-1)
    if lo_left.shape != lo_right.shape:
        raise ValueError("slice_lines expects one corner per unit on each side")
    n = max(int(count), 1)
    k = lo_left.shape[0]
    margin = float(angle_margin)
    if sampling == "stratified":
        offsets = rng.uniform(0.0, 1.0, size=(k, n))
        fraction = margin + (1.0 - 2.0 * margin) * (np.arange(n)[None, :] + offsets) / n
    else:
        fraction = np.sort(rng.uniform(margin, 1.0 - margin, size=(k, n)), axis=1)
    angles = fraction * (np.pi / 2.0)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=2).astype(np.float64)
    corners = np.stack([lo_left, lo_right], axis=1)[:, None, :]
    basepoints = np.broadcast_to(corners, (k, n, 2))
    return np.ascontiguousarray(basepoints), np.ascontiguousarray(directions)
