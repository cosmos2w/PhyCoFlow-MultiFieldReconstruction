"""Superlevel H0 pairing as a batched, device-resident tensor pipeline.

:func:`superlevel_pairs_on_device` reproduces :func:`..topology_betti.betti_curves.superlevel_pairs_from_values`
bar for bar, without leaving the device and without a Python loop over vertices. It
exists because the pairing is the whole cost of the topology family: everything else
the family does is already vectorized, and the host reduction is what pins the term to
a thread pool.

The filtration handed to both backends is each vertex's *rank* in `(value descending,
index ascending)` order, so the insertion sequence is a strict total order with no ties
left to resolve. That is what makes an exact parallel formulation possible.

Working in rank space, an edge of the 4-connected grid enters the filtration when its
later endpoint does, so ordering edges by `key = later * N + earlier` is a strict total
order refining the insertion sequence. Components are represented by their *minimum
rank*, which is exactly their birth vertex, so the elder rule reduces to comparing two
integers and union-find state collapses to a single `parent` array.

Merging uses Borůvka rounds. Pointing every component at its cheapest outgoing edge
builds a forest whose every tree holds exactly one mutually-best pair, and keys never
increase along a pointer, so a tree's edges are already in Kruskal order reading outward
from that pair. Contracting a whole tree is nonetheless *not* order-correct in general:
the group can absorb a component at a key above an edge leaving the tree, which Kruskal
would have processed first, and that swaps which birth pairs with which death. The guard
is `k_out`, the cheapest edge leaving the tree -- absorbing only the nodes whose own best
key falls below it is exactly the prefix Kruskal would reach before any outside merge
interferes. The mutually-best pair always clears that bar, so every tree retires at least
one merge per round and the loop terminates.

Within an absorbed prefix each node is still a singleton when its turn comes, so the
group's birth is a running minimum over node ids and the bar retired at each step is the
younger of that running minimum and the node itself. That makes the whole round a
segmented exclusive prefix-minimum, which one `cummin` delivers: sorting segments by
descending tree id and encoding `tree * (N + 1) + id` keeps every earlier segment's value
above everything in the current one, so a global scan cannot leak across a boundary.

Two details keep the round cheap. The ordering a round needs -- by tree, then by key
within a tree -- comes from one sort on a composite key rather than two stable sorts,
which is why the grid size is bounded by that key fitting in int64; a grid past that
bound declines the batch rather than wrapping, and the host walk pairs it. And the pointer
doubling that labels the forest stops when it settles, measured at about five passes
against a worst case of `log2(N)`; the check runs every third pass, because testing
every pass costs more than the passes it saves.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import NamedTuple

import torch

#: Cap on `K * E` worked at once, so device memory stays bounded for large batches.
DEVICE_ELEMENT_BUDGET = 1 << 26


class MergeTreeUnavailable(RuntimeError):
    """The tensor pipeline declines this batch; the caller runs the host walk instead.

    Every subclass is recoverable by construction: the host walk pairs the same batch
    into the same bars, so which backend ends up running is an execution detail and
    nothing scientific rides on it. This is deliberately *not* raised for a malformed
    input -- a wrong shape or point count stays a `ValueError`, because falling back
    would hide it.
    """


class MergeTreeNotConverged(MergeTreeUnavailable):
    """Raised when the round budget runs out with components still to merge."""


class MergeTreeTooLarge(MergeTreeUnavailable):
    """Raised when a grid overflows the packed `(tree, key)` sort key."""


@lru_cache(maxsize=32)
def _grid_edges(height: int, width: int, periodic: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Undirected 4-neighbour edges of an `[H,W]` grid as two flat vertex-id arrays.

    A wrap edge is emitted only when its axis is longer than two: at extent one it
    would be a self-loop and at extent two it would duplicate the interior edge.
    Both cases leave the *graph* identical to what :func:`.._neighbors` describes, so
    the pairing is unaffected, and dropping them keeps every edge key distinct.
    """
    ids = torch.arange(height * width, dtype=torch.long).reshape(height, width)
    left: list[torch.Tensor] = []
    right: list[torch.Tensor] = []
    if width > 1:
        left.append(ids[:, :-1].reshape(-1))
        right.append(ids[:, 1:].reshape(-1))
    if height > 1:
        left.append(ids[:-1, :].reshape(-1))
        right.append(ids[1:, :].reshape(-1))
    if periodic and width > 2:
        left.append(ids[:, -1])
        right.append(ids[:, 0])
    if periodic and height > 2:
        left.append(ids[-1, :])
        right.append(ids[0, :])
    if not left:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty
    return torch.cat(left).contiguous(), torch.cat(right).contiguous()


def default_round_budget(point_count: int) -> int:
    """Rounds allowed before :class:`MergeTreeNotConverged`.

    Every tree retires at least its mutually-best pair per round, so the loop terminates
    within `N - 1` rounds, and measured round counts sit far below this budget. A batch
    that somehow exhausts it falls back to the host pairing rather than returning
    something wrong.
    """
    return max(64, 8 * max(1, point_count).bit_length())


def sort_key_fits(point_count: int) -> bool:
    """Whether a grid's packed `(tree, key)` sort key stays inside int64.

    A round orders nodes by one composite key that scales an edge key -- itself below
    `point_count ** 2` -- by a tree id running up to `point_count`. Past this bound the
    product would wrap silently, so the pipeline declines the batch instead.
    """
    return point_count * point_count * (point_count + 1) < torch.iinfo(torch.long).max


def _merge_chunk(
    edge_left: torch.Tensor,
    edge_right: torch.Tensor,
    key: torch.Tensor,
    death_label: torch.Tensor,
    node_count: int,
    key_span: int,
    max_rounds: int,
    check_from: int = 0,
) -> tuple[torch.Tensor, int]:
    """Elder-rule Kruskal on `K` graphs at once; returns `(death_of, rounds)`.

    Nodes are numbered in insertion order, so a component's oldest member is its
    smallest id and that id is its birth. `edge_left` and `edge_right` are `[K, E]`
    node ids, `key` is a `[K, E]` strict total order on the edges refining the
    filtration (every key below `key_span - 1`), and `death_label[k, e]` is what to
    record against the birth node an edge `e` retires: the later endpoint's rank for
    a lower-star grid, the edge's own index for the planar dual. `death_of[k, i]` is
    that label for node `i`, or `-1` if the node's component survives; the array is
    one column wider than the graph, a trash slot the masked scatters write into.

    This is the round structure documented at the top of the module; nothing here
    depends on where the graph came from. Each termination check is a device
    synchronization, and a round on a graph with no live edge changes nothing, so a
    caller that knows its problems take several rounds passes `check_from` to run the
    first rounds unchecked; `rounds` then counts executed rounds, no-ops included.
    """
    count = edge_left.shape[0]
    device = edge_left.device
    trash = node_count
    width_plus = node_count + 1
    rows = torch.arange(width_plus, device=device).expand(count, width_plus)
    parent = rows.clone()
    death_of = torch.full((count, width_plus), -1, dtype=torch.long, device=device)
    pad = torch.full_like(edge_left, trash)
    pad_column = torch.full((count, width_plus), trash, dtype=torch.long, device=device)
    infinity = torch.iinfo(torch.long).max

    rounds = 0
    jumps = width_plus.bit_length()
    huge = torch.full((), infinity, dtype=torch.long, device=device)
    for round_index in range(max_rounds):
        root_left = parent.gather(1, edge_left)
        root_right = parent.gather(1, edge_right)
        live = root_left != root_right
        if round_index >= check_from and not bool(live.any()):
            break
        rounds += 1
        live_key = torch.where(live, key, infinity)

        # ---- each live component's cheapest outgoing edge -----------------------
        best = torch.full((count, width_plus), infinity, dtype=torch.long, device=device)
        best.scatter_reduce_(1, root_left, live_key, "amin")
        best.scatter_reduce_(1, root_right, live_key, "amin")
        # Keys are distinct, so exactly one edge realizes a root's best and these
        # scatters never collide on a live slot.
        partner = torch.full((count, width_plus), -1, dtype=torch.long, device=device)
        death = torch.full((count, width_plus), -1, dtype=torch.long, device=device)
        chosen_left = torch.where(live & (live_key == best.gather(1, root_left)), root_left, pad)
        partner.scatter_(1, chosen_left, root_right)
        death.scatter_(1, chosen_left, death_label)
        chosen_right = torch.where(live & (live_key == best.gather(1, root_right)), root_right, pad)
        partner.scatter_(1, chosen_right, root_left)
        death.scatter_(1, chosen_right, death_label)

        # ---- label the best-edge forest -----------------------------------------
        # Each tree holds one mutually-best pair; sending both of its members to the
        # smaller id turns that 2-cycle into the fixed point the rest of the tree
        # converges on under pointer doubling.
        live_root = partner >= 0
        mutual = live_root & (partner.gather(1, partner.clamp_min(0)) == rows)
        follow = torch.where(live_root, partner, rows)
        follow = torch.where(mutual, torch.minimum(rows, partner), follow)
        # Each pass doubles the reach, so the bound below is the worst case; measured
        # forests settle in about five. Every test is a device synchronization and a
        # pass on a settled forest is a no-op, so the first test comes after six
        # passes and then every third.
        for index in range(jumps):
            jumped = follow.gather(1, follow)
            settled = index >= 5 and index % 3 == 2 and torch.equal(jumped, follow)
            follow = jumped
            if settled:
                break
        tree = follow

        # ---- how far Kruskal may run before an outside edge interferes ----------
        tree_left = tree.gather(1, root_left)
        tree_right = tree.gather(1, root_right)
        cross_key = torch.where(live & (tree_left != tree_right), key, infinity)
        outside = torch.full((count, width_plus), infinity, dtype=torch.long, device=device)
        outside.scatter_reduce_(1, tree_left, cross_key, "amin")
        outside.scatter_reduce_(1, tree_right, cross_key, "amin")
        absorbed = live_root & (best < outside.gather(1, tree))

        # ---- order each tree's prefix by key and scan its births -----------------
        # One composite key does the work of two stable sorts: keys are below
        # `key_span`, so scaling the descending tree id by it orders by tree first and
        # by key within a tree, and a node that is not being absorbed sorts last in its
        # tree because `key_span - 1` exceeds every real key.
        order_in_tree = torch.argsort(
            (width_plus - 1 - tree) * key_span + torch.where(absorbed, best, key_span - 1),
            dim=1,
            stable=True,
        )
        tree_sorted = tree.gather(1, order_in_tree)
        encoded = tree_sorted * width_plus + order_in_tree
        running = torch.cummin(encoded, dim=1).values
        leading = torch.full((count, 1), infinity, dtype=torch.long, device=device)
        earlier_min = torch.cat((leading, running[:, :-1]), dim=1) % width_plus
        follows = torch.cat(
            (
                torch.zeros((count, 1), dtype=torch.bool, device=device),
                tree_sorted[:, 1:] == tree_sorted[:, :-1],
            ),
            dim=1,
        )
        # The first node of a tree starts the group; every later one retires the
        # younger of the group's birth so far and its own.
        emit = absorbed.gather(1, order_in_tree) & follows
        retired = torch.where(emit, torch.maximum(earlier_min, order_in_tree), pad_column)
        death_of.scatter_(1, retired, death.gather(1, order_in_tree))

        # ---- contract every absorbed prefix onto its oldest member ---------------
        group = torch.full((count, width_plus), infinity, dtype=torch.long, device=device)
        # The index needs no mask: a node that is not being absorbed contributes
        # `huge`, and an "amin" against a slot already holding `huge` changes nothing.
        group.scatter_reduce_(1, tree, torch.where(absorbed, rows, huge), "amin")
        parent = torch.where(absorbed, group.gather(1, tree), parent)
        # A contraction deepens a path by one level, so one jump recompresses it and
        # the second is margin.
        parent = parent.gather(1, parent)
        parent = parent.gather(1, parent)
    else:
        if bool((parent.gather(1, edge_left) != parent.gather(1, edge_right)).any()):
            raise MergeTreeNotConverged(f"pairing did not converge in {max_rounds} rounds")
    return death_of, rounds


def _pair_chunk(
    values: torch.Tensor, height: int, width: int, periodic: bool, max_rounds: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pair one chunk of fields; returns `(order, death_of, essential, rounds)`."""
    count, point_count = values.shape
    device = values.device
    # `stable=True` on the negated values is exactly `lexsort((arange, -values))`:
    # descending by value, ties resolved by ascending vertex id.
    order = torch.argsort(-values, dim=1, stable=True)
    positions = torch.arange(point_count, device=device).expand(count, point_count)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, positions)

    left, right = _grid_edges(height, width, periodic)
    left = left.to(device)
    right = right.to(device)
    edge_left = rank.index_select(1, left)
    edge_right = rank.index_select(1, right)
    later = torch.maximum(edge_left, edge_right)
    key = torch.minimum(edge_left, edge_right) + later * point_count
    # Edge keys are below `point_count ** 2`; the composite sort key multiplies that
    # by a tree id, so the product must stay inside int64.
    if not sort_key_fits(point_count):
        raise MergeTreeTooLarge(
            f"a {point_count}-point grid overflows the packed merge-tree sort key"
        )
    death_of, rounds = _merge_chunk(
        edge_left, edge_right, key, later, point_count, point_count * point_count, max_rounds
    )
    return order, death_of[:, :point_count], order[:, 0], rounds


def superlevel_pairs_on_device(
    values: torch.Tensor,
    height: int,
    width: int,
    periodic: bool,
    *,
    max_rounds: int | None = None,
    element_budget: int = DEVICE_ELEMENT_BUDGET,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor, int]], int]:
    """Pair a `[K,N]` block of detached fields; returns `(pairings, rounds)`.

    Each pairing is `(finite birth vertex ids, finite death vertex ids, essential id)`,
    matching :data:`..topology_betti.betti_curves.SuperlevelPairs` with device tensors in place of
    lists. Bars come out ordered by birth rank rather than in GUDHI's emission order;
    every consumer sums over bars, so the order carries no meaning.
    """
    if values.ndim != 2:
        raise ValueError("superlevel_pairs_on_device expects [K,N]")
    count, point_count = values.shape
    if point_count != height * width:
        raise ValueError("superlevel_pairs_on_device expects height*width values")
    if max_rounds is None:
        max_rounds = default_round_budget(point_count)
    edge_count = max(1, _grid_edges(height, width, periodic)[0].numel())
    chunk = max(1, min(count, element_budget // edge_count))

    pairings: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    total_rounds = 0
    for start in range(0, count, chunk):
        block = values[start : start + chunk]
        order, death_of, essential, rounds = _pair_chunk(
            block, height, width, periodic, max_rounds
        )
        total_rounds = max(total_rounds, rounds)
        positions = torch.arange(point_count, device=values.device)
        # A vertex's own class is born and dies at that vertex; the host walk never
        # records it, so the zero-length bar is dropped here too.
        finite = (death_of >= 0) & (death_of != positions)
        counts = finite.sum(dim=1).tolist()
        row, column = finite.nonzero(as_tuple=True)
        births = order[row, column]
        deaths = order[row, death_of[row, column]]
        essentials = essential.tolist()
        birth_split = torch.split(births, counts)
        death_split = torch.split(deaths, counts)
        for index in range(block.shape[0]):
            pairings.append((birth_split[index], death_split[index], int(essentials[index])))
    return pairings, total_rounds


# ------------------------------------------------------------------ planar dual


@lru_cache(maxsize=32)
def _dual_graph(
    height: int, width: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Planar dual of the nonperiodic `[H,W]` grid, as `(corners, endpoints, side_a, side_b)`.

    Dual nodes are the `(H-1)(W-1)` unit squares, numbered row-major, plus one outside
    node numbered last. Every primal edge is a dual edge joining the two squares it
    separates, or a square and the outside along the border. `corners[s]` holds a
    square's four vertex ids, `endpoints[e]` a primal edge's two, and `side_a[e]`,
    `side_b[e]` the dual nodes the edge joins. Edges come in the same order as
    :func:`_grid_edges` emits them for a nonperiodic grid.
    """
    squares = max(height - 1, 0) * max(width - 1, 0)
    outside = squares
    ids = torch.arange(height * width, dtype=torch.long).reshape(height, width)
    square = torch.arange(squares, dtype=torch.long).reshape(max(height - 1, 0), max(width - 1, 0))
    corners = torch.stack(
        (ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]), dim=-1
    ).reshape(-1, 4)
    endpoints: list[torch.Tensor] = []
    side_a: list[torch.Tensor] = []
    side_b: list[torch.Tensor] = []
    if width > 1:
        # Horizontal edge (i, j)-(i, j+1) separates the square above from the one below.
        endpoints.append(torch.stack((ids[:, :-1].reshape(-1), ids[:, 1:].reshape(-1)), dim=1))
        above = torch.full((height, width - 1), outside, dtype=torch.long)
        below = torch.full((height, width - 1), outside, dtype=torch.long)
        if height > 1:
            above[1:] = square
            below[:-1] = square
        side_a.append(above.reshape(-1))
        side_b.append(below.reshape(-1))
    if height > 1:
        # Vertical edge (i, j)-(i+1, j) separates the square on its left from the right.
        endpoints.append(torch.stack((ids[:-1, :].reshape(-1), ids[1:, :].reshape(-1)), dim=1))
        left = torch.full((height - 1, width), outside, dtype=torch.long)
        right = torch.full((height - 1, width), outside, dtype=torch.long)
        if width > 1:
            left[:, 1:] = square
            right[:, :-1] = square
        side_a.append(left.reshape(-1))
        side_b.append(right.reshape(-1))
    if not endpoints:
        empty = torch.zeros(0, dtype=torch.long)
        return corners, empty.reshape(0, 2), empty, empty
    return (
        corners.contiguous(),
        torch.cat(endpoints).contiguous(),
        torch.cat(side_a).contiguous(),
        torch.cat(side_b).contiguous(),
    )


def dual_sort_key_fits(point_count: int, edge_count: int, node_count: int) -> bool:
    """Whether the dual pairing's packed `(tree, key)` sort key stays inside int64."""
    return node_count * point_count * edge_count < torch.iinfo(torch.long).max


class _Problem(NamedTuple):
    """One degree's merge problem in the generic core's terms, plus its decoding."""

    edge_left: torch.Tensor
    edge_right: torch.Tensor
    key: torch.Tensor
    death_label: torch.Tensor
    node_count: int
    key_span: int
    decode: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    essential: torch.Tensor


def _h0_problem(values: torch.Tensor, height: int, width: int) -> _Problem:
    """H0 sublevel as a merge problem: the superlevel grid pairing of the negated field.

    Ranks are `(value ascending, id ascending)`, node ids are ranks, an edge enters
    with its later endpoint and its death label is that endpoint's rank.
    """
    count, point_count = values.shape
    device = values.device
    order = torch.argsort(values, dim=1, stable=True)
    positions = torch.arange(point_count, device=device).expand(count, point_count)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, positions)
    left, right = (t.to(device) for t in _grid_edges(height, width, False))
    edge_left = rank.index_select(1, left)
    edge_right = rank.index_select(1, right)
    later = torch.maximum(edge_left, edge_right)
    key = torch.minimum(edge_left, edge_right) + later * point_count
    if not sort_key_fits(point_count):
        raise MergeTreeTooLarge(
            f"a {point_count}-point grid overflows the packed merge-tree sort key"
        )

    def decode(death_of: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        death_of = death_of[:, :point_count]
        # A vertex's own class is born and dies at that vertex; the host walk never
        # records it, so the zero-length bar is dropped here too.
        finite = (death_of >= 0) & (death_of != positions)
        return order, order.gather(1, death_of.clamp_min(0)), finite

    return _Problem(edge_left, edge_right, key, later, point_count, point_count * point_count, decode, order[:, 0])


def _h1_problem(values: torch.Tensor, height: int, width: int) -> _Problem | None:
    """H1 sublevel as a merge problem on the planar dual; `None` when there are no squares.

    In the plane, the holes of a sublevel set are the bounded components of its
    complement, and the complement lives on the dual grid: a square is outside the
    sublevel set until its highest corner enters, a primal edge's dual crossing is
    open until the edge's higher endpoint enters, and the unbounded face is always
    outside. Sweeping the level *down*, the complement grows: squares appear at their
    top corner's rank, crossings open at their top endpoint's rank, and components
    merge under the elder rule with the oldest being the one that appeared at the
    highest rank. A component born at square `s` and retired by crossing `e` is the
    primal H1 class born when `e` closed its loop and killed when `s` filled the hole,
    so its generators are `e`'s top endpoint and `s`'s top corner. The outside node
    appears first and never retires, which is why a nonperiodic grid has no essential
    H1 class.

    Ranks are `(value ascending, id ascending)`, a strict total order, so a square's
    and a crossing's top vertex are unique; squares and crossings that share a top
    vertex tie in rank, and any tie-break gives the same generators.
    """
    count, point_count = values.shape
    device = values.device
    corners, endpoints, side_a, side_b = (t.to(device) for t in _dual_graph(height, width))
    squares = corners.shape[0]
    node_count = squares + 1
    edge_count = endpoints.shape[0]
    if squares == 0 or edge_count == 0:
        return None
    if not dual_sort_key_fits(point_count, edge_count, node_count):
        raise MergeTreeTooLarge(
            f"a {point_count}-point grid overflows the packed dual merge-tree sort key"
        )
    order = torch.argsort(values, dim=1, stable=True)
    positions = torch.arange(point_count, device=device).expand(count, point_count)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, positions)

    corner_rank = rank[:, corners]  # [K, S, 4]
    square_level, top_corner = corner_rank.max(dim=2)
    top_vertex = corners[None].expand(count, squares, 4).gather(2, top_corner[..., None])[..., 0]
    endpoint_rank = rank[:, endpoints]  # [K, E, 2]
    edge_level, top_end = endpoint_rank.max(dim=2)
    top_endpoint = endpoints[None].expand(count, edge_count, 2).gather(2, top_end[..., None])[..., 0]

    # Dual nodes numbered in order of appearance: the outside first, then squares by
    # descending top rank, ties by square id.
    level = torch.cat((square_level, torch.full((count, 1), point_count, device=device)), dim=1)
    node_order = torch.argsort(-level, dim=1, stable=True)
    node_id = torch.empty_like(node_order)
    node_id.scatter_(1, node_order, torch.arange(node_count, device=device).expand(count, node_count))
    edge_left = node_id.gather(1, side_a[None].expand(count, edge_count))
    edge_right = node_id.gather(1, side_b[None].expand(count, edge_count))
    edge_index = torch.arange(edge_count, device=device).expand(count, edge_count)
    # Crossings open in descending top rank; the edge index makes the order strict.
    key = (point_count - 1 - edge_level) * edge_count + edge_index

    def decode(death_of: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        death_of = death_of[:, :node_count]
        finite = (death_of >= 0) & (node_order < squares)
        birth = top_endpoint.gather(1, death_of.clamp_min(0))
        death = top_vertex.gather(1, node_order.clamp_max(squares - 1))
        return birth, death, finite

    essential = torch.full((count,), -1, dtype=torch.long, device=device)
    return _Problem(edge_left, edge_right, key, edge_index, node_count, point_count * edge_count, decode, essential)


#: Rounds run before the first termination check on the persistence path: measured
#: problems take five or six, so the early checks never fire.
_UNCHECKED_ROUNDS = 3

#: Largest `K * E` per problem at which a chunk's degrees are stacked into one merge.
#: Below it the round loop is bound by kernel launches and synchronizations, which
#: stacking halves; above it the loop is bound by memory traffic, and stacking only
#: makes the shorter problem run the longer one's rounds on a larger working set.
FUSED_ELEMENT_LIMIT = 1 << 20


def sublevel_generators_on_device(
    values: torch.Tensor,
    height: int,
    width: int,
    dimensions: tuple[int, ...],
    *,
    max_rounds: int | None = None,
    element_budget: int = DEVICE_ELEMENT_BUDGET,
    min_persistence: float = 0.0,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], int]:
    """Lower-star generators of a `[K,N]` block on a nonperiodic grid, all degrees at once.

    Returns `({degree: (row, birth, death, essential)}, rounds)`: `row[m]` is the unit
    of bar `m`, `birth[m]` and `death[m]` the primal vertex ids at whose values it is
    born and dies, and `essential[k]` the vertex of unit `k`'s essential class (`-1`
    for H1, which has none on a nonperiodic grid). Bars of zero persistence *in
    value* are dropped, as GUDHI's `min_persistence=0` drops them, and so are bars
    shorter than `min_persistence`, so this is the device counterpart of a rank-fed
    cubical reduction with vertex generators followed by a persistence floor.

    H0 is the superlevel pairing of the negated block and H1 runs on the planar dual;
    both are problems for the same generic core. A small chunk's degrees are stacked
    along the batch axis and merged in one pass, which halves the launches and
    synchronizations that bound it; a large chunk, bound by memory traffic instead,
    merges each degree on its own (`FUSED_ELEMENT_LIMIT`). Rounds are counted per
    pass, including the unchecked leading rounds.
    """
    if values.ndim != 2:
        raise ValueError("sublevel_generators_on_device expects [K,N]")
    count, point_count = values.shape
    if point_count != height * width:
        raise ValueError("sublevel_generators_on_device expects height*width values")
    if not dimensions or not set(dimensions) <= {0, 1} or len(set(dimensions)) != len(dimensions):
        raise ValueError("sublevel_generators_on_device supports distinct degrees from {0,1}")
    if max_rounds is None:
        max_rounds = default_round_budget(point_count)
    device = values.device
    edge_count = max(1, _grid_edges(height, width, False)[0].numel())
    chunk = max(1, min(count, element_budget // (edge_count * len(dimensions))))
    parts: dict[int, list[list[torch.Tensor]]] = {d: [[], [], [], []] for d in dimensions}
    total_rounds = 0
    for start in range(0, count, chunk):
        block = values[start : start + chunk]
        problems: dict[int, _Problem] = {}
        for dimension in dimensions:
            problem = _h0_problem(block, height, width) if dimension == 0 else _h1_problem(block, height, width)
            if problem is None:
                parts[dimension][3].append(torch.full((block.shape[0],), -1, dtype=torch.long, device=device))
                continue
            problems[dimension] = problem
        fuse = block.shape[0] * edge_count <= FUSED_ELEMENT_LIMIT
        groups = [list(problems.items())] if fuse else [[item] for item in problems.items()]
        for group in groups:
            if not group:  # every requested degree was empty on this grid
                continue
            stacked = [problem for _, problem in group]
            node_count = max(problem.node_count for problem in stacked)
            key_span = max(problem.key_span for problem in stacked)
            if node_count * key_span >= torch.iinfo(torch.long).max:
                raise MergeTreeTooLarge("stacked merge problems overflow the packed sort key")
            death_of, rounds = _merge_chunk(
                torch.cat([problem.edge_left for problem in stacked]),
                torch.cat([problem.edge_right for problem in stacked]),
                torch.cat([problem.key for problem in stacked]),
                torch.cat([problem.death_label for problem in stacked]),
                node_count,
                key_span,
                max_rounds,
                check_from=_UNCHECKED_ROUNDS,
            )
            total_rounds = max(total_rounds, rounds)
            rows_per_problem = block.shape[0]
            for index, (dimension, problem) in enumerate(group):
                own = death_of[index * rows_per_problem : (index + 1) * rows_per_problem]
                birth_of, death_of_nodes, finite = problem.decode(own)
                row, column = finite.nonzero(as_tuple=True)
                birth = birth_of[row, column]
                death = death_of_nodes[row, column]
                persistence = (block[row, death] - block[row, birth]).abs()
                keep = (persistence > 0) & (persistence >= float(min_persistence))
                parts[dimension][0].append(row[keep] + start)
                parts[dimension][1].append(birth[keep])
                parts[dimension][2].append(death[keep])
                parts[dimension][3].append(problem.essential)
    out = {}
    for dimension in dimensions:
        rows, births, deaths, essentials = parts[dimension]
        empty = torch.zeros(0, dtype=torch.long, device=device)
        out[dimension] = (
            torch.cat(rows) if rows else empty,
            torch.cat(births) if births else empty,
            torch.cat(deaths) if deaths else empty,
            torch.cat(essentials),
        )
    return out, total_rounds


def sublevel_pairs_on_device(
    values: torch.Tensor,
    height: int,
    width: int,
    dimension: int,
    *,
    max_rounds: int | None = None,
    element_budget: int = DEVICE_ELEMENT_BUDGET,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """One degree of :func:`sublevel_generators_on_device`: `(row, birth, death, essential, rounds)`."""
    if dimension not in (0, 1):
        raise ValueError("sublevel_pairs_on_device supports H0 and H1 only")
    out, rounds = sublevel_generators_on_device(
        values, height, width, (dimension,), max_rounds=max_rounds, element_budget=element_budget
    )
    return (*out[dimension], rounds)
