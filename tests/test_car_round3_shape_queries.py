from scripts.analysis.diagnose_car_round3_shape_queries import query_plan
from phycoflow_reconstruction.coherence.query_sampling import SharedQuerySampler


def test_query_plans_preserve_production_law_and_coupled_batch_anchor():
    fields = ('temperature', 'co', 'oxygen', 'velocity', 'pressure')
    shared, individual = query_plan(fields, 7, 70, 64)
    assert len(shared) == len(individual) == 70
    assert len(set(shared[:64])) == 1
    assert shared[0] == individual[0]
    assert shared[64] == individual[64]
    for i, query in enumerate(individual):
        batch_start = i // 64 * 64
        seed = 7 * 1000003 + batch_start + (i - batch_start) * 7919
        assert query == SharedQuerySampler(fields, seed=seed, epsilon=1.0).draw()
        assert sum(d.count for d in query.pair_queries) == 3
        assert all(d.weight == d.count / 3 for d in query.pair_queries)
        assert len({d.index for d in query.pair_queries}) == len(query.pair_queries)
