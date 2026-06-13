# tests/test_panel.py
import numpy as np
from geneal.metrics.panel import MaxValue, CumulativeDiversity, AlphaNDCG, evaluate_panel


def test_maxvalue_is_best_revealed_over_global_max():
    target = np.array([1.0, 5.0, 2.0, 4.0])  # global max 5.0
    m = MaxValue()
    assert m.evaluate([0, 2], target, None) == 2.0 / 5.0   # best revealed 2.0
    assert m.evaluate([0, 1], target, None) == 1.0          # found the max
    assert m.name == "max_value"


def test_maxvalue_handles_nonpositive_globalmax():
    target = np.array([-3.0, -1.0, -2.0])  # global max -1.0
    # normalized by max(|global max|, eps); best revealed -3 -> negative ratio ok
    v = MaxValue().evaluate([0], target, None)
    assert np.isfinite(v)


def test_cumulative_diversity_mean_pairwise():
    emb = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])  # 0&2 identical, 1 far
    d = CumulativeDiversity()
    assert d.evaluate([0, 2], None, emb) == 0.0      # two identical -> 0
    assert np.isclose(d.evaluate([0, 1], None, emb), 5.0)  # dist (0,0)-(3,4)=5
    assert d.name == "diversity"


def test_alpha_ndcg_penalizes_same_cluster_redundancy():
    # 4 genes: 0,1 in a high-lethal cluster A; 2 in lethal cluster B; 3 inert.
    # embeddings make clusters obvious; lethality high for 0,1,2.
    emb = np.array([[0.0, 0.0], [0.05, 0.0], [10.0, 0.0], [20.0, 0.0]])
    target = np.array([5.0, 5.0, 5.0, 0.0])
    m = AlphaNDCG(k=2, alpha=0.5, n_clusters=3, seed=0)
    # picking one-from-A then B (diverse) should score >= picking both from A
    diverse = m.evaluate([0, 2], target, emb)
    redundant = m.evaluate([0, 1], target, emb)
    assert diverse >= redundant
    assert 0.0 <= diverse <= 1.0 and 0.0 <= redundant <= 1.0
    assert m.name == "alpha_ndcg@2"


def test_evaluate_panel_returns_named_dict():
    emb = np.random.default_rng(0).standard_normal((10, 3))
    target = np.arange(10.0)
    panel = [MaxValue(), CumulativeDiversity(), AlphaNDCG(k=5, n_clusters=3, seed=0)]
    out = evaluate_panel(panel, [0, 1, 2], target, emb)
    assert set(out) == {"max_value", "diversity", "alpha_ndcg@5"}
    assert all(np.isfinite(v) for v in out.values())
