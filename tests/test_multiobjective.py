# tests/test_multiobjective.py
import numpy as np
from geneal.models.multiobjective import pareto_front, hypervolume2d, mc_ehvi


def test_hypervolume_mc_unit_and_corner():
    from geneal.models.multiobjective import hypervolume_mc
    rng = np.random.default_rng(0)
    assert np.isclose(hypervolume_mc(np.array([[1.0, 1.0, 1.0]]), rng, 20000), 1.0, atol=0.02)
    assert np.isclose(hypervolume_mc(np.array([[0.5, 0.5, 0.5]]), rng, 40000), 0.125, atol=0.02)


def test_pareto_front_nd():
    from geneal.models.multiobjective import pareto_front_nd
    P = np.array([[1, 0, 0], [0, 1, 0], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]])
    keep = set(pareto_front_nd(P))
    assert 3 not in keep                 # dominated by [0.5,0.5,0.5]
    assert {0, 1, 2} <= keep


def test_ehvi_nominate_nd_runs_and_excludes_dominated():
    from geneal.models.multiobjective import ehvi_nominate_nd
    rng = np.random.default_rng(0)
    mean = np.array([[0.9, 0.4, 0.5], [0.4, 0.9, 0.5], [0.1, 0.1, 0.1]])
    cov = np.broadcast_to(np.eye(3) * 0.001, (3, 3, 3)).copy()
    pick = ehvi_nominate_nd(mean, cov, K=2, rng=rng, n_post=16, n_hv=2000)
    assert len(pick) == len(set(pick)) == 2
    assert 2 not in pick                  # the low corner is not chosen


def test_greedy_hv_nominate_excludes_dominated():
    from geneal.models.multiobjective import greedy_hv_nominate
    rng = np.random.default_rng(0)
    # two non-dominated trade-off points + one strictly dominated point.
    Y = np.array([[0.9, 0.4], [0.4, 0.9], [0.3, 0.3]])
    pick = greedy_hv_nominate(Y, K=2, rng=rng, n_samples=40000)
    assert len(pick) == len(set(pick)) == 2
    assert set(pick) == {0, 1}        # picks the two non-dominated, drops dominated 2


def test_pareto_front_2d():
    # points (maximize both); (2,2) dominates (1,1); (3,0),(0,3) non-dominated
    P = np.array([[1, 1], [2, 2], [3, 0], [0, 3], [2, 1]])
    idx = pareto_front(P)
    assert set(idx) == {1, 2, 3}  # (2,2),(3,0),(0,3)


def test_hypervolume2d_monotone():
    ref = np.array([0.0, 0.0])
    hv1 = hypervolume2d(np.array([[2, 2]]), ref)
    assert np.isclose(hv1, 4.0)
    hv2 = hypervolume2d(np.array([[2, 2], [3, 1]]), ref)  # adds region
    assert hv2 > hv1


def test_mc_ehvi_prefers_improving_candidate():
    rng = np.random.default_rng(0)
    ref = np.array([0.0, 0.0])
    front = np.array([[2.0, 2.0]])
    # cand A: mean (3,3) clearly extends the front; cand B: mean (1,1) dominated
    mean = np.array([[3.0, 3.0], [1.0, 1.0]])
    std = np.array([[0.2, 0.2], [0.2, 0.2]])
    ehvi = mc_ehvi(mean, std, front, ref, rng, n_samples=200)
    assert ehvi[0] > ehvi[1]
    assert ehvi[1] >= 0.0
