# tests/test_multiobjective.py
import numpy as np
from geneal.models.multiobjective import pareto_front, hypervolume2d, mc_ehvi


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
