# tests/test_ablation.py
"""Tests for the default-ablation engine (Plan 7).

Toy separable problem: efficacy ~ X[:,0], toxicity ~ X[:,1], independent. A small
GP recovers each axis. Exercises the 5 acquisitions, the AL loop, the two-stage
nomination (safety filter then diversity operator), and evaluate()."""
from __future__ import annotations
import numpy as np
import pytest

from geneal.runner.ablation import (
    run_acquisition, nominate, evaluate, build_string_S, build_embedding_S,
    ACQUISITIONS,
)
from geneal.models.surrogate import GPRSurrogate


def _toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    eff = X[:, 0].copy()                 # efficacy along dim0 (higher better)
    tox = (X[:, 1] - X[:, 1].min())      # toxicity along dim1, >= 0
    tox = tox / tox.max()                # 0..1
    # membership: two pathways split by dim0 sign + a few unannotated
    membership = {}
    for i in range(n):
        if i % 7 == 0:
            membership[i] = set()        # unannotated
        else:
            membership[i] = {1 if X[i, 0] >= 0 else 2}
    return X, eff, tox, membership


def _factory():
    return GPRSurrogate(n_iters=20)


@pytest.mark.parametrize("kind", ACQUISITIONS)
def test_acquisition_grows_revealed_distinct(kind):
    X, eff, tox, _ = _toy()
    rev = run_acquisition(kind, X, eff, tox, n_init=8, n_rounds=3, batch=4,
                          seed=0, surr_factory=_factory)
    assert len(rev) == len(set(rev))                 # distinct
    assert len(rev) == 8 + 3 * 4                      # init + rounds*batch
    assert all(0 <= i < len(eff) for i in rev)


def test_run_acquisition_deterministic():
    X, eff, tox, _ = _toy()
    a = run_acquisition("greedy", X, eff, tox, 8, 2, 4, seed=1, surr_factory=_factory)
    b = run_acquisition("greedy", X, eff, tox, 8, 2, 4, seed=1, surr_factory=_factory)
    assert a == b
    # common random numbers: all kinds share the same initial set for a seed
    g = run_acquisition("greedy", X, eff, tox, 8, 0, 4, seed=3, surr_factory=_factory)
    r = run_acquisition("random", X, eff, tox, 8, 0, 4, seed=3, surr_factory=_factory)
    assert set(g) == set(r)                           # n_rounds=0 -> just init


def test_nominate_known_respects_quantile_ceiling():
    X, eff, tox, membership = _toy()
    rev = run_acquisition("greedy", X, eff, tox, 8, 3, 4, seed=0, surr_factory=_factory)
    tau = 0.6
    sel = nominate(rev, X, eff, tox, membership, S=None, K=6, safety="known",
                   diversity="none", tau=tau, surr_factory=_factory)
    assert len(sel) == 6
    # 'known' filters on the known-toxicity tau-QUANTILE -> picks below that ceiling
    thr = float(np.quantile(tox, tau))
    assert all(tox[i] <= thr + 1e-9 for i in sel)


def test_nominate_pred_safety_runs():
    X, eff, tox, membership = _toy()
    rev = run_acquisition("greedy", X, eff, tox, 8, 3, 4, seed=0, surr_factory=_factory)
    sel = nominate(rev, X, eff, tox, membership, S=None, K=6, safety="pred",
                   diversity="none", tau=0.5, surr_factory=_factory)
    assert len(sel) == 6 and len(set(sel)) == 6


def test_diversity_operator_changes_pick():
    X, eff, tox, membership = _toy()
    rev = run_acquisition("greedy", X, eff, tox, 8, 3, 4, seed=0, surr_factory=_factory)
    base = nominate(rev, X, eff, tox, membership, S=None, K=6, safety="none",
                    diversity="none", tau=1.0, surr_factory=_factory)
    capped = nominate(rev, X, eff, tox, membership, S=None, K=6, safety="none",
                      diversity="cap", tau=1.0, surr_factory=_factory, cap=2)
    # capping at 2/pathway must de-concentrate vs unconstrained top-K
    from geneal.metrics.portfolio import pathway_concentration
    assert pathway_concentration(capped, membership) <= pathway_concentration(base, membership)


def test_evaluate_returns_finite_metrics():
    X, eff, tox, membership = _toy()
    rev = run_acquisition("greedy", X, eff, tox, 8, 3, 4, seed=0, surr_factory=_factory)
    sel = nominate(rev, X, eff, tox, membership, S=None, K=6, safety="known",
                   diversity="cap", tau=0.7, surr_factory=_factory, cap=2)
    m = evaluate(sel, eff, tox, membership, X)
    for key in ("mean_efficacy", "max_efficacy", "mean_toxicity", "concentration",
                "robustness", "n_pathways", "alpha_ndcg"):
        assert key in m and np.isfinite(m[key])


def test_build_embedding_S_dense_unit_diag_in_range():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4))
    S = build_embedding_S(X)
    assert S.shape == (10, 10)
    assert np.allclose(np.diag(S), 1.0)
    assert np.allclose(S, S.T)
    assert S.min() >= 0.0 and S.max() <= 1.0 + 1e-9
    # dense: identical rows -> S=1, opposite rows -> S=0
    X2 = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    S2 = build_embedding_S(X2)
    assert np.isclose(S2[0, 1], 1.0) and np.isclose(S2[0, 2], 0.0)


def test_kdpp_with_embedding_S_changes_pick():
    X, eff, tox, membership = _toy()
    rev = run_acquisition("greedy", X, eff, tox, 8, 3, 4, seed=0, surr_factory=_factory)
    S = build_embedding_S(X)
    base = nominate(rev, X, eff, tox, membership, S=None, K=6, safety="none",
                    diversity="none", tau=1.0, surr_factory=_factory)
    kdpp = nominate(rev, X, eff, tox, membership, S=S, K=6, safety="none",
                    diversity="kdpp", tau=1.0, surr_factory=_factory)
    assert len(kdpp) == 6
    assert set(kdpp) != set(base)   # diversity reshuffles away from pure top-K


def test_joint_gp_nominate_and_acquire():
    from geneal.models.multitask import MultiTaskGPR
    jf = lambda: MultiTaskGPR(n_iters=25)
    X, eff, tox, membership = _toy()
    # joint EHVI acquisition runs and grows the revealed set
    rev = run_acquisition("ehvi", X, eff, tox, 8, 2, 4, seed=0, surr_factory=_factory,
                          ehvi_samples=8, shortlist=30, joint_factory=jf)
    assert len(rev) == len(set(rev)) and len(rev) == 8 + 2 * 4
    # joint pred-safety nomination runs and returns K
    sel = nominate(rev, X, eff, tox, membership, S=None, K=6, safety="pred",
                   diversity="none", tau=0.5, surr_factory=_factory, joint_factory=jf)
    assert len(sel) == 6 and len(set(sel)) == 6


def test_build_string_S_shape_and_diag():
    # gene_names with embedded entrez ids; S is symmetric with unit diagonal.
    names = ["AAA (1)", "BBB (2)", "CCC (999999999)"]  # last absent from STRING
    S = build_string_S(names)
    assert S.shape == (3, 3)
    assert np.allclose(np.diag(S), 1.0)
    assert np.allclose(S, S.T)
