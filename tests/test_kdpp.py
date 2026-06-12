# tests/test_kdpp.py
import numpy as np
from geneal.models.selection import KDPP
from geneal.models.acquisition import GreedyMean, UCB
from geneal.data.dataset import make_synthetic
from geneal.models.surrogate import GPRSurrogate


def test_kdpp_returns_q_distinct_absolute_idx():
    ds = make_synthetic(n_genes=40, dim=4, seed=1)
    Xtr, ytr = ds.embeddings[:10], ds.target[:10]
    surr = GPRSurrogate(n_iters=60).fit(Xtr, ytr)
    cand = list(range(10, 40))
    Xc = ds.embeddings[cand]
    mean, std = surr.predict(Xc)
    sel = KDPP(solver="greedy").select(
        candidate_idx=cand, X_candidates=Xc, mean=mean, std=std,
        best=float(ytr.max()), q=5, rng=np.random.default_rng(0),
        surrogate=surr, acquisition=UCB(beta=1.0), X_train=Xtr, y_train=ytr)
    assert len(sel) == 5
    assert len(set(sel)) == 5
    assert set(sel).issubset(set(cand))


def test_kdpp_avoids_duplicate_directions():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((1, 4))
    Xc = np.vstack([base, base, rng.standard_normal((1, 4))])  # 0,1 identical; 2 distinct
    ds_x = np.vstack([Xc, rng.standard_normal((8, 4))])
    ds_y = ds_x @ rng.standard_normal(4)
    surr = GPRSurrogate(n_iters=60).fit(ds_x[3:], ds_y[3:])
    cand = [100, 101, 102]
    mean, std = surr.predict(Xc)
    sel = KDPP(solver="greedy").select(
        candidate_idx=cand, X_candidates=Xc, mean=mean, std=std,
        best=0.0, q=2, rng=np.random.default_rng(0),
        surrogate=surr, acquisition=GreedyMean(), X_train=ds_x[3:], y_train=ds_y[3:])
    assert 102 in sel  # distinct candidate chosen over the duplicate pair
    assert not (100 in sel and 101 in sel)  # never both identical ones


def test_kdpp_sampling_solver_runs_and_is_seeded():
    ds = make_synthetic(n_genes=30, dim=4, seed=3)
    Xtr, ytr = ds.embeddings[:8], ds.target[:8]
    surr = GPRSurrogate(n_iters=50).fit(Xtr, ytr)
    cand = list(range(8, 30)); Xc = ds.embeddings[cand]
    mean, std = surr.predict(Xc)
    kw = dict(candidate_idx=cand, X_candidates=Xc, mean=mean, std=std, best=0.0,
              q=4, surrogate=surr, acquisition=UCB(1.0), X_train=Xtr, y_train=ytr)
    a = KDPP(solver="sampling").select(rng=np.random.default_rng(7), **kw)
    b = KDPP(solver="sampling").select(rng=np.random.default_rng(7), **kw)
    assert len(a) == 4 and len(set(a)) == 4
    assert a == b  # same seed -> same sample
