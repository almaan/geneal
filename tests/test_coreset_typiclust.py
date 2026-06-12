# tests/test_coreset_typiclust.py
import numpy as np
from geneal.models.selection import CoreSet, TypiClust
from geneal.models.acquisition import GreedyMean
from geneal.data.dataset import make_synthetic
from geneal.models.surrogate import GPRSurrogate


def _setup(n_genes=40, n_init=10, dim=4, seed=1):
    ds = make_synthetic(n_genes=n_genes, dim=dim, seed=seed)
    Xtr, ytr = ds.embeddings[:n_init], ds.target[:n_init]
    surr = GPRSurrogate(n_iters=50).fit(Xtr, ytr)
    cand = list(range(n_init, n_genes))
    Xc = ds.embeddings[cand]
    mean, std = surr.predict(Xc)
    return dict(candidate_idx=cand, X_candidates=Xc, mean=mean, std=std,
               best=float(ytr.max()), surrogate=surr, acquisition=GreedyMean(),
               X_train=Xtr, y_train=ytr)


def test_coreset_returns_q_distinct():
    kw = _setup()
    sel = CoreSet().select(q=5, rng=np.random.default_rng(0), **kw)
    assert len(sel) == 5 and len(set(sel)) == 5
    assert set(sel).issubset(set(kw["candidate_idx"]))


def test_coreset_picks_spread_points():
    X = np.linspace(0, 1, 11).reshape(-1, 1)
    cand = list(range(11))
    sel = CoreSet().select(
        candidate_idx=cand, X_candidates=X, mean=np.zeros(11), std=np.zeros(11),
        best=0.0, q=2, rng=np.random.default_rng(0), surrogate=None,
        acquisition=GreedyMean(), X_train=np.array([[0.5]]), y_train=np.array([0.0]))
    assert set(sel) == {0, 10} or set(sel) == {10, 0}


def test_typiclust_returns_q_distinct():
    kw = _setup()
    sel = TypiClust().select(q=5, rng=np.random.default_rng(0), **kw)
    assert len(sel) == 5 and len(set(sel)) == 5
    assert set(sel).issubset(set(kw["candidate_idx"]))
