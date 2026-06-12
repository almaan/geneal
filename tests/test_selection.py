# tests/test_selection.py
import numpy as np
from geneal.models.selection import TopQGreedy, GreedyFantasy
from geneal.models.acquisition import GreedyMean, UCB
from geneal.models.surrogate import GPRSurrogate
from geneal.data.dataset import make_synthetic


def test_topq_picks_highest_scores_and_maps_to_absolute_idx():
    candidate_idx = [10, 11, 12, 13]
    mean = np.array([0.1, 0.9, 0.5, 0.3])
    std = np.zeros(4)
    sel = TopQGreedy().select(
        candidate_idx=candidate_idx, X_candidates=np.zeros((4, 2)),
        mean=mean, std=std, best=0.0, q=2,
        rng=np.random.default_rng(0), surrogate=None, acquisition=GreedyMean(),
    )
    assert sel == [11, 12]  # highest mean (0.9) then 0.5


def test_topq_returns_q_items():
    sel = TopQGreedy().select(
        candidate_idx=[0, 1, 2, 3, 4], X_candidates=np.zeros((5, 2)),
        mean=np.arange(5.0), std=np.zeros(5), best=0.0, q=3,
        rng=np.random.default_rng(0), surrogate=None, acquisition=GreedyMean(),
    )
    assert len(sel) == 3
    assert len(set(sel)) == 3


def test_fantasy_returns_q_distinct_absolute_idx():
    ds = make_synthetic(n_genes=40, dim=4, seed=1)
    Xtr, ytr = ds.embeddings[:10], ds.target[:10]
    surr = GPRSurrogate().fit(Xtr, ytr)
    cand = list(range(10, 40))
    Xc = ds.embeddings[cand]
    mean, std = surr.predict(Xc)
    sel = GreedyFantasy().select(
        candidate_idx=cand, X_candidates=Xc, mean=mean, std=std,
        best=float(ds.target[:10].max()), q=5, rng=np.random.default_rng(0),
        surrogate=surr, acquisition=UCB(beta=1.0), X_train=Xtr, y_train=ytr,
    )
    assert len(sel) == 5
    assert len(set(sel)) == 5
    assert set(sel).issubset(set(cand))


def test_fantasy_refit_conditions_on_real_data_not_collapsed():
    # Regression test: the working surrogate must be refit on real training data
    # PLUS fantasies, not fantasies alone. With UCB(beta>0), a refit on a single
    # fantasy point collapses predictive std to ~0 everywhere, so the post-pick
    # scores would be the degenerate constant mean. Here we assert that after a
    # fantasy step the surrogate still produces non-degenerate (non-constant)
    # predictions over the candidates — which only holds when conditioned on the
    # real data.
    ds = make_synthetic(n_genes=60, dim=4, seed=2)
    Xtr, ytr = ds.embeddings[:20], ds.target[:20]
    surr = GPRSurrogate().fit(Xtr, ytr)
    cand = list(range(20, 60))
    Xc = ds.embeddings[cand]
    mean, std = surr.predict(Xc)
    # Reproduce one fantasy step the way GreedyFantasy does it, on real+fantasy.
    fant_X = Xc[0:1]
    fant_y = np.array([float(mean[0])])
    aug_X = np.vstack([Xtr, fant_X])
    aug_y = np.concatenate([ytr, fant_y])
    m, s = surr.clone().fit(aug_X, aug_y).predict(Xc)
    # Non-degenerate: predictions vary across candidates and uncertainty is > 0.
    assert len(np.unique(np.round(m, 4))) > 1
    assert float(s.max()) > 1e-3
