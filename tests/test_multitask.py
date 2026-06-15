# tests/test_multitask.py
"""Tests for the multitask (coregionalized) GP surrogate."""
from __future__ import annotations
import numpy as np
from geneal.models.multitask import MultiTaskGPR


def _corr_data(n=60, rho=0.9, seed=0):
    """Two correlated outputs over a 1-D feature: y0 = f, y1 = f shifted, plus
    correlated noise -> strong positive task correlation."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    f = X[:, 0]
    y0 = f + 0.05 * rng.standard_normal(n)
    y1 = rho * f + np.sqrt(1 - rho**2) * rng.standard_normal(n) * 0.3
    Y = np.column_stack([y0, y1])
    return X, Y


def test_fit_predict_shapes():
    X, Y = _corr_data()
    s = MultiTaskGPR(n_iters=40).fit(X, Y)
    m, sd = s.predict(X)
    assert m.shape == (len(X), 2) and sd.shape == (len(X), 2)
    assert np.all(np.isfinite(m)) and np.all(sd >= 0)


def test_predict_taskcov_shape_and_psd():
    X, Y = _corr_data()
    s = MultiTaskGPR(n_iters=40).fit(X, Y)
    m, cov = s.predict_taskcov(X[:5])
    assert m.shape == (5, 2) and cov.shape == (5, 2, 2)
    for c in cov:
        assert np.allclose(c, c.T, atol=1e-5)        # symmetric
        assert np.linalg.eigvalsh(c).min() > -1e-6   # PSD


def test_captures_positive_task_covariation():
    # for rho=0.9 data (both tasks track the same latent f), the joint model's
    # predicted task MEANS should be strongly positively correlated across genes.
    X, Y = _corr_data(rho=0.9, n=80)
    s = MultiTaskGPR(n_iters=150).fit(X, Y)
    m, _ = s.predict(X)
    assert float(np.corrcoef(m[:, 0], m[:, 1])[0, 1]) > 0.5
