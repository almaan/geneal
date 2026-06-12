# tests/test_predict_cov.py
import numpy as np
from geneal.data.dataset import make_synthetic
from geneal.models.surrogate import GPRSurrogate, BNNSurrogate


def _xy():
    ds = make_synthetic(n_genes=120, dim=5, seed=2, noise_sd=0.05)
    return ds.embeddings[:80], ds.target[:80], ds.embeddings[80:]


def test_gpr_predict_cov_shapes_and_psd():
    Xtr, ytr, Xte = _xy()
    surr = GPRSurrogate(n_iters=80).fit(Xtr, ytr)
    mean, cov = surr.predict_cov(Xte)
    n = len(Xte)
    assert mean.shape == (n,)
    assert cov.shape == (n, n)
    assert np.allclose(cov, cov.T, atol=1e-4)
    _, std = surr.predict(Xte)
    assert np.allclose(np.diag(cov), std**2, rtol=0.2, atol=1e-3)
    eig = np.linalg.eigvalsh((cov + cov.T) / 2)
    assert eig.min() > -1e-4


def test_bnn_predict_cov_shapes():
    Xtr, ytr, Xte = _xy()
    surr = BNNSurrogate(hidden=16, n_steps=150, seed=0).fit(Xtr, ytr)
    mean, cov = surr.predict_cov(Xte)
    n = len(Xte)
    assert mean.shape == (n,)
    assert cov.shape == (n, n)
    assert np.allclose(cov, cov.T, atol=1e-4)
