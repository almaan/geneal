# tests/test_surrogate.py
import numpy as np
import pytest
from geneal.data.dataset import make_synthetic
from geneal.models.surrogate import GPRSurrogate, BNNSurrogate


def _train_test():
    ds = make_synthetic(n_genes=120, dim=5, seed=2, noise_sd=0.05)
    Xtr, ytr = ds.embeddings[:80], ds.target[:80]
    Xte, yte = ds.embeddings[80:], ds.target[80:]
    return Xtr, ytr, Xte, yte


def test_gpr_predict_shapes_and_clone():
    Xtr, ytr, Xte, yte = _train_test()
    surr = GPRSurrogate().fit(Xtr, ytr)
    mean, std = surr.predict(Xte)
    assert mean.shape == (len(Xte),)
    assert std.shape == (len(Xte),)
    assert np.all(std >= 0)
    # clone is unfitted and independent
    clone = surr.clone()
    assert isinstance(clone, GPRSurrogate)


def test_gpr_learns_signal():
    Xtr, ytr, Xte, yte = _train_test()
    mean, _ = GPRSurrogate().fit(Xtr, ytr).predict(Xte)
    corr = np.corrcoef(mean, yte)[0, 1]
    assert corr > 0.7


def test_bnn_predict_shapes():
    Xtr, ytr, Xte, yte = _train_test()
    mean, std = BNNSurrogate(hidden=16, n_steps=200, seed=0).fit(Xtr, ytr).predict(Xte)
    assert mean.shape == (len(Xte),)
    assert std.shape == (len(Xte),)
    assert np.all(std >= 0)
