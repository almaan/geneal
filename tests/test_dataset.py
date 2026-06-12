# tests/test_dataset.py
import numpy as np
from geneal.data.dataset import Dataset, make_synthetic


def test_make_synthetic_shapes():
    ds = make_synthetic(n_genes=50, dim=8, seed=0)
    assert ds.embeddings.shape == (50, 8)
    assert ds.target.shape == (50,)
    assert len(ds.gene_names) == 50
    assert ds.n_genes == 50


def test_make_synthetic_is_deterministic():
    a = make_synthetic(n_genes=20, dim=4, seed=7)
    b = make_synthetic(n_genes=20, dim=4, seed=7)
    np.testing.assert_array_equal(a.target, b.target)
    np.testing.assert_array_equal(a.embeddings, b.embeddings)


def test_synthetic_target_is_learnable_from_embeddings():
    # Target is a (noisy) linear function of embeddings, so correlation with the
    # true linear signal should be strong -> a surrogate can learn it.
    ds = make_synthetic(n_genes=500, dim=6, seed=1, noise_sd=0.1)
    # Fit least squares; R^2 should be high.
    X = np.hstack([ds.embeddings, np.ones((ds.n_genes, 1))])
    beta, *_ = np.linalg.lstsq(X, ds.target, rcond=None)
    pred = X @ beta
    ss_res = np.sum((ds.target - pred) ** 2)
    ss_tot = np.sum((ds.target - ds.target.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.8
