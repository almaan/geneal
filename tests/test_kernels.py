# tests/test_kernels.py
import numpy as np
from geneal.models.kernels import posterior_correlation


def test_correlation_diag_is_one():
    cov = np.array([[4.0, 1.0], [1.0, 9.0]])
    corr = posterior_correlation(cov)
    assert np.allclose(np.diag(corr), 1.0)


def test_correlation_values():
    cov = np.array([[4.0, 2.0], [2.0, 9.0]])
    corr = posterior_correlation(cov)
    assert np.isclose(corr[0, 1], 1.0 / 3.0)
    assert np.allclose(corr, corr.T)


def test_zero_variance_safe():
    cov = np.array([[0.0, 0.0], [0.0, 4.0]])
    corr = posterior_correlation(cov)
    assert np.all(np.isfinite(corr))
    assert np.isclose(corr[1, 1], 1.0)
