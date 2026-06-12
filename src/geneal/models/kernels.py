# src/geneal/models/kernels.py
from __future__ import annotations
import numpy as np


def posterior_correlation(cov: np.ndarray) -> np.ndarray:
    """Correlation matrix from a covariance matrix.

    S_ij = cov_ij / (sqrt(cov_ii) * sqrt(cov_jj)). Zero-variance rows/cols are
    treated as self-correlation 1 and zero cross-correlation (they carry no
    diversity information). Output is symmetric with unit diagonal.
    """
    cov = np.asarray(cov, dtype=float)
    d = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    n = cov.shape[0]
    corr[np.diag_indices(n)] = 1.0
    return np.clip((corr + corr.T) / 2, -1.0, 1.0)
