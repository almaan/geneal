# src/geneal/models/acquisition.py
from __future__ import annotations
import numpy as np
from scipy.stats import norm

# All acquisitions MAXIMIZE the target. Higher score => more desirable.


class UCB:
    """Upper confidence bound: mean + beta * std."""
    def __init__(self, beta: float = 2.0) -> None:
        self.beta = beta

    def score(self, mean, std, best, rng):
        return np.asarray(mean) + self.beta * np.asarray(std)


class ExpectedImprovement:
    """Expected improvement over current best (maximization)."""
    def __init__(self, xi: float = 0.0) -> None:
        self.xi = xi

    def score(self, mean, std, best, rng):
        mean = np.asarray(mean, dtype=float)
        std = np.asarray(std, dtype=float)
        imp = mean - best - self.xi
        out = np.zeros_like(mean)
        mask = std > 1e-12
        z = np.zeros_like(mean)
        z[mask] = imp[mask] / std[mask]
        out[mask] = imp[mask] * norm.cdf(z[mask]) + std[mask] * norm.pdf(z[mask])
        # where std==0: improvement only if mean exceeds best
        out[~mask] = np.maximum(imp[~mask], 0.0)
        return np.maximum(out, 0.0)


class MaxVariance:
    """Pure-exploration / information-gain proxy: predictive std."""
    def score(self, mean, std, best, rng):
        return np.asarray(std)


class GreedyMean:
    """Pure exploitation: predicted mean."""
    def score(self, mean, std, best, rng):
        return np.asarray(mean)


class RandomAcquisition:
    """Uniform random scores — used by the random baseline method."""
    def score(self, mean, std, best, rng):
        return rng.random(len(mean))
