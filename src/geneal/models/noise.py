# src/geneal/models/noise.py
from __future__ import annotations
import numpy as np


class NoNoise:
    """Reveals true target values unchanged."""
    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.zeros(n)


class GaussianNoise:
    """Homoscedastic Gaussian measurement noise, N(0, sigma^2)."""
    def __init__(self, sigma: float = 1.0) -> None:
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        self.sigma = sigma

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, self.sigma, size=n)
