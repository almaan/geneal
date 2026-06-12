# src/geneal/interfaces.py
from __future__ import annotations
from typing import Protocol, runtime_checkable, Sequence
import numpy as np


@runtime_checkable
class Surrogate(Protocol):
    """Maps gene embeddings -> predicted target with uncertainty."""
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Surrogate": ...
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...  # (mean, std)
    def predict_cov(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...  # (mean, full covariance)
    def clone(self) -> "Surrogate": ...


@runtime_checkable
class Acquisition(Protocol):
    """Scores candidates from posterior predictive. Higher score = more desirable."""
    def score(self, mean: np.ndarray, std: np.ndarray, best: float,
              rng: np.random.Generator) -> np.ndarray: ...


@runtime_checkable
class Selection(Protocol):
    """Chooses q candidate indices to acquire this round.

    Receives everything any strategy might need; impls use the subset they care
    about. `candidate_idx` are absolute dataset indices; the return value is a
    list of absolute dataset indices (a subset of candidate_idx), length q.
    `X_train`/`y_train` are the surrogate's current fitted training inputs and
    targets — strategies that refit (e.g. fantasy batching) need them so the
    refit conditions on real data plus fantasies, not fantasies alone.
    """
    def select(self, *, candidate_idx: Sequence[int], X_candidates: np.ndarray,
               mean: np.ndarray, std: np.ndarray, best: float, q: int,
               rng: np.random.Generator, surrogate: Surrogate,
               acquisition: Acquisition, X_train: np.ndarray,
               y_train: np.ndarray) -> list[int]: ...


@runtime_checkable
class NoiseModel(Protocol):
    """Draws a per-gene noise vector once per seed (common random numbers)."""
    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray: ...


@runtime_checkable
class Metric(Protocol):
    """Evaluates progress given the revealed set and ground-truth target."""
    def evaluate(self, revealed_idx: Sequence[int], target: np.ndarray) -> float: ...
    @property
    def name(self) -> str: ...
