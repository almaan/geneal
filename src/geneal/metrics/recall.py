# src/geneal/metrics/recall.py
from __future__ import annotations
from typing import Sequence
import numpy as np


class RecallAtK:
    """Fraction of the true top-k (highest target) that has been revealed."""

    def __init__(self, k: int = 10) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k

    @property
    def name(self) -> str:
        return f"recall@{self.k}"

    def true_top(self, target: np.ndarray) -> set[int]:
        k = min(self.k, len(target))
        return set(np.argsort(target)[::-1][:k].tolist())

    def evaluate(self, revealed_idx: Sequence[int], target: np.ndarray) -> float:
        top = self.true_top(target)
        if not top:
            return 0.0
        hit = len(set(int(i) for i in revealed_idx) & top)
        return hit / len(top)
