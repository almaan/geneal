# src/geneal/metrics/diagnostics.py
from __future__ import annotations
from typing import Sequence
import numpy as np


def batch_quality(batch_idx: Sequence[int], target: np.ndarray) -> float:
    """Mean true target (higher = better) of the genes selected in a batch."""
    batch_idx = list(batch_idx)
    if not batch_idx:
        return 0.0
    return float(np.mean(np.asarray(target)[batch_idx]))


def batch_diversity(batch_idx: Sequence[int], embeddings: np.ndarray) -> float:
    """Mean pairwise Euclidean distance among the batch in embedding space.

    A simple, model-free diversity readout for the quality-diversity Pareto plot.
    Zero for a singleton or identical points; larger for a more spread batch.
    """
    batch_idx = list(batch_idx)
    if len(batch_idx) < 2:
        return 0.0
    X = np.asarray(embeddings)[batch_idx]
    n = len(batch_idx)
    total, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(np.linalg.norm(X[i] - X[j]))
            cnt += 1
    return total / cnt if cnt else 0.0
