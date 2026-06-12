# tests/test_diagnostics.py
import numpy as np
from geneal.metrics.diagnostics import batch_quality, batch_diversity


def test_batch_quality_is_mean_true_effect():
    target = np.array([1.0, 2.0, 3.0, 4.0])
    assert batch_quality([1, 3], target) == np.mean([2.0, 4.0])


def test_batch_diversity_zero_for_identical_rows():
    X = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert batch_diversity([0, 1, 2], X) == 0.0


def test_batch_diversity_positive_for_spread():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert batch_diversity([0, 1, 2], X) > 0.0


def test_batch_diversity_singleton_is_zero():
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert batch_diversity([0], X) == 0.0
