# tests/test_recall.py
import numpy as np
from geneal.metrics.recall import RecallAtK


def test_recall_name():
    assert RecallAtK(k=5).name == "recall@5"


def test_recall_full_when_all_top_revealed():
    target = np.array([10.0, 9.0, 8.0, 1.0, 0.0])  # top-2 are idx 0,1
    m = RecallAtK(k=2)
    assert m.evaluate([0, 1, 4], target) == 1.0


def test_recall_partial():
    target = np.array([10.0, 9.0, 8.0, 1.0, 0.0])
    m = RecallAtK(k=2)
    assert m.evaluate([0, 4], target) == 0.5  # found idx 0 of {0,1}


def test_recall_zero():
    target = np.array([10.0, 9.0, 8.0, 1.0, 0.0])
    m = RecallAtK(k=2)
    assert m.evaluate([3, 4], target) == 0.0
