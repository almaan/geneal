# tests/test_interfaces.py
import numpy as np
from geneal.interfaces import Surrogate, Acquisition, Selection, NoiseModel, Metric


class _Surr:
    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X)), np.ones(len(X))
    def clone(self): return _Surr()


def test_surrogate_protocol_runtime_check():
    assert isinstance(_Surr(), Surrogate)


def test_non_surrogate_fails_check():
    class NotSurr:
        def fit(self, X, y): return self
    assert not isinstance(NotSurr(), Surrogate)
