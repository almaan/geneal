# tests/test_acquisition.py
import numpy as np
from geneal.models.acquisition import (
    UCB, ExpectedImprovement, MaxVariance, GreedyMean, RandomAcquisition,
)


def test_ucb_prefers_high_mean_and_high_std():
    mean = np.array([0.0, 1.0, 0.0])
    std = np.array([0.0, 0.0, 1.0])
    s = UCB(beta=2.0).score(mean, std, best=0.0, rng=np.random.default_rng(0))
    # idx1 (mean=1) and idx2 (2*std=2) both beat idx0
    assert s[1] > s[0]
    assert s[2] > s[0]


def test_greedy_mean_is_mean():
    mean = np.array([3.0, 1.0])
    std = np.array([5.0, 5.0])
    s = GreedyMean().score(mean, std, best=0.0, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(s, mean)


def test_max_variance_is_std():
    mean = np.array([3.0, 1.0])
    std = np.array([0.2, 0.9])
    s = MaxVariance().score(mean, std, best=0.0, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(s, std)


def test_ei_nonnegative_and_zero_when_std_zero_below_best():
    mean = np.array([1.0, 2.0])
    std = np.array([0.0, 1.0])
    s = ExpectedImprovement().score(mean, std, best=5.0, rng=np.random.default_rng(0))
    assert np.all(s >= 0)
    assert s[0] == 0.0  # std 0 and mean < best -> no improvement


def test_random_is_deterministic_given_seed():
    mean = np.zeros(10); std = np.zeros(10)
    a = RandomAcquisition().score(mean, std, 0.0, np.random.default_rng(5))
    b = RandomAcquisition().score(mean, std, 0.0, np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)
