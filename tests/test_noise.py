# tests/test_noise.py
import numpy as np
from geneal.models.noise import NoNoise, GaussianNoise


def test_nonoise_is_zeros():
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(NoNoise().draw(5, rng), np.zeros(5))


def test_gaussian_shape_and_determinism():
    a = GaussianNoise(sigma=0.5).draw(100, np.random.default_rng(3))
    b = GaussianNoise(sigma=0.5).draw(100, np.random.default_rng(3))
    assert a.shape == (100,)
    np.testing.assert_array_equal(a, b)


def test_gaussian_scale_roughly_correct():
    x = GaussianNoise(sigma=2.0).draw(50000, np.random.default_rng(1))
    assert abs(x.std() - 2.0) < 0.1
