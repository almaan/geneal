# tests/test_bivariate.py
import numpy as np
from geneal.runner.bivariate import BivariateALRunner
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise


def _toy():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 5))
    # efficacy correlated with dim0; toxicity with dim1 (so they're separable)
    eff = X[:, 0] + 0.1 * rng.standard_normal(120)
    tox = X[:, 1] + 0.1 * rng.standard_normal(120)
    return X, eff, tox


def test_bivariate_loop_runs_and_grows_hypervolume():
    X, eff, tox = _toy()
    r = BivariateALRunner(lambda: GPRSurrogate(n_iters=40), GaussianNoise(0.01))
    hist = r.run(X, eff, tox, n_initial=15, n_rounds=4, batch_size=5,
                 seed=0)
    assert len(hist) == 5  # round 0..4
    hv = [h["hypervolume"] for h in hist]
    assert hv[-1] >= hv[0] - 1e-9      # hypervolume non-decreasing overall
    assert all(set(h) >= {"round", "hypervolume", "n_revealed"} for h in hist)


def test_bivariate_deterministic():
    X, eff, tox = _toy()
    def run():
        return BivariateALRunner(lambda: GPRSurrogate(n_iters=40), GaussianNoise(0.01)
                                 ).run(X, eff, tox, 15, 3, 5, seed=1)
    a, b = run(), run()
    assert [h["hypervolume"] for h in a] == [h["hypervolume"] for h in b]
