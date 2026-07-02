import numpy as np
from geneal.runner.gp_cv import gp_cv_fold_metrics, gp_cv_table, cv_cache_key
from geneal.models.surrogate import GPRSurrogate


def _factory():
    return GPRSurrogate(n_iters=20)


def test_gp_cv_fold_metrics_learnable_signal():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 3))
    y = X[:, 0] * 2.0 + 0.05 * rng.standard_normal(60)   # strong learnable signal
    folds = gp_cv_fold_metrics(X, y, n_splits=5, seed=0, surr_factory=_factory)
    assert len(folds) == 5
    assert all({"fold", "r2", "spearman", "rmse"} <= set(f) for f in folds)
    assert np.mean([f["r2"] for f in folds]) > 0.5          # GP recovers the signal
    assert np.mean([f["spearman"] for f in folds]) > 0.7


def test_gp_cv_table_caches(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 2)); y = X[:, 0]
    axes = [("L1", "target_pop", X, y)]
    cache = tmp_path / "gp_cv_x.parquet"
    df1 = gp_cv_table(axes, n_splits=4, seed=0, surr_factory=_factory, cache_path=str(cache))
    assert cache.exists()
    assert set(df1.columns) >= {"cell_line", "axis", "fold", "r2", "spearman", "rmse"}
    # second call loads cache (returns identical frame even with bogus axes)
    df2 = gp_cv_table([("BOGUS", "x", X, y)], n_splits=4, seed=0,
                      surr_factory=_factory, cache_path=str(cache))
    assert df2["cell_line"].tolist() == df1["cell_line"].tolist()


def test_cv_cache_key_stable_and_sensitive():
    a = cv_cache_key("emb", "ge", "panel", ["L1", "L2"], ["contrast"], 5, -0.5, "C")
    b = cv_cache_key("emb", "ge", "panel", ["L1", "L2"], ["contrast"], 5, -0.5, "C")
    c = cv_cache_key("emb", "ge", "panel", ["L1", "L2"], ["contrast"], 5, -0.5, "D")
    assert a == b and a != c
