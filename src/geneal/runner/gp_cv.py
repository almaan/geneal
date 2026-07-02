# src/geneal/runner/gp_cv.py
"""Method-independent surrogate fit-quality: 5-fold cross-validation of the GP on
each objective axis, per cell line. This depends only on the data + surrogate, NOT
on the acquisition/diversity method, so it never changes across method ablations
and is cached to disk (keyed by a hash of the inputs) and loaded when present."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

from geneal.models.surrogate import GPRSurrogate


def _default_factory():
    return GPRSurrogate(n_iters=120)


def gp_cv_fold_metrics(X, y, n_splits: int = 5, seed: int = 0,
                       surr_factory=None) -> list[dict]:
    """5-fold CV of the surrogate predicting y from X. Returns one dict per fold
    with R^2, Spearman rho, and RMSE on the held-out fold."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    surr_factory = surr_factory or _default_factory
    out = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    for fold, (tr, te) in enumerate(kf.split(X)):
        m, _ = surr_factory().fit(X[tr], y[tr]).predict(X[te])
        m = np.asarray(m, float)
        resid = y[te] - m
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((y[te] - y[te].mean()) ** 2).sum()) or 1e-12
        rho = spearmanr(y[te], m).correlation
        out.append(dict(fold=int(fold),
                        r2=float(1.0 - ss_res / ss_tot),
                        spearman=float(rho if rho == rho else 0.0),
                        rmse=float(np.sqrt((resid ** 2).mean()))))
    return out


def cv_cache_key(*parts) -> str:
    """Stable short hash of the inputs that determine the CV result."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def gp_cv_table(axes, n_splits: int = 5, seed: int = 0, surr_factory=None,
                cache_path=None) -> pd.DataFrame:
    """Compute (or load) the per-(cell_line, axis) CV table.

    `axes`: iterable of (cell_line, axis_name, X, y). If `cache_path` exists, it is
    loaded and returned without recomputation; otherwise the table is computed and
    written there. Columns: cell_line, axis, fold, r2, spearman, rmse."""
    if cache_path is not None and Path(cache_path).exists():
        return pd.read_parquet(cache_path)
    rows = []
    for cl, axis, X, y in axes:
        for r in gp_cv_fold_metrics(X, y, n_splits=n_splits, seed=seed,
                                    surr_factory=surr_factory):
            rows.append(dict(cell_line=cl, axis=axis, **r))
    df = pd.DataFrame(rows)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
    return df
