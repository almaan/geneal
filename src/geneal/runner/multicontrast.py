# src/geneal/runner/multicontrast.py
"""Multi-contrast selectivity nomination (joint-GP N-D EHVI) for one cell line.

Maximize target-line efficacy and minimize lethality in EACH of several contrast
(non-target reference) lines simultaneously -- an (1+n_contrasts)-objective
problem solved with a JOINT multitask GP + posterior-integrated N-D EHVI. Shared
by the standalone exploratory script and the main ablation runner so a normal
sweep produces the multi-contrast tables natively."""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler

from geneal.data.selective import build_selective_dataset, toxicity_vector
from geneal.models.multitask import MultiTaskGPR
from geneal.models.selection import CoreSet, TypiClust, _greedy_map_logdet  # noqa: F401
from geneal.models.multiobjective import (hypervolume_mc, ehvi_nominate_nd,
                                          pareto_front_nd)


def _norm01(a):
    a = np.asarray(a, float)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-12)


def run_line(ge, emb, cl, contrasts, n_init, K, seed, hv_samples):
    """Nominate K targets selective against ALL `contrasts` at once, by several
    methods (greedy / random / cluster / info_div / ehvi / ehvi_pareto; the
    Section-A methods with NO safety filter). Returns a list of metric dicts (one
    per method) evaluated on TRUE values: target efficacy, lethality in each
    contrast (tox_c1..), their average, and the (1+n_contrasts)-D hypervolume.
    The joint GP's full task covariance is computed only on the EHVI shortlist
    pool so this scales to the whole genome."""
    from geneal.runner.ablation import build_embedding_S
    ds, _ = build_selective_dataset(ge, emb, cl, lam=0.0, thresh=-0.5)
    X = StandardScaler().fit_transform(ds.embeddings)
    eff = np.asarray(ds.target, float)                       # target-line lethality (true)
    tox = [toxicity_vector(ge, ds.gene_names, "contrast", target_line=cl,
                           contrast_line=c, thresh=-0.5) for c in contrasts]
    nobj = 1 + len(tox)
    n = len(eff)
    rng = np.random.default_rng((seed, 7))
    init = rng.permutation(n)[:n_init]
    cand = np.array([i for i in range(n) if i not in set(init)])
    Xc = X[cand]

    Yj = np.column_stack([eff] + tox)
    jg = MultiTaskGPR(n_iters=120, num_tasks=nobj).fit(X[init], Yj[init])
    pmean, _pstd = jg.predict(Xc)                            # diagonal, all candidates
    pe = pmean[:, 0]
    D = np.diag([1.0] + [-1.0] * len(tox))                   # flip tox -> maximize all
    obj_mean = pmean @ D
    lo = obj_mean.min(0); span = obj_mean.max(0) - lo + 1e-12
    promise = obj_mean.mean(1)

    nom_rng = np.random.default_rng((seed, 11))
    m = len(cand)
    pool = min(120, m)
    pool_idx = np.argsort(promise)[::-1][:pool]
    _, pcov_pool = jg.predict_taskcov(Xc[pool_idx])          # 4x4 cov on the pool only
    obj_cov_pool = np.einsum("ij,mjk,kl->mil", D, pcov_pool, D)
    mean_n_pool = (obj_mean[pool_idx] - lo) / span
    cov_n_pool = obj_cov_pool / np.outer(span, span)[None]

    eff_pick = list(np.argsort(pe)[::-1][:K])
    rand_pick = list(nom_rng.permutation(m)[:K])
    clust_pick = list(TypiClust().select(candidate_idx=list(range(m)), X_candidates=Xc,
                                         mean=None, std=None, best=None, q=K, rng=nom_rng,
                                         surrogate=None, acquisition=None,
                                         X_train=X[init], y_train=eff[init]))
    q = pe[pool_idx] - pe[pool_idx].min() + 1e-6
    Sc = build_embedding_S(Xc[pool_idx])
    L = (q[:, None] * Sc) * q[None, :]; L = (L + L.T) / 2 + 1e-9 * np.eye(len(q))
    info_pick = [int(pool_idx[j]) for j in _greedy_map_logdet(L, K)]
    ehvi_local = ehvi_nominate_nd(mean_n_pool, cov_n_pool, K=K, rng=nom_rng,
                                  n_post=40, n_hv=3000)
    ehvi_pick = [int(pool_idx[j]) for j in ehvi_local]
    front = pareto_front_nd(obj_mean)
    front_sorted = sorted(front, key=lambda i: promise[i], reverse=True)
    if len(front_sorted) >= K:
        par_pick = front_sorted[:K]
    else:
        rest = [i for i in np.argsort(promise)[::-1] if i not in set(front_sorted)]
        par_pick = front_sorted + rest[:K - len(front_sorted)]

    eff_c = eff[cand]; tox_c = [t[cand] for t in tox]
    Ytrue = np.column_stack([_norm01(eff_c)] + [_norm01(-np.asarray(t)) for t in tox_c])
    hv_rng = np.random.default_rng(20260626)
    rows = []
    for name, pick in [("greedy", eff_pick), ("random", rand_pick), ("cluster", clust_pick),
                       ("info_div", info_pick), ("ehvi", ehvi_pick), ("ehvi_pareto", par_pick)]:
        pick = list(pick)
        row = dict(cell_line=cl, seed=seed, method=name,
                   mean_efficacy=float(np.mean(eff_c[pick])),
                   hv4d=float(hypervolume_mc(Ytrue[pick], hv_rng, hv_samples)))
        toxes = [float(np.mean(t[pick])) for t in tox_c]
        for j, tv in enumerate(toxes, 1):
            row[f"tox_c{j}"] = tv
        row["tox_avg"] = float(np.mean(toxes))
        rows.append(row)
    return rows
