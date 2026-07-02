# src/geneal/models/multiobjective.py
from __future__ import annotations
import numpy as np


def pareto_front(P: np.ndarray) -> list[int]:
    """Indices of non-dominated rows (maximize both columns). 2-D."""
    P = np.asarray(P, float)
    n = len(P)
    keep = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if j == i:
                continue
            if (P[j, 0] >= P[i, 0] and P[j, 1] >= P[i, 1] and
                    (P[j, 0] > P[i, 0] or P[j, 1] > P[i, 1])):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def pareto_indices(P: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated rows (maximize both columns), O(n log n).
    Sort by col0 desc (ties col1 desc), sweep keeping the running max of col1;
    a point is non-dominated iff its col1 exceeds every earlier (higher-col0) one."""
    P = np.asarray(P, float)
    n = len(P)
    if n == 0:
        return np.array([], dtype=int)
    order = np.lexsort((-P[:, 1], -P[:, 0]))   # primary: col0 desc; secondary: col1 desc
    keep, best_y = [], -np.inf
    for i in order:
        if P[i, 1] > best_y:
            keep.append(i); best_y = P[i, 1]
    return np.array(keep, dtype=int)


def hypervolume2d(P: np.ndarray, ref: np.ndarray) -> float:
    """Dominated hypervolume (area) of point set P above reference ref (maximize
    both). Only non-dominated points above ref contribute."""
    P = np.asarray(P, float)
    if len(P) == 0:
        return 0.0
    pf = P[pareto_front(P)]
    pf = pf[(pf[:, 0] > ref[0]) & (pf[:, 1] > ref[1])]
    if len(pf) == 0:
        return 0.0
    # sort by x descending; sweep
    pf = pf[np.argsort(-pf[:, 0])]
    area = 0.0
    prev_y = ref[1]
    for x, y in pf:
        if y > prev_y:
            area += (x - ref[0]) * (y - prev_y)
            prev_y = y
    return float(area)


def hypervolume_mc(Y, rng, n_samples: int = 40000) -> float:
    """Monte-Carlo dominated hypervolume of points Y in the unit box [0,1]^d
    (maximize every objective, reference = origin). HV = fraction of uniform box
    samples dominated by at least one point. Dimension-agnostic (used for the
    >2-objective multi-contrast selectivity experiment)."""
    Y = np.asarray(Y, float)
    if len(Y) == 0:
        return 0.0
    d = Y.shape[1]
    S = rng.random((n_samples, d))
    dom = np.zeros(n_samples, dtype=bool)
    for p in Y:
        dom |= np.all(p >= S, axis=1)
    return float(dom.mean())


def greedy_hv_nominate(Y, K, rng, n_samples: int = 40000) -> list[int]:
    """Greedy hypervolume-maximizing batch of K rows of Y (each in [0,1]^d,
    maximize all objectives, reference = origin). At each step add the candidate
    giving the largest marginal dominated-volume gain over the current set. This
    is the N-objective EHVI-style nominator evaluated at the predicted means
    (deterministic HVI); shared MC samples make the gains comparable across steps."""
    Y = np.asarray(Y, float)
    n = len(Y)
    if n == 0:
        return []
    d = Y.shape[1]
    S = rng.random((n_samples, d))
    # per-candidate mask of which box samples it dominates (n x n_samples)
    Dmask = np.empty((n, n_samples), dtype=bool)
    for i in range(n):
        Dmask[i] = np.all(Y[i] >= S, axis=1)
    dom = np.zeros(n_samples, dtype=bool)
    chosen: list[int] = []
    for _ in range(min(K, n)):
        gains = (Dmask & ~dom).sum(axis=1).astype(float)
        gains[chosen] = -1.0
        i = int(gains.argmax())
        chosen.append(i)
        dom |= Dmask[i]
    return chosen


def pareto_front_nd(P) -> list[int]:
    """Indices of non-dominated rows of P (maximize ALL columns), any dimension."""
    P = np.asarray(P, float)
    n = len(P)
    keep = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if j != i and np.all(P[j] >= P[i]) and np.any(P[j] > P[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def ehvi_nominate_nd(mean, cov, K, rng, n_post: int = 48, n_hv: int = 3000) -> list[int]:
    """Greedy N-objective EHVI batch over a JOINT-GP posterior. `mean` (n,d) and
    `cov` (n,d,d) are the per-candidate posterior of the maximize-objectives,
    normalized so the relevant region is the unit box [0,1]^d (reference = origin).
    For each candidate, EHVI = expected fraction of box samples newly dominated,
    estimated by sampling N(mean_i, cov_i) (n_post draws) against fixed box samples
    (n_hv). Greedy: add the max-EHVI candidate, fantasize its MEAN into the front,
    repeat. This is the posterior-integrated EHVI (the 'E'), generalized to d>2."""
    mean = np.asarray(mean, float); cov = np.asarray(cov, float)
    n, d = mean.shape
    if n == 0:
        return []
    S = rng.random((n_hv, d))
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(d)[None])          # (n,d,d)
    z = rng.standard_normal((n, n_post, d))
    draws = mean[:, None, :] + np.einsum("nij,nkj->nki", L, z)    # (n,n_post,d)
    front_dom = np.zeros(n_hv, dtype=bool)
    chosen: list[int] = []
    for _ in range(min(K, n)):
        best, bi = -1.0, -1
        avail = front_dom.sum()  # noqa: F841 (kept for clarity)
        for i in range(n):
            if i in chosen:
                continue
            dommask = np.all(draws[i][:, None, :] >= S[None, :, :], axis=2)  # (n_post,n_hv)
            ehvi = float((dommask & ~front_dom[None, :]).mean())
            if ehvi > best:
                best, bi = ehvi, i
        chosen.append(bi)
        front_dom |= np.all(mean[bi] >= S, axis=1)                # fantasize at the mean
    return chosen


def mc_ehvi(mean, std, front, ref, rng, n_samples: int = 128, cov=None) -> np.ndarray:
    """Monte-Carlo Expected Hypervolume Improvement per candidate (maximize both
    objectives). mean: (n_cand, 2) posterior means. std: (n_cand, 2) marginal
    stds for INDEPENDENT-Gaussian sampling. cov: optional (n_cand, 2, 2) per-
    candidate covariance for CORRELATED sampling (joint GP) -- if given, draws are
    sampled from N(mean_i, cov_i) and `std` is ignored. front: current Pareto set
    (m, 2). Returns EHVI >= 0 per candidate.

    Vectorized: the 2-D hypervolume improvement of a point q=(a,b) over the front
    is the band integral  int_{ref_x}^{a} max(0, b - h(x)) dx,  where h(x) is the
    front's (non-increasing) upper-boundary step function. Precompute the bands
    once, evaluate all (sample x candidate) draws against them with numpy."""
    mean = np.asarray(mean, float)
    front = np.asarray(front, float).reshape(-1, 2)
    n = len(mean)
    z = rng.standard_normal((n_samples, n, 2))
    if cov is not None:
        cov = np.asarray(cov, float)
        # Cholesky per candidate (jitter for PSD safety); draws = mean + L z
        L = np.linalg.cholesky(cov + 1e-9 * np.eye(2)[None])      # (n,2,2)
        draws = mean[None] + np.einsum("nij,snj->sni", L, z)      # (S,n,2)
    else:
        std = np.asarray(std, float)
        draws = mean[None] + std[None] * z                       # (S,n,2)
    A = draws[..., 0]; B = draws[..., 1]                                      # (S,n)

    if len(front):
        pf = front[pareto_front(front)]
        pf = pf[(pf[:, 0] > ref[0]) & (pf[:, 1] > ref[1])]
    else:
        pf = np.empty((0, 2))
    if len(pf) == 0:                       # empty front: HVI = box area above ref
        imp = np.clip(A - ref[0], 0, None) * np.clip(B - ref[1], 0, None)
        return imp.mean(axis=0)

    pf = pf[np.argsort(pf[:, 0])]          # x ascending -> y descending
    fx, fy = pf[:, 0], pf[:, 1]
    lo = np.concatenate([[ref[0]], fx])    # band left edges   (m+1,)
    hi = np.concatenate([fx, [np.inf]])    # band right edges  (m+1,)
    h = np.concatenate([fy, [ref[1]]])     # band heights      (m+1,)

    out = np.zeros((n_samples, n))
    for k in range(len(h)):
        width = np.clip(np.minimum(hi[k], A) - lo[k], 0.0, None)
        height = np.clip(B - h[k], 0.0, None)
        out += width * height
    return out.mean(axis=0)
