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


def mc_ehvi(mean, std, front, ref, rng, n_samples: int = 128) -> np.ndarray:
    """Monte-Carlo Expected Hypervolume Improvement per candidate (maximize both
    objectives). mean/std: (n_cand, 2) independent-Gaussian posteriors. front:
    current Pareto set (m, 2). Returns EHVI >= 0 per candidate.

    Vectorized: the 2-D hypervolume improvement of a point q=(a,b) over the front
    is the band integral  int_{ref_x}^{a} max(0, b - h(x)) dx,  where h(x) is the
    front's (non-increasing) upper-boundary step function. Precompute the bands
    once, evaluate all (sample x candidate) draws against them with numpy."""
    mean = np.asarray(mean, float); std = np.asarray(std, float)
    front = np.asarray(front, float).reshape(-1, 2)
    n = len(mean)
    draws = mean[None] + std[None] * rng.standard_normal((n_samples, n, 2))  # (S,n,2)
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
