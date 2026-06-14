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
    current Pareto set (m, 2). Returns EHVI >= 0 per candidate."""
    mean = np.asarray(mean, float); std = np.asarray(std, float)
    front = np.asarray(front, float).reshape(-1, 2)
    base = hypervolume2d(front, ref)
    n = len(mean)
    out = np.zeros(n)
    for s in range(n_samples):
        draw = mean + std * rng.standard_normal(mean.shape)  # (n,2)
        for i in range(n):
            hv = hypervolume2d(np.vstack([front, draw[i]]), ref)
            out[i] += max(hv - base, 0.0)
    return out / n_samples
