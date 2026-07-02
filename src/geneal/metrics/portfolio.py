# src/geneal/metrics/portfolio.py
from __future__ import annotations
from typing import Sequence
import numpy as np


def _pathways_of(sel, membership):
    return {g: membership.get(g, set()) for g in sel}


def n_pathways_covered(sel: Sequence[int], membership: dict) -> int:
    """Number of DISTINCT pathways/complexes the selected genes touch."""
    paths = set()
    for g in sel:
        paths |= membership.get(g, set())
    return len(paths)


def pathway_concentration(sel: Sequence[int], membership: dict) -> float:
    """Max fraction of the (pathway-annotated) portfolio in any single pathway.

    1.0 = all annotated picks in one pathway (max concentration risk);
    low = spread across pathways. Genes with no pathway are excluded from the
    denominator (can't assess their pathway risk)."""
    counts: dict = {}
    annotated = 0
    for g in sel:
        ps = membership.get(g, set())
        if not ps:
            continue
        annotated += 1
        for p in ps:
            counts[p] = counts.get(p, 0) + 1
    if annotated == 0:
        return 0.0
    return max(counts.values()) / annotated


def dropout_robustness(sel: Sequence[int], membership: dict,
                       value: np.ndarray) -> float:
    """Expected fraction of portfolio VALUE that survives if a single random
    pathway is eliminated (e.g. found toxic/undruggable).

    For each pathway present, dropping it removes all selected genes in it;
    robustness = 1 - mean over present pathways of (lost value / total value).
    Higher = more robust to pathway-level failure. Genes with no pathway always
    survive (no pathway risk)."""
    sel = list(sel)
    value = np.asarray(value, dtype=float)
    total = float(value[sel].sum())
    if total <= 0:
        return 0.0
    present: dict = {}
    for g in sel:
        for p in membership.get(g, set()):
            present.setdefault(p, 0.0)
            present[p] += float(value[g])
    if not present:
        return 1.0  # nothing annotated -> nothing to drop
    lost_fracs = [v / total for v in present.values()]
    return 1.0 - float(np.mean(lost_fracs))


def portfolio_risk(picks, S) -> float:
    """Equal-weight portfolio variance of the selected set under similarity S.

    R = wᵀ S w with w = 1/K over the selected submatrix:
        R = 1/K + (1/K²) · Σ_{i≠j in picks} S_ij.
    Markowitz diversification: if each gene carries some adverse-event risk
    correlated across genes by S, R is the variance of equal-weight portfolio
    loss. A diverse set (off-diagonals ~0) -> R ≈ 1/K (floor, all-idiosyncratic);
    a concentrated set (many similar picks) -> high R (up to 1.0 if all S_ij=1).
    Lower = better hedged. S should have unit diagonal and off-diagonal in [0,1]."""
    picks = list(picks)
    K = len(picks)
    if K == 0:
        return 0.0
    Sub = np.asarray(S, dtype=float)[np.ix_(picks, picks)]
    w = np.full(K, 1.0 / K)
    return float(w @ Sub @ w)


def effective_bets(picks, S) -> float:
    """Effective number of independent bets in the selected set under S:
        N_eff = K² / (1ᵀ S 1)  over the selected submatrix.
    Range [1, K]: K when the picks are mutually dissimilar (S=I, fully
    diversified), 1 when all mutually identical (S all-ones, one real bet).
    The interpretable companion to `portfolio_risk` (N_eff = 1 / R when w=1/K)."""
    picks = list(picks)
    K = len(picks)
    if K == 0:
        return 0.0
    Sub = np.asarray(S, dtype=float)[np.ix_(picks, picks)]
    denom = float(Sub.sum())
    if denom <= 0:
        return float(K)
    return float(K * K / denom)


def dropout_curve(sel, membership, value, max_drop: int = 5):
    """Worst-case value-of-diversity curve: fraction of portfolio VALUE retained
    when the d most-valuable pathways fail (d = 0..max_drop). A gene's value is
    split across the complexes it belongs to; unannotated genes always survive.
    A diversified portfolio loses less when its top pathways drop. Returns a list
    of length max_drop+1, retained[0]=1.0, monotone non-increasing."""
    sel = list(sel)
    v = np.clip(np.asarray(value, dtype=float), 0.0, None)
    total = float(v[sel].sum())
    if total <= 0:
        return [1.0] + [0.0] * max_drop
    pathval: dict = {}
    for g in sel:
        ps = membership.get(g, set())
        if not ps:
            continue
        for p in ps:
            pathval[p] = pathval.get(p, 0.0) + float(v[g]) / len(ps)
    ranked = sorted(pathval.values(), reverse=True)
    retained, lost = [1.0], 0.0
    for d in range(1, max_drop + 1):
        if d <= len(ranked):
            lost += ranked[d - 1]
        retained.append(max(0.0, (total - lost) / total))
    return retained
