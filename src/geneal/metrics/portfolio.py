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
