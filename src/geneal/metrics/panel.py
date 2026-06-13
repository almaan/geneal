# src/geneal/metrics/panel.py
from __future__ import annotations
from typing import Sequence
import numpy as np


class MaxValue:
    """Best (highest) true target among revealed genes, normalized by the global
    max so it lands in a comparable range. The 'best lethal value found so far'
    curve — the classic simple-regret-style readout."""
    @property
    def name(self) -> str:
        return "max_value"

    def evaluate(self, revealed_order, target, embeddings) -> float:
        target = np.asarray(target)
        if len(revealed_order) == 0:
            return 0.0
        best = float(np.max(target[list(revealed_order)]))
        denom = max(abs(float(np.max(target))), 1e-8)
        return best / denom


class CumulativeDiversity:
    """Mean pairwise Euclidean distance of the cumulative revealed set in
    embedding space. How spread the collected genes are over time."""
    @property
    def name(self) -> str:
        return "diversity"

    def evaluate(self, revealed_order, target, embeddings) -> float:
        idx = list(revealed_order)
        if len(idx) < 2:
            return 0.0
        X = np.asarray(embeddings)[idx]
        n = len(idx)
        total, cnt = 0.0, 0
        for i in range(n):
            diff = X[i + 1:] - X[i]
            if len(diff):
                total += float(np.sum(np.linalg.norm(diff, axis=1)))
                cnt += len(diff)
        return total / cnt if cnt else 0.0


class AlphaNDCG:
    """alpha-NDCG over the acquisition-ordered revealed list (Clarke et al. 2008).

    Nuggets = k-means clusters of gene embeddings; a gene 'covers' its cluster
    with graded relevance = max(0, lethality). The gain of revealing a gene is
    discounted by (1-alpha)^(# previously-revealed genes in the same cluster), so
    piling into one lethal cluster is penalized and covering many distinct lethal
    clusters is rewarded. Normalized by a greedy 'ideal' ordering -> [0, 1].

    Clusters are computed once per (embeddings) call from ALL genes so the nugget
    structure is fixed; this is a fast k-means (numpy)."""
    def __init__(self, k: int = 50, alpha: float = 0.5, n_clusters: int = 20,
                 seed: int = 0) -> None:
        self.k = k
        self.alpha = alpha
        self.n_clusters = n_clusters
        self.seed = seed

    @property
    def name(self) -> str:
        return f"alpha_ndcg@{self.k}"

    def _labels(self, embeddings):
        X = np.asarray(embeddings, dtype=float)
        n = X.shape[0]
        kk = min(self.n_clusters, n)
        rng = np.random.default_rng(self.seed)
        centers = X[rng.choice(n, size=kk, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(25):
            d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            new = d.argmin(1)
            if np.array_equal(new, labels):
                break
            labels = new
            for c in range(kk):
                pts = X[labels == c]
                if len(pts):
                    centers[c] = pts.mean(0)
        return labels

    def _dcg(self, order, labels, rel):
        seen: dict[int, int] = {}
        dcg = 0.0
        for i, g in enumerate(order[: self.k]):
            c = int(labels[g])
            r = seen.get(c, 0)
            gain = rel[g] * ((1.0 - self.alpha) ** r)
            dcg += gain / np.log2(i + 2)
            seen[c] = r + 1
        return dcg

    def evaluate(self, revealed_order, target, embeddings) -> float:
        order = list(revealed_order)
        if not order:
            return 0.0
        labels = self._labels(embeddings)
        rel = np.clip(np.asarray(target, dtype=float), 0.0, None)
        actual = self._dcg(order, labels, rel)
        # ideal: greedily order the revealed genes to maximize alpha-discounted gain
        ideal_order, seen, pool = [], {}, list(order)
        while pool and len(ideal_order) < self.k:
            best_g, best_gain = None, -np.inf
            for g in pool:
                c = int(labels[g])
                gain = rel[g] * ((1.0 - self.alpha) ** seen.get(c, 0))
                if gain > best_gain:
                    best_gain, best_g = gain, g
            ideal_order.append(best_g)
            seen[int(labels[best_g])] = seen.get(int(labels[best_g]), 0) + 1
            pool.remove(best_g)
        ideal = self._dcg(ideal_order, labels, rel)
        return float(actual / ideal) if ideal > 0 else 0.0


def evaluate_panel(panel, revealed_order, target, embeddings) -> dict:
    """Evaluate every metric in the panel on the cumulative revealed set."""
    return {m.name: float(m.evaluate(revealed_order, target, embeddings))
            for m in panel}
