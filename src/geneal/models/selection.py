# src/geneal/models/selection.py
from __future__ import annotations
from typing import Sequence
import numpy as np

from geneal.models.kernels import posterior_correlation


class TopQGreedy:
    """Score all candidates once, take the q highest-scoring."""
    def select(self, *, candidate_idx: Sequence[int], X_candidates, mean, std,
               best, q, rng, surrogate, acquisition, X_train=None,
               y_train=None) -> list[int]:
        scores = acquisition.score(mean, std, best, rng)
        candidate_idx = list(candidate_idx)
        q = min(q, len(candidate_idx))
        order = np.argsort(scores)[::-1][:q]
        return [candidate_idx[i] for i in order]


class GreedyFantasy:
    """Sequential greedy batch with fantasized outcomes for diversity.

    Pick the best candidate, fantasize its outcome as the surrogate's predictive
    mean, refit a clone of the surrogate on the REAL training data plus the
    accumulated fantasies, rescore, repeat q times. Conditioning on real data is
    essential: refitting on fantasies alone collapses the GP to its prior (a
    single-point fit has zero target variance), which destroys the diversity
    signal. Encourages spread instead of q near-identical picks.
    """
    def select(self, *, candidate_idx: Sequence[int], X_candidates, mean, std,
               best, q, rng, surrogate, acquisition, X_train,
               y_train) -> list[int]:
        candidate_idx = list(candidate_idx)
        X_candidates = np.asarray(X_candidates)
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        q = min(q, len(candidate_idx))

        fant_X: list[np.ndarray] = []
        fant_y: list[float] = []

        remaining = list(range(len(candidate_idx)))
        chosen: list[int] = []
        cur_mean, cur_std = np.asarray(mean).copy(), np.asarray(std).copy()
        cur_best = best

        for _ in range(q):
            scores = acquisition.score(cur_mean[remaining], cur_std[remaining],
                                       cur_best, rng)
            local = remaining[int(np.argmax(scores))]
            chosen.append(candidate_idx[local])
            remaining.remove(local)
            if not remaining:
                break
            # Fantasize the outcome at the chosen point = its predictive mean.
            fant_X.append(X_candidates[local])
            fant_y.append(float(cur_mean[local]))
            cur_best = max(cur_best, fant_y[-1])
            # Refit working model on REAL data + fantasies, re-predict candidates.
            aug_X = np.vstack([X_train, np.asarray(fant_X)])
            aug_y = np.concatenate([y_train, np.asarray(fant_y)])
            work = surrogate.clone().fit(aug_X, aug_y)
            m, s = work.predict(X_candidates)
            cur_mean, cur_std = np.asarray(m), np.asarray(s)
        return chosen


class KDPP:
    """Quality-weighted k-DPP batch acquisition (the geneal method).

    Builds an L-ensemble kernel over candidates:  L = diag(q) S diag(q),
    where q_i = acquisition score (quality) and S_ij = surrogate posterior
    correlation (outcome similarity). Selecting the size-q subset that maximizes
    det(L_B) yields a batch that is simultaneously high-quality (large diagonal)
    and non-redundant (near-orthogonal rows). Because
    log det(posterior cov of a batch) is the Gaussian joint entropy, max-det is
    an information-theoretic batch acquisition with outcome-diversity built in.

    solver:
      "greedy"   -> greedy MAP: repeatedly add the candidate with the largest
                    marginal gain in log det(L_B). Deterministic, near-optimal.
      "sampling" -> exact k-DPP sampling via the eigendecomposition of L.
    """

    def __init__(self, solver: str = "greedy", jitter: float = 1e-9) -> None:
        if solver not in ("greedy", "sampling"):
            raise ValueError("solver must be 'greedy' or 'sampling'")
        self.solver = solver
        self.jitter = jitter

    def _build_L(self, *, mean, std, best, rng, surrogate, acquisition,
                 X_candidates):
        q = np.asarray(acquisition.score(mean, std, best, rng), dtype=float)
        q = q - q.min() + 1e-6 if q.min() < 0 else q + 1e-6
        _, cov = surrogate.predict_cov(np.asarray(X_candidates))
        S = posterior_correlation(cov)
        L = (q[:, None] * S) * q[None, :]
        L = (L + L.T) / 2 + self.jitter * np.eye(L.shape[0])
        return L

    def select(self, *, candidate_idx, X_candidates, mean, std, best, q, rng,
               surrogate, acquisition, X_train, y_train) -> list[int]:
        candidate_idx = list(candidate_idx)
        n = len(candidate_idx)
        q = min(q, n)
        L = self._build_L(mean=mean, std=std, best=best, rng=rng,
                          surrogate=surrogate, acquisition=acquisition,
                          X_candidates=X_candidates)
        if self.solver == "greedy":
            local = _greedy_map_logdet(L, q)
        else:
            local = _sample_kdpp(L, q, rng)
        return [candidate_idx[i] for i in local]


def _greedy_map_logdet(L: np.ndarray, k: int) -> list[int]:
    """Greedy MAP for max log det(L_S), |S|=k. O(k^2 n). Cholesky-style update."""
    n = L.shape[0]
    selected: list[int] = []
    d2 = np.diag(L).astype(float).copy()
    c = np.zeros((n, n))
    for _ in range(min(k, n)):
        j = int(np.argmax(np.where(d2 > 0, d2, -np.inf)))
        if d2[j] <= 0:
            leftover = [i for i in range(n) if i not in selected]
            selected.extend(leftover[: k - len(selected)])
            break
        selected.append(j)
        if len(selected) == min(k, n):
            break
        ci = (L[:, j] - c[:, :len(selected) - 1] @ c[j, :len(selected) - 1]) / np.sqrt(d2[j])
        c[:, len(selected) - 1] = ci
        d2 = d2 - ci ** 2
        d2[selected] = -np.inf
    return selected[:k]


def _sample_kdpp(L: np.ndarray, k: int, rng) -> list[int]:
    """Exact k-DPP sampling via eigendecomposition (Kulesza & Taskar)."""
    vals, vecs = np.linalg.eigh(L)
    vals = np.clip(vals, 0.0, None)
    n = L.shape[0]
    k = min(k, n)
    E = np.zeros((k + 1, n + 1))
    E[0, :] = 1.0
    for l in range(1, k + 1):
        for m in range(1, n + 1):
            E[l, m] = E[l, m - 1] + vals[m - 1] * E[l - 1, m - 1]
    chosen_vecs = []
    l = k
    for m in range(n, 0, -1):
        if l == 0:
            break
        if E[l, m] <= 0:
            continue
        if rng.random() < vals[m - 1] * E[l - 1, m - 1] / E[l, m]:
            chosen_vecs.append(m - 1)
            l -= 1
    V = vecs[:, chosen_vecs]
    selected: list[int] = []
    for _ in range(len(chosen_vecs)):
        probs = (V ** 2).sum(axis=1)
        probs = probs / probs.sum()
        i = int(rng.choice(n, p=probs))
        selected.append(i)
        if V.shape[1] == 1:
            V = np.zeros((n, 0))
            continue
        col = np.argmax(np.abs(V[i, :]))
        Vj = V[:, col].copy()
        V = np.delete(V, col, axis=1)
        V = V - np.outer(Vj, V[i, :] / Vj[i])
        Q, _ = np.linalg.qr(V)
        V = Q
    return sorted(set(selected))[:k] if selected else list(range(k))


class CoreSet:
    """Greedy farthest-point (k-center) selection in embedding space.

    Diversity-only baseline representing IterPert's "greedy distance
    maximization" selection rule. Quality (acquisition) is ignored — each pick is
    the candidate maximally far (Euclidean, in embedding space) from the already
    chosen set, seeded by the existing training points.
    """
    def select(self, *, candidate_idx, X_candidates, mean, std, best, q, rng,
               surrogate, acquisition, X_train, y_train) -> list[int]:
        candidate_idx = list(candidate_idx)
        Xc = np.asarray(X_candidates, dtype=float)
        n = len(candidate_idx)
        q = min(q, n)
        anchors = np.asarray(X_train, dtype=float)
        if anchors.ndim == 1:
            anchors = anchors.reshape(1, -1)
        min_d = np.full(n, np.inf)
        if len(anchors):
            min_d = np.min(np.linalg.norm(Xc[:, None, :] - anchors[None, :, :], axis=2), axis=1)
        chosen: list[int] = []
        for _ in range(q):
            j = int(np.argmax(min_d))
            chosen.append(j)
            min_d = np.minimum(min_d, np.linalg.norm(Xc - Xc[j], axis=1))
            min_d[chosen] = -np.inf
        return [candidate_idx[j] for j in chosen]


class TypiClust:
    """Typicality-in-clusters selection (Hacohen et al.), IterPert's best baseline.

    Cluster candidates (k-means, k=q) in embedding space; from each cluster pick
    the most "typical" point (highest local density = smallest mean distance to
    its K nearest neighbours within the cluster). Diversity-only, quality-free.
    """
    def __init__(self, n_neighbors: int = 5) -> None:
        self.n_neighbors = n_neighbors

    def select(self, *, candidate_idx, X_candidates, mean, std, best, q, rng,
               surrogate, acquisition, X_train, y_train) -> list[int]:
        candidate_idx = list(candidate_idx)
        Xc = np.asarray(X_candidates, dtype=float)
        n = len(candidate_idx)
        q = min(q, n)
        labels = _kmeans_labels(Xc, q, rng)
        chosen: list[int] = []
        for c in range(q):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            chosen.append(int(members[_most_typical(Xc[members], self.n_neighbors)]))
        if len(chosen) < q:
            remaining = [i for i in range(n) if i not in chosen]
            chosen.extend(remaining[: q - len(chosen)])
        return [candidate_idx[j] for j in chosen[:q]]


def _kmeans_labels(X: np.ndarray, k: int, rng, n_iter: int = 25) -> np.ndarray:
    """Minimal seeded k-means (Lloyd). Returns cluster label per row."""
    n = X.shape[0]
    k = min(k, n)
    centers = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new = d.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            pts = X[labels == c]
            if len(pts):
                centers[c] = pts.mean(0)
    return labels


def _most_typical(X: np.ndarray, n_neighbors: int) -> int:
    """Index of the densest point: smallest mean distance to its K nearest."""
    m = X.shape[0]
    if m == 1:
        return 0
    K = min(n_neighbors, m - 1)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    knn_mean = np.sort(D, axis=1)[:, :K].mean(axis=1)
    return int(np.argmin(knn_mean))
