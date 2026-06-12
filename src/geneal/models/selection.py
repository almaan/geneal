# src/geneal/models/selection.py
from __future__ import annotations
from typing import Sequence
import numpy as np


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
