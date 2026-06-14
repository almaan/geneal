# src/geneal/models/hedged_selection.py
from __future__ import annotations
import numpy as np


class HedgedSelect:
    """Select K targets by quality q with optional pathway-hedging.

    mode='greedy' : top-K by q (concentration-blind baseline).
    mode='cap'    : greedy by q but at most `cap` genes per pathway (hard hedge).
    mode='dpp'    : quality-tempered k-DPP with an external similarity S
                    (continuous mechanism graph). Provided for completeness; the
                    practical hedging lever is 'cap' (DPP frontier is flat in
                    sparse graphs — see CORUM_VALIDATION.md)."""

    def __init__(self, mode: str = "cap", cap: int = 2, beta: float = 1.0,
                 jitter: float = 1e-9):
        self.mode = mode
        self.cap = cap
        self.beta = beta
        self.jitter = jitter

    def select_idx(self, q, membership, K, S=None) -> list[int]:
        q = np.asarray(q, dtype=float)
        order = np.argsort(q)[::-1]
        if self.mode == "greedy":
            return order[:K].tolist()
        if self.mode == "cap":
            used: dict = {}
            chosen: list[int] = []
            for i in order:
                ps = membership.get(int(i), set())
                # a gene with no pathway is always allowed
                if ps and any(used.get(p, 0) >= self.cap for p in ps):
                    continue
                chosen.append(int(i))
                for p in ps:
                    used[p] = used.get(p, 0) + 1
                if len(chosen) == K:
                    break
            # if cap was too tight to reach K, fill with next-best leftovers
            if len(chosen) < K:
                for i in order:
                    if int(i) not in chosen:
                        chosen.append(int(i))
                        if len(chosen) == K:
                            break
            return chosen
        if self.mode == "dpp":
            from geneal.models.selection import _greedy_map_logdet
            qb = ((q - q.min()) / (q.max() - q.min() + 1e-9) + 1e-3) ** self.beta
            Sc = np.array(S, dtype=float).copy()
            np.fill_diagonal(Sc, 1.0)
            L = (qb[:, None] * Sc) * qb[None, :]
            L = (L + L.T) / 2 + self.jitter * np.eye(len(q))
            return _greedy_map_logdet(L, K)
        raise ValueError(self.mode)
