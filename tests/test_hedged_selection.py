# tests/test_hedged_selection.py
import numpy as np
from geneal.models.hedged_selection import HedgedSelect


def test_capped_selection_limits_per_pathway():
    # 6 genes, q descending; genes 0,1,2 in pathway A, 3,4,5 singletons.
    membership = {0: {"A"}, 1: {"A"}, 2: {"A"}, 3: {"B"}, 4: {"C"}, 5: {"D"}}
    q = np.array([0.9, 0.85, 0.8, 0.4, 0.3, 0.2])
    sel = HedgedSelect(mode="cap", cap=1).select_idx(q, membership, K=3)
    # cap=1 per pathway: can take only 1 from A -> picks gene0(A), then 3(B),4(C)
    assert 0 in sel
    assert not (1 in sel and 2 in sel)  # never 2 from pathway A beyond cap
    assert len(sel) == 3


def test_greedy_mode_ignores_pathways():
    membership = {i: {"A"} for i in range(6)}
    q = np.arange(6.0)[::-1]
    sel = HedgedSelect(mode="greedy").select_idx(q, membership, K=3)
    assert sorted(sel) == [0, 1, 2]  # top-3 by q regardless of pathway
