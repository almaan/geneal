# tests/test_portfolio.py
import numpy as np
from geneal.metrics.portfolio import (n_pathways_covered, pathway_concentration,
                                       dropout_robustness)

# gene -> set of pathway ids
MEMBERSHIP = {0: {"A"}, 1: {"A"}, 2: {"A"}, 3: {"B"}, 4: {"C"}, 5: set()}


def test_n_pathways_covered():
    # genes 0,1,2 in A; 3 in B; 4 in C -> 3 pathways (gene 5 has none)
    assert n_pathways_covered([0, 1, 2, 3, 4, 5], MEMBERSHIP) == 3


def test_pathway_concentration_max_fraction():
    # 0,1,2,3 -> A has 3 of 4 genes-with-pathways -> 3/4 = 0.75
    assert np.isclose(pathway_concentration([0, 1, 2, 3], MEMBERSHIP), 0.75)


def test_pathway_concentration_spread_is_low():
    # 2(A),3(B),4(C) -> each pathway 1/3
    assert np.isclose(pathway_concentration([2, 3, 4], MEMBERSHIP), 1/3)


def test_dropout_robustness_counts_survivors():
    # value per gene; if pathway A is dropped (toxic), genes 0,1,2 lost.
    # robustness = expected fraction of portfolio VALUE surviving a random
    # single-pathway dropout. With pathways {A:3 genes, B:1, C:1}, dropping A
    # (prob 1/3) loses 3/5 genes; B or C loses 1/5 each. Equal-value genes:
    # expected survivors = 1 - mean over pathways of (genes_in_pathway/total)
    sel = [0, 1, 2, 3, 4]  # gene 5 excluded (no pathway)
    val = np.ones(6)
    r = dropout_robustness(sel, MEMBERSHIP, val)
    # pathways A(3),B(1),C(1); total value 5; drop A->lose3, B->lose1, C->lose1
    # expected surviving fraction = 1 - mean(3/5,1/5,1/5) = 1 - (5/15)=1-0.333=0.667
    assert np.isclose(r, 0.667, atol=1e-2)


def test_portfolio_risk_concentrated_vs_diverse():
    from geneal.metrics.portfolio import portfolio_risk
    # 3 picks. Concentrated: all mutually similar (S=1 off-diag).
    Sc = np.ones((3, 3))
    # equal-weight variance w^T S w, w=1/3 -> all-ones -> 1.0
    assert np.isclose(portfolio_risk([0, 1, 2], Sc), 1.0)
    # Diverse: orthogonal (identity) -> 1/K floor = 1/3
    Sd = np.eye(3)
    assert np.isclose(portfolio_risk([0, 1, 2], Sd), 1 / 3)
    # Partial: one similar pair among three -> between floor and 1
    Sp = np.eye(4)
    Sp[0, 1] = Sp[1, 0] = 1.0
    r = portfolio_risk([0, 1, 2], Sp)        # picks 0,1 similar; 2 idiosyncratic
    # 1/3 + (1/9)*sum_offdiag(=2) = 1/3 + 2/9
    assert np.isclose(r, 1 / 3 + 2 / 9)


def test_portfolio_risk_subsets_by_picks():
    from geneal.metrics.portfolio import portfolio_risk
    S = np.eye(5)
    S[3, 4] = S[4, 3] = 1.0
    # picking the idiosyncratic 0,1,2 -> floor 1/3 regardless of the 3-4 pair
    assert np.isclose(portfolio_risk([0, 1, 2], S), 1 / 3)


def test_effective_bets_range():
    from geneal.metrics.portfolio import effective_bets
    # fully diverse (identity) -> K independent bets
    assert np.isclose(effective_bets([0, 1, 2], np.eye(3)), 3.0)
    # fully concentrated (all-ones) -> 1 effective bet
    assert np.isclose(effective_bets([0, 1, 2], np.ones((3, 3))), 1.0)
