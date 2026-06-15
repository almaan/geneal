# src/geneal/runner/ablation.py
"""Default-ablation engine (Plan 7): two analyses on real DepMap.

Two distinct stages, deliberately NOT entangled into one grid:

  Analysis A (safety vs efficacy): one axis, the safety rule applied at
    nomination -- none / truncation (known-toxicity ceiling) / ehvi
    (learned-toxicity ceiling, fed by a dual-objective EHVI acquisition) -- plus
    the random / coreset / typiclust acquisition baselines.

  Analysis B (diversity / robustness): the diversity OPERATORS -- none / cap
    (per-pathway, CORUM membership) / kdpp (quality-weighted k-DPP with a
    STRING-derived similarity S) -- layered on top of a base nomination. Capping
    is a bolt-on-any-method operator; it needs only the graph.

Key reuse: GPRSurrogate (efficacy/toxicity GPs), BivariateALRunner (EHVI loop),
CoreSet/TypiClust (IterPert-style baselines), HedgedSelect.select_idx (the three
diversity operators), portfolio + AlphaNDCG metrics. STRING is a *similarity*,
never a prediction embedding: PubMedBERT predicts; STRING/CORUM structure the
diversity step."""
from __future__ import annotations
from pathlib import Path
import numpy as np

from geneal.models.surrogate import GPRSurrogate
from geneal.models.hedged_selection import HedgedSelect
from geneal.models.selection import CoreSet, TypiClust
from geneal.runner.bivariate import BivariateALRunner
from geneal.metrics.portfolio import (pathway_concentration, dropout_robustness,
                                      n_pathways_covered)
from geneal.metrics.panel import AlphaNDCG

# Generic names for representative prior-work acquisitions (not exact
# reimplementations):
#   'greedy'   -- quality/effect-driven (UCB on predicted efficacy); the
#                 quality-only family (e.g. NAIAD).
#   'farthest' -- farthest-point / coverage diversity (quality-blind).
#   'cluster'  -- cluster-representative diversity (quality-blind).
#   'info_div' -- INFORMATIVE-DIVERSE batch: quality-weighted k-DPP at acquisition
#                 (GP-UCB quality x embedding-cosine diversity). The faithful
#                 stand-in for informativeness+diversity AL (e.g. IterPert).
ACQUISITIONS = ("random", "greedy", "farthest", "cluster", "info_div", "ehvi")
_NEG = -1e9


class _ZeroNoise:
    """Observation noise for the EHVI loop: noiseless assays (ablation default)."""
    def draw(self, n, rng):
        return np.zeros(n)


def _default_factory():
    return GPRSurrogate(n_iters=120)


# --------------------------------------------------------------------------- #
# Acquisition (what to assay each round)                                       #
# --------------------------------------------------------------------------- #
def _ei(mean, std, incumbent):
    """Expected Improvement over the incumbent (maximization)."""
    from scipy.stats import norm
    mean = np.asarray(mean, float); std = np.clip(np.asarray(std, float), 1e-12, None)
    z = (mean - incumbent) / std
    return (mean - incumbent) * norm.cdf(z) + std * norm.pdf(z)


def acquire(kind, X, eff, revealed, cand, batch, rng, surr_factory, score="ucb"):
    """One round of a single-objective acquisition. Returns `batch` new global
    indices drawn from `cand` (the unrevealed set). `score` selects the
    quality-aware rule for kind='greedy': 'ucb' (mu+2 sigma) or 'ei' (expected
    improvement). EHVI is handled by run_acquisition (it is intrinsically
    bivariate)."""
    X = np.asarray(X, float)
    cand = list(cand)
    batch = min(batch, len(cand))
    if kind == "random":
        return [int(i) for i in rng.choice(cand, size=batch, replace=False)]
    if kind == "greedy":
        eff = np.asarray(eff, float)
        s = surr_factory().fit(X[revealed], eff[revealed])
        m, sd = s.predict(X[cand])
        if score == "ei":
            a = _ei(m, sd, float(eff[revealed].max()))
        else:
            a = np.asarray(m) + 2.0 * np.asarray(sd)
        order = np.argsort(a)[::-1][:batch]
        return [int(cand[i]) for i in order]
    if kind in ("farthest", "cluster"):
        sel = (CoreSet() if kind == "farthest" else TypiClust())
        local = sel.select(candidate_idx=list(range(len(cand))),
                           X_candidates=X[cand], mean=None, std=None, best=None,
                           q=batch, rng=rng, surrogate=None, acquisition=None,
                           X_train=X[revealed], y_train=np.asarray(eff)[revealed])
        return [int(cand[i]) for i in local]
    if kind == "info_div":
        # informative-diverse batch: quality-weighted k-DPP at acquisition time.
        # quality = UCB/EI; diversity = dense embedding-cosine S over candidates.
        from geneal.models.selection import _greedy_map_logdet
        eff = np.asarray(eff, float)
        s = surr_factory().fit(X[revealed], eff[revealed])
        m, sd = s.predict(X[cand])
        if score == "ei":
            q = _ei(m, sd, float(eff[revealed].max()))
        else:
            q = np.asarray(m) + 2.0 * np.asarray(sd)
        q = np.asarray(q, float); q = q - q.min() + 1e-6
        Sc = build_embedding_S(X[cand])
        L = (q[:, None] * Sc) * q[None, :]
        L = (L + L.T) / 2 + 1e-9 * np.eye(len(q))
        local = _greedy_map_logdet(L, batch)
        return [int(cand[i]) for i in local]
    raise ValueError(f"unknown acquisition {kind!r}")


def run_acquisition(kind, X, eff, tox, n_init, n_rounds, batch, seed,
                    surr_factory=None, ehvi_samples=32, shortlist=150,
                    noise=None, score="ucb", joint_factory=None):
    """Full active-learning loop for one acquisition. Returns the final revealed
    global indices. Common random numbers: every kind seeds its initial set from
    default_rng(seed).permutation, so kinds are paired per (line, seed)."""
    surr_factory = surr_factory or _default_factory
    X = np.asarray(X, float)
    eff = np.asarray(eff, float); tox = np.asarray(tox, float)
    n = len(eff)
    if kind == "ehvi":
        runner = BivariateALRunner(surr_factory, noise or _ZeroNoise(),
                                   ehvi_samples=ehvi_samples, shortlist=shortlist,
                                   joint_factory=joint_factory)
        hist = runner.run(X, eff, tox, n_init, n_rounds, batch, seed)
        return [int(i) for i in hist[-1]["revealed"]]
    rng = np.random.default_rng(seed)
    acq_rng = np.random.default_rng((seed, 99))
    revealed = [int(i) for i in rng.permutation(n)[:n_init]]
    for _ in range(n_rounds):
        rset = set(revealed)
        cand = [i for i in range(n) if i not in rset]
        if not cand:
            break
        revealed += acquire(kind, X, eff, revealed, cand, batch, acq_rng,
                            surr_factory, score=score)
    return revealed


# --------------------------------------------------------------------------- #
# Nomination (final shortlist): safety filter, then diversity operator         #
# --------------------------------------------------------------------------- #
_DIV_MODE = {"none": "greedy", "cap": "cap", "kdpp": "dpp"}


def nominate(revealed, X, eff, tox, membership, K, safety, diversity, tau,
             S=None, surr_factory=None, cap=2, pool=200, joint_factory=None):
    """Nominate K targets. Fit a final efficacy GP on revealed labels, predict
    genome-wide; apply the safety filter; then the diversity operator over the
    top-`pool` eligible candidates (none=top-K, cap=per-pathway, kdpp=k-DPP on S).

    Safety filter source (`safety`):
      'none'  -> no filter.
      'known' -> filter on the KNOWN toxicity (a-priori annotation; the oracle
                 ceiling). Used by truncation_known.
      'pred'  -> fit a toxicity GP on revealed toxicity labels, filter on the
                 PREDICTED toxicity (the learned regime). Used by truncation_pred
                 and the EHVI method.
    `tau` is a QUANTILE in [0,1] (scale-free, comparable across toxicity
    definitions): keep genes whose toxicity is at/below the tau-quantile of the
    relevant toxicity distribution (tau=0.5 = 'the safest half')."""
    surr_factory = surr_factory or _default_factory
    X = np.asarray(X, float)
    eff = np.asarray(eff, float); tox = np.asarray(tox, float)
    n = len(eff)

    if safety == "pred" and joint_factory is not None:
        # joint (multitask) GP: efficacy + toxicity share strength. q (efficacy)
        # and the toxicity filter both come from the one correlated fit.
        m2 = np.asarray(joint_factory().fit(
            X[revealed], np.column_stack([eff[revealed], tox[revealed]])).predict(X)[0])
        q = m2[:, 0].astype(float).copy()
        m_tox = m2[:, 1]
        thr = float(np.quantile(m_tox, tau))
        q[m_tox > thr] = _NEG
    else:
        q = np.asarray(surr_factory().fit(X[revealed], eff[revealed]).predict(X)[0],
                       dtype=float).copy()
        if safety == "known":
            thr = float(np.quantile(tox, tau))
            q[tox > thr] = _NEG
        elif safety == "pred":
            m_tox = np.asarray(surr_factory().fit(X[revealed], tox[revealed]).predict(X)[0])
            thr = float(np.quantile(m_tox, tau))
            q[m_tox > thr] = _NEG
        elif safety != "none":
            raise ValueError(f"unknown safety {safety!r}")

    # restrict the diversity operator to the top-`pool` eligible by quality:
    # bounds k-DPP cost (O(K^2 pool)) and diversifies only among high-efficacy genes.
    eligible = int(np.sum(q > _NEG / 2))
    pool_n = min(pool, max(eligible, K))
    order = np.argsort(q)[::-1][:pool_n]
    qp = q[order]
    memp = {li: membership.get(int(order[li]), set()) for li in range(len(order))}
    Sp = S[np.ix_(order, order)] if (S is not None and diversity == "kdpp") else None

    mode = _DIV_MODE[diversity]
    local = HedgedSelect(mode=mode, cap=cap).select_idx(qp, memp, K=K, S=Sp)
    return [int(order[li]) for li in local]


# --------------------------------------------------------------------------- #
# Evaluate (true-value metrics for a nominated portfolio)                      #
# --------------------------------------------------------------------------- #
def evaluate(pick, eff, tox, membership, X):
    """TRUE-value metrics for a nominated set. eff/tox are ground truth; X the
    PubMedBERT panel embeddings (for alpha-NDCG nugget clustering)."""
    eff = np.asarray(eff, float); tox = np.asarray(tox, float)
    pick = list(pick)
    # alpha-NDCG over the picks ordered by true efficacy (deterministic)
    order = sorted(pick, key=lambda g: -eff[g])
    andcg = AlphaNDCG(k=len(pick) or 1).evaluate(order, eff, np.asarray(X, float))
    return {
        "mean_efficacy": float(np.mean(eff[pick])),
        "max_efficacy": float(np.max(eff[pick])),
        "mean_toxicity": float(np.mean(tox[pick])),
        "concentration": float(pathway_concentration(pick, membership)),
        # robustness is a fraction-of-VALUE-surviving; value must be non-negative
        # (raw lethality can go negative for growth-promoting knockouts), else the
        # ratios blow past 1. Clip at 0.
        "robustness": float(dropout_robustness(pick, membership, np.clip(eff, 0.0, None))),
        "n_pathways": int(n_pathways_covered(pick, membership)),
        "alpha_ndcg": float(andcg),
    }


# --------------------------------------------------------------------------- #
# Diversity similarity S for the k-DPP operator                                #
#                                                                              #
# S is a MECHANISM/REPRESENTATION similarity, NOT outcome similarity: we want  #
# the picks to share the (high-efficacy) outcome, so diversity must live in a  #
# space other than the outcome. Embedding cosine is dense (tunable); STRING is #
# biological but sparse.                                                       #
# --------------------------------------------------------------------------- #
def build_embedding_S(X):
    """Dense (n,n) similarity from PubMedBERT embeddings: S_ij = (1+cos)/2 in
    [0,1]. Unit diagonal. Dense -> the k-DPP diversity term varies smoothly, so
    the quality/diversity trade-off is genuinely tunable (unlike the sparse
    STRING graph). This is the DEFAULT k-DPP similarity."""
    X = np.asarray(X, float)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.clip(norm, 1e-12, None)
    cos = Xn @ Xn.T
    S = 0.5 * (1.0 + np.clip(cos, -1.0, 1.0))
    np.fill_diagonal(S, 1.0)
    return S



def build_string_S(gene_names,
                   edges_path="data/processed/depmap/string_edges_all.parquet"):
    """Dense (n,n) STRING combined-score similarity over a gene panel, built from
    the genome-wide edge list (entrez_i, entrez_j, weight in 0..1). Symmetric,
    unit diagonal; gene pairs with no STRING edge -> 0; genes absent from STRING
    are isolated (only self-similar). STRING is a similarity, NOT an embedding."""
    from geneal.data.depmap import parse_entrez
    ents = [parse_entrez(g) for g in gene_names]
    pos = {int(e): i for i, e in enumerate(ents) if e is not None}
    n = len(gene_names)
    S = np.eye(n, dtype=float)
    if not Path(edges_path).exists():
        return S
    import pandas as pd
    e = pd.read_parquet(edges_path)
    a = e["entrez_i"].to_numpy(); b = e["entrez_j"].to_numpy()
    w = e["weight"].to_numpy(dtype=float)
    for ei, ej, wij in zip(a, b, w):
        i = pos.get(int(ei)); j = pos.get(int(ej))
        if i is None or j is None:
            continue
        S[i, j] = wij
        S[j, i] = wij
    return S
