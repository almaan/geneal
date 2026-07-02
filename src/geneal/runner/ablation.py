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
                                      n_pathways_covered, portfolio_risk,
                                      effective_bets)

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


def acquire(kind, X, eff, revealed, cand, batch, rng, surr_factory, score="ucb",
            info_pool=300):
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
        # SHORTLIST to the top-`info_pool` by quality before building the (N^2)
        # similarity, so this stays cheap at genome scale.
        from geneal.models.selection import _greedy_map_logdet
        eff = np.asarray(eff, float)
        s = surr_factory().fit(X[revealed], eff[revealed])
        m, sd = s.predict(X[cand])
        if score == "ei":
            qa = _ei(m, sd, float(eff[revealed].max()))
        else:
            qa = np.asarray(m) + 2.0 * np.asarray(sd)
        qa = np.asarray(qa, float)
        if len(cand) > info_pool:
            keep = np.argsort(qa)[::-1][:info_pool]
            cand = [cand[i] for i in keep]; qa = qa[keep]
        q = qa - qa.min() + 1e-6
        Sc = build_embedding_S(X[cand])
        L = (q[:, None] * Sc) * q[None, :]
        L = (L + L.T) / 2 + 1e-9 * np.eye(len(q))
        local = _greedy_map_logdet(L, batch)
        return [int(cand[i]) for i in local]
    raise ValueError(f"unknown acquisition {kind!r}")


def _safe_candidates(cand, X, eff, tox, revealed, acq_safety, tau, tau_mode,
                     surr_factory, joint_factory, batch):
    """Restrict candidates to those BELIEVED safe this round (constrained
    acquisition). 'known' -> true toxicity <= ceiling; 'pred' -> predicted
    toxicity <= ceiling (one joint multitask GP when joint_factory is given, else
    a separate toxicity GP). Falls back to the full pool if the safe set is
    smaller than the batch (don't stall the loop)."""
    if acq_safety == "none":
        return cand
    cand = list(cand)
    if acq_safety == "known":
        m_tox = tox[cand]
    else:  # 'pred'
        if joint_factory is not None:
            jg = joint_factory().fit(X[revealed], np.column_stack([eff[revealed], tox[revealed]]))
            m_tox = np.asarray(jg.predict(X[cand])[0])[:, 1]
        else:
            m_tox = np.asarray(surr_factory().fit(X[revealed], tox[revealed]).predict(X[cand])[0])
    thr = _tox_threshold(m_tox, tau, tau_mode)
    safe = [c for c, mt in zip(cand, m_tox) if mt <= thr]
    return safe if len(safe) >= batch else cand


def run_acquisition(kind, X, eff, tox, n_init, n_rounds, batch, seed,
                    surr_factory=None, ehvi_samples=32, shortlist=150,
                    noise=None, score="ucb", joint_factory=None,
                    acq_safety="none", tau=0.5, tau_mode="absolute",
                    return_history=False):
    """Full active-learning loop for one acquisition. Returns the final revealed
    global indices (or, with return_history=True, (final, [revealed-after-round-r
    for r in 0..R]) where round 0 is the initial set). `acq_safety` constrains the
    per-round candidate pool to believed-safe genes (none/known/pred). Common
    random numbers: every kind seeds its initial set from default_rng(seed)."""
    surr_factory = surr_factory or _default_factory
    X = np.asarray(X, float)
    eff = np.asarray(eff, float); tox = np.asarray(tox, float)
    n = len(eff)
    if kind == "ehvi":
        runner = BivariateALRunner(surr_factory, noise or _ZeroNoise(),
                                   ehvi_samples=ehvi_samples, shortlist=shortlist,
                                   joint_factory=joint_factory,
                                   acq_safety=acq_safety, tau=tau, tau_mode=tau_mode)
        hist = runner.run(X, eff, tox, n_init, n_rounds, batch, seed)
        history = [[int(i) for i in h["revealed"]] for h in hist]
        final = history[-1]
        return (final, history) if return_history else final
    rng = np.random.default_rng(seed)
    acq_rng = np.random.default_rng((seed, 99))
    revealed = [int(i) for i in rng.permutation(n)[:n_init]]
    history = [list(revealed)]
    for _ in range(n_rounds):
        rset = set(revealed)
        cand = [i for i in range(n) if i not in rset]
        if not cand:
            history.append(list(revealed)); continue
        cand = _safe_candidates(cand, X, eff, tox, revealed, acq_safety, tau,
                                tau_mode, surr_factory, joint_factory, batch)
        revealed += acquire(kind, X, eff, revealed, cand, batch, acq_rng,
                            surr_factory, score=score)
        history.append(list(revealed))
    return (revealed, history) if return_history else revealed


# --------------------------------------------------------------------------- #
# Nomination (final shortlist): safety filter, then diversity operator         #
# --------------------------------------------------------------------------- #
_DIV_MODE = {"none": "greedy", "cap": "cap", "kdpp": "dpp"}


def _pareto_select(m_eff, m_tox, K, pool=200):
    """Nominate K genes that best cover the predicted (efficacy, -toxicity) Pareto
    front: greedily add the gene with the largest marginal hypervolume gain. Gives
    a BALANCED set spread along the frontier (high-efficacy/high-tox ... low-eff/
    low-tox) rather than max-efficacy. Restricted to a quality pool for speed."""
    from geneal.models.multiobjective import hypervolume2d
    m_eff = np.asarray(m_eff, float); m_tox = np.asarray(m_tox, float)
    n = len(m_eff)
    score = m_eff - m_tox                          # optimistic balance, for the pool
    order = np.argsort(score)[::-1][:min(pool, n)]
    P = np.column_stack([m_eff[order], -m_tox[order]])
    ref = np.array([P[:, 0].min() - 1e-6, P[:, 1].min() - 1e-6])
    chosen, cur = [], np.empty((0, 2))
    for _ in range(min(K, len(order))):
        best, best_hv = -1, -np.inf
        for i in range(len(order)):
            if i in chosen:
                continue
            hv = hypervolume2d(np.vstack([cur, P[i]]), ref)
            if hv > best_hv:
                best_hv, best = hv, i
        chosen.append(best); cur = np.vstack([cur, P[best]])
    return [int(order[i]) for i in chosen]


def _tox_threshold(tox_values, tau, tau_mode):
    """Toxicity ceiling. 'absolute' (default, biological): tau is a toxicity value
    on the Chronos scale -- e.g. tau=0.5 keeps genes with toxicity < 0.5, i.e.
    contrast-line effect > -0.5 (the standard 'not essential' cutoff), or common-
    essential fraction < 0.5. 'quantile': tau is a fraction, keep the safest tau
    of candidates (scale-free, for frontier sweeps)."""
    return float(tau) if tau_mode == "absolute" else float(np.quantile(tox_values, tau))


def nominate(revealed, X, eff, tox, membership, K, safety, diversity, tau,
             S=None, S_builder=None, surr_factory=None, cap=2, pool=200,
             joint_factory=None, tau_mode="absolute", quality="eff"):
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
    relevant toxicity distribution (tau=0.5 = 'the safest half').

    `quality` selects the scalar the diversity operator ranks/temper-weights by,
    AFTER the safety filter (eligibility) is applied:
      'eff' -> predicted target-population efficacy (default; the historical q).
      'sel' -> predicted selectivity = predicted target-population efficacy minus
               predicted non-target-population efficacy (the old `eff - tox`).
    Both reuse the same final fitted predictions, so 'sel' adds only one extra
    selection pass (and a toxicity GP fit only when safety='none'/'known' did not
    already predict toxicity)."""
    surr_factory = surr_factory or _default_factory
    X = np.asarray(X, float)
    eff = np.asarray(eff, float); tox = np.asarray(tox, float)
    n = len(eff)

    if safety == "pareto":
        # nominate the predicted (efficacy, -toxicity) Pareto front (balanced),
        # NOT max-efficacy. No tau filter; this is the no-threshold nomination.
        if joint_factory is not None:
            m2 = np.asarray(joint_factory().fit(
                X[revealed], np.column_stack([eff[revealed], tox[revealed]])).predict(X)[0])
            m_eff, m_tox = m2[:, 0], m2[:, 1]
        else:
            m_eff = np.asarray(surr_factory().fit(X[revealed], eff[revealed]).predict(X)[0])
            m_tox = np.asarray(surr_factory().fit(X[revealed], tox[revealed]).predict(X)[0])
        return _pareto_select(m_eff, m_tox, K, pool)

    m_tox = None   # predicted non-target-population efficacy, when available
    if safety == "pred" and joint_factory is not None:
        # joint (multitask) GP: efficacy + toxicity share strength. q (efficacy)
        # and the toxicity filter both come from the one correlated fit.
        m2 = np.asarray(joint_factory().fit(
            X[revealed], np.column_stack([eff[revealed], tox[revealed]])).predict(X)[0])
        m_eff = m2[:, 0].astype(float)
        m_tox = m2[:, 1].astype(float)
        q = m_eff.copy()
        q[m_tox > _tox_threshold(m_tox, tau, tau_mode)] = _NEG
    else:
        m_eff = np.asarray(surr_factory().fit(X[revealed], eff[revealed]).predict(X)[0],
                           dtype=float)
        q = m_eff.copy()
        if safety == "known":
            q[tox > _tox_threshold(tox, tau, tau_mode)] = _NEG
        elif safety == "pred":
            m_tox = np.asarray(surr_factory().fit(X[revealed], tox[revealed]).predict(X)[0],
                               dtype=float)
            q[m_tox > _tox_threshold(m_tox, tau, tau_mode)] = _NEG
        elif safety != "none":
            raise ValueError(f"unknown safety {safety!r}")

    # quality variant: rank/temper by predicted SELECTIVITY (eff - tox) instead of
    # efficacy, keeping the same eligibility (filtered genes stay at _NEG).
    if quality == "sel":
        if m_tox is None:   # safety none/known didn't predict toxicity -> fit it now
            m_tox = np.asarray(surr_factory().fit(X[revealed], tox[revealed]).predict(X)[0],
                               dtype=float)
        eligible_mask = q > _NEG / 2
        q = np.where(eligible_mask, m_eff - m_tox, _NEG)
    elif quality != "eff":
        raise ValueError(f"unknown quality {quality!r}")

    # restrict the diversity operator to the top-`pool` eligible by quality:
    # bounds k-DPP cost (O(K^2 pool)) and diversifies only among high-efficacy genes.
    eligible = int(np.sum(q > _NEG / 2))
    pool_n = min(pool, max(eligible, K))
    order = np.argsort(q)[::-1][:pool_n]
    qp = q[order]
    memp = {li: membership.get(int(order[li]), set()) for li in range(len(order))}
    # k-DPP similarity over the top-`pool` eligible genes only. Prefer S_builder
    # (builds the (pool,pool) submatrix directly, O(pool^2) memory); fall back to
    # slicing a pre-built full matrix. Both give the identical submatrix.
    if diversity == "kdpp":
        if S_builder is not None:
            Sp = S_builder([int(i) for i in order])
        elif S is not None:
            Sp = S[np.ix_(order, order)]
        else:
            Sp = None
    else:
        Sp = None

    mode = _DIV_MODE[diversity]
    local = HedgedSelect(mode=mode, cap=cap).select_idx(qp, memp, K=K, S=Sp)
    return [int(order[li]) for li in local]


# --------------------------------------------------------------------------- #
# Evaluate (true-value metrics for a nominated portfolio)                      #
# --------------------------------------------------------------------------- #
def evaluate(pick, eff, tox, membership, X=None, tox_ceiling=None, risk_S=None,
             membership_string=None):
    """TRUE-value metrics for a nominated set. eff/tox are ground truth. If
    `tox_ceiling` is given, also count nominees whose TRUE toxicity is at/below
    (safe) vs above (toxic) the ceiling -- a direct count of how many of the K
    picks are actually tolerable. (X is accepted for API compatibility; unused.)

    `risk_S`: optional dict {source -> (N,N) similarity matrix} (e.g.
    {'corum': S_corum, 'string': S_string}). For each source, adds two
    portfolio-risk readouts of the picked set: `risk_<src>` = equal-weight
    portfolio variance wᵀSw (lower = better hedged) and `neff_<src>` = effective
    number of independent bets K²/1ᵀS1.

    `membership_string`: optional gene->{module} dict (STRING discrete modules).
    When given, the discrete diversity metrics (concentration, robustness,
    distinct groups) are ALSO computed on STRING -> `*_string` keys, so the same
    columns can be reported evaluated on CORUM (the hedge graph) and on STRING
    (held-out). Tests whether the CORUM hedge generalizes."""
    eff = np.asarray(eff, float); tox = np.asarray(tox, float)
    pick = list(pick)
    out = {
        "mean_efficacy": float(np.mean(eff[pick])),
        "max_efficacy": float(np.max(eff[pick])),
        "mean_toxicity": float(np.mean(tox[pick])),
        "concentration": float(pathway_concentration(pick, membership)),
        # robustness is a fraction-of-VALUE-surviving; value must be non-negative
        # (raw lethality can go negative for growth-promoting knockouts), else the
        # ratios blow past 1. Clip at 0.
        "robustness": float(dropout_robustness(pick, membership, np.clip(eff, 0.0, None))),
        "n_pathways": int(n_pathways_covered(pick, membership)),
    }
    if membership_string is not None:
        out["concentration_string"] = float(pathway_concentration(pick, membership_string))
        out["robustness_string"] = float(dropout_robustness(
            pick, membership_string, np.clip(eff, 0.0, None)))
        out["n_pathways_string"] = int(n_pathways_covered(pick, membership_string))
    if risk_S:
        for src, S in risk_S.items():
            if callable(S):
                # S(pick) returns the (K,K) picks submatrix directly (memory-
                # efficient path). Equivalent to portfolio_risk/effective_bets on
                # the full matrix sliced to the picks.
                sub = np.asarray(S(pick), float)
                Kp = len(pick)
                w = np.full(Kp, 1.0 / Kp) if Kp else np.zeros(0)
                out[f"risk_{src}"] = float(w @ sub @ w) if Kp else 0.0
                denom = float(sub.sum())
                out[f"neff_{src}"] = float(Kp * Kp / denom) if denom > 0 else float(Kp)
            else:
                out[f"risk_{src}"] = float(portfolio_risk(pick, S))
                out[f"neff_{src}"] = float(effective_bets(pick, S))
    if tox_ceiling is not None:
        pe = np.asarray(pick)
        safe = tox[pe] <= tox_ceiling
        out["n_toxic"] = int(np.sum(~safe))
        out["n_safe"] = int(np.sum(safe))
        # mean efficacy among PERMISSIBLE picks (below the safety ceiling)
        out["mean_efficacy_safe"] = float(np.mean(eff[pe[safe]])) if safe.any() else float("nan")
        # USEFUL efficacy: mean over all K picks with toxic (non-permissible) picks
        # scored 0 -- rewards being potent AND keeping the shortlist safe, so a few
        # very-potent-but-toxic picks no longer inflate the score.
        out["useful_efficacy"] = float(np.mean(np.where(safe, eff[pe], 0.0)))
    # nominee hypervolume in TRUE (efficacy, -toxicity) space vs a global reference
    # (efficacy/safety BALANCE -- the right readout when there is no hard threshold).
    from geneal.models.multiobjective import hypervolume2d
    ref = np.array([eff.min() - 0.1 * (np.ptp(eff) + 1e-9),
                    -(tox.max()) - 0.1 * (np.ptp(tox) + 1e-9)])
    out["hypervolume"] = float(hypervolume2d(
        np.column_stack([eff[pick], -tox[pick]]), ref))
    return out


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


# --------------------------------------------------------------------------- #
# SUBSET similarity builders (memory-efficient path)                           #
#                                                                              #
# The k-DPP only ever touches the top-`pool` (~200) eligible submatrix and the #
# risk/Neff metrics only the K (~30) picks. Building the full (n,n) matrix and #
# slicing it wastes O(n^2) memory (~8 GB at genome scale). Because embedding    #
# cosine, CORUM Jaccard, and STRING edge weight are all PAIRWISE, the submatrix #
# of the full matrix EQUALS the matrix built on the subset: build_*(X)[ix,ix]  #
# == sub_*(X, ix), exactly. These builders return that submatrix directly.      #
# --------------------------------------------------------------------------- #
def sub_embedding_S(X, idx):
    """(m,m) embedding-cosine similarity over global rows `idx`. Equals
    build_embedding_S(X)[np.ix_(idx, idx)] exactly."""
    Xs = np.asarray(X, float)[np.asarray(idx, int)]
    nrm = np.linalg.norm(Xs, axis=1, keepdims=True)
    Xn = Xs / np.clip(nrm, 1e-12, None)
    S = 0.5 * (1.0 + np.clip(Xn @ Xn.T, -1.0, 1.0))
    np.fill_diagonal(S, 1.0)
    return S


def sub_corum_S(membership, idx):
    """(m,m) CORUM Jaccard similarity over global genes `idx`. Equals
    build_corum_S(membership, n)[np.ix_(idx, idx)] exactly."""
    ms = [membership.get(int(i), set()) for i in idx]
    m = len(ms)
    S = np.eye(m, dtype=float)
    for a in range(m):
        sa = ms[a]
        if not sa:
            continue
        for b in range(a + 1, m):
            sb = ms[b]
            if not sb:
                continue
            u = len(sa | sb)
            if u:
                v = len(sa & sb) / u
                if v:
                    S[a, b] = S[b, a] = v
    return S


def string_adjacency(gene_names,
                     edges_path="data/processed/depmap/string_edges_all.parquet"):
    """Sparse STRING adjacency keyed by POSITION in `gene_names`: {i: {j: w}}.
    Loads the genome-wide edge list once (~1e6 edges) without ever materializing
    the dense (n,n) matrix. Feeds sub_string_S and build_string_membership."""
    from geneal.data.depmap import parse_entrez
    ents = [parse_entrez(g) for g in gene_names]
    pos = {int(e): i for i, e in enumerate(ents) if e is not None}
    adj: dict = {}
    if not Path(edges_path).exists():
        return adj
    import pandas as pd
    e = pd.read_parquet(edges_path)
    a = e["entrez_i"].to_numpy(); b = e["entrez_j"].to_numpy()
    w = e["weight"].to_numpy(dtype=float)
    for ei, ej, wij in zip(a, b, w):
        i = pos.get(int(ei)); j = pos.get(int(ej))
        if i is None or j is None:
            continue
        adj.setdefault(i, {})[j] = wij
        adj.setdefault(j, {})[i] = wij
    return adj


def sub_string_S(adj, idx):
    """(m,m) STRING combined-score similarity over global genes `idx`, from the
    sparse adjacency. Equals build_string_S(...)[np.ix_(idx, idx)] exactly."""
    idx = [int(i) for i in idx]
    loc = {g: k for k, g in enumerate(idx)}
    m = len(idx)
    S = np.eye(m, dtype=float)
    for k, g in enumerate(idx):
        for j, wij in adj.get(g, {}).items():
            kk = loc.get(j)
            if kk is not None:
                S[k, kk] = wij
    return S



def build_string_membership(S_string=None, thresh: float = 0.7, adj=None,
                            n=None, cache_path=None):
    """Discrete STRING 'modules' for the diversity metrics: greedy-modularity
    communities of the STRING graph thresholded at `thresh` (combined score in
    [0,1]). Connected components are NOT used — the STRING network is dense and
    chains into one giant component at any threshold; modularity communities give
    balanced co-functional modules instead. Returns gene_idx -> {module_id}; genes
    in singleton communities get an empty set (unannotated, mirroring CORUM). Lets
    concentration / robustness / distinct-group metrics be computed on STRING the
    same way as on CORUM complexes.

    Graph source (equivalent partition either way):
      `adj` (sparse {i: {j: w}}, from string_adjacency) + `n` -> genome-safe path
        that never materializes the dense (n,n) matrix (was ~13.5 GB at genome
        scale). PREFERRED.
      `S_string` (dense) -> legacy path.
    `cache_path`: if given, the (idx, module) table is loaded when present and
    written otherwise -- the communities are method-independent, so the O(n^2)
    modularity is paid once per (graph, threshold)."""
    if cache_path is not None and Path(cache_path).exists():
        import pandas as pd
        mdf = pd.read_parquet(cache_path)
        # only annotated genes stored; unannotated default to set() via .get()
        return {int(i): {int(mid)} for i, mid in zip(mdf["idx"], mdf["module"])}

    import networkx as nx
    if adj is not None:
        if n is None:
            n = (max(adj) + 1) if adj else 0
        G = nx.Graph()
        G.add_nodes_from(range(int(n)))
        for i, nbrs in adj.items():
            for j, w in nbrs.items():
                if i < j and w >= thresh:
                    G.add_edge(i, j, weight=float(w))
    else:
        S = np.asarray(S_string, float).copy()
        np.fill_diagonal(S, 0.0)
        A = (S >= thresh) * S                  # weighted, thresholded adjacency
        n = len(S)
        G = nx.from_numpy_array(A)
    comms = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
    mem = {i: set() for i in range(int(n))}
    for mid, c in enumerate(comms):
        if len(c) <= 1:
            continue                           # singleton -> no module (unannotated)
        for i in c:
            mem[i] = {int(mid)}
    if cache_path is not None:
        import pandas as pd
        rows = [dict(idx=i, module=next(iter(s))) for i, s in mem.items() if s]
        mdf = pd.DataFrame(rows, columns=["idx", "module"])
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        mdf.to_parquet(cache_path)
    return mem


def build_corum_S(membership, n):
    """Dense (n,n) CORUM pathway similarity: S_ij = Jaccard overlap of the two
    genes' complex-membership sets (|Ci∩Cj| / |Ci∪Cj|), unit diagonal. Genes
    sharing no complex -> 0; unannotated genes are isolated. Built from a sparse
    gene×complex incidence (M Mᵀ gives intersection counts). EXTERNAL knowledge
    (CORUM) similarity for the k-DPP ablation vs the learned embedding cosine."""
    import scipy.sparse as sp
    comps = sorted({c for s in membership.values() for c in s})
    if not comps:
        return np.eye(n, dtype=float)
    cidx = {c: j for j, c in enumerate(comps)}
    rows, cols = [], []
    for i in range(n):
        for c in membership.get(i, ()):
            rows.append(i); cols.append(cidx[c])
    M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(comps)))
    inter = np.asarray((M @ M.T).todense(), dtype=float)
    sizes = np.asarray(M.sum(1)).ravel()
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(union > 0, inter / union, 0.0)
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
