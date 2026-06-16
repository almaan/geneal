# scripts/run_ablation.py
"""Default ablation experiment (geneal Plan 7, revision v2).

TWO analyses on real DepMap, full active-learning loop per method, run under TWO
toxicity definitions (faceted in one report):

  A) Safety vs efficacy -- the safety RULE: none / truncation_known (filter on
     the known toxicity = oracle ceiling) / truncation_pred (filter on a learned
     toxicity GP) / ehvi (dual-objective EHVI acquisition + learned filter), plus
     random / farthest / cluster diversity baselines. 7 methods on the
     efficacy-toxicity tradeoff. truncation_known is the upper limit truncation_pred
     chases; the trunc_known-vs-trunc_pred/ehvi gap measures the cost of having to
     LEARN safety from the embedding.

  B) Diversity / robustness -- operators none / cap (CORUM per-pathway) / kdpp
     (embedding-cosine k-DPP) layered on THREE bases {greedy, truncation, ehvi}.

Toxicity definitions (both run, faceted):
  contrast  -- lethality in one fixed contrast cell line (normal-tissue stand-in;
               the PRIMARY definition). Auto-picked from a ranked candidate list.
  aggregate -- common-essential, EXCLUDING the target line (leakage fix).

tau is a QUANTILE (scale-free): keep genes below the tau-quantile of toxicity.
PubMedBERT predicts; CORUM=pathways (cap); embedding cosine = k-DPP similarity S
(mechanism, never outcome). Full genome via --panel none.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from geneal.data.depmap import load_gene_effect, parse_entrez
from geneal.data.selective import (build_selective_dataset, corum_membership,
                                   toxicity_vector, rank_contrast_lines)
from geneal.runner.ablation import (run_acquisition, nominate, evaluate,
                                    build_string_S, build_embedding_S, build_corum_S,
                                    _tox_threshold)
from geneal.metrics.portfolio import dropout_curve
from geneal.models.multiobjective import pareto_indices


def _string_spread(sel, S):
    """STRING pairwise-spread metrics for a portfolio: mean & max pairwise STRING
    similarity among the picks (lower = more mechanistically spread/hedged)."""
    sel = list(sel)
    if S is None or len(sel) < 2:
        return dict(string_redundancy=float("nan"), string_max_sim=float("nan"))
    sub = S[np.ix_(sel, sel)]
    iu = np.triu_indices(len(sel), 1)
    pair = sub[iu]
    return dict(string_redundancy=float(pair.mean()), string_max_sim=float(pair.max()))


def _pareto_recall(sel, eff, tox, pareto_set):
    """(n hit, recall): how many of the nominees lie on the TRUE (eff,-tox) Pareto
    front, and the fraction of that front recovered."""
    hit = len(set(sel) & pareto_set)
    return hit, (hit / len(pareto_set) if pareto_set else float("nan"))

# Acquisition specs: acq_key -> (kind, acq_safety). acq_safety constrains the
# per-round candidate pool to believed-safe genes (none/known/pred).
ACQ_SPECS = {
    "random":       ("random",   "none"),
    "greedy":       ("greedy",   "none"),
    "farthest":     ("farthest", "none"),
    "cluster":      ("cluster",  "none"),
    "info_div":     ("info_div", "none"),
    "ehvi":         ("ehvi",     "none"),
    "greedy_safe":  ("greedy",   "pred"),   # truncate-each-round (predicted tox)
    "ehvi_safe":    ("ehvi",     "pred"),   # EHVI, truncate-each-round (predicted)
    "known_safe":   ("greedy",   "known"),  # truncate-each-round (KNOWN tox) = upper bound
}
# acquisitions that do NOT depend on the toxicity definition (run once per line,seed).
# 'farthest' is implemented (ACQ_SPECS/CoreSet) but omitted from the analysis for
# now (its pairwise-distance step is memory-heavy at genome scale).
_TOX_INDEP_ACQ = ["random", "greedy", "cluster", "info_div"]
_TOX_DEP_ACQ = ["ehvi", "greedy_safe", "ehvi_safe", "known_safe"]

# Analysis A method: (label, acq_key, nominate_safety). diversity = none.
ANALYSIS_A = [
    ("greedy",       "greedy",      "none"),   # naive top-K efficacy
    ("trunc_known",  "greedy",      "known"),  # known ceiling at NOMINATION (oracle limit)
    ("trunc_pred",   "greedy",      "pred"),   # learned ceiling at nomination
    ("greedy_safe",  "greedy_safe", "pred"),   # + truncate-each-round (predicted)
    ("ehvi",         "ehvi",        "none"),   # BASELINE: EHVI acq, NO filter (max-eff nomination)
    ("ehvi_pareto",  "ehvi",        "pareto"), # EHVI acq, nominate the predicted Pareto front (balanced)
    ("ehvi_trunc",   "ehvi",        "pred"),   # EHVI acq + nomination filter
    ("ehvi_safe",    "ehvi_safe",   "pred"),   # EHVI + per-round filter (predicted)
    ("known_safe",   "known_safe",  "known"),  # KNOWN truncation throughout (upper bound)
    ("random",       "random",      "none"),
    ("cluster",      "cluster",     "none"),
    ("info_div",     "info_div",    "none"),
]
# Analysis B bases (aligned with Section-A method keys): greedy vs EHVI, each with
# nomination filtering and per-round filtering (all predicted-tox). Diversity
# operators are layered on each. (base_key, acq_key, nominate_safety).
B_BASES = [
    ("trunc_pred",  "greedy",      "pred"),   # G·nom·P  (greedy + nomination filter)
    ("greedy_safe", "greedy_safe", "pred"),   # G·RT·P   (greedy + per-round filter)
    ("ehvi_trunc",  "ehvi",        "pred"),   # E·nom·P  (EHVI + nomination filter)
    ("ehvi_safe",   "ehvi_safe",   "pred"),   # E·RT·P   (EHVI + per-round filter)
]
# Section-B diversity operators. (op_label, mode, similarity-source).
# kdpp splits by the k-DPP similarity S: learned embedding cosine vs external
# CORUM pathway matrix (the similarity ablation).
DIVERSITY_OPS = [
    ("none",        "none", None),
    ("kdpp_string", "kdpp", "string"),     # k-DPP, external STRING network S
    ("kdpp_corum",  "kdpp", "corum"),      # k-DPP, external CORUM complex S
]   # (embedding-S k-DPP and per-pathway 'cap' are implemented but excluded from B)
MAX_DROP = 5   # failure-simulation horizon (# pathways dropped)
TOX_SOURCES = ["contrast", "aggregate"]


def _factory(n_iters):
    from geneal.models.surrogate import GPRSurrogate
    return lambda: GPRSurrogate(n_iters=n_iters)


def _mean_ci(x):
    x = np.asarray(x, float); n = len(x); m = float(x.mean()) if n else float("nan")
    return m, (0.0 if n < 2 else 1.96 * float(x.std(ddof=1)) / np.sqrt(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-effect", default="data/processed/depmap/gene_effect.parquet")
    ap.add_argument("--embeddings", default="data/processed/embeddings/pubmedbert_all.parquet",
                    help="PubMedBERT embedding parquet (genome-wide cache; use --panel to subset)")
    ap.add_argument("--panel-a", default="none",
                    help="candidate panel for Analysis A (safety/efficacy); 'none' = full genome (default)")
    ap.add_argument("--panel-b", default="data/processed/depmap/panel_5k.txt",
                    help="candidate panel for Analysis B (diversity); default = 5k subset "
                         "(the N^2 k-DPP similarities are built on this pool)")
    ap.add_argument("--string", default="data/processed/depmap/string_edges_all.parquet")
    ap.add_argument("--kdpp-sim", choices=["embedding", "string"], default="embedding",
                    help="k-DPP similarity S: 'embedding' = dense PubMedBERT cosine "
                         "(default, tunable); 'string' = sparse STRING. Never outcome similarity.")
    ap.add_argument("--acq-score", choices=["ucb", "ei"], default="ucb",
                    help="greedy-efficacy acquisition score (UCB=mu+2sigma, or EI)")
    ap.add_argument("--joint-gp", action="store_true",
                    help="learned-safety methods (trunc_pred, ehvi) use a JOINT multitask "
                         "GP over (efficacy, toxicity) instead of two independent GPs. "
                         "Borrows strength via the ~0.8 eff-tox correlation.")
    ap.add_argument("--n-cell-lines", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--n-initial", type=int, default=40)
    ap.add_argument("--n-rounds", type=int, default=8)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.5,
                    help="safety ceiling. With --tau-mode absolute (default): a toxicity "
                         "value on the Chronos scale (0.5 = contrast effect > -0.5 = 'not "
                         "essential in the normal stand-in', or common-essential < 50%% of "
                         "lines). With quantile: keep the safest tau fraction.")
    ap.add_argument("--tau-mode", choices=["absolute", "quantile"], default="absolute",
                    help="absolute (biological, default) or quantile (scale-free sweeps)")
    ap.add_argument("--thresh", type=float, default=-0.5, help="strongly-lethal threshold")
    ap.add_argument("--cap", type=int, default=2, help="per-pathway cap for the cap operator")
    ap.add_argument("--pool", type=int, default=200, help="quality pool the diversity operator selects within")
    ap.add_argument("--ehvi-samples", type=int, default=32)
    ap.add_argument("--shortlist", type=int, default=150)
    ap.add_argument("--n-iters", type=int, default=120, help="GP training iterations")
    ap.add_argument("--contrast-line", default=None,
                    help="fixed contrast (normal-tissue stand-in) ModelID; default = top of the ranked candidates")
    ap.add_argument("--contrast-candidates", type=int, default=10,
                    help="how many ranked contrast-line candidates to record in meta")
    ap.add_argument("--cell-lines", type=str, nargs="+", default=None,
                    help="explicit target ModelIDs (overrides --n-cell-lines)")
    ap.add_argument("--out-root", default="res/runs_ablation")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--export-figs", action="store_true",
                    help="also write vector PDF+PNG of every figure (kaleido). OFF by "
                         "default -- it is slow (per-figure chromium) and would stall the "
                         "run; the interactive HTML always renders. Turn on for manuscript figs.")
    args = ap.parse_args()
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)   # live progress under SLURM
    except Exception:
        pass
    factory = _factory(args.n_iters)
    joint_factory = None
    if args.joint_gp:
        from geneal.models.multitask import MultiTaskGPR
        joint_factory = lambda: MultiTaskGPR(n_iters=args.n_iters)
        print("joint multitask GP enabled for learned-safety methods (trunc_pred, ehvi)")

    ge = load_gene_effect(args.gene_effect)
    emb_full = pd.read_parquet(args.embeddings)

    def subset_emb(panel):
        p = (panel or "").strip()
        if not p or p.lower() == "none":
            return emb_full
        keep = set(int(x) for x in Path(p).read_text().split())
        return emb_full[emb_full.index.isin(keep)]

    emb_a = subset_emb(args.panel_a)        # Analysis A candidate pool (default full genome)
    emb_b = subset_emb(args.panel_b)        # Analysis B candidate pool (default 5k subset)
    print(f"Analysis A panel: {len(emb_a)} genes ({args.panel_a}); "
          f"Analysis B panel: {len(emb_b)} genes ({args.panel_b})")
    embset = set(emb_a.index)
    labs = [g for g in ge.index if parse_entrez(g) in embset]

    # ranked contrast-line candidates (most normal-like first)
    contrast_ranked = rank_contrast_lines(ge, args.thresh, n=args.contrast_candidates)
    contrast_line = args.contrast_line or contrast_ranked[0]
    print(f"contrast-line candidates (best first): {contrast_ranked}")
    print(f"using contrast line: {contrast_line}")

    # target lines: lowest-NaN, EXCLUDING the contrast line (no self-toxicity)
    if args.cell_lines:
        lines = [c for c in args.cell_lines if c != contrast_line]
    else:
        ranked = ge.loc[labs].isna().sum(axis=0).sort_values().index.tolist()
        lines = [c for c in ranked if c != contrast_line][:args.n_cell_lines]
    print(f"target lines ({len(lines)}): {lines}")

    common = dict(n_init=args.n_initial, n_rounds=args.n_rounds, batch=args.batch,
                  surr_factory=factory)
    nom_common = dict(K=args.K, tau=args.tau, surr_factory=factory, cap=args.cap,
                      pool=args.pool, joint_factory=joint_factory, tau_mode=args.tau_mode)

    def quick(idx, eff, tox, ceiling):
        """Cheap true-value summary of a gene set (no alpha-NDCG/kmeans) for the
        per-round curves."""
        idx = list(idx)
        if not idx:
            return dict(mean_efficacy=float("nan"), mean_toxicity=float("nan"),
                        n_safe=0, mean_efficacy_safe=float("nan"))
        e = eff[idx]; t = tox[idx]; safe = t <= ceiling
        return dict(mean_efficacy=float(e.mean()), mean_toxicity=float(t.mean()),
                    n_safe=int(safe.sum()),
                    mean_efficacy_safe=float(e[safe].mean()) if safe.any() else float("nan"))

    def build_line(emb_x, cl):
        ds, _ = build_selective_dataset(ge, emb_x, cl, lam=0.0, thresh=args.thresh)
        X = StandardScaler().fit_transform(ds.embeddings)
        eff = np.asarray(ds.target, float)
        mem = corum_membership(ds.gene_names)
        tox = {src: toxicity_vector(ge, ds.gene_names, src, target_line=cl,
                                    contrast_line=contrast_line, thresh=args.thresh)
               for src in TOX_SOURCES}
        return X, eff, mem, tox, ds.gene_names

    B_ACQ = sorted({acq for _, acq, _ in B_BASES})   # acquisitions Analysis B needs
    rows, scatters, assayed_rows, round_rows = [], [], [], []
    for cl in lines:
        Xa, effa, mema, toxa_by_src, _ = build_line(emb_a, cl)       # A: full genome
        Xb, effb, memb, toxb_by_src, names_b = build_line(emb_b, cl)  # B: subset
        # k-DPP similarity sources (N^2 only on the B subset): learned embedding
        # cosine, external STRING network, external CORUM complexes.
        S_by_src = {"embedding": build_embedding_S(Xb),
                    "string": build_string_S(names_b, args.string),
                    "corum": build_corum_S(memb, len(effb))}
        print(f"[{cl}] A={len(effa)} genes, B={len(effb)} genes")

        for seed in args.seeds:
            # ===================== Analysis A (full-genome pool) =====================
            histA = {k: run_acquisition(k, Xa, effa, None, seed=seed, score=args.acq_score,
                                        return_history=True, **common)[1]
                     for k in _TOX_INDEP_ACQ}
            for src in TOX_SOURCES:
                toxa = toxa_by_src[src]; ceil_a = _tox_threshold(toxa, args.tau, args.tau_mode)
                pareto_a = set(int(i) for i in pareto_indices(
                    np.column_stack([effa, -toxa])))   # true genome-wide Pareto front
                for k in _TOX_DEP_ACQ:
                    kind, acqsaf = ACQ_SPECS[k]
                    histA[k] = run_acquisition(kind, Xa, effa, toxa, seed=seed,
                                               ehvi_samples=args.ehvi_samples,
                                               shortlist=args.shortlist, score=args.acq_score,
                                               joint_factory=joint_factory, acq_safety=acqsaf,
                                               tau=args.tau, tau_mode=args.tau_mode,
                                               return_history=True, **common)[1]
                for k, h in histA.items():
                    m = evaluate(h[-1], effa, toxa, mema, tox_ceiling=ceil_a)
                    assayed_rows.append(dict(tox_source=src, acq=k, cell_line=cl,
                                             seed=seed, n_assayed=len(h[-1]), **m))
                want_scatter = (seed == args.seeds[0]); picks = {}
                for label, acq_key, safety in ANALYSIS_A:
                    h = histA[acq_key]
                    for r, rev_r in enumerate(h):
                        a = quick(rev_r, effa, toxa, ceil_a)
                        seln = nominate(rev_r, Xa, effa, toxa, mema, safety=safety,
                                        diversity="none", S=None, **nom_common)
                        nq = quick(seln, effa, toxa, ceil_a)
                        round_rows.append(dict(
                            tox_source=src, method=label, cell_line=cl, seed=seed, round=r,
                            n_assayed=len(rev_r), assayed_mean_efficacy=a["mean_efficacy"],
                            assayed_mean_toxicity=a["mean_toxicity"], assayed_n_safe=a["n_safe"],
                            nom_mean_efficacy=nq["mean_efficacy"],
                            nom_mean_efficacy_safe=nq["mean_efficacy_safe"],
                            nom_mean_toxicity=nq["mean_toxicity"], nom_n_safe=nq["n_safe"]))
                    sel = nominate(h[-1], Xa, effa, toxa, mema, safety=safety,
                                   diversity="none", S=None, **nom_common)
                    m = evaluate(sel, effa, toxa, mema, tox_ceiling=ceil_a)
                    p_hit, p_rec = _pareto_recall(sel, effa, toxa, pareto_a)
                    rows.append(dict(analysis="A", tox_source=src, method=label, base=label,
                                     operator="none", acq=acq_key, safety=safety, cell_line=cl,
                                     seed=seed, n_novel=len(set(sel) - set(h[-1])),
                                     pareto_hit=p_hit, pareto_recall=p_rec, **m))
                    if want_scatter:
                        picks[label] = set(sel)
                if want_scatter:
                    sc = pd.DataFrame({"efficacy": effa, "toxicity": toxa})
                    for label, pk in picks.items():
                        sc[f"pick_{label}"] = [i in pk for i in range(len(effa))]
                    sc["cell_line"] = cl; sc["tox_source"] = src
                    scatters.append(sc)

            # ===================== Analysis B (subset pool) =====================
            # tox-independent B acquisitions (run once)
            histB = {k: run_acquisition(k, Xb, effb, None, seed=seed, score=args.acq_score,
                                        return_history=True, **common)[1]
                     for k in B_ACQ if ACQ_SPECS[k][1] == "none" and k != "ehvi"}
            for src in TOX_SOURCES:
                toxb = toxb_by_src[src]; ceil_b = _tox_threshold(toxb, args.tau, args.tau_mode)
                pareto_b = set(int(i) for i in pareto_indices(np.column_stack([effb, -toxb])))
                for k in [a for a in B_ACQ if a not in histB]:    # tox-dependent B acqs
                    kind, acqsaf = ACQ_SPECS[k]
                    histB[k] = run_acquisition(kind, Xb, effb, toxb, seed=seed,
                                               ehvi_samples=args.ehvi_samples,
                                               shortlist=args.shortlist, score=args.acq_score,
                                               joint_factory=joint_factory, acq_safety=acqsaf,
                                               tau=args.tau, tau_mode=args.tau_mode,
                                               return_history=True, **common)[1]
                for base_label, acq_key, safety in B_BASES:
                    for op_label, mode, simsrc in DIVERSITY_OPS:
                        Sb = S_by_src.get(simsrc) if simsrc else None
                        rev_final = histB[acq_key][-1]
                        sel = nominate(rev_final, Xb, effb, toxb, memb, safety=safety,
                                       diversity=mode, S=Sb, **nom_common)
                        m = evaluate(sel, effb, toxb, memb, tox_ceiling=ceil_b)
                        dc = dropout_curve(sel, memb, effb, MAX_DROP)
                        p_hit, p_rec = _pareto_recall(sel, effb, toxb, pareto_b)
                        sm = _string_spread(sel, S_by_src["string"])   # STRING pairwise spread
                        rows.append(dict(analysis="B", tox_source=src,
                                         method=f"{base_label}+{op_label}", base=base_label,
                                         operator=op_label, acq=acq_key, safety=safety,
                                         cell_line=cl, seed=seed,
                                         n_novel=len(set(sel) - set(rev_final)),
                                         pareto_hit=p_hit, pareto_recall=p_rec, **m, **sm,
                                         **{f"drop_{i}": dc[i] for i in range(len(dc))}))
        print(f"[{cl}] done ({len(args.seeds)} seeds x {len(TOX_SOURCES)} tox-sources)")

    # ---- Persist + report ----
    df = pd.DataFrame(rows)
    run = args.run_name or pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out_root) / run
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "ablation.parquet")
    scatter_df = pd.concat(scatters, ignore_index=True) if scatters else None
    if scatter_df is not None:
        scatter_df.to_parquet(out / "scatter.parquet")
    assayed_df = pd.DataFrame(assayed_rows); assayed_df.to_parquet(out / "assayed.parquet")
    rounds_df = pd.DataFrame(round_rows); rounds_df.to_parquet(out / "rounds.parquet")
    meta = dict(panel_a=args.panel_a, panel_b=args.panel_b,
                n_genes_a=int(len(emb_a)), n_genes_b=int(len(emb_b)),
                lines=lines, seeds=list(args.seeds),
                K=args.K, tau=args.tau, cap=args.cap, n_rounds=args.n_rounds,
                batch=args.batch, kdpp_sim=args.kdpp_sim, acq_score=args.acq_score,
                tau_mode=args.tau_mode, joint_gp=bool(args.joint_gp),
                tox_sources=TOX_SOURCES, contrast_line=contrast_line,
                contrast_ranked=contrast_ranked)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    for src in TOX_SOURCES:
        d = df[df.tox_source == src]
        print(f"\n=== [{src} toxicity] {len(lines)} lines x {len(args.seeds)} seeds (mean +/- 95% CI) ===")
        print("[A: safety vs efficacy]")
        for label, _, _ in ANALYSIS_A:
            sub = d[(d.analysis == "A") & (d.method == label)]
            e, ec = _mean_ci(sub["mean_efficacy"]); t, tc = _mean_ci(sub["mean_toxicity"])
            print(f"  {label:12s} eff {e:.3f}+/-{ec:.3f}  tox {t:.3f}+/-{tc:.3f}")
        print("[B: diversity / robustness]")
        for base_label, _, _ in B_BASES:
            for op_label, _, _ in DIVERSITY_OPS:
                sub = d[(d.analysis == "B") & (d.base == base_label) & (d.operator == op_label)]
                c, _ = _mean_ci(sub["concentration"]); r, _ = _mean_ci(sub["robustness"])
                e, _ = _mean_ci(sub["mean_efficacy"])
                print(f"  {base_label:11s}+{op_label:11s} conc {c:.3f} robust {r:.3f} eff {e:.3f}")

    try:
        from geneal.report.ablation_report import build_ablation_report
        build_ablation_report(df, out / "report.html", scatter=scatter_df, meta=meta,
                              assayed=assayed_df, rounds=rounds_df,
                              fig_dir=(out / "figs") if args.export_figs else None)
        print(f"\nreport -> {out / 'report.html'}")
    except Exception as e:
        import traceback; traceback.print_exc(); print("report skipped:", e)
    print(f"run dir: {out}")


if __name__ == "__main__":
    main()
