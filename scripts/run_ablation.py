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
                                    build_string_S, build_embedding_S)

# Analysis A: (label, acquisition, safety-filter). diversity = none.
ANALYSIS_A = [
    ("greedy",       "greedy",   "none"),   # naive top-K efficacy
    ("trunc_known",  "greedy",   "known"),  # known-toxicity ceiling (oracle limit)
    ("trunc_pred",   "greedy",   "pred"),   # learned-toxicity ceiling, greedy acq
    ("ehvi",         "ehvi",     "pred"),   # dual-objective AL + learned ceiling
    ("random",       "random",   "none"),   # naive baseline
    ("farthest",     "farthest", "none"),   # coverage diversity baseline
    ("cluster",      "cluster",  "none"),   # cluster-representative baseline
    ("info_div",     "info_div", "none"),   # informative-diverse (IterPert-family)
]
# Analysis B bases: (label, acquisition, safety-filter); each x {none,cap,kdpp}.
B_BASES = [
    ("greedy",      "greedy", "none"),
    ("truncation",  "greedy", "known"),
    ("ehvi",        "ehvi",   "pred"),
]
DIVERSITY_OPS = ["none", "cap", "kdpp"]
TOX_SOURCES = ["contrast", "aggregate"]
# acquisitions that do NOT depend on the toxicity definition (computed once/line,seed)
_TOX_INDEP_ACQ = ["random", "greedy", "farthest", "cluster", "info_div"]


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
    ap.add_argument("--panel", default="data/processed/depmap/panel_5k.txt",
                    help="entrez-id panel file; pass '' or 'none' for the full genome")
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
                    help="safety QUANTILE: keep genes below the tau-quantile of toxicity")
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
    emb = pd.read_parquet(args.embeddings)
    panel = (args.panel or "").strip()
    if panel and panel.lower() != "none":
        keep = set(int(x) for x in Path(panel).read_text().split())
        emb = emb[emb.index.isin(keep)]
        print(f"panel subset: {len(emb)} genes ({panel})")
    else:
        print(f"full genome: {len(emb)} genes")
    embset = set(emb.index)
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
                      pool=args.pool, joint_factory=joint_factory)

    rows, scatters = [], []
    for cl in lines:
        ds, aux = build_selective_dataset(ge, emb, cl, lam=0.0, thresh=args.thresh)
        X = StandardScaler().fit_transform(ds.embeddings)
        eff = np.asarray(ds.target, float)            # raw lethality in this line
        mem = corum_membership(ds.gene_names)
        S = (build_embedding_S(X) if args.kdpp_sim == "embedding"
             else build_string_S(ds.gene_names, args.string))
        tox_by_source = {
            src: toxicity_vector(ge, ds.gene_names, src, target_line=cl,
                                 contrast_line=contrast_line, thresh=args.thresh)
            for src in TOX_SOURCES}
        print(f"[{cl}] {len(eff)} genes, {sum(bool(v) for v in mem.values())} CORUM-annotated")

        for seed in args.seeds:
            # tox-independent acquisitions: run once per (line, seed)
            rev_indep = {k: run_acquisition(k, X, eff, None, seed=seed,
                                            score=args.acq_score, **common)
                         for k in _TOX_INDEP_ACQ}
            for src in TOX_SOURCES:
                tox = tox_by_source[src]
                rev = dict(rev_indep)
                rev["ehvi"] = run_acquisition("ehvi", X, eff, tox, seed=seed,
                                              ehvi_samples=args.ehvi_samples,
                                              shortlist=args.shortlist,
                                              joint_factory=joint_factory, **common)
                want_scatter = (seed == args.seeds[0])
                picks = {}
                # Analysis A
                for label, acq, safety in ANALYSIS_A:
                    sel = nominate(rev[acq], X, eff, tox, mem, safety=safety,
                                   diversity="none", S=S, **nom_common)
                    m = evaluate(sel, eff, tox, mem, X)
                    rows.append(dict(analysis="A", tox_source=src, method=label,
                                     base=label, operator="none", acq=acq,
                                     safety=safety, cell_line=cl, seed=seed, **m))
                    if want_scatter:
                        picks[label] = set(sel)
                # Analysis B
                for base_label, acq, safety in B_BASES:
                    for op in DIVERSITY_OPS:
                        sel = nominate(rev[acq], X, eff, tox, mem, safety=safety,
                                       diversity=op, S=S, **nom_common)
                        m = evaluate(sel, eff, tox, mem, X)
                        rows.append(dict(analysis="B", tox_source=src,
                                         method=f"{base_label}+{op}", base=base_label,
                                         operator=op, acq=acq, safety=safety,
                                         cell_line=cl, seed=seed, **m))
                if want_scatter:
                    sc = pd.DataFrame({"efficacy": eff, "toxicity": tox})
                    for label, pk in picks.items():
                        sc[f"pick_{label}"] = [i in pk for i in range(len(eff))]
                    sc["cell_line"] = cl; sc["tox_source"] = src
                    scatters.append(sc)
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
    meta = dict(panel=panel, n_genes=int(len(eff)), lines=lines, seeds=list(args.seeds),
                K=args.K, tau=args.tau, cap=args.cap, n_rounds=args.n_rounds,
                batch=args.batch, kdpp_sim=args.kdpp_sim, acq_score=args.acq_score,
                joint_gp=bool(args.joint_gp), tox_sources=TOX_SOURCES,
                contrast_line=contrast_line, contrast_ranked=contrast_ranked)
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
            for op in DIVERSITY_OPS:
                sub = d[(d.analysis == "B") & (d.base == base_label) & (d.operator == op)]
                c, _ = _mean_ci(sub["concentration"]); r, _ = _mean_ci(sub["robustness"])
                e, _ = _mean_ci(sub["mean_efficacy"])
                print(f"  {base_label:11s}+{op:5s} conc {c:.3f} robust {r:.3f} eff {e:.3f}")

    try:
        from geneal.report.ablation_report import build_ablation_report
        build_ablation_report(df, out / "report.html", scatter=scatter_df, meta=meta,
                              fig_dir=(out / "figs") if args.export_figs else None)
        print(f"\nreport -> {out / 'report.html'}")
    except Exception as e:
        import traceback; traceback.print_exc(); print("report skipped:", e)
    print(f"run dir: {out}")


if __name__ == "__main__":
    main()
