# scripts/run_ablation.py
"""Default ablation experiment (geneal Plan 7).

TWO analyses on real DepMap, full active-learning loop per method:

  A) Safety vs efficacy -- one axis (the safety rule): none / truncation
     (known-toxicity ceiling) / ehvi (learned-toxicity ceiling via a
     dual-objective EHVI acquisition) + random / coreset / typiclust baselines.
     6 methods, all on the efficacy-toxicity tradeoff.

  B) Diversity / robustness -- the operators none / cap (CORUM per-pathway) /
     kdpp (STRING-similarity k-DPP), layered on TWO bases: greedy (naive top-K)
     and best-safety (the winner of Analysis A). 6 variants.

Efficiency: the 5 distinct acquisitions run ONCE per (line, seed); both analyses'
nomination rules are applied to the revealed sets. Toxicity ground truth =
common-essential (known a priori); EHVI additionally LEARNS a toxicity GP.

PubMedBERT is the efficacy/toxicity predictor; CORUM gives pathway membership;
STRING gives the k-DPP similarity. Run on a 5k panel (default) or the full
genome (--panel off) -- same report.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from geneal.data.depmap import load_gene_effect, parse_entrez
from geneal.data.selective import build_selective_dataset, corum_membership
from geneal.runner.ablation import (run_acquisition, nominate, evaluate,
                                    build_string_S, ACQUISITIONS)

# Analysis A: (label, acquisition, safety). diversity is always "none" here.
ANALYSIS_A = [
    ("greedy",     "greedy",    "none"),        # naive top-K efficacy
    ("truncation", "greedy",    "truncation"),  # known-toxicity ceiling
    ("ehvi",       "ehvi",      "ehvi"),         # dual-objective AL + learned ceiling
    ("random",     "random",    "none"),         # naive baseline
    ("coreset",    "coreset",   "none"),         # IterPert-style
    ("typiclust",  "typiclust", "none"),         # IterPert best baseline
]
# safety label -> (acquisition, safety) for resolving the Analysis-B base
_SAFETY_BASE = {"none": ("greedy", "none"),
                "truncation": ("greedy", "truncation"),
                "ehvi": ("ehvi", "ehvi")}
DIVERSITY_OPS = ["none", "cap", "kdpp"]


def _factory(n_iters):
    from geneal.models.surrogate import GPRSurrogate
    return lambda: GPRSurrogate(n_iters=n_iters)


def _mean_ci(x):
    x = np.asarray(x, float); n = len(x); m = float(x.mean())
    return m, (0.0 if n < 2 else 1.96 * float(x.std(ddof=1)) / np.sqrt(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-effect", default="data/processed/depmap/gene_effect.parquet")
    ap.add_argument("--embeddings", default="data/processed/embeddings/pubmedbert_all.parquet",
                    help="PubMedBERT embedding parquet (genome-wide cache; use --panel to subset)")
    ap.add_argument("--panel", default="data/processed/depmap/panel_5k.txt",
                    help="entrez-id panel file; pass '' or 'none' for the full genome")
    ap.add_argument("--string", default="data/processed/depmap/string_edges_all.parquet")
    ap.add_argument("--n-cell-lines", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--n-initial", type=int, default=40)
    ap.add_argument("--n-rounds", type=int, default=8)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.5, help="toxicity ceiling")
    ap.add_argument("--thresh", type=float, default=-0.5, help="common-essential lethality threshold")
    ap.add_argument("--cap", type=int, default=2, help="per-pathway cap for the cap operator")
    ap.add_argument("--pool", type=int, default=200, help="quality pool the diversity operator selects within")
    ap.add_argument("--ehvi-samples", type=int, default=32)
    ap.add_argument("--shortlist", type=int, default=150)
    ap.add_argument("--n-iters", type=int, default=120, help="GP training iterations")
    ap.add_argument("--cell-lines", type=str, nargs="+", default=None,
                    help="explicit ModelIDs (overrides --n-cell-lines)")
    ap.add_argument("--out-root", default="res/runs_ablation")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    factory = _factory(args.n_iters)

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
    if args.cell_lines:
        lines = list(args.cell_lines)
    else:
        lines = ge.loc[labs].isna().sum(0).sort_values().index[:args.n_cell_lines].tolist()
    print(f"cell lines ({len(lines)}): {lines}")

    common = dict(n_init=args.n_initial, n_rounds=args.n_rounds, batch=args.batch,
                  surr_factory=factory, ehvi_samples=args.ehvi_samples,
                  shortlist=args.shortlist)
    nom_common = dict(K=args.K, tau=args.tau, surr_factory=factory, cap=args.cap,
                      pool=args.pool)

    rows, scatters = [], []
    # cache per line: X, eff, tox, membership, S; per (line,seed): revealed[kind]
    revealed_cache = {}
    line_cache = {}

    # ---- Pass 1: run acquisitions once per (line, seed); build Analysis A ----
    for cl in lines:
        ds, aux = build_selective_dataset(ge, emb, cl, lam=0.0, thresh=args.thresh)
        X = StandardScaler().fit_transform(ds.embeddings)
        eff = np.asarray(ds.target, float)            # raw lethality (lam=0)
        tox = np.asarray(aux["common_essential"], float)
        mem = corum_membership(ds.gene_names)
        S = build_string_S(ds.gene_names, args.string)
        line_cache[cl] = (X, eff, tox, mem, S)
        print(f"[{cl}] {len(eff)} genes, {sum(bool(v) for v in mem.values())} CORUM-annotated")

        for seed in args.seeds:
            rev = {k: run_acquisition(k, X, eff, tox, seed=seed, **common)
                   for k in ACQUISITIONS}
            revealed_cache[(cl, seed)] = rev
            want_scatter = (cl == lines[0] and seed == args.seeds[0])
            picks_for_scatter = {}
            for label, acq, safety in ANALYSIS_A:
                sel = nominate(rev[acq], X, eff, tox, mem, safety=safety,
                               diversity="none", S=S, **nom_common)
                m = evaluate(sel, eff, tox, mem, X)
                rows.append(dict(analysis="A", method=label, base=label,
                                 operator="none", acq=acq, safety=safety,
                                 cell_line=cl, seed=seed, **m))
                if want_scatter:
                    picks_for_scatter[label] = set(sel)
            if want_scatter:
                sc = pd.DataFrame({"efficacy": eff, "toxicity": tox})
                for label in picks_for_scatter:
                    sc[f"pick_{label}"] = [i in picks_for_scatter[label]
                                           for i in range(len(eff))]
                sc["cell_line"] = cl
                scatters.append(sc)
        print(f"[{cl}] Analysis A done ({len(args.seeds)} seeds)")

    # ---- Resolve the winning SAFETY RULE for Analysis B's second base ----
    # Candidates are the two real safety rules (truncation, ehvi); greedy(none) is
    # already the other base. The winner is the rule that retains the most efficacy
    # while meeting the safety bar (mean toxicity <= tau); if neither meets it, the
    # safest one.
    dfA = pd.DataFrame([r for r in rows if r["analysis"] == "A"])
    stat = {}
    for safety in ("truncation", "ehvi"):
        sub = dfA[dfA["safety"] == safety]
        if len(sub):
            stat[safety] = (float(sub["mean_efficacy"].mean()),
                            float(sub["mean_toxicity"].mean()))
    valid = {s: v for s, v in stat.items() if v[1] <= args.tau}
    if valid:
        best_safety = max(valid, key=lambda s: valid[s][0])      # most efficacy, safe
    elif stat:
        best_safety = min(stat, key=lambda s: stat[s][1])        # else safest
    else:
        best_safety = "truncation"
    util = {s: round(e - t, 3) for s, (e, t) in stat.items()}
    print(f"\nsafety rules (eff, tox): { {s: (round(e,3),round(t,3)) for s,(e,t) in stat.items()} } "
          f"-> best-safety base = {best_safety}")

    # ---- Pass 2: Analysis B -- operators on TWO bases (greedy + best-safety) ----
    bases = [("greedy", "greedy", "none")]
    if best_safety != "none":
        b_acq, b_saf = _SAFETY_BASE[best_safety]
        bases.append((f"best-safety[{best_safety}]", b_acq, b_saf))
    for cl in lines:
        X, eff, tox, mem, S = line_cache[cl]
        for seed in args.seeds:
            rev = revealed_cache[(cl, seed)]
            for base_label, acq, safety in bases:
                for op in DIVERSITY_OPS:
                    sel = nominate(rev[acq], X, eff, tox, mem, safety=safety,
                                   diversity=op, S=S, **nom_common)
                    m = evaluate(sel, eff, tox, mem, X)
                    rows.append(dict(analysis="B", method=f"{base_label}+{op}",
                                     base=base_label, operator=op, acq=acq,
                                     safety=safety, cell_line=cl, seed=seed, **m))
    print("Analysis B done")

    # ---- Persist + report ----
    df = pd.DataFrame(rows)
    run = args.run_name or pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out_root) / run
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "ablation.parquet")
    scatter_df = pd.concat(scatters, ignore_index=True) if scatters else None
    if scatter_df is not None:
        scatter_df.to_parquet(out / "scatter.parquet")
    meta = dict(panel=panel, n_genes=int(len(line_cache[lines[0]][1])),
                lines=lines, seeds=list(args.seeds), K=args.K, tau=args.tau,
                cap=args.cap, n_rounds=args.n_rounds, batch=args.batch,
                best_safety=best_safety, safety_utility=util)
    import json
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n=== aggregate over {len(lines)} lines x {len(args.seeds)} seeds (mean +/- 95% CI) ===")
    print("\n[Analysis A: safety vs efficacy]")
    for label, _, _ in ANALYSIS_A:
        sub = df[(df.analysis == "A") & (df.method == label)]
        e, ec = _mean_ci(sub["mean_efficacy"]); t, tc = _mean_ci(sub["mean_toxicity"])
        print(f"  {label:12s} eff {e:.3f}+/-{ec:.3f}  tox {t:.3f}+/-{tc:.3f}")
    print("\n[Analysis B: diversity / robustness]")
    for base_label, _, _ in bases:
        for op in DIVERSITY_OPS:
            sub = df[(df.analysis == "B") & (df.base == base_label) & (df.operator == op)]
            c, cc = _mean_ci(sub["concentration"]); r, rc = _mean_ci(sub["robustness"])
            e, _ = _mean_ci(sub["mean_efficacy"])
            print(f"  {base_label:18s}+{op:5s}  conc {c:.3f}+/-{cc:.3f}  "
                  f"robust {r:.3f}+/-{rc:.3f}  eff {e:.3f}")

    try:
        from geneal.report.ablation_report import build_ablation_report
        build_ablation_report(df, out / "report.html", scatter=scatter_df, meta=meta)
        print(f"\nreport -> {out / 'report.html'}")
    except Exception as e:
        import traceback; traceback.print_exc(); print("report skipped:", e)
    print(f"run dir: {out}")


if __name__ == "__main__":
    main()
