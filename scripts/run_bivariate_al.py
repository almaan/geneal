# scripts/run_bivariate_al.py
"""Bivariate efficacy-toxicity active learning (geneal Plan 6).

Toxicity is UNKNOWN and explored jointly with efficacy via EHVI. Two toxicity
references x two regimes:
  contrast_learned / contrast_known   : toxicity = lethality in a FIXED proxy line
  population_learned / population_known: toxicity = mean lethality across the population
'known' = toxicity oracle (the a-priori ceiling); 'learned' = toxicity surrogate
(explored). Tracks dominated hypervolume + true selective-top-k recall over rounds.
Efficacy = -effect in the target line (learned in all conditions)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from geneal.data.depmap import load_gene_effect, parse_entrez
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.runner.bivariate import BivariateALRunner


def _mean_ci(x):
    x = np.asarray(x, float); n = len(x); m = float(x.mean())
    return m, (0.0 if n < 2 else 1.96 * float(x.std(ddof=1)) / np.sqrt(n))


def selective_recall_history(hist_revealed, eff, tox, k):
    """recall of the true selective-top-k (by efficacy - toxicity) per round."""
    sel = eff - tox
    true_top = set(np.argsort(sel)[::-1][:k].tolist())
    return [len(set(rv) & true_top) / k for rv in hist_revealed]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-effect", default="data/processed/depmap/gene_effect.parquet")
    ap.add_argument("--embeddings", default="data/processed/embeddings/pubmedbert_all.parquet")
    ap.add_argument("--panel", default="data/processed/depmap/panel_hvg.txt")
    ap.add_argument("--n-cell-lines", type=int, default=4)
    ap.add_argument("--cell-lines", type=str, nargs="+", default=None)
    ap.add_argument("--contrast-line", default=None,
                    help="fixed normal-ish proxy line (default: highest-coverage line)")
    ap.add_argument("--n-initial", type=int, default=40)
    ap.add_argument("--n-rounds", type=int, default=8)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--ehvi-samples", type=int, default=48)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out-root", default="res/runs_bivariate")
    args = ap.parse_args()

    ge = load_gene_effect(args.gene_effect)
    emb = pd.read_parquet(args.embeddings)
    keep = set(int(x) for x in Path(args.panel).read_text().split())
    emb = emb[emb.index.isin(keep)]
    embset = set(emb.index)
    labs = [g for g in ge.index if parse_entrez(g) in embset]
    cov = ge.loc[labs].isna().sum(0).sort_values()
    contrast = args.contrast_line or cov.index[0]   # highest-coverage line as proxy
    target_lines = (args.cell_lines or
                    [c for c in cov.index if c != contrast][:args.n_cell_lines])
    print(f"contrast (toxicity proxy) line: {contrast}")
    print(f"target lines: {target_lines}")

    # gene matrix aligned to embedded genes
    sub = ge.loc[labs]
    ent = np.array([parse_entrez(g) for g in labs])
    X = StandardScaler().fit_transform(emb.loc[ent].to_numpy())
    pop_tox = -(sub.mean(axis=1).to_numpy())        # population mean lethality (toxicity)
    tox_contrast = -(sub[contrast].to_numpy())      # lethality in proxy line (toxicity)

    runner = BivariateALRunner(lambda: GPRSurrogate(n_iters=80),
                               GaussianNoise(args.sigma), ehvi_samples=args.ehvi_samples)
    rows = []
    for cl in target_lines:
        eff = -(sub[cl].to_numpy())
        ok = ~np.isnan(eff)
        for ref_name, tox in [("contrast", tox_contrast), ("population", pop_tox)]:
            ok2 = ok & ~np.isnan(tox)
            Xs, es, ts = X[ok2], eff[ok2], tox[ok2]
            for regime, known in [("learned", False), ("known", True)]:
                for sd in args.seeds:
                    # capture revealed sets per round by wrapping: re-run reveals via hist
                    hist = runner.run(Xs, es, ts, args.n_initial, args.n_rounds,
                                      args.batch, sd, tox_known=known)
                    for h in hist:
                        rows.append(dict(cell_line=cl, ref=ref_name, regime=regime,
                                         condition=f"{ref_name}_{regime}", seed=sd,
                                         round=h["round"], hypervolume=h["hypervolume"],
                                         n_revealed=h["n_revealed"]))
            print(f"  done {cl} / {ref_name}")
    df = pd.DataFrame(rows)
    run = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out_root) / run; out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "bivariate.parquet")

    try:
        from geneal.report.bivariate_report import build_bivariate_report
        build_bivariate_report(df, out / "report.html", contrast=contrast)
        print(f"report -> {out / 'report.html'}")
    except Exception as e:
        import traceback; traceback.print_exc(); print("report skipped:", e)

    last = df["round"].max()
    print(f"\n=== final hypervolume (round {last}) mean +/- 95%% CI ===")
    for cond in ["contrast_known", "contrast_learned", "population_known", "population_learned"]:
        s = df[(df.condition == cond) & (df["round"] == last)]["hypervolume"]
        if len(s):
            mn, ci = _mean_ci(s); print(f"  {cond:20s} {mn:.4f} +/- {ci:.4f}")
    print(f"\nrun dir: {out}")


if __name__ == "__main__":
    main()
