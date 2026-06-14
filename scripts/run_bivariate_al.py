# scripts/run_bivariate_al.py
"""Bivariate efficacy-toxicity active learning (geneal Plan 6).

Toxicity is UNKNOWN, explored jointly with efficacy via EHVI. Produces BOTH:
  (1) AL dynamics: dominated hypervolume over rounds, for regimes
      learned / known(tox-oracle) / oracle(full-info ceiling), x two toxicity
      references (contrast line, population).
  (2) Final nomination comparison (the risk-report elements): from the learned
      bivariate surrogates, nominate K targets two ways — efficacy_only (top-K by
      predicted efficacy, ignoring toxicity) vs joint (top-K by predicted
      selectivity = efficacy − toxicity) — and evaluate both on TRUE efficacy,
      toxicity, pathway concentration, dropout robustness, + the efficacy-vs-
      toxicity scatter."""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from geneal.data.depmap import load_gene_effect, parse_entrez
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.runner.bivariate import BivariateALRunner
from geneal.models.hedged_selection import HedgedSelect
from geneal.metrics.portfolio import pathway_concentration, dropout_robustness

_SYM = re.compile(r"^(.*?)\s*\(\d+\)$")


def corum_membership(entrez_ids, cache="data/processed/depmap/corum_membership_all.parquet"):
    m = pd.read_parquet(cache)
    e2c = {}
    for ent, cid in zip(m["entrez"], m["complex_id"]):
        e2c.setdefault(int(ent), set()).add(int(cid))
    return {i: e2c.get(int(e), set()) for i, e in enumerate(entrez_ids)}


def _mean_ci(x):
    x = np.asarray(x, float); n = len(x); m = float(x.mean())
    return m, (0.0 if n < 2 else 1.96 * float(x.std(ddof=1)) / np.sqrt(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-effect", default="data/processed/depmap/gene_effect.parquet")
    ap.add_argument("--embeddings", default="data/processed/embeddings/pubmedbert_all.parquet")
    ap.add_argument("--panel", default="data/processed/depmap/panel_hvg.txt")
    ap.add_argument("--n-cell-lines", type=int, default=4)
    ap.add_argument("--cell-lines", type=str, nargs="+", default=None)
    ap.add_argument("--contrast-line", default=None)
    ap.add_argument("--n-initial", type=int, default=40)
    ap.add_argument("--n-rounds", type=int, default=8)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--ehvi-samples", type=int, default=32)
    ap.add_argument("--shortlist", type=int, default=150)
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
    contrast = args.contrast_line or cov.index[0]
    target_lines = (args.cell_lines or
                    [c for c in cov.index if c != contrast][:args.n_cell_lines])
    print(f"contrast (toxicity proxy) line: {contrast}\ntarget lines: {target_lines}")

    sub = ge.loc[labs]
    ent = np.array([parse_entrez(g) for g in labs])
    X = StandardScaler().fit_transform(emb.loc[ent].to_numpy())
    membership = corum_membership(ent)
    pop_tox = -(sub.mean(axis=1).to_numpy())
    tox_contrast = -(sub[contrast].to_numpy())
    runner = BivariateALRunner(lambda: GPRSurrogate(n_iters=80),
                               GaussianNoise(args.sigma),
                               ehvi_samples=args.ehvi_samples, shortlist=args.shortlist)

    hv_rows, nom_rows, scatters = [], [], []
    for cl in target_lines:
        eff_full = -(sub[cl].to_numpy())
        for ref_name, tox_full in [("contrast", tox_contrast), ("population", pop_tox)]:
            ok = ~np.isnan(eff_full) & ~np.isnan(tox_full)
            Xs, es, ts, ent_s = X[ok], eff_full[ok], tox_full[ok], ent[ok]
            mem = {i: membership[j] for i, j in enumerate(np.where(ok)[0])}
            for regime, kw in [("learned", {}), ("known", {"tox_known": True}),
                               ("oracle", {"tox_known": True, "eff_known": True})]:
                for sd in args.seeds:
                    hist = runner.run(Xs, es, ts, args.n_initial, args.n_rounds,
                                      args.batch, sd, **kw)
                    for h in hist:
                        hv_rows.append(dict(cell_line=cl, ref=ref_name,
                                            condition=f"{ref_name}_{regime}", seed=sd,
                                            round=h["round"], hypervolume=h["hypervolume"]))
                    # --- final nomination (only for the 'learned' regime) ---
                    if regime == "learned":
                        revealed = hist[-1]["revealed"]
                        se = GPRSurrogate(n_iters=80).fit(Xs[revealed], es[revealed])
                        st = GPRSurrogate(n_iters=80).fit(Xs[revealed], ts[revealed])
                        pe, _ = se.predict(Xs); pt, _ = st.predict(Xs)
                        K = args.K
                        sel_score = pe - pt
                        eff_pick = list(np.argsort(pe)[::-1][:K])               # efficacy-only (uncapped)
                        joint_pick = list(np.argsort(sel_score)[::-1][:K])      # selectivity (uncapped, any pathways)
                        joint_cap_pick = HedgedSelect(mode="cap", cap=2).select_idx(
                            sel_score, mem, K=K)                                 # selectivity + pathway hedge
                        for mname, pick in [("efficacy_only", eff_pick),
                                            ("joint", joint_pick),
                                            ("joint_cap2", joint_cap_pick)]:
                            nom_rows.append(dict(
                                cell_line=cl, ref=ref_name, seed=sd, method=mname,
                                mean_efficacy=float(es[pick].mean()),
                                max_efficacy=float(es[pick].max()),
                                mean_toxicity=float(ts[pick].mean()),
                                concentration=pathway_concentration(pick, mem),
                                robustness=dropout_robustness(pick, mem, es)))
                        if sd == args.seeds[0] and ref_name == "contrast":
                            scatters.append(pd.DataFrame({
                                "efficacy": es, "toxicity": ts,
                                "picked_efficacy": [i in set(eff_pick) for i in range(len(es))],
                                "picked_selective": [i in set(joint_pick) for i in range(len(es))],
                                "cell_line": cl}))
            print(f"  done {cl} / {ref_name}")

    hv = pd.DataFrame(hv_rows); nom = pd.DataFrame(nom_rows)
    scatter = pd.concat(scatters, ignore_index=True) if scatters else None
    run = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out_root) / run; out.mkdir(parents=True, exist_ok=True)
    hv.to_parquet(out / "hypervolume.parquet"); nom.to_parquet(out / "nomination.parquet")
    if scatter is not None:
        scatter.to_parquet(out / "scatter.parquet")
    try:
        from geneal.report.bivariate_report import build_bivariate_report
        build_bivariate_report(hv, nom, out / "report.html", scatter=scatter,
                               contrast=contrast, K=args.K)
        print(f"report -> {out / 'report.html'}")
    except Exception as e:
        import traceback; traceback.print_exc(); print("report skipped:", e)
    print(f"\nrun dir: {out}")


if __name__ == "__main__":
    main()
