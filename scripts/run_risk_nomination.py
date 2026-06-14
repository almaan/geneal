# scripts/run_risk_nomination.py
"""Risk-aware target nomination experiment (geneal Plan 5).

Select K knockout targets that are (1) lethal in a cancer line, (2) selective
(not common-essential = toxicity hedge), (3) mechanistically hedged (per-pathway
cap, to insure against pathway-level toxicity/failure). Compares greedy vs
per-pathway-cap selection across a cap sweep; reports the efficacy-risk frontier
(selective-lethality vs pathway-concentration vs dropout-robustness)."""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from geneal.data.depmap import load_gene_effect, parse_entrez
from geneal.data.selective import build_selective_dataset
from geneal.models.surrogate import GPRSurrogate
from geneal.models.hedged_selection import HedgedSelect
from geneal.metrics.portfolio import (n_pathways_covered, pathway_concentration,
                                      dropout_robustness)

_SYM = re.compile(r"^(.*?)\s*\(\d+\)$")


def gene_symbol(label: str) -> str:
    m = _SYM.match(label)
    return (m.group(1) if m else label).strip()


def corum_membership(gene_names, corum_path="data/corum_dl/humanComplexes.txt"):
    """gene_idx -> set of CORUM complex_ids the gene belongs to (matched by symbol)."""
    df = pd.read_csv(corum_path, sep="\t")
    sym2complexes: dict = {}
    for _, r in df.iterrows():
        cid = int(r["complex_id"])
        subs = str(r.get("subunits_gene_name", "") or "")
        for s in re.split(r"[;,]", subs):
            s = s.strip()
            if s:
                sym2complexes.setdefault(s, set()).add(cid)
    membership = {}
    for i, lab in enumerate(gene_names):
        membership[i] = sym2complexes.get(gene_symbol(lab), set())
    return membership


def _mean_ci(x):
    x = np.asarray(x, float); n = len(x); m = float(x.mean())
    return m, (0.0 if n < 2 else 1.96 * float(x.std(ddof=1)) / np.sqrt(n))


def run_line(ge, emb, cl, membership_full, K, n_init, caps, lam, thresh, seed):
    ds, aux = build_selective_dataset(ge, emb, cl, lam=lam, thresh=thresh)
    X = StandardScaler().fit_transform(ds.embeddings); y = ds.target
    # membership re-indexed to THIS dataset's gene order
    mem = corum_membership(ds.gene_names)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y)); init = idx[:n_init]
    cand = np.array([i for i in range(len(y)) if i not in set(init)])
    surr = GPRSurrogate(n_iters=120).fit(X[init], y[init])
    mean, std = surr.predict(X[cand]); ucb = mean + 2 * std
    # selection methods operate over candidate-local indices; map membership/value local
    mem_local = {li: mem[int(cand[li])] for li in range(len(cand))}
    yval_local = y[cand]
    rows = []
    methods = [("greedy", dict(mode="greedy"))]
    for c in caps:
        methods.append((f"cap{c}", dict(mode="cap", cap=c)))
    for name, kw in methods:
        sel_local = HedgedSelect(**kw).select_idx(ucb, mem_local, K=K)
        sel_local = list(sel_local)
        # portfolio value = TRUE selective lethality of picked genes (honest eval)
        val = yval_local
        rows.append(dict(
            cell_line=cl, seed=seed, method=name,
            selective_lethality=float(np.mean(yval_local[sel_local])),
            concentration=pathway_concentration(sel_local, mem_local),
            robustness=dropout_robustness(sel_local, mem_local, val),
            n_pathways=n_pathways_covered(sel_local, mem_local),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-effect", default="data/processed/depmap/gene_effect.parquet")
    ap.add_argument("--embeddings", default="data/processed/embeddings/pubmedbert_hvg.parquet")
    ap.add_argument("--n-cell-lines", type=int, default=4)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--n-initial", type=int, default=60)
    ap.add_argument("--caps", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--thresh", type=float, default=-0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out-root", default="res/runs_risk")
    args = ap.parse_args()

    ge = load_gene_effect(args.gene_effect)
    emb = pd.read_parquet(args.embeddings)
    embset = set(emb.index)
    labs = [g for g in ge.index if parse_entrez(g) in embset]
    lines = ge.loc[labs].isna().sum(0).sort_values().index[:args.n_cell_lines].tolist()
    print("cell lines:", lines)

    all_rows = []
    for cl in lines:
        for sd in args.seeds:
            all_rows += run_line(ge, emb, cl, None, args.K, args.n_initial,
                                 args.caps, args.lam, args.thresh, sd)
    df = pd.DataFrame(all_rows)
    run = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out_root) / run; out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "risk.parquet")

    order = ["greedy"] + [f"cap{c}" for c in args.caps]
    print(f"\n=== aggregate over {len(lines)} lines x {len(args.seeds)} seeds (mean +/- 95%% CI) ===")
    for m in ["selective_lethality", "concentration", "robustness", "n_pathways"]:
        print(f"\n[{m}]")
        for meth in order:
            sub = df[df.method == meth][m]
            mn, ci = _mean_ci(sub)
            print(f"  {meth:8s} {mn:.3f} +/- {ci:.3f}")

    # figures
    try:
        import plotly.graph_objects as go
        agg = df.groupby("method").agg(
            leth=("selective_lethality", "mean"), conc=("concentration", "mean"),
            rob=("robustness", "mean")).reindex(order)
        # Fig: efficacy-risk Pareto (concentration x, lethality y)
        fig = go.Figure(go.Scatter(x=agg["conc"], y=agg["leth"], mode="markers+text",
                                   text=agg.index, textposition="top center",
                                   marker=dict(size=12)))
        fig.update_layout(template="simple_white",
                          xaxis_title="pathway concentration (max frac in one pathway) — RISK",
                          yaxis_title="selective lethality (efficacy)",
                          title="Efficacy–risk frontier: greedy vs per-pathway cap")
        fig.write_html(out / "pareto.html")
    except Exception as e:
        print("figure skipped:", e)
    print(f"\nrun dir: {out}")


if __name__ == "__main__":
    main()
