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


def run_line(ge, emb, cl, K, n_init, tau, caps, thresh, seed, want_scatter=False):
    """Constrained nomination on one cell line: maximize PREDICTED efficacy
    subject to (known) toxicity <= tau, optionally with a per-pathway cap.
    Strategies, all evaluated on the same ground-truth axes:
      efficacy            : top-K by predicted efficacy, NO safety constraint (baseline)
      constrained         : top-K by predicted efficacy s.t. toxicity <= tau
      constrained_cap{c}  : constrained + at most c genes per pathway (full hedge)
    Toxicity = common-essential score (known pan-cancer annotation); efficacy in
    THIS line is predicted by the surrogate (the AL quantity)."""
    ds, aux = build_selective_dataset(ge, emb, cl, lam=0.0, thresh=thresh)  # lam=0: target = raw lethality
    X = StandardScaler().fit_transform(ds.embeddings)
    toxicity = np.asarray(aux["common_essential"])          # 0..1, KNOWN; higher = more toxic
    raw_leth = ds.target                                    # = raw lethality (lam=0)
    mem = corum_membership(ds.gene_names)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(raw_leth)); init = idx[:n_init]
    cand = np.array([i for i in range(len(raw_leth)) if i not in set(init)])
    mem_local = {li: mem[int(cand[li])] for li in range(len(cand))}
    eff_c = raw_leth[cand]; tox_c = toxicity[cand]

    s = GPRSurrogate(n_iters=120).fit(X[init], raw_leth[init])
    m, sd = s.predict(X[cand]); ucb_eff = m + 2 * sd        # PREDICTED efficacy

    def constrain(q, t):
        # disqualify candidates whose KNOWN toxicity exceeds the ceiling
        u = q.copy(); u[tox_c > t] = -1e9
        return u

    methods = [("efficacy", ucb_eff, dict(mode="greedy")),
               ("constrained", constrain(ucb_eff, tau), dict(mode="greedy"))]
    for c in caps:
        methods.append((f"constrained_cap{c}", constrain(ucb_eff, tau), dict(mode="cap", cap=c)))

    rows, picks = [], {}
    for name, q, kw in methods:
        sl = list(HedgedSelect(**kw).select_idx(q, mem_local, K=K))
        picks[name] = sl
        rows.append(dict(
            cell_line=cl, seed=seed, method=name,
            mean_efficacy=float(np.mean(eff_c[sl])),
            max_efficacy=float(np.max(eff_c[sl])),
            mean_toxicity=float(np.mean(tox_c[sl])),
            frac_over_tau=float(np.mean(tox_c[sl] > tau)),
            concentration=pathway_concentration(sl, mem_local),
            robustness=dropout_robustness(sl, mem_local, eff_c),
        ))
    scatter = None
    if want_scatter:
        scatter = pd.DataFrame({"efficacy": eff_c, "toxicity": tox_c,
                                "picked_efficacy": [i in set(picks["efficacy"]) for i in range(len(cand))],
                                "picked_selective": [i in set(picks["constrained"]) for i in range(len(cand))]})
        scatter["cell_line"] = cl
    return rows, scatter


def tau_frontier(ge, emb, cl, K, n_init, taus, thresh, seed):
    """Sweep the safety ceiling tau: maximize PREDICTED efficacy s.t. toxicity<=tau,
    record (mean efficacy, mean toxicity) of the K nominees. tau=1.0 == no
    constraint (efficacy-only). Traces the constrained efficacy-toxicity frontier."""
    ds, aux = build_selective_dataset(ge, emb, cl, lam=0.0, thresh=thresh)
    X = StandardScaler().fit_transform(ds.embeddings)
    tox = np.asarray(aux["common_essential"])
    eff = ds.target
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(eff)); init = idx[:n_init]
    cand = np.array([i for i in range(len(eff)) if i not in set(init)])
    s = GPRSurrogate(n_iters=120).fit(X[init], eff[init])
    m, sd = s.predict(X[cand]); ucb = m + 2 * sd
    ec, tc = eff[cand], tox[cand]
    rows = []
    for tau in taus:
        q = ucb.copy(); q[tc > tau] = -1e9
        sl = list(np.argsort(q)[::-1][:K])
        rows.append(dict(cell_line=cl, seed=seed, tau=tau,
                         mean_efficacy=float(ec[sl].mean()),
                         mean_toxicity=float(tc[sl].mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-effect", default="data/processed/depmap/gene_effect.parquet")
    ap.add_argument("--embeddings", default="data/processed/embeddings/pubmedbert_hvg.parquet")
    ap.add_argument("--n-cell-lines", type=int, default=4)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--n-initial", type=int, default=60)
    ap.add_argument("--caps", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--tau", type=float, default=0.5,
                    help="safety ceiling: max allowed toxicity (common-essential score)")
    ap.add_argument("--thresh", type=float, default=-0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--cell-lines", type=str, nargs="+", default=None,
                    help="explicit ModelIDs to run (overrides --n-cell-lines); "
                         "used by the sharded launcher")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0],
                    help="safety-ceiling sweep for the constrained efficacy-toxicity frontier")
    ap.add_argument("--out-root", default="res/runs_risk")
    ap.add_argument("--run-name", default=None,
                    help="output subdir name (default = timestamp)")
    args = ap.parse_args()

    ge = load_gene_effect(args.gene_effect)
    emb = pd.read_parquet(args.embeddings)
    embset = set(emb.index)
    labs = [g for g in ge.index if parse_entrez(g) in embset]
    if args.cell_lines:
        lines = list(args.cell_lines)
    else:
        lines = ge.loc[labs].isna().sum(0).sort_values().index[:args.n_cell_lines].tolist()
    print("cell lines:", lines)

    all_rows, scatters, pareto = [], [], []
    for cl in lines:
        for sd in args.seeds:
            want_sc = (sd == args.seeds[0])   # one scatter per line (first seed)
            r, sc = run_line(ge, emb, cl, args.K, args.n_initial, args.tau,
                             args.caps, args.thresh, sd, want_scatter=want_sc)
            all_rows += r
            if sc is not None:
                scatters.append(sc)
            pareto += tau_frontier(ge, emb, cl, args.K, args.n_initial,
                                   args.taus, args.thresh, sd)
    df = pd.DataFrame(all_rows)
    run = args.run_name or pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out_root) / run; out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "risk.parquet")
    scatter_df = pd.concat(scatters, ignore_index=True) if scatters else None
    if scatter_df is not None:
        scatter_df.to_parquet(out / "scatter.parquet")
    pareto_df = pd.DataFrame(pareto); pareto_df.to_parquet(out / "tau_frontier.parquet")
    # detailed HTML report
    try:
        from geneal.report.risk_report import build_risk_report
        build_risk_report(df, out / "report.html", scatter=scatter_df,
                          tau_frontier=pareto_df, K=args.K)
        print(f"report -> {out / 'report.html'}")
    except Exception as e:
        import traceback; traceback.print_exc(); print("report skipped:", e)

    order = (["efficacy", "constrained"] +
             [f"constrained_cap{c}" for c in args.caps])
    print(f"\n=== aggregate over {len(lines)} lines x {len(args.seeds)} seeds (mean +/- 95%% CI) ===")
    for m in ["mean_efficacy", "max_efficacy", "mean_toxicity",
              "concentration", "robustness"]:
        print(f"\n[{m}]")
        for meth in order:
            sub = df[df.method == meth][m]
            if len(sub):
                mn, ci = _mean_ci(sub)
                print(f"  {meth:16s} {mn:.3f} +/- {ci:.3f}")
    print(f"\nrun dir: {out}")


if __name__ == "__main__":
    main()
