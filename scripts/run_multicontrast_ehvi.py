# scripts/run_multicontrast_ehvi.py
"""EXPLORATORY (Part 1 only): multi-objective selectivity against MULTIPLE contrast
lines simultaneously, via hypervolume optimization.

EHVI/hypervolume optimization is not limited to 2 objectives. Here we nominate K
targets that are simultaneously (a) lethal in the target cancer line and (b) NOT
lethal in each of 3 fixed contrast (normal-stand-in) lines — i.e. a 4-objective
problem: maximize (target efficacy, -tox_c1, -tox_c2, -tox_c3). We compare the
multi-objective hypervolume nominator against efficacy-only and a single-objective
'avg-selectivity' baseline, and report mean target efficacy, mean lethality in
EACH contrast line, and their average, plus the 4-D dominated hypervolume.

Standalone report (does NOT touch the main ablation). If it looks good we integrate.
Surrogate predicts each objective from embeddings (single-shot: fit on a random
init set, predict the candidate pool); metrics are evaluated on TRUE values.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from geneal.data.depmap import load_gene_effect, parse_entrez
from geneal.data.selective import rank_contrast_lines
from geneal.runner.multicontrast import run_line


def _mean_ci(x):
    x = np.asarray(x, float); n = len(x)
    m = float(x.mean())
    return m, (0.0 if n < 2 else 1.96 * float(x.std(ddof=1)) / np.sqrt(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-effect", default="data/processed/depmap/gene_effect.parquet")
    ap.add_argument("--embeddings", default="data/processed/embeddings/pubmedbert_all.parquet")
    ap.add_argument("--panel", default="data/processed/depmap/panel_5k.txt")
    ap.add_argument("--n-cell-lines", type=int, default=12)
    ap.add_argument("--cell-lines", nargs="+", default=None,
                    help="explicit target ModelIDs (overrides --n-cell-lines; for sharding)")
    ap.add_argument("--n-contrasts", type=int, default=3)
    ap.add_argument("--contrast-lines", nargs="+", default=None)
    ap.add_argument("--n-initial", type=int, default=300)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--hv-samples", type=int, default=40000)
    ap.add_argument("--out-root", default="res/runs_multicontrast")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    ge = load_gene_effect(args.gene_effect)
    emb = pd.read_parquet(args.embeddings)
    if args.panel and args.panel.lower() != "none":          # 'none' = full genome
        keep = set(int(x) for x in Path(args.panel).read_text().split())
        emb = emb[emb.index.isin(keep)]
    labs = [g for g in ge.index if parse_entrez(g) in set(emb.index)]
    contrasts = (args.contrast_lines or
                 rank_contrast_lines(ge, n=max(10, args.n_contrasts))[:args.n_contrasts])
    cset = set(contrasts)
    ranked = ge.loc[labs].isna().sum(axis=0).sort_values().index.tolist()
    if args.cell_lines:
        lines = [c for c in args.cell_lines if c not in cset]
    else:
        lines = [c for c in ranked if c not in cset][:args.n_cell_lines]
    print(f"contrasts ({len(contrasts)}): {contrasts}")
    print(f"target lines ({len(lines)}): {lines}")

    rows = []
    for cl in lines:
        for sd in args.seeds:
            rows += run_line(ge, emb, cl, contrasts, args.n_initial, args.K, sd, args.hv_samples)
        print(f"[{cl}] done")
    df = pd.DataFrame(rows)
    run = args.run_name or pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out_root) / run; out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "multicontrast.parquet")

    order = ["greedy", "random", "cluster", "info_div", "ehvi", "ehvi_pareto"]
    metrics = ["mean_efficacy"] + [f"tox_c{j+1}" for j in range(len(contrasts))] + ["tox_avg", "hv4d"]
    print(f"\n=== {len(lines)} lines x {len(args.seeds)} seeds (mean +/- 95% CI) ===")
    for m in metrics:
        print(f"\n[{m}]")
        for meth in order:
            sub = df[df.method == meth][m]
            mn, ci = _mean_ci(sub)
            print(f"  {meth:18s} {mn:.3f} +/- {ci:.3f}")

    _write_report(df, out / "report.html", contrasts, lines, args.seeds, order, metrics)
    print(f"\nreport -> {out / 'report.html'}")
    print(f"run dir: {out}")


def _write_report(df, path, contrasts, lines, seeds, order, metrics):
    nobj = 1 + len(contrasts)
    def cell(meth, m):
        mn, ci = _mean_ci(df[df.method == meth][m])
        return f"{mn:.3f} ± {ci:.3f}"
    head = "".join(f"<th>{m}</th>" for m in metrics)
    body = ""
    for meth in order:
        tds = "".join(f"<td>{cell(meth, m)}</td>" for m in metrics)
        hl = ' style="background:#eaf2fb"' if meth in ("ehvi", "ehvi_pareto") else ""
        body += f"<tr{hl}><td>{meth}</td>{tds}</tr>"
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;color:#1a1a1a}}
table{{border-collapse:collapse}} td,th{{border-bottom:1px solid #e8ebef;padding:.4rem .8rem;text-align:right}}
td:first-child,th:first-child{{text-align:left;font-weight:600}} .note{{color:#6b7280;font-size:.9rem;max-width:60rem}}</style></head><body>
<h1>Multi-contrast selectivity via hypervolume (exploratory, Part 1)</h1>
<div class="note">{nobj}-objective nomination via a JOINT multitask GP over (target efficacy, lethality in each of {len(contrasts)} contrast lines: {', '.join(contrasts)}); maximize efficacy, minimize each contrast lethality simultaneously. Methods mirror the Section-A main table WITHOUT a safety filter: <b>ehvi</b> = posterior-integrated N-D EHVI over the joint-GP posterior; <b>ehvi_pareto</b> = predicted 4-D Pareto-front nomination; <b>greedy</b> = top-K target efficacy; <b>random/cluster/info_div</b> = naive/diversity baselines. Metrics on TRUE values, mean ± 95% CI over {len(lines)} target lines × {len(seeds)} seeds. tox_c* = mean lethality in contrast line * (↓ better); hv4d = {nobj}-D dominated hypervolume (↑ better).</div>
<table><thead><tr><th>method</th>{head}</tr></thead><tbody>{body}</tbody></table>
</body></html>"""
    Path(path).write_text(html)


if __name__ == "__main__":
    main()
