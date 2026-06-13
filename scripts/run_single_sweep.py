# scripts/run_single_sweep.py
"""Single-objective metric-panel sweep over the lowest-NaN cell lines.

Runs all 6 methods x seeds for each chosen cell line in parallel (one process
per cell line) and records the evaluation-metric panel (max-value, diversity,
alpha-NDCG) alongside recall@k. Builds an elegant v2 report and prints the
final-round mean +/- 95% CI per metric per method (across lines and seeds).

The build_experiment factory is a TOP-LEVEL picklable callable; under the
'spawn' start method used by run_sweep it reconstructs the Experiment inside
each worker from a module-level CFG dict populated in main() before the sweep.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from geneal.data.depmap import load_gene_effect, parse_entrez, build_cell_line_dataset
from geneal.experiment.objects import ObjectiveObject, DesignObject, Method, Experiment
from geneal.metrics.recall import RecallAtK
from geneal.metrics.panel import MaxValue, CumulativeDiversity, AlphaNDCG
from geneal.models.acquisition import UCB, RandomAcquisition, GreedyMean
from geneal.models.selection import KDPP, TopQGreedy, GreedyFantasy, CoreSet, TypiClust
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.runner.sweep import run_sweep
from geneal.report.report2 import build_report2


def _build_methods() -> list[Method]:
    """Six methods, each with its OWN fresh GPRSurrogate(n_iters=100)."""
    def gpr():
        return GPRSurrogate(n_iters=100)
    return [
        Method("kdpp", UCB(2.0), KDPP("greedy"), gpr()),
        Method("greedy", UCB(2.0), TopQGreedy(), gpr()),
        Method("fantasy", UCB(2.0), GreedyFantasy(), gpr()),
        Method("coreset", GreedyMean(), CoreSet(), gpr()),
        Method("typiclust", GreedyMean(), TypiClust(), gpr()),
        Method("random", RandomAcquisition(), TopQGreedy(), gpr()),
    ]


class ExperimentBuilder:
    """Top-level, picklable factory carrying its config in instance state.

    run_sweep pickles the (builder, cell_line, ...) task tuple and ships it to a
    worker started under the 'spawn' method. Because the config lives on the
    instance (not a module global), it survives pickling and is available in the
    worker WITHOUT main() re-running there. Confirmed picklable under spawn.
    """
    def __init__(self, gene_effect, embeddings, k, rounds, batch, n_initial,
                 sigma):
        self.gene_effect = gene_effect
        self.embeddings = embeddings
        self.k = k
        self.rounds = rounds
        self.batch = batch
        self.n_initial = n_initial
        self.sigma = sigma

    def __call__(self, cell_line: str) -> Experiment:
        ge = load_gene_effect(self.gene_effect)
        emb = pd.read_parquet(self.embeddings)
        ds = build_cell_line_dataset(ge, emb, cell_line=cell_line)
        return Experiment(
            dataset=ds,
            objective=ObjectiveObject(RecallAtK(k=self.k), "maximize"),
            design=DesignObject(n_rounds=self.rounds, batch_size=self.batch,
                                n_initial=self.n_initial, seed=0),
            noise=GaussianNoise(self.sigma),
            methods=_build_methods(),
        )


def _mean_ci(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = float(np.mean(x))
    if n < 2:
        return mean, 0.0
    se = float(np.std(x, ddof=1)) / np.sqrt(n)
    return mean, 1.96 * se


def choose_lowest_nan_lines(ge: pd.DataFrame, emb: pd.DataFrame, n: int) -> list[str]:
    """Pick the n cell-line columns with the FEWEST NaNs among embedded genes."""
    emb_index = set(emb.index)
    embedded = [g for g in ge.index if parse_entrez(g) in emb_index]
    sub = ge.loc[embedded]
    nan_counts = sub.isna().sum(axis=0).sort_values()
    return list(nan_counts.index[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cell-lines", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--n-initial", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n-clusters", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--gene-effect", default="data/processed/depmap/gene_effect.parquet")
    ap.add_argument("--embeddings", default="data/processed/embeddings/esm2_small.parquet")
    ap.add_argument("--model-csv", default="data/raw/depmap/Model.csv")
    ap.add_argument("--out-root", default="res/runs_single_panel")
    ap.add_argument("--max-workers", type=int, default=None)
    args = ap.parse_args()

    ge = load_gene_effect(args.gene_effect)
    emb = pd.read_parquet(args.embeddings)
    lines = choose_lowest_nan_lines(ge, emb, args.n_cell_lines)

    model = pd.read_csv(args.model_csv).set_index("ModelID")
    print(f"chosen {len(lines)} cell lines (fewest NaNs among embedded genes):")
    for cl in lines:
        lineage = model.loc[cl, "OncotreeLineage"] if cl in model.index else "?"
        n_genes = build_cell_line_dataset(ge, emb, cl).n_genes
        print(f"  {cl}  lineage={lineage}  n_genes={n_genes}")

    builder = ExperimentBuilder(
        gene_effect=args.gene_effect, embeddings=args.embeddings, k=args.k,
        rounds=args.rounds, batch=args.batch, n_initial=args.n_initial,
        sigma=args.sigma)
    panel = [MaxValue(), CumulativeDiversity(),
             AlphaNDCG(k=args.k, alpha=args.alpha,
                       n_clusters=args.n_clusters, seed=0)]
    df = run_sweep(builder, lines, list(args.seeds),
                   eval_panel=panel, max_workers=args.max_workers)

    run = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_root) / run
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "rounds.parquet")
    report = out_dir / "report.html"
    build_report2(df, report, title="geneal — single-objective metric panel")

    # Final-round mean +/- CI per metric per method (across lines + seeds).
    metric_cols = ["metric"] + sorted(c for c in df.columns if c.startswith("eval_"))
    final = df[df["round"] == df["round"].max()]
    methods = ["kdpp", "greedy", "fantasy", "coreset", "typiclust", "random"]
    print("\n=== FINAL-ROUND mean +/- 95%% CI (across %d lines x %d seeds) ===" %
          (len(lines), len(args.seeds)))
    label = {"metric": f"recall@{args.k}", "eval_max_value": "max_value",
             "eval_diversity": "diversity", f"eval_alpha_ndcg@{args.k}": f"alpha_ndcg@{args.k}"}
    for col in metric_cols:
        print(f"\n[{label.get(col, col)}]")
        for m in methods:
            vals = final[final["method"] == m][col].to_numpy()
            if len(vals) == 0:
                continue
            mean, ci = _mean_ci(vals)
            print(f"  {m:10s} {mean:.4f} +/- {ci:.4f}")

    print(f"\nreport: {report}")
    print(f"rounds parquet: {out_dir / 'rounds.parquet'}")


if __name__ == "__main__":
    main()
