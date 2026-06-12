# scripts/run_real_experiment.py
"""Run all methods on a real DepMap cell line and write a report + Pareto."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from geneal.data.depmap import load_gene_effect, build_cell_line_dataset
from geneal.experiment.objects import ObjectiveObject, DesignObject, Method, Experiment
from geneal.metrics.recall import RecallAtK
from geneal.models.acquisition import UCB, RandomAcquisition, GreedyMean
from geneal.models.selection import KDPP, TopQGreedy, GreedyFantasy, CoreSet, TypiClust
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.runner.runner import Runner
from geneal.runner.logging import write_run_dir
from geneal.report.report import build_report
from geneal.report.pareto import pareto_summary, pareto_plot_div


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-effect", default="data/processed/depmap/gene_effect.parquet")
    ap.add_argument("--embeddings", default="data/processed/embeddings/esm2_small.parquet")
    ap.add_argument("--cell-line", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--n-initial", type=int, default=50)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out-root", default="res/runs_real")
    args = ap.parse_args()

    ge = load_gene_effect(args.gene_effect)
    emb = pd.read_parquet(args.embeddings)
    ds = build_cell_line_dataset(ge, emb, cell_line=args.cell_line)
    print(f"cell line {args.cell_line}: {ds.n_genes} genes with embeddings")

    def gpr():
        return GPRSurrogate(n_iters=100)
    methods = [
        Method("kdpp", UCB(2.0), KDPP("greedy"), gpr()),
        Method("greedy", UCB(2.0), TopQGreedy(), gpr()),
        Method("fantasy", UCB(2.0), GreedyFantasy(), gpr()),
        Method("coreset", GreedyMean(), CoreSet(), gpr()),
        Method("typiclust", GreedyMean(), TypiClust(), gpr()),
        Method("random", RandomAcquisition(), TopQGreedy(), gpr()),
    ]
    exp = Experiment(ds, ObjectiveObject(RecallAtK(k=args.k), "maximize"),
                     DesignObject(n_rounds=args.rounds, batch_size=args.batch,
                                  n_initial=args.n_initial, seed=0),
                     GaussianNoise(args.sigma), methods)
    df = Runner().run(exp, seeds=args.seeds)
    run_dir = write_run_dir(out_root=args.out_root, df=df,
                            config=vars(args), extra_manifest={"cell_line": args.cell_line})
    build_report(pd.read_parquet(run_dir / "rounds.parquet"), run_dir / "report.html")
    summ = pareto_summary(df)
    (run_dir / "pareto.html").write_text(pareto_plot_div(summ))
    final = df[df["round"] == df["round"].max()].groupby("method")["metric"].mean().sort_values(ascending=False)
    print("=== final recall@%d ===" % args.k); print(final.to_string())
    print("=== pareto ==="); print(summ.to_string())
    print(f"run dir: {run_dir}")


if __name__ == "__main__":
    main()
