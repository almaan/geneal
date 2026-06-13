# scripts/run_dual_powerup.py
"""Power-up of the dual/selectivity finding across MANY cell-line pairs + seeds.

Tests whether the single-pair observation — on the differential (selectivity)
target lethality_A - lethality_B, greedy collapses and quality+diversity methods
(k-DPP/fantasy) win recall — holds across multiple contrasting-lineage pairs.

DepMap is all cancer lines, so this is "selective lethality across cell contexts",
NOT true normal-tissue toxicity; we pair DIFFERENT lineages and state the caveat.

For each pair: run all 6 methods over `seeds`, record final-round recall@k on the
differential target. Then a paired comparison ACROSS pairs (per-pair mean over
seeds): Wilcoxon signed-rank of greedy vs each of {kdpp, fantasy}, plus the
per-pair win counts. Honest: prints everything, tunes nothing.
"""
from __future__ import annotations
import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from geneal.data.depmap import load_gene_effect, parse_entrez
from geneal.data.dual import build_differential_dataset
from geneal.experiment.objects import ObjectiveObject, DesignObject, Method, Experiment
from geneal.metrics.recall import RecallAtK
from geneal.metrics.panel import MaxValue, CumulativeDiversity, AlphaNDCG
from geneal.models.acquisition import UCB, RandomAcquisition, GreedyMean
from geneal.models.selection import KDPP, TopQGreedy, GreedyFantasy, CoreSet, TypiClust
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.runner.runner import Runner

ORDER = ["kdpp", "greedy", "fantasy", "coreset", "typiclust", "random"]


def _methods(n_iters):
    def gpr():
        return GPRSurrogate(n_iters=n_iters)
    return [
        Method("kdpp", UCB(2.0), KDPP("greedy"), gpr()),
        Method("greedy", UCB(2.0), TopQGreedy(), gpr()),
        Method("fantasy", UCB(2.0), GreedyFantasy(), gpr()),
        Method("coreset", GreedyMean(), CoreSet(), gpr()),
        Method("typiclust", GreedyMean(), TypiClust(), gpr()),
        Method("random", RandomAcquisition(), TopQGreedy(), gpr()),
    ]


def choose_pairs(ge, emb, model, n_pairs):
    """Pick n_pairs cell-line pairs from DIFFERENT lineages, preferring the
    lowest-NaN embedded lines (one representative line per lineage)."""
    emb_index = set(emb.index)
    embedded = [g for g in ge.index if parse_entrez(g) in emb_index]
    sub = ge.loc[embedded]
    ranked = sub.isna().sum(axis=0).sort_values().index.tolist()
    # one representative (lowest-NaN) line per lineage
    rep_by_lineage = {}
    for cl in ranked:
        lin = model.loc[cl, "OncotreeLineage"] if cl in model.index else None
        if lin and lin not in rep_by_lineage:
            rep_by_lineage[lin] = cl
    reps = list(rep_by_lineage.items())  # [(lineage, line), ...]
    pairs = []
    for (lin_a, a), (lin_b, b) in itertools.combinations(reps, 2):
        pairs.append((a, b, lin_a, lin_b))
        if len(pairs) >= n_pairs:
            break
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--n-initial", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n-clusters", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--n-iters", type=int, default=100)
    ap.add_argument("--gene-effect", default="data/processed/depmap/gene_effect.parquet")
    ap.add_argument("--embeddings", default="data/processed/embeddings/esm2_small.parquet")
    ap.add_argument("--model-csv", default="data/raw/depmap/Model.csv")
    ap.add_argument("--out-root", default="res/runs_dual_powerup")
    args = ap.parse_args()

    ge = load_gene_effect(args.gene_effect)
    emb = pd.read_parquet(args.embeddings)
    model = pd.read_csv(args.model_csv).set_index("ModelID")
    pairs = choose_pairs(ge, emb, model, args.n_pairs)

    print(f"selective lethality across cell contexts (differential target); "
          f"{len(pairs)} pairs x {len(args.seeds)} seeds")
    print("NOTE: all DepMap lines are cancer — this is cross-context selectivity, "
          "not normal-tissue toxicity.")

    panel = [MaxValue(), CumulativeDiversity(),
             AlphaNDCG(k=args.k, alpha=args.alpha, n_clusters=args.n_clusters, seed=0)]
    frames = []
    for (a, b, lin_a, lin_b) in pairs:
        ds, aux = build_differential_dataset(ge, emb, a, b)
        exp = Experiment(ds, ObjectiveObject(RecallAtK(k=args.k), "maximize"),
                         DesignObject(n_rounds=args.rounds, batch_size=args.batch,
                                      n_initial=args.n_initial, seed=0),
                         GaussianNoise(args.sigma), _methods(args.n_iters))
        df = Runner().run(exp, seeds=list(args.seeds), eval_panel=panel)
        df["pair"] = f"{a}_vs_{b}"
        df["lineage_a"] = lin_a
        df["lineage_b"] = lin_b
        frames.append(df)
        print(f"  done: {lin_a} vs {lin_b} ({a} vs {b}), {ds.n_genes} genes")

    alldf = pd.concat(frames, ignore_index=True)
    run = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_root) / run
    out_dir.mkdir(parents=True, exist_ok=True)
    alldf.to_parquet(out_dir / "rounds.parquet")

    # per-pair mean-over-seeds final recall per method
    last = alldf["round"].max()
    fin = alldf[alldf["round"] == last]
    per_pair = fin.groupby(["pair", "method"])["metric"].mean().unstack("method")
    per_pair = per_pair[[m for m in ORDER if m in per_pair.columns]]

    print(f"\n=== per-pair final recall@{args.k} (mean over {len(args.seeds)} seeds) ===")
    print(per_pair.round(3).to_string())

    print(f"\n=== mean across {len(pairs)} pairs ===")
    print(per_pair.mean().round(4).sort_values(ascending=False).to_string())

    # paired tests across pairs: greedy vs kdpp, greedy vs fantasy
    print("\n=== paired comparison across pairs (per-pair mean recall) ===")
    g = per_pair["greedy"].to_numpy()
    for chal in ("kdpp", "fantasy"):
        if chal not in per_pair.columns:
            continue
        c = per_pair[chal].to_numpy()
        wins = int(np.sum(c > g)); ties = int(np.sum(c == g))
        diff = c - g
        try:
            stat, p = wilcoxon(c, g)
            ptxt = f"wilcoxon p={p:.4f}"
        except ValueError as e:
            ptxt = f"wilcoxon n/a ({e})"
        print(f"  {chal} vs greedy: mean diff {diff.mean():+.4f}, "
              f"{chal} wins {wins}/{len(g)} (ties {ties}), {ptxt}")

    print(f"\nrounds parquet: {out_dir / 'rounds.parquet'}")


if __name__ == "__main__":
    main()
