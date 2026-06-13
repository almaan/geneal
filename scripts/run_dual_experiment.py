# scripts/run_dual_experiment.py
"""Dual / selectivity (efficacy-toxicity) experiment.

target = lethality_A - lethality_B (high => lethal in cancer line A, safe in
contrast line B). Runs all 6 methods over seeds with the evaluation-metric
panel on the differential target, builds the elegant v2 report with a 2D
selectivity scatter, and prints the differential-target per-metric final
mean +/- 95% CI per method.

Selected-genes capture for the scatter (seed 0): we replicate the Runner's
round loop locally (`_revealed_indices`) using the SAME surrogate / acquisition
/ selection objects, returning the cumulative revealed gene-index set at the
final round. This is the honest capture the plan calls for -- no proxy, no
posterior-mean substitute -- it is literally what the method would reveal.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

from geneal.data.depmap import load_gene_effect, parse_entrez, build_cell_line_dataset
from geneal.data.dual import build_differential_dataset
from geneal.experiment.objects import ObjectiveObject, DesignObject, Method, Experiment
from geneal.metrics.recall import RecallAtK
from geneal.metrics.panel import MaxValue, CumulativeDiversity, AlphaNDCG
from geneal.models.acquisition import UCB, RandomAcquisition, GreedyMean
from geneal.models.selection import KDPP, TopQGreedy, GreedyFantasy, CoreSet, TypiClust
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.runner.runner import Runner
from geneal.report.report2 import build_report2
from geneal.report.dual_plot import dual_scatter_div


def _build_methods() -> list[Method]:
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


def _revealed_indices(ds, design, method: Method, sigma: float, seed: int) -> list[int]:
    """Replicate Runner._run_one's acquisition loop for ONE method+seed and
    return the cumulative revealed gene indices at the final round.

    Mirrors runner.py exactly: same CRN for noise + initial set, same per-method
    acquisition RNG (stable sha256 name hash), same surrogate/acquisition/
    selection call. The Runner returns only metrics, so we re-derive the
    revealed set here for the scatter."""
    crn = np.random.default_rng(seed)
    noise_vec = GaussianNoise(sigma).draw(ds.n_genes, crn)
    init_idx = crn.permutation(ds.n_genes)[:design.n_initial].tolist()

    name_hash = int.from_bytes(
        hashlib.sha256(method.name.encode()).digest()[:4], "big")
    acq_rng = np.random.default_rng((seed, name_hash))

    revealed = list(init_idx)
    revealed_y = (ds.target[revealed] + noise_vec[revealed]).tolist()
    for _ in range(1, design.n_rounds + 1):
        X_train = ds.embeddings[revealed]
        y_train = np.asarray(revealed_y)
        surr = method.surrogate.clone().fit(X_train, y_train)
        seen = set(revealed)
        cand = [i for i in range(ds.n_genes) if i not in seen]
        if not cand:
            break
        Xc = ds.embeddings[cand]
        mean, std = surr.predict(Xc)
        best = float(max(revealed_y))
        sel = method.selection.select(
            candidate_idx=cand, X_candidates=Xc, mean=mean, std=std,
            best=best, q=design.batch_size, rng=acq_rng,
            surrogate=surr, acquisition=method.acquisition,
            X_train=X_train, y_train=y_train)
        for idx in sel:
            revealed.append(idx)
            revealed_y.append(float(ds.target[idx] + noise_vec[idx]))
    return revealed


def _mean_ci(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = float(np.mean(x))
    if n < 2:
        return mean, 0.0
    se = float(np.std(x, ddof=1)) / np.sqrt(n)
    return mean, 1.96 * se


def choose_two_lines_diff_lineage(ge, emb, model) -> tuple[str, str]:
    """Two lowest-NaN embedded lines from DIFFERENT OncotreeLineages."""
    emb_index = set(emb.index)
    embedded = [g for g in ge.index if parse_entrez(g) in emb_index]
    sub = ge.loc[embedded]
    ranked = sub.isna().sum(axis=0).sort_values().index.tolist()
    line_a = ranked[0]
    lin_a = model.loc[line_a, "OncotreeLineage"] if line_a in model.index else None
    for cl in ranked[1:]:
        lin = model.loc[cl, "OncotreeLineage"] if cl in model.index else None
        if lin != lin_a:
            return line_a, cl
    return ranked[0], ranked[1]  # fallback: just the two lowest-NaN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line-a", default=None)
    ap.add_argument("--line-b", default=None)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--n-initial", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n-clusters", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--gene-effect", default="data/processed/depmap/gene_effect.parquet")
    ap.add_argument("--embeddings", default="data/processed/embeddings/esm2_small.parquet")
    ap.add_argument("--model-csv", default="data/raw/depmap/Model.csv")
    ap.add_argument("--out-root", default="res/runs_dual")
    args = ap.parse_args()

    ge = load_gene_effect(args.gene_effect)
    emb = pd.read_parquet(args.embeddings)
    model = pd.read_csv(args.model_csv).set_index("ModelID")

    if args.line_a and args.line_b:
        line_a, line_b = args.line_a, args.line_b
    else:
        line_a, line_b = choose_two_lines_diff_lineage(ge, emb, model)

    ds, aux = build_differential_dataset(ge, emb, line_a, line_b)
    lin_a = model.loc[line_a, "OncotreeLineage"] if line_a in model.index else "?"
    lin_b = model.loc[line_b, "OncotreeLineage"] if line_b in model.index else "?"
    print(f"dual selectivity: target = lethality_A - lethality_B")
    print(f"  line A (efficacy):  {line_a}  lineage={lin_a}")
    print(f"  line B (toxicity):  {line_b}  lineage={lin_b}")
    print(f"  #genes (effect in both lines + embedding): {ds.n_genes}")

    design = DesignObject(n_rounds=args.rounds, batch_size=args.batch,
                          n_initial=args.n_initial, seed=0)
    methods = _build_methods()
    exp = Experiment(ds, ObjectiveObject(RecallAtK(k=args.k), "maximize"),
                     design, GaussianNoise(args.sigma), methods)
    panel = [MaxValue(), CumulativeDiversity(),
             AlphaNDCG(k=args.k, alpha=args.alpha,
                       n_clusters=args.n_clusters, seed=0)]
    df = Runner().run(exp, seeds=list(args.seeds), eval_panel=panel)
    df["cell_line"] = f"{line_a}_vs_{line_b}"

    # Honest selected-genes capture for the scatter: seed 0 revealed set per
    # method, via the local replica of the Runner's loop (same objects/RNG).
    selected_by_method = {
        m.name: _revealed_indices(ds, design, m, args.sigma, seed=0)
        for m in _build_methods()
    }

    run = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_root) / run
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "rounds.parquet")
    report = out_dir / "report.html"
    div = dual_scatter_div(aux, ds.target, top_k=args.k,
                           selected_by_method=selected_by_method)
    build_report2(df, report,
                  title="geneal — dual selectivity (efficacy-toxicity)",
                  dual_div=div)

    metric_cols = ["metric"] + sorted(c for c in df.columns if c.startswith("eval_"))
    final = df[df["round"] == df["round"].max()]
    order = ["kdpp", "greedy", "fantasy", "coreset", "typiclust", "random"]
    label = {"metric": f"recall@{args.k}", "eval_max_value": "max_value",
             "eval_diversity": "diversity", f"eval_alpha_ndcg@{args.k}": f"alpha_ndcg@{args.k}"}
    print("\n=== DIFFERENTIAL-TARGET FINAL-ROUND mean +/- 95%% CI (%d seeds) ===" %
          len(args.seeds))
    for col in metric_cols:
        print(f"\n[{label.get(col, col)}]")
        for m in order:
            vals = final[final["method"] == m][col].to_numpy()
            if len(vals) == 0:
                continue
            mean, ci = _mean_ci(vals)
            print(f"  {m:10s} {mean:.4f} +/- {ci:.4f}")

    print(f"\nreport: {report}")
    print(f"rounds parquet: {out_dir / 'rounds.parquet'}")


if __name__ == "__main__":
    main()
