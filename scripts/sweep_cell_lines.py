# scripts/sweep_cell_lines.py
"""Multi-cell-line sweep: does greedy >= k-DPP hold beyond OVCAR-8?

Diligence experiment. Tune nothing to favor any method. Pick the 8 cell lines
with the most coverage among the embedded genes, race all six methods on each,
and report final recall@50 (mean over seeds) per method, plus aggregate stats.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from geneal.data.depmap import load_gene_effect, build_cell_line_dataset, parse_entrez
from geneal.experiment.objects import ObjectiveObject, DesignObject, Method, Experiment
from geneal.metrics.recall import RecallAtK
from geneal.models.acquisition import UCB, RandomAcquisition, GreedyMean
from geneal.models.selection import KDPP, TopQGreedy, GreedyFantasy, CoreSet, TypiClust
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.runner.runner import Runner

GENE_EFFECT = "data/processed/depmap/gene_effect.parquet"
EMBEDDINGS = "data/processed/embeddings/esm2_small.parquet"
MODEL_CSV = "data/raw/depmap/Model.csv"
OUT_MD = "data/REAL_RESULT_MULTI.md"

K = 50
N_ROUNDS = 10
BATCH = 10
N_INITIAL = 50
SIGMA = 0.1
SEEDS = [0, 1, 2]
N_LINES = 8
METHOD_ORDER = ["kdpp", "greedy", "fantasy", "coreset", "typiclust", "random"]


def make_methods():
    """Fresh method list with its own GPRSurrogate per method."""
    return [
        Method("kdpp", UCB(2.0), KDPP("greedy"), GPRSurrogate(n_iters=100)),
        Method("greedy", UCB(2.0), TopQGreedy(), GPRSurrogate(n_iters=100)),
        Method("fantasy", UCB(2.0), GreedyFantasy(), GPRSurrogate(n_iters=100)),
        Method("coreset", GreedyMean(), CoreSet(), GPRSurrogate(n_iters=100)),
        Method("typiclust", GreedyMean(), TypiClust(), GPRSurrogate(n_iters=100)),
        Method("random", RandomAcquisition(), TopQGreedy(), GPRSurrogate(n_iters=100)),
    ]


def main():
    ge = load_gene_effect(GENE_EFFECT)
    emb = pd.read_parquet(EMBEDDINGS)
    model = pd.read_csv(MODEL_CSV)
    lineage = dict(zip(model["ModelID"], model["OncotreeLineage"]))

    # Restrict gene_effect to genes present in the embedding cache (by entrez).
    emb_entrez = set(emb.index)
    keep_rows = [lbl for lbl in ge.index if parse_entrez(lbl) in emb_entrez]
    ge_sub = ge.loc[keep_rows]
    print(f"genes in embedding cache present in gene_effect: {len(keep_rows)}")

    # 8 cell-line columns with fewest NaNs (most coverage) among those genes.
    nan_counts = ge_sub.isna().sum(axis=0).sort_values()
    chosen = list(nan_counts.index[:N_LINES])
    print("\n=== chosen cell lines (most coverage among embedded genes) ===")
    for cl in chosen:
        cov = len(ge_sub) - int(nan_counts[cl])
        print(f"  {cl:14s} lineage={lineage.get(cl, '?'):28s} covered={cov}/{len(ge_sub)}")

    # Race all six methods per cell line.
    rows = []  # per-line per-method final recall
    line_meta = []
    for cl in chosen:
        try:
            ds = build_cell_line_dataset(ge, emb, cell_line=cl)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {cl}: dataset build failed: {e}")
            continue
        exp = Experiment(
            ds,
            ObjectiveObject(RecallAtK(k=K), "maximize"),
            DesignObject(n_rounds=N_ROUNDS, batch_size=BATCH, n_initial=N_INITIAL, seed=0),
            GaussianNoise(SIGMA),
            make_methods(),
        )
        try:
            df = Runner().run(exp, seeds=SEEDS)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {cl}: runner failed: {e}")
            continue
        final = (
            df[df["round"] == df["round"].max()]
            .groupby("method")["metric"].mean()
        )
        rec = {m: float(final.get(m, np.nan)) for m in METHOD_ORDER}
        rec["cell_line"] = cl
        rec["lineage"] = lineage.get(cl, "?")
        rows.append(rec)
        line_meta.append((cl, lineage.get(cl, "?"), ds.n_genes))
        print(
            f"[done] {cl} ({lineage.get(cl,'?')}): "
            + " ".join(f"{m}={rec[m]:.3f}" for m in METHOD_ORDER)
        )

    if not rows:
        print("no cell lines completed; aborting")
        return

    # Summary table: rows = cell lines, cols = methods + kdpp-greedy.
    tbl = pd.DataFrame(rows).set_index("cell_line")
    tbl = tbl[["lineage"] + METHOD_ORDER]
    tbl["kdpp - greedy"] = tbl["kdpp"] - tbl["greedy"]

    mean_row = tbl[METHOD_ORDER + ["kdpp - greedy"]].mean()
    mean_row["lineage"] = "(all lines)"
    tbl_full = pd.concat([tbl, pd.DataFrame([mean_row], index=["MEAN"])])

    pd.set_option("display.width", 200, "display.max_columns", 20)
    fmt = {m: "{:.4f}".format for m in METHOD_ORDER + ["kdpp - greedy"]}
    table_str = tbl_full.to_string(formatters=fmt)
    print("\n=== final recall@%d (mean over seeds %s) ===" % (K, SEEDS))
    print(table_str)

    # Aggregate win counts.
    q = tbl[["kdpp", "greedy", "fantasy"]].max(axis=1)
    d = tbl[["coreset", "typiclust"]].max(axis=1)
    n = len(tbl)
    kdpp_ge_greedy = int((tbl["kdpp"] >= tbl["greedy"]).sum())
    quality_beats_div = int((q > d).sum())
    kdpp_beats_random = int((tbl["kdpp"] > tbl["random"]).sum())

    mean_per_method = tbl[METHOD_ORDER].mean()
    print("\n=== aggregate ===")
    print(f"lines completed: {n}")
    print(f"kdpp >= greedy:  {kdpp_ge_greedy}/{n} lines")
    print(f"quality (max kdpp/greedy/fantasy) > diversity-only (max coreset/typiclust): {quality_beats_div}/{n} lines")
    print(f"kdpp > random:   {kdpp_beats_random}/{n} lines")
    print(f"mean kdpp-greedy gap: {tbl['kdpp - greedy'].mean():+.4f}")
    print("\nmean recall@50 per method across lines:")
    print(mean_per_method.sort_values(ascending=False).to_string(float_format="{:.4f}".format))

    # Write the markdown report.
    write_md(tbl, tbl_full, table_str, mean_per_method, n,
             kdpp_ge_greedy, quality_beats_div, kdpp_beats_random, line_meta)
    print(f"\nwrote {OUT_MD}")


def write_md(tbl, tbl_full, table_str, mean_per_method, n,
             kdpp_ge_greedy, quality_beats_div, kdpp_beats_random, line_meta):
    gap_mean = tbl["kdpp - greedy"].mean()
    best = mean_per_method.sort_values(ascending=False)
    lines = []
    lines.append("# Real DepMap multi-cell-line sweep: k-DPP vs baselines\n")
    lines.append("Diligence experiment. Tests whether the single-line verdict "
                 "(greedy beats k-DPP on OVCAR-8) holds across cell lines or is "
                 "OVCAR-8-specific. Nothing was tuned to favor any method.\n")
    lines.append("## Config\n")
    lines.append(f"- recall@{K}; {N_ROUNDS} rounds, batch {BATCH}, n_initial {N_INITIAL}, "
                 f"GaussianNoise({SIGMA}), seeds {SEEDS}\n")
    lines.append(f"- surrogate: fresh GPRSurrogate(n_iters=100) per method per line\n")
    lines.append("- methods: kdpp=UCB(2.0)+KDPP('greedy'), greedy=UCB(2.0)+TopQGreedy, "
                 "fantasy=UCB(2.0)+GreedyFantasy, coreset=GreedyMean+CoreSet, "
                 "typiclust=GreedyMean+TypiClust, random=RandomAcquisition+TopQGreedy\n")
    lines.append("\n## Cell lines (most coverage among embedded genes)\n")
    for cl, lin, ng in line_meta:
        lines.append(f"- `{cl}` — {lin} ({int(ng)} genes)")
    lines.append("\n## Final recall@%d per line (mean over seeds)\n" % K)
    lines.append("```")
    lines.append(table_str)
    lines.append("```\n")
    lines.append("## Aggregate\n")
    lines.append(f"- Lines completed: **{n}**")
    lines.append(f"- **kdpp >= greedy: {kdpp_ge_greedy}/{n} lines**")
    lines.append(f"- quality (max kdpp/greedy/fantasy) > diversity-only (max coreset/typiclust): "
                 f"**{quality_beats_div}/{n} lines**")
    lines.append(f"- kdpp > random: **{kdpp_beats_random}/{n} lines**")
    lines.append(f"- mean (kdpp - greedy) gap across lines: **{gap_mean:+.4f}**\n")
    lines.append("Mean recall@%d per method across lines (sorted):\n" % K)
    lines.append("```")
    lines.append(best.to_string(float_format="{:.4f}".format))
    lines.append("```\n")

    # Honest read.
    lines.append("## Honest read\n")
    kdpp_better_lines = tbl.index[tbl["kdpp - greedy"] > 1e-6].tolist()
    greedy_at_least = n - kdpp_ge_greedy + int((tbl["kdpp"] == tbl["greedy"]).sum())
    lines.append(
        f"**Does greedy >= k-DPP generalize?** Across {n} cell lines, kdpp matched or "
        f"beat greedy in {kdpp_ge_greedy}/{n} lines; the mean kdpp-greedy gap is "
        f"{gap_mean:+.4f} recall@{K}. "
        + ("This confirms the OVCAR-8 verdict generalizes: greedy is at least as good as "
           "k-DPP on the great majority of lines, and on average."
           if gap_mean < 0 and kdpp_ge_greedy <= n // 2 else
           "The picture is mixed: see the per-line table — the OVCAR-8 verdict does NOT "
           "cleanly generalize.")
        + "\n"
    )
    if kdpp_better_lines:
        det = ", ".join(f"`{cl}` ({tbl.loc[cl,'lineage']}, +{tbl.loc[cl,'kdpp - greedy']:.4f})"
                        for cl in kdpp_better_lines)
        lines.append(f"**Lines where k-DPP clearly beats greedy:** {det}.\n")
    else:
        lines.append("**Lines where k-DPP clearly beats greedy:** none "
                     "(k-DPP never strictly wins).\n")
    lines.append(
        f"**Does quality >> diversity-only hold?** A quality method beat the best "
        f"diversity-only method in {quality_beats_div}/{n} lines. "
        + ("This holds broadly across lines."
           if quality_beats_div >= max(1, int(0.75 * n)) else
           "This is NOT consistent across lines.") + "\n"
    )
    lines.append(
        f"**k-DPP vs random:** k-DPP beat random in {kdpp_beats_random}/{n} lines.\n"
    )
    Path(OUT_MD).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
