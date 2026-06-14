# scripts/validate_string.py
"""Validate whether STRING graph-similarity carries knockout-outcome-redundancy.

Reproducible diagnostic from the saved STRING similarity matrix
(data/processed/depmap/string_sim.parquet, built by the STRING fetch step — see
REPRODUCE.md). Reports the NN/random lethality-diff ratio (lower = more outcome
structure) for STRING neighbours, vs the FM baseline (~0.86) and co-dependency
(0.39)."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from geneal.data.depmap import load_gene_effect, build_cell_line_dataset, parse_entrez


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-effect", default="data/processed/depmap/gene_effect.parquet")
    ap.add_argument("--sim", default="data/processed/depmap/string_sim.parquet")
    ap.add_argument("--embeddings", default="data/processed/embeddings/pubmedbert_hvg.parquet")
    ap.add_argument("--cell-line", default="ACH-000147")
    ap.add_argument("--n-neighbors", type=int, default=10)
    ap.add_argument("--thresh", type=float, default=0.7)
    args = ap.parse_args()

    ge = load_gene_effect(args.gene_effect)
    S = pd.read_parquet(args.sim)
    emb = pd.read_parquet(args.embeddings)
    ds = build_cell_line_dataset(ge, emb, args.cell_line)
    ent = [parse_entrez(g) for g in ds.gene_names]
    y = ds.target
    Sg = np.array(S.reindex(index=ent, columns=ent).fillna(0).to_numpy(), copy=True)
    np.fill_diagonal(Sg, 0)
    rng = np.random.default_rng(0)
    K = args.n_neighbors
    nn_d, rd_d, n_used = [], [], 0
    for i in range(len(y)):
        nbrs = np.argsort(Sg[i])[::-1]
        nbrs = [j for j in nbrs if Sg[i, j] > args.thresh][:K]
        if not nbrs:
            continue
        n_used += 1
        nn_d.append(np.abs(y[i] - y[nbrs]).mean())
        rd_d.append(np.abs(y[i] - y[rng.choice(len(y), K)]).mean())
    ratio = float(np.mean(nn_d) / np.mean(rd_d))
    print(f"STRING redundancy diagnostic on {args.cell_line} (thresh>{args.thresh}):")
    print(f"  genes with >=1 STRING neighbor: {n_used}/{len(y)}")
    print(f"  NN/random lethality-diff ratio: {ratio:.3f}  "
          f"(FM baseline ~0.86, co-dependency 0.39)")


if __name__ == "__main__":
    main()
