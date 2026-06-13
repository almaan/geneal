# scripts/select_gene_panel.py
"""Select a gene panel = top-N highly-variable UNION top-N most-lethal genes,
from the full DepMap gene-effect matrix. Writes Entrez ids (one per line).

Rationale: the first-500-by-ID panel was 97% inert (no lethal signal). HVG +
most-lethal genes give a panel that is ~99% strongly-lethal in some cell line,
so top-k recovery is a meaningful task and methods can differentiate."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from geneal.data.depmap import load_gene_effect, parse_entrez


def select_panel(gene_effect_path: str, n_hvg: int = 2000,
                 n_lethal: int = 1000) -> list[int]:
    ge = load_gene_effect(gene_effect_path)
    eff = ge.to_numpy()
    gene_var = np.nanvar(eff, axis=1)
    gene_min = np.nanmin(eff, axis=1)  # most negative = most lethal somewhere
    hvg = np.argsort(gene_var)[::-1][:n_hvg]
    leth = np.argsort(gene_min)[:n_lethal]
    keep = np.union1d(hvg, leth)
    return [parse_entrez(ge.index[i]) for i in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-effect", default="data/processed/depmap/gene_effect.parquet")
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--n-lethal", type=int, default=1000)
    ap.add_argument("--out", default="data/processed/depmap/panel_hvg.txt")
    args = ap.parse_args()
    entrez = select_panel(args.gene_effect, args.n_hvg, args.n_lethal)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(str(e) for e in entrez))
    print(f"panel: {len(entrez)} entrez ids -> {args.out}")


if __name__ == "__main__":
    main()
