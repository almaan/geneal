# src/geneal/data/depmap.py
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from geneal.data.dataset import Dataset

_ENTREZ_RE = re.compile(r"\((\d+)\)\s*$")


def parse_entrez(gene_label: str) -> int:
    """Extract the Entrez id from a DepMap gene label like 'TP53 (7157)'."""
    m = _ENTREZ_RE.search(gene_label)
    if not m:
        raise ValueError(f"no entrez id in gene label: {gene_label!r}")
    return int(m.group(1))


def load_gene_effect(path) -> pd.DataFrame:
    """Load the curated genes x cell_lines Chronos gene-effect matrix."""
    return pd.read_parquet(path)


def build_cell_line_dataset(gene_effect: pd.DataFrame, embeddings: pd.DataFrame,
                            cell_line: str) -> Dataset:
    """Build a per-cell-line Dataset.

    gene_effect: genes (index 'SYMBOL (Entrez)') x cell_lines.
    embeddings:  rows indexed by Entrez id, columns = embedding dims.
    target = -gene_effect for the chosen cell line (higher = more lethal).
    Genes with NaN effect for this cell line OR no embedding are dropped.
    """
    if cell_line not in gene_effect.columns:
        raise KeyError(f"cell line {cell_line!r} not in gene-effect matrix")
    col = gene_effect[cell_line].dropna()
    rows, target, names = [], [], []
    emb_index = set(embeddings.index)
    for label, effect in col.items():
        ent = parse_entrez(label)
        if ent not in emb_index:
            continue
        rows.append(embeddings.loc[ent].to_numpy(dtype=float))
        target.append(-float(effect))  # lethality = -effect
        names.append(label)
    if not names:
        raise ValueError("no genes with both an effect and an embedding")
    return Dataset(embeddings=np.vstack(rows),
                   target=np.asarray(target), gene_names=names)
