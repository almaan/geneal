# src/geneal/data/dual.py
from __future__ import annotations
import numpy as np
import pandas as pd
from geneal.data.dataset import Dataset
from geneal.data.depmap import parse_entrez


def build_differential_dataset(gene_effect: pd.DataFrame, embeddings: pd.DataFrame,
                               line_a: str, line_b: str):
    """Selectivity (efficacy-toxicity) dataset: target = lethality_A - lethality_B.

    lethality = -(Chronos effect). target high => lethal in A, safe in B. Genes
    need a non-NaN effect in BOTH lines and an embedding (by Entrez id).
    Returns (Dataset, aux) where aux has per-gene 'lethality_a'/'lethality_b'
    arrays (aligned to ds.gene_names) for the 2D scatter visualization.
    """
    for ln in (line_a, line_b):
        if ln not in gene_effect.columns:
            raise KeyError(f"cell line {ln!r} not in gene-effect matrix")
    sub = gene_effect[[line_a, line_b]].dropna()
    emb_index = set(embeddings.index)
    rows, target, names, leth_a, leth_b = [], [], [], [], []
    for label, r in sub.iterrows():
        ent = parse_entrez(label)
        if ent not in emb_index:
            continue
        la = -float(r[line_a]); lb = -float(r[line_b])
        rows.append(embeddings.loc[ent].to_numpy(dtype=float))
        target.append(la - lb); names.append(label)
        leth_a.append(la); leth_b.append(lb)
    if not names:
        raise ValueError("no genes with effects in both lines and an embedding")
    ds = Dataset(embeddings=np.vstack(rows), target=np.asarray(target),
                 gene_names=names)
    aux = {"lethality_a": np.asarray(leth_a), "lethality_b": np.asarray(leth_b),
           "line_a": line_a, "line_b": line_b}
    return ds, aux
