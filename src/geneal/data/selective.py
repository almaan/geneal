# src/geneal/data/selective.py
from __future__ import annotations
import numpy as np
import pandas as pd
from geneal.data.dataset import Dataset
from geneal.data.depmap import parse_entrez


def common_essential_score(gene_effect: pd.DataFrame, thresh: float = -0.5) -> pd.Series:
    """Fraction of cell lines where each gene is strongly lethal (effect < thresh).

    High = pan-essential = toxicity proxy (would kill normal cells too)."""
    return (gene_effect < thresh).mean(axis=1)


def build_selective_dataset(gene_effect: pd.DataFrame, embeddings: pd.DataFrame,
                            cell_line: str, lam: float = 1.0,
                            thresh: float = -0.5):
    """Dataset whose target is SELECTIVE lethality:
        target = lethality_in_line - lam * (common-essential lethality)
    where lethality = -effect, and the common-essential penalty is the gene's
    MEAN lethality across all lines weighted by how pan-essential it is. High
    target = lethal in THIS line but not a general essential (lower toxicity).
    Genes need an effect in this line and an embedding (by Entrez)."""
    if cell_line not in gene_effect.columns:
        raise KeyError(cell_line)
    ces = common_essential_score(gene_effect, thresh)         # 0..1 per gene
    mean_leth = -(gene_effect.mean(axis=1))                    # avg lethality across lines
    col = gene_effect[cell_line].dropna()
    emb_index = set(embeddings.index)
    rows, target, names, ce_list = [], [], [], []
    for label, eff in col.items():
        ent = parse_entrez(label)
        if ent not in emb_index:
            continue
        leth = -float(eff)
        # penalty: pan-essential genes (high ces, high mean lethality) are toxic
        penalty = lam * float(ces[label]) * float(mean_leth[label])
        rows.append(embeddings.loc[ent].to_numpy(dtype=float))
        target.append(leth - penalty)
        names.append(label); ce_list.append(float(ces[label]))
    if not names:
        raise ValueError("no genes with effect+embedding")
    ds = Dataset(np.vstack(rows), np.asarray(target), names)
    aux = {"common_essential": np.asarray(ce_list)}
    return ds, aux
