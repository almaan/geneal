# src/geneal/data/selective.py
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from geneal.data.dataset import Dataset
from geneal.data.depmap import parse_entrez

_SYM = re.compile(r"^(.*?)\s*\(\d+\)$")


def corum_membership(gene_names,
                     cache="data/processed/depmap/corum_membership_all.parquet",
                     corum_path="data/corum_dl/humanComplexes.txt") -> dict:
    """gene_idx -> set of CORUM complex_ids, keyed by position in `gene_names`.

    Prefers the genome-wide membership cache (entrez -> complex_id); falls back to
    parsing the raw CORUM file by gene symbol. Used by the cap operator (per-
    pathway hedging) and the concentration / robustness / n_pathways metrics."""
    if Path(cache).exists():
        mdf = pd.read_parquet(cache)
        ent2c: dict = {}
        for ent, cid in zip(mdf["entrez"], mdf["complex_id"]):
            ent2c.setdefault(int(ent), set()).add(int(cid))
        return {i: ent2c.get(int(parse_entrez(lab) or -1), set())
                for i, lab in enumerate(gene_names)}
    df = pd.read_csv(corum_path, sep="\t")
    sym2c: dict = {}
    for _, r in df.iterrows():
        cid = int(r["complex_id"])
        for s in re.split(r"[;,]", str(r.get("subunits_gene_name", "") or "")):
            if s.strip():
                sym2c.setdefault(s.strip(), set()).add(cid)
    def _sym(label):
        m = _SYM.match(label)
        return (m.group(1) if m else label).strip()
    return {i: sym2c.get(_sym(lab), set()) for i, lab in enumerate(gene_names)}


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
