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


def common_essential_score(gene_effect: pd.DataFrame, thresh: float = -0.5,
                           exclude=None) -> pd.Series:
    """Fraction of cell lines where each gene is strongly lethal (effect < thresh).

    High = pan-essential = toxicity proxy (would kill normal cells too).
    `exclude` (a ModelID or list) drops those lines from the average -- pass the
    TARGET line to prevent leakage (the gene's own lethality in the line we are
    learning must not feed its toxicity label)."""
    g = gene_effect
    if exclude is not None:
        ex = {exclude} if isinstance(exclude, str) else set(exclude)
        g = g[[c for c in g.columns if c not in ex]]
    return (g < thresh).mean(axis=1)


def contrast_toxicity(gene_effect: pd.DataFrame, contrast_line: str) -> pd.Series:
    """Toxicity = lethality (-effect) in ONE fixed contrast cell line that stands
    in for normal tissue. Higher = the knockout also kills the contrast line =
    less selective / more dangerous. This is the per-line toxicity definition
    (arguably the more meaningful one than the pan-essential aggregate)."""
    if contrast_line not in gene_effect.columns:
        raise KeyError(contrast_line)
    return -gene_effect[contrast_line]


def rank_contrast_lines(gene_effect: pd.DataFrame, thresh: float = -0.5,
                        n: int = 10, min_frac_measured: float = 0.5) -> list:
    """Rank candidate contrast (normal-tissue stand-in) cell lines, best first.

    Heuristic: a good stand-in tolerates most knockouts, i.e. has the FEWEST
    strongly-lethal genes (a line where few knockouts kill behaves least like a
    fragile cancer line). We require a line to have measured at least
    `min_frac_measured` of genes (drop sparsely-screened lines), then sort by the
    count of strongly-lethal knockouts ascending. Returns the top `n` ModelIDs."""
    measured = gene_effect.notna().mean(axis=0)
    keep = measured[measured >= min_frac_measured].index
    leth = (gene_effect[keep] < thresh).sum(axis=0)   # per line: #strongly-lethal
    return leth.sort_values().index[:n].tolist()


def toxicity_vector(gene_effect: pd.DataFrame, gene_names, source: str,
                    target_line: str = None, contrast_line: str = None,
                    thresh: float = -0.5) -> np.ndarray:
    """Toxicity per gene, aligned to `gene_names` (DepMap labels), for a source:
      'aggregate' -> common-essential, EXCLUDING the target line (leakage fix).
      'contrast'  -> lethality in the fixed contrast line.
    Missing values are filled with the source's median (neutral)."""
    if source == "aggregate":
        s = common_essential_score(gene_effect, thresh, exclude=target_line)
    elif source == "contrast":
        s = contrast_toxicity(gene_effect, contrast_line)
    else:
        raise ValueError(f"unknown toxicity source {source!r}")
    med = float(s.median())
    return np.array([float(s.get(g, med)) if pd.notna(s.get(g, med)) else med
                     for g in gene_names], dtype=float)


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
