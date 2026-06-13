# tests/test_dual.py
import numpy as np
import pandas as pd
from geneal.data.dual import build_differential_dataset


def _toy():
    ge = pd.DataFrame(
        {"ACH-A": [-3.0, 0.0, -1.0], "ACH-B": [0.0, -2.0, -1.0]},
        index=pd.Index(["TP53 (7157)", "A1BG (1)", "EGFR (1956)"], name="gene"))
    ge.columns.name = "cell_line"
    emb = pd.DataFrame(np.eye(3), index=pd.Index([7157, 1, 1956], name="entrez"))
    return ge, emb


def test_differential_target_is_lethalityA_minus_lethalityB():
    ge, emb = _toy()
    ds, aux = build_differential_dataset(ge, emb, line_a="ACH-A", line_b="ACH-B")
    # lethality = -effect; differential = (-effA) - (-effB) = effB - effA
    # TP53: effA -3, effB 0 -> lethalityA 3, lethalityB 0 -> diff +3 (selective to A)
    i = ds.gene_names.index("TP53 (7157)")
    assert np.isclose(ds.target[i], 3.0)
    # A1BG: effA 0, effB -2 -> lethalityA 0, lethalityB 2 -> diff -2 (toxic to B)
    j = ds.gene_names.index("A1BG (1)")
    assert np.isclose(ds.target[j], -2.0)
    # aux carries per-line lethality for the 2D scatter
    assert "lethality_a" in aux and "lethality_b" in aux
    assert len(aux["lethality_a"]) == ds.n_genes


def test_dual_drops_genes_missing_either_line_or_embedding():
    ge, emb = _toy()
    ge.loc["EGFR (1956)", "ACH-B"] = np.nan  # missing in B -> dropped
    ds, aux = build_differential_dataset(ge, emb, line_a="ACH-A", line_b="ACH-B")
    assert "EGFR (1956)" not in ds.gene_names
    assert ds.n_genes == 2
