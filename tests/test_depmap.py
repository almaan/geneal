# tests/test_depmap.py
import numpy as np
import pandas as pd
import pytest
from geneal.data.depmap import load_gene_effect, build_cell_line_dataset, parse_entrez


def test_parse_entrez():
    assert parse_entrez("A1BG (1)") == 1
    assert parse_entrez("TP53 (7157)") == 7157


def _toy_effect(tmp_path):
    df = pd.DataFrame(
        {"ACH-001": [-2.0, 0.1, np.nan], "ACH-002": [-1.0, 0.0, -3.0]},
        index=pd.Index(["TP53 (7157)", "A1BG (1)", "EGFR (1956)"], name="gene"),
    )
    df.columns.name = "cell_line"
    p = tmp_path / "ge.parquet"
    df.to_parquet(p)
    return p


def test_load_gene_effect_roundtrip(tmp_path):
    p = _toy_effect(tmp_path)
    df = load_gene_effect(p)
    assert df.shape == (3, 2)
    assert df.index.name == "gene"


def test_build_dataset_target_is_negated_effect_and_drops_nan(tmp_path):
    p = _toy_effect(tmp_path)
    emb = pd.DataFrame(
        np.arange(9).reshape(3, 3).astype(float),
        index=pd.Index([7157, 1, 1956], name="entrez"),
    )
    ds = build_cell_line_dataset(load_gene_effect(p), emb, cell_line="ACH-001")
    assert ds.n_genes == 2
    i = ds.gene_names.index("TP53 (7157)")
    assert np.isclose(ds.target[i], 2.0)
    assert ds.embeddings.shape == (2, 3)


def test_build_dataset_requires_embedding_coverage(tmp_path):
    p = _toy_effect(tmp_path)
    emb = pd.DataFrame(np.zeros((1, 3)), index=pd.Index([7157], name="entrez"))
    ds = build_cell_line_dataset(load_gene_effect(p), emb, cell_line="ACH-001")
    assert ds.n_genes == 1
    assert ds.gene_names == ["TP53 (7157)"]
