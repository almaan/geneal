# tests/test_selective.py
import numpy as np
import pandas as pd
from geneal.data.selective import common_essential_score, build_selective_dataset


def _ge(tmp_path):
    # 3 genes x 4 lines. g0 pan-essential (lethal everywhere)=toxic.
    # g1 selectively lethal in L1 only. g2 inert.
    df = pd.DataFrame(
        {"L1": [-2.0, -2.0, 0.0], "L2": [-2.0, 0.0, 0.0],
         "L3": [-2.0, 0.0, 0.1], "L4": [-2.0, 0.1, 0.0]},
        index=pd.Index(["A (1)", "B (2)", "C (3)"], name="gene"))
    df.columns.name = "cell_line"
    p = tmp_path / "ge.parquet"; df.to_parquet(p); return p


def test_common_essential_score(tmp_path):
    ge = pd.read_parquet(_ge(tmp_path))
    ces = common_essential_score(ge, thresh=-0.5)
    # A lethal in all 4 lines -> 1.0 ; B lethal in 1/4 -> 0.25 ; C -> 0
    assert np.isclose(ces["A (1)"], 1.0)
    assert np.isclose(ces["B (2)"], 0.25)
    assert np.isclose(ces["C (3)"], 0.0)


def test_build_selective_target_downweights_common_essential(tmp_path):
    ge = pd.read_parquet(_ge(tmp_path))
    emb = pd.DataFrame(np.eye(3), index=pd.Index([1, 2, 3], name="entrez"))
    ds, aux = build_selective_dataset(ge, emb, cell_line="L1", lam=1.0, thresh=-0.5)
    i_a = ds.gene_names.index("A (1)"); i_b = ds.gene_names.index("B (2)")
    # A: lethality 2.0 but common-ess 1.0 -> selective ~ 2 - 1*(mean lethality of A across lines=2) = 0
    # B: lethality 2.0 in L1, common-ess 0.25 -> selective stays high
    # so B (selective) should outrank A (pan-essential/toxic)
    assert ds.target[i_b] > ds.target[i_a]
    assert "common_essential" in aux
