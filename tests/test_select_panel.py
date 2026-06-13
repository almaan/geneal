# tests/test_select_panel.py
import numpy as np
import pandas as pd
from geneal.data.depmap import parse_entrez
import importlib.util, pathlib

_spec = importlib.util.spec_from_file_location(
    "select_gene_panel", pathlib.Path("scripts/select_gene_panel.py"))
sel = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sel)


def test_select_panel_picks_variable_and_lethal(tmp_path):
    # 5 genes x 4 lines: g0 inert, g1 highly variable, g2 strongly lethal everywhere
    df = pd.DataFrame(
        {"L1": [0.0, -3.0, -4.0, 0.1, 0.0],
         "L2": [0.0, 3.0, -4.0, 0.0, 0.1],
         "L3": [0.0, -3.0, -4.0, 0.0, 0.0],
         "L4": [0.0, 3.0, -4.0, 0.1, 0.0]},
        index=pd.Index(["A (1)", "B (2)", "C (3)", "D (4)", "E (5)"], name="gene"))
    df.columns.name = "cell_line"
    p = tmp_path / "ge.parquet"; df.to_parquet(p)
    entrez = sel.select_panel(str(p), n_hvg=2, n_lethal=2)
    # g1 (entrez 2, highly variable) and g2 (entrez 3, strongly lethal) must be in
    assert 2 in entrez and 3 in entrez
    assert isinstance(entrez, list) and all(isinstance(e, int) for e in entrez)
