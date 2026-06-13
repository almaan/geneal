# tests/test_report2.py
import numpy as np
import pandas as pd
from geneal.report.report2 import (metric_columns, stage_table, build_report2)
from geneal.report.dual_plot import dual_scatter_div


def _df():
    rows = []
    for method in ("kdpp", "greedy"):
        for seed in (0, 1):
            for rnd in range(3):
                rows.append({"method": method, "seed": seed, "round": rnd,
                             "metric": 0.1 * rnd, "metric_name": "recall@50",
                             "eval_max_value": 0.2 * rnd + 0.1,
                             "eval_diversity": 1.0 + rnd,
                             "eval_alpha_ndcg@50": 0.3 * rnd})
    return pd.DataFrame(rows)


def test_metric_columns_discovers_eval_and_recall():
    cols = metric_columns(_df())
    assert "metric" in cols  # recall
    assert "eval_max_value" in cols and "eval_alpha_ndcg@50" in cols


def test_stage_table_model_rows_round_cols_mean_ci():
    t = stage_table(_df(), "eval_max_value")
    # rows = methods, columns = rounds
    assert set(t.index) == {"kdpp", "greedy"}
    assert list(t.columns) == [0, 1, 2]
    # cells are 'mean ± ci' strings
    assert "±" in t.loc["kdpp", 0]


def test_build_report2_writes_rich_html(tmp_path):
    out = tmp_path / "report.html"
    build_report2(_df(), out, title="geneal single-objective")
    html = out.read_text()
    assert "<html" in html.lower()
    assert "recall@50" in html
    assert "max_value" in html and "alpha_ndcg@50" in html and "diversity" in html
    # stage tables present (one per metric) and error-bar plots embedded
    assert "±" in html
    assert "plotly" in html.lower() or "<div" in html.lower()


def test_dual_scatter_div_is_html():
    rng = np.random.default_rng(0)
    aux = {"lethality_a": rng.standard_normal(50), "lethality_b": rng.standard_normal(50),
           "line_a": "ACH-A", "line_b": "ACH-B"}
    target = aux["lethality_a"] - aux["lethality_b"]
    selected = {"kdpp": [0, 1, 2, 3], "greedy": [4, 5, 6, 7]}
    div = dual_scatter_div(aux, target, top_k=10, selected_by_method=selected)
    assert isinstance(div, str) and ("<div" in div.lower() or "plotly" in div.lower())
