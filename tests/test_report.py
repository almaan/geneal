# tests/test_report.py
import pandas as pd
from geneal.report.report import build_report, aggregate, to_latex_table


def _df():
    rows = []
    for method in ("al_ucb", "baseline"):
        for seed in (0, 1):
            for rnd in range(3):
                val = 0.1 * rnd + (0.2 if method == "al_ucb" else 0.0)
                rows.append({"method": method, "seed": seed, "round": rnd,
                             "n_revealed": 10 + 5 * rnd, "metric": val,
                             "metric_name": "recall@10"})
    return pd.DataFrame(rows)


def test_aggregate_mean_ci():
    agg = aggregate(_df())
    # one row per (method, round) = 2 * 3 = 6
    assert len(agg) == 6
    assert {"method", "round", "mean", "ci_low", "ci_high"}.issubset(agg.columns)


def test_latex_table_is_string_with_tabular():
    latex = to_latex_table(aggregate(_df()))
    assert "\\begin{tabular}" in latex
    assert "al_ucb" in latex


def test_build_report_writes_html(tmp_path):
    out = tmp_path / "report.html"
    build_report(_df(), out)
    assert out.exists()
    html = out.read_text()
    assert "<html" in html.lower()
    assert "recall@10" in html
    assert "\\begin{tabular}" in html  # latex embedded in a dropdown
