# tests/test_pareto.py
import numpy as np
import pandas as pd
from geneal.report.pareto import pareto_summary, pareto_plot_div


def _df():
    rows = []
    for method, (qual, div) in {"kdpp": (3.0, 2.0), "greedy": (3.0, 0.2),
                                "coreset": (0.5, 2.5)}.items():
        for seed in (0, 1):
            for rnd in range(3):
                rows.append({
                    "method": method, "seed": seed, "round": rnd,
                    "metric": 0.1 * rnd + (0.2 if method == "kdpp" else 0.0),
                    "metric_name": "recall@10",
                    "batch_quality": float("nan") if rnd == 0 else qual,
                    "batch_diversity": float("nan") if rnd == 0 else div,
                })
    return pd.DataFrame(rows)


def test_pareto_summary_one_row_per_method():
    s = pareto_summary(_df())
    assert set(s["method"]) == {"kdpp", "greedy", "coreset"}
    assert {"mean_quality", "mean_diversity", "final_recall"}.issubset(s.columns)
    kd = s[s["method"] == "kdpp"].iloc[0]
    assert np.isclose(kd["mean_quality"], 3.0)
    assert np.isclose(kd["mean_diversity"], 2.0)


def test_pareto_plot_div_is_html():
    div = pareto_plot_div(pareto_summary(_df()))
    assert isinstance(div, str)
    assert "plotly" in div.lower() or "<div" in div.lower()
