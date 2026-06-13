# src/geneal/report/report2.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, BaseLoader

_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{{ title }}</title>
<style>
  :root { --fg:#1a1a2e; --muted:#6b7280; --line:#e5e7eb; --accent:#3b4cca; }
  body { font-family: 'Inter', system-ui, sans-serif; color: var(--fg);
         margin: 0; padding: 2.5rem 3rem; background: #fafafb; line-height: 1.5; }
  h1 { font-weight: 700; letter-spacing: -0.02em; margin-bottom: .25rem; }
  .sub { color: var(--muted); margin-bottom: 2rem; }
  h2 { margin-top: 2.5rem; font-weight: 650; border-bottom: 2px solid var(--accent);
       display: inline-block; padding-bottom: 2px; }
  .card { background:#fff; border:1px solid var(--line); border-radius:14px;
          padding:1.25rem 1.5rem; margin:1rem 0; box-shadow:0 1px 3px rgba(0,0,0,.04); }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  th, td { border-bottom: 1px solid var(--line); padding: 7px 12px; text-align: right; }
  th:first-child, td:first-child { text-align: left; font-weight: 600; }
  thead th { color: var(--muted); font-weight: 600; border-bottom: 2px solid var(--line); }
  caption { text-align:left; font-weight:650; margin-bottom:.5rem; color:var(--fg); }
  .metric-note { color: var(--muted); font-size: .9rem; margin:.2rem 0 .8rem; }
</style></head><body>
  <h1>{{ title }}</h1>
  <div class="sub">{{ methods|length }} methods · {{ n_seeds }} seeds · {{ n_rounds }} rounds
       · metrics: {{ metric_labels|join(', ') }}</div>
  {% if dual_div %}<h2>Selectivity landscape</h2><div class="card">{{ dual_div|safe }}</div>{% endif %}
  {% for m in metrics %}
    <h2>{{ m.label }}</h2>
    <div class="metric-note">{{ m.desc }}</div>
    <div class="card">{{ m.plot|safe }}</div>
    <div class="card"><table>
      <caption>{{ m.label }} by stage — mean ± 95% CI (rows: model, cols: round)</caption>
      <thead><tr><th>model</th>{% for c in m.cols %}<th>{{ c }}</th>{% endfor %}</tr></thead>
      <tbody>{% for r in m.rows %}<tr><td>{{ r.name }}</td>
        {% for v in r.cells %}<td>{{ v }}</td>{% endfor %}</tr>{% endfor %}</tbody>
    </table></div>
  {% endfor %}
</body></html>"""

_METRIC_DESC = {
    "metric": "Fraction of the true top-k most-lethal genes recovered (recall@k).",
    "eval_max_value": "Best true lethality found so far, normalized by the global max.",
    "eval_diversity": "Mean pairwise embedding distance of the cumulative revealed set.",
}


def metric_columns(df: pd.DataFrame) -> list[str]:
    cols = ["metric"] + sorted(c for c in df.columns if c.startswith("eval_"))
    return [c for c in cols if c in df.columns]


def _label(col: str, df: pd.DataFrame) -> str:
    if col == "metric":
        return str(df["metric_name"].iloc[0]) if "metric_name" in df else "recall"
    return col[len("eval_"):]


def _mean_ci(series: pd.Series) -> tuple[float, float]:
    x = series.to_numpy(dtype=float)
    n = len(x)
    mean = float(np.mean(x))
    if n < 2:
        return mean, 0.0
    se = float(np.std(x, ddof=1)) / np.sqrt(n)
    return mean, 1.96 * se


def stage_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    methods = sorted(df["method"].unique())
    rounds = sorted(df["round"].unique())
    data = {}
    for r in rounds:
        cells = []
        for m in methods:
            sub = df[(df["method"] == m) & (df["round"] == r)][col]
            mean, ci = _mean_ci(sub)
            cells.append(f"{mean:.3f} ± {ci:.3f}")
        data[r] = cells
    return pd.DataFrame(data, index=methods)


def _error_bar_plot(df: pd.DataFrame, col: str, label: str) -> str:
    fig = go.Figure()
    rounds = sorted(df["round"].unique())
    for m in sorted(df["method"].unique()):
        means, cis = [], []
        for r in rounds:
            mean, ci = _mean_ci(df[(df["method"] == m) & (df["round"] == r)][col])
            means.append(mean); cis.append(ci)
        fig.add_trace(go.Scatter(
            x=rounds, y=means, mode="lines+markers", name=m,
            error_y=dict(type="data", array=cis, visible=True, thickness=1.2, width=3)))
    fig.update_layout(template="simple_white", xaxis_title="stage (round)",
                      yaxis_title=label, legend_title="model",
                      margin=dict(l=60, r=20, t=10, b=50), height=420)
    fig.update_xaxes(dtick=1)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def build_report2(df: pd.DataFrame, out_path, title: str = "geneal report",
                  dual_div: str | None = None) -> Path:
    cols = metric_columns(df)
    metrics = []
    for c in cols:
        label = _label(c, df)
        t = stage_table(df, c)
        metrics.append({
            "label": label,
            "desc": _METRIC_DESC.get(c, "alpha-NDCG: quality-weighted, redundancy-discounted coverage." if "alpha" in c else ""),
            "plot": _error_bar_plot(df, c, label),
            "cols": list(t.columns),
            "rows": [{"name": idx, "cells": list(t.loc[idx])} for idx in t.index],
        })
    env = Environment(loader=BaseLoader())
    html = env.from_string(_TEMPLATE).render(
        title=title, metrics=metrics,
        metric_labels=[m["label"] for m in metrics],
        methods=sorted(df["method"].unique()),
        n_seeds=df["seed"].nunique(), n_rounds=df["round"].nunique(),
        dual_div=dual_div)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
