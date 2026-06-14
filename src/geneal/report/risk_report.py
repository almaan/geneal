# src/geneal/report/risk_report.py
"""Detailed HTML report for the risk-aware target-nomination experiment.

Consumes a risk parquet (rows: cell_line, seed, method, selective_lethality,
concentration, robustness, n_pathways) and emits an elegant self-contained HTML:
per-metric mean+/-95%CI tables (method rows), error-bar curves over cap level,
the efficacy-risk Pareto, and a per-cell-line breakdown."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, BaseLoader

_METRICS = [
    ("selective_lethality", "Selective lethality (efficacy)",
     "Mean true selective-lethality of the nominated portfolio (higher = better targets)."),
    ("concentration", "Pathway concentration (RISK)",
     "Max fraction of the portfolio in any single pathway (lower = better hedged)."),
    ("robustness", "Dropout robustness",
     "Expected fraction of portfolio value surviving a random single-pathway dropout (higher = safer)."),
]

_TEMPLATE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<title>{{ title }}</title><style>
 body{font-family:'Inter',system-ui,sans-serif;color:#1a1a2e;margin:0;padding:2.5rem 3rem;background:#fafafb;line-height:1.5}
 h1{font-weight:700;letter-spacing:-.02em;margin-bottom:.2rem}
 .sub{color:#6b7280;margin-bottom:1.5rem}
 h2{margin-top:2.2rem;font-weight:650;border-bottom:2px solid #3b4cca;display:inline-block;padding-bottom:2px}
 .card{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:1.1rem 1.4rem;margin:1rem 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}
 table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
 th,td{border-bottom:1px solid #e5e7eb;padding:7px 12px;text-align:right}
 th:first-child,td:first-child{text-align:left;font-weight:600}
 thead th{color:#6b7280;border-bottom:2px solid #e5e7eb}
 caption{text-align:left;font-weight:650;margin-bottom:.5rem}
 .note{color:#6b7280;font-size:.9rem;margin:.2rem 0 .7rem}
 .key{background:#eef2ff;border-left:3px solid #3b4cca;padding:.6rem 1rem;border-radius:6px;margin:.5rem 0}
</style></head><body>
<h1>{{ title }}</h1>
<div class="sub">{{ n_lines }} cell lines &middot; {{ n_seeds }} seeds &middot; K={{ K }} targets/portfolio &middot; methods: {{ methods|join(', ') }}</div>
<div class="key">{{ headline }}</div>
<h2>Efficacy–risk frontier</h2>
<div class="note">Each point a selection method; greedy (no hedge) vs per-pathway caps. Down-right = better-hedged at efficacy cost.</div>
<div class="card">{{ pareto_div|safe }}</div>
{% for m in metrics %}
<h2>{{ m.label }}</h2><div class="note">{{ m.desc }}</div>
<div class="card">{{ m.plot|safe }}</div>
<div class="card"><table><caption>{{ m.label }} — mean ± 95% CI</caption>
<thead><tr><th>method</th><th>mean ± CI</th></tr></thead><tbody>
{% for r in m.rows %}<tr><td>{{ r.method }}</td><td>{{ r.cell }}</td></tr>{% endfor %}
</tbody></table></div>{% endfor %}
<h2>Per-cell-line breakdown</h2>
<div class="card"><table><caption>selective-lethality / concentration / robustness, per line (greedy → cap2)</caption>
<thead><tr><th>cell line</th>{% for h in per_line_cols %}<th>{{ h }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in per_line_rows %}<tr><td>{{ r.line }}</td>{% for c in r.cells %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table></div>
</body></html>"""


def _ci(x):
    x = np.asarray(x, float); n = len(x); m = float(x.mean())
    return m, (0.0 if n < 2 else 1.96 * float(x.std(ddof=1)) / np.sqrt(n))


def _order(df):
    caps = sorted({int(m[3:]) for m in df.method.unique() if m.startswith("cap")})
    return [x for x in (["greedy"] + [f"cap{c}" for c in caps]) if x in set(df.method)]


def _curve(df, col, label, order):
    fig = go.Figure()
    means, cis = [], []
    for meth in order:
        mn, ci = _ci(df[df.method == meth][col]); means.append(mn); cis.append(ci)
    fig.add_trace(go.Scatter(x=order, y=means, mode="lines+markers",
                  error_y=dict(type="data", array=cis, visible=True, thickness=1.2, width=4)))
    fig.update_layout(template="simple_white", xaxis_title="selection method",
                      yaxis_title=label, height=380, margin=dict(l=60, r=20, t=10, b=50))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _pareto(df, order):
    agg = df.groupby("method").agg(leth=("selective_lethality", "mean"),
                                   conc=("concentration", "mean")).reindex(order)
    fig = go.Figure(go.Scatter(x=agg["conc"], y=agg["leth"], mode="markers+text",
                    text=agg.index, textposition="top center", marker=dict(size=13)))
    fig.update_layout(template="simple_white",
                      xaxis_title="pathway concentration (max frac one pathway) — RISK",
                      yaxis_title="selective lethality — EFFICACY",
                      height=440, margin=dict(l=60, r=20, t=10, b=50))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def build_risk_report(df: pd.DataFrame, out_path, title="geneal — risk-aware target nomination") -> Path:
    order = _order(df)
    g = df[df.method == "greedy"]; gc, _ = _ci(g["concentration"]); gr, _ = _ci(g["robustness"])
    cap2 = df[df.method == "cap2"]
    headline = (f"Greedy concentrates {gc:.0%} of the portfolio in one pathway "
                f"(robustness {gr:.2f}). Per-pathway capping (cap2) cuts concentration to "
                f"{_ci(cap2['concentration'])[0]:.0%} and lifts robustness to "
                f"{_ci(cap2['robustness'])[0]:.2f}, at "
                f"{_ci(g['selective_lethality'])[0]-_ci(cap2['selective_lethality'])[0]:+.3f} "
                f"selective-lethality.") if len(cap2) else "greedy concentration baseline."
    metrics = []
    for col, label, desc in _METRICS:
        rows = [{"method": m, "cell": f"{_ci(df[df.method==m][col])[0]:.3f} ± {_ci(df[df.method==m][col])[1]:.3f}"} for m in order]
        metrics.append({"label": label, "desc": desc, "plot": _curve(df, col, label, order), "rows": rows})
    # per-line breakdown (greedy vs cap2)
    lines = sorted(df.cell_line.unique())
    per_line_rows = []
    for ln in lines:
        sub = df[df.cell_line == ln]
        cells = []
        for meth in ("greedy", "cap2"):
            s = sub[sub.method == meth]
            if len(s):
                cells.append(f"{s.selective_lethality.mean():.2f}/{s.concentration.mean():.2f}/{s.robustness.mean():.2f}")
            else:
                cells.append("—")
        per_line_rows.append({"line": ln, "cells": cells})
    html = Environment(loader=BaseLoader()).from_string(_TEMPLATE).render(
        title=title, n_lines=df.cell_line.nunique(), n_seeds=df.seed.nunique(),
        K=30, methods=order, headline=headline, pareto_div=_pareto(df, order),
        metrics=metrics, per_line_cols=["greedy", "cap2"], per_line_rows=per_line_rows)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path


if __name__ == "__main__":
    import sys
    df = pd.read_parquet(sys.argv[1])
    p = build_risk_report(df, sys.argv[2] if len(sys.argv) > 2 else "risk_report.html")
    print(f"report -> {p}")
