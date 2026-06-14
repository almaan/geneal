# src/geneal/report/bivariate_report.py
"""Report for bivariate efficacy-toxicity active learning (Plan 6).

Combines the risk-report nomination elements (efficacy-only vs joint comparison,
efficacy-vs-toxicity scatter, per-metric mean±CI tables) with the AL-specific
hypervolume-over-rounds curve (learned vs known-toxicity vs full oracle)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, BaseLoader

_GLOSS = """
<b>Setup.</b> Efficacy (lethality in the target line) AND toxicity are both unknown and
explored jointly by an active-learning loop using <b>EHVI</b> (Expected Hypervolume
Improvement) over the (efficacy↑, toxicity↓) objective space.<br>
<b>Toxicity reference</b> — <b>contrast</b>: lethality in a fixed proxy cell line
(different-domain stand-in for normal tissue); <b>population</b>: mean lethality across
all lines.<br>
<b>Hypervolume regimes</b> — <b>learned</b>: toxicity predicted + explored;
<b>known</b>: toxicity oracle (a-priori), only efficacy learned; <b>oracle</b>:
full information (both known) = absolute ceiling.<br>
<b>Final nomination</b> — from the learned bivariate surrogates we nominate K targets two
ways: <b>efficacy_only</b> (top-K by predicted lethality, ignoring toxicity) vs
<b>joint</b> (top-K by predicted selectivity = efficacy − toxicity). Metrics below are
TRUE values of the nominated set. <b>concentration</b> = max fraction in one pathway
(↓ better); <b>robustness</b> = expected value surviving a random pathway dropout (↑).
"""

_METRICS = [
    ("mean_efficacy", "Mean efficacy (lethality)", True),
    ("max_efficacy", "Max efficacy (best hit)", True),
    ("mean_toxicity", "Mean toxicity", False),
    ("concentration", "Pathway concentration", False),
    ("robustness", "Dropout robustness", True),
]


def _ci(x):
    x = np.asarray(x, float); n = len(x); m = float(x.mean())
    return m, (0.0 if n < 2 else 1.96 * float(x.std(ddof=1)) / np.sqrt(n))


def _hv_curve(hv):
    fig = go.Figure()
    dash = {"known": "dot", "learned": "solid", "oracle": "dashdot"}
    for cond in sorted(hv.condition.unique()):
        sub = hv[hv.condition == cond]
        rounds = sorted(sub["round"].unique())
        means = [_ci(sub[sub["round"] == r]["hypervolume"])[0] for r in rounds]
        cis = [_ci(sub[sub["round"] == r]["hypervolume"])[1] for r in rounds]
        reg = cond.split("_")[-1]
        fig.add_trace(go.Scatter(x=rounds, y=means, mode="lines+markers", name=cond,
                      line=dict(dash=dash.get(reg, "solid")),
                      error_y=dict(type="data", array=cis, visible=True, thickness=1, width=3)))
    fig.update_layout(template="simple_white", xaxis_title="active-learning round",
                      yaxis_title="dominated hypervolume (↑ better)", height=440,
                      margin=dict(l=60, r=20, t=10, b=70),
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(dtick=1)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _scatter(sc):
    if sc is None or sc.empty:
        return None
    cl = sc.cell_line.iloc[0]; d = sc[sc.cell_line == cl]
    eff, tox = d.efficacy.to_numpy(), d.toxicity.to_numpy()
    pe, ps = d.picked_efficacy.to_numpy(), d.picked_selective.to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eff, y=tox, mode="markers", name="all candidates",
                  marker=dict(size=4, color="#cbd5e1"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=eff[pe], y=tox[pe], mode="markers", name="efficacy_only picks",
                  marker=dict(size=10, color="#ef4444", symbol="x")))
    fig.add_trace(go.Scatter(x=eff[ps], y=tox[ps], mode="markers", name="joint picks",
                  marker=dict(size=10, color="#2563eb", symbol="circle-open", line=dict(width=2))))
    fig.update_layout(template="simple_white",
                      xaxis_title="efficacy: lethality in target line (→ potent)",
                      yaxis_title="toxicity: lethality in contrast line (↑ toxic)",
                      height=520, margin=dict(l=70, r=30, t=20, b=110),
                      legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _metric_curve(nom, col, label, hib, order):
    fig = go.Figure()
    means = [_ci(nom[nom.method == m][col])[0] for m in order]
    cis = [_ci(nom[nom.method == m][col])[1] for m in order]
    fig.add_trace(go.Scatter(x=order, y=means, mode="lines+markers",
                  error_y=dict(type="data", array=cis, visible=True, thickness=1.2, width=4)))
    fig.update_layout(template="simple_white", xaxis_title="nomination method",
                      yaxis_title=f"{label} ({'↑' if hib else '↓'} better)",
                      height=340, margin=dict(l=60, r=20, t=10, b=50))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


_TEMPLATE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<title>{{ title }}</title><style>
 body{font-family:'Inter',system-ui,sans-serif;color:#1a1a2e;margin:0;padding:2.5rem 3rem;background:#fafafb;line-height:1.55}
 h1{font-weight:700;margin-bottom:.2rem} .sub{color:#6b7280;margin-bottom:1.2rem}
 h2{margin-top:2rem;font-weight:650;border-bottom:2px solid #3b4cca;display:inline-block;padding-bottom:2px}
 .card{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:1.1rem 1.4rem;margin:1rem 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}
 .gloss{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:.8rem 1.1rem;font-size:.92rem}
 .key{background:#eef2ff;border-left:3px solid #3b4cca;padding:.6rem 1rem;border-radius:6px;margin:.5rem 0}
 .note{color:#6b7280;font-size:.9rem;margin:.2rem 0 .7rem}
 table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
 th,td{border-bottom:1px solid #e5e7eb;padding:7px 12px;text-align:right} th:first-child,td:first-child{text-align:left;font-weight:600}
 thead th{color:#6b7280}
</style></head><body>
<h1>{{ title }}</h1>
<div class="sub">{{ n_lines }} target lines &middot; {{ n_seeds }} seeds &middot; K={{ K }} &middot; toxicity proxy = {{ contrast }}</div>
<div class="key">{{ headline }}</div>
<h2>How to read this</h2><div class="gloss">{{ gloss|safe }}</div>

<h2>Joint discovery: hypervolume over rounds</h2>
<div class="note">How fast each regime discovers the efficacy–toxicity frontier. learned (solid) vs known-toxicity (dot) vs full oracle (dash-dot); error bars over seeds/lines.</div>
<div class="card">{{ hv_curve|safe }}</div>

{% if scatter_div %}
<h2>Efficacy vs toxicity — efficacy-only vs joint nomination</h2>
<div class="note">Final nominees from the learned surrogates. Red ✕ = efficacy-only (ignores toxicity); blue ○ = joint (efficacy−toxicity). Efficacy-only reaches into the toxic top-right; joint stays lethal-but-safe.</div>
<div class="card">{{ scatter_div|safe }}</div>
{% endif %}

<h2>Nomination metrics: efficacy_only vs joint (TRUE values, mean ± 95% CI)</h2>
<div class="card"><table>
<thead><tr><th>metric</th>{% for m in order %}<th>{{ m }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in table_rows %}<tr><td>{{ r.label }}</td>{% for c in r.cells %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table></div>
{% for m in metric_figs %}
<h3 style="margin-top:1.5rem;color:#374151">{{ m.label }}</h3><div class="card">{{ m.plot|safe }}</div>
{% endfor %}
</body></html>"""


def build_bivariate_report(hv, nom, out_path, scatter=None, contrast="?", K=30,
                           title="geneal — bivariate efficacy-toxicity active learning") -> Path:
    order = [m for m in ["efficacy_only", "joint"] if m in set(nom.method)]
    # cost-of-learning headline
    last = hv["round"].max()
    def hvf(cond):
        return _ci(hv[(hv.condition == cond) & (hv["round"] == last)]["hypervolume"])[0]
    ck, cl = hvf("contrast_known"), hvf("contrast_learned")
    eo_tox = _ci(nom[nom.method == "efficacy_only"]["mean_toxicity"])[0] if len(nom) else 0
    jo_tox = _ci(nom[nom.method == "joint"]["mean_toxicity"])[0] if len(nom) else 0
    headline = (f"Joint efficacy–toxicity AL: learning toxicity reaches "
                f"{100*cl/ck:.0f}% of the known-toxicity hypervolume (contrast). "
                f"At nomination, joint selection cuts toxicity {eo_tox:.2f}→{jo_tox:.2f} "
                f"vs efficacy-only.") if ck else "bivariate efficacy-toxicity AL."
    table_rows = []
    for col, label, _hib in _METRICS:
        cells = [f"{_ci(nom[nom.method==m][col])[0]:.3f} ± {_ci(nom[nom.method==m][col])[1]:.3f}"
                 for m in order]
        table_rows.append({"label": label, "cells": cells})
    metric_figs = [{"label": label, "plot": _metric_curve(nom, col, label, hib, order)}
                   for col, label, hib in _METRICS]
    html = Environment(loader=BaseLoader()).from_string(_TEMPLATE).render(
        title=title, n_lines=hv.cell_line.nunique(), n_seeds=hv.seed.nunique(),
        contrast=contrast, K=K, headline=headline, gloss=_GLOSS,
        hv_curve=_hv_curve(hv), scatter_div=_scatter(scatter),
        order=order, table_rows=table_rows, metric_figs=metric_figs)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
