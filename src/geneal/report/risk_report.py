# src/geneal/report/risk_report.py
"""Detailed HTML report for risk-aware target nomination.

Compares three nomination strategies and evaluates each on independent
ground-truth axes (efficacy, toxicity, pathway risk):
  efficacy         - maximize predicted lethality only (ignores toxicity + pathway)
  selective        - maximize lethality MINUS a common-essential toxicity penalty
  selective_cap{c} - selective, with at most c genes per pathway (portfolio hedge)

Emits: a glossary; an efficacy-vs-toxicity scatter (the dual-optimization story);
a strategy-progression table; per-metric error-bar curves + mean+/-95%CI tables;
and a per-cell-line breakdown."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, BaseLoader

# (column, label, higher_is_better, one-line definition)
_METRICS = [
    ("mean_efficacy", "Mean efficacy (lethality)", True,
     "Mean true knockout-lethality (−Chronos effect) of the nominated targets in THIS cell line. Higher = more potent portfolio."),
    ("max_efficacy", "Max efficacy (best single hit)", True,
     "Lethality of the single most-lethal target in the portfolio. Shows the best hit is preserved even when the mean drops."),
    ("mean_toxicity", "Mean toxicity (common-essential)", False,
     "Mean common-essential score = fraction of ALL cell lines where the gene is lethal. High = pan-essential = likely toxic to normal tissue. LOWER is safer."),
    ("concentration", "Pathway concentration", False,
     "Max fraction of the portfolio sitting in a single pathway/complex. LOWER = better hedged against a pathway turning out toxic/undruggable."),
    ("robustness", "Dropout robustness", True,
     "Expected fraction of portfolio value surviving if one random pathway is eliminated (toxicity/failure). Higher = safer bet."),
]

_GLOSSARY = """
<b>Strategies</b><br>
&bull; <b>efficacy</b> — pick the K targets with highest predicted lethality, NO safety constraint. The naive baseline (&ldquo;maximize efficacy, check safety later&rdquo;).<br>
&bull; <b>constrained</b> — maximize predicted lethality SUBJECT TO toxicity ≤ τ (a hard safety ceiling). Toxicity = common-essential score (known pan-cancer annotation); efficacy in this line is predicted.<br>
&bull; <b>constrained_cap&lt;c&gt;</b> — constrained, but allow at most <b>c</b> genes per pathway/complex (e.g. cap2 = max 2 per pathway). Adds the portfolio hedge.<br><br>
<b>Why a constraint, not a weighted penalty?</b> Safety is usually a bar, not a tradeable quantity — a too-toxic target is disqualified. <b>Why also hedge pathways?</b> If many targets sit in one pathway and it proves toxic/undruggable, the whole bet fails together; capping spreads it across independent mechanisms.
"""


def _ci(x):
    x = np.asarray(x, float); n = len(x); m = float(x.mean())
    return m, (0.0 if n < 2 else 1.96 * float(x.std(ddof=1)) / np.sqrt(n))


def _order(df):
    caps = sorted({int(m.split("cap")[1]) for m in df.method.unique() if "cap" in m})
    return [x for x in (["efficacy", "constrained"] +
            [f"constrained_cap{c}" for c in caps]) if x in set(df.method)]


def _curve(df, col, label, order, higher_better):
    fig = go.Figure()
    means, cis = [], []
    for meth in order:
        mn, ci = _ci(df[df.method == meth][col]); means.append(mn); cis.append(ci)
    fig.add_trace(go.Scatter(x=order, y=means, mode="lines+markers",
                  error_y=dict(type="data", array=cis, visible=True, thickness=1.2, width=4)))
    arrow = "↑ better" if higher_better else "↓ better"
    fig.update_layout(template="simple_white", xaxis_title="strategy",
                      yaxis_title=f"{label}  ({arrow})", height=360,
                      margin=dict(l=60, r=20, t=10, b=70))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _scatter(scatter_df):
    """Efficacy (x) vs toxicity (y); overlay efficacy-picks vs selective-picks.
    The dual-optimization story: selective avoids the high-toxicity top-right."""
    if scatter_df is None or scatter_df.empty:
        return None
    # pool one representative cell line for clarity (the first)
    cl = scatter_df["cell_line"].iloc[0]
    d = scatter_df[scatter_df["cell_line"] == cl]
    eff, tox = d["efficacy"].to_numpy(), d["toxicity"].to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eff, y=tox, mode="markers", name="all candidates",
                  marker=dict(size=4, color="#cbd5e1"), hoverinfo="skip"))
    pe = d["picked_efficacy"].to_numpy()
    ps = d["picked_selective"].to_numpy()
    fig.add_trace(go.Scatter(x=eff[pe], y=tox[pe], mode="markers",
                  name="picked by EFFICACY (ignores toxicity)",
                  marker=dict(size=10, color="#ef4444", symbol="x")))
    fig.add_trace(go.Scatter(x=eff[ps], y=tox[ps], mode="markers",
                  name="picked by SELECTIVE (efficacy+safety)",
                  marker=dict(size=10, color="#2563eb", symbol="circle-open",
                              line=dict(width=2))))
    fig.update_layout(template="simple_white",
                      xaxis_title="efficacy: lethality in this line  (→ more potent)",
                      yaxis_title="toxicity: common-essential score  (↑ more toxic)",
                      height=540, margin=dict(l=70, r=30, t=20, b=110),
                      legend=dict(orientation="h", yanchor="top", y=-0.22,
                                  xanchor="left", x=0))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


_TEMPLATE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<title>{{ title }}</title><style>
 body{font-family:'Inter',system-ui,sans-serif;color:#1a1a2e;margin:0;padding:2.5rem 3rem;background:#fafafb;line-height:1.55}
 h1{font-weight:700;letter-spacing:-.02em;margin-bottom:.2rem}
 .sub{color:#6b7280;margin-bottom:1.2rem}
 h2{margin-top:2.2rem;font-weight:650;border-bottom:2px solid #3b4cca;display:inline-block;padding-bottom:2px}
 .card{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:1.1rem 1.4rem;margin:1rem 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}
 table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
 th,td{border-bottom:1px solid #e5e7eb;padding:7px 12px;text-align:right}
 th:first-child,td:first-child{text-align:left;font-weight:600}
 thead th{color:#6b7280;border-bottom:2px solid #e5e7eb}
 caption{text-align:left;font-weight:650;margin-bottom:.5rem}
 .note{color:#6b7280;font-size:.9rem;margin:.2rem 0 .7rem}
 .key{background:#eef2ff;border-left:3px solid #3b4cca;padding:.6rem 1rem;border-radius:6px;margin:.5rem 0}
 .gloss{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:.8rem 1.1rem;font-size:.92rem}
</style></head><body>
<h1>{{ title }}</h1>
<div class="sub">{{ n_lines }} cell lines &middot; {{ n_seeds }} seeds &middot; K={{ K }} targets/portfolio</div>
<div class="key">{{ headline }}</div>

<h2>How to read this</h2>
<div class="gloss">{{ glossary|safe }}</div>

{% if scatter_div %}
<h2>Efficacy vs toxicity — why optimize jointly</h2>
<div class="note">Each grey point a candidate gene. Red ✕ = chosen by efficacy-only (ignores toxicity). Blue ○ = chosen by the selective (efficacy+safety) objective. Efficacy-only reaches into the toxic top-right; selective stays in the lethal-but-safe lower-right.</div>
<div class="card">{{ scatter_div|safe }}</div>
{% endif %}

{% if tau_div %}
<h2>Constrained efficacy–toxicity frontier (τ sweep)</h2>
<div class="note">τ is the safety ceiling: nominees must have toxicity ≤ τ. τ=1 admits everything (pure efficacy); tightening τ buys safety at the cost of potency. The shape shows the real tension — the most lethal knockouts are often pan-essential (toxic), so a tight ceiling forces lower efficacy.</div>
<div class="card">{{ tau_div|safe }}</div>
{% endif %}

<h2>Strategy progression (efficacy → +safety → +pathway-hedge)</h2>
<div class="note">Each row adds a constraint. Watch efficacy trade against toxicity and concentration. Max-efficacy shows the best single hit is largely preserved.</div>
<div class="card"><table>
<thead><tr><th>strategy</th>{% for h in prog_cols %}<th>{{ h }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in prog_rows %}<tr><td>{{ r.m }}</td>{% for c in r.cells %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table></div>

{% for m in metrics %}
<h2>{{ m.label }}</h2><div class="note">{{ m.desc }}</div>
<div class="card">{{ m.plot|safe }}</div>
<div class="card"><table><caption>{{ m.label }} — mean ± 95% CI</caption>
<thead><tr><th>strategy</th><th>mean ± CI</th></tr></thead><tbody>
{% for r in m.rows %}<tr><td>{{ r.method }}</td><td>{{ r.cell }}</td></tr>{% endfor %}
</tbody></table></div>{% endfor %}
</body></html>"""


def _tau_frontier_div(lp):
    """Constrained efficacy-vs-toxicity frontier over the safety ceiling tau."""
    if lp is None or len(lp) == 0:
        return None
    agg = lp.groupby("tau").agg(eff=("mean_efficacy", "mean"),
                                tox=("mean_toxicity", "mean")).reset_index().sort_values("tau")
    fig = go.Figure(go.Scatter(x=agg["tox"], y=agg["eff"], mode="lines+markers+text",
                    text=[f"τ={t:g}" for t in agg["tau"]], textposition="top right",
                    marker=dict(size=11)))
    fig.update_layout(template="simple_white",
                      xaxis_title="mean toxicity of nominees (common-essential)  (↓ safer)",
                      yaxis_title="mean efficacy (lethality)  (↑ more potent)",
                      height=440, margin=dict(l=60, r=20, t=10, b=50),
                      title="Constrained frontier: tightening the safety ceiling τ trades potency for safety")
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def build_risk_report(df: pd.DataFrame, out_path, scatter=None, tau_frontier=None,
                      K=30, title="geneal — risk-aware target nomination") -> Path:
    order = _order(df)
    def mean_of(meth, col):
        s = df[df.method == meth][col]
        return _ci(s)[0] if len(s) else float("nan")
    cap2 = "selective_cap2" if "selective_cap2" in order else (order[-1] if order else None)
    headline = (
        f"Efficacy-only nomination puts {mean_of('efficacy','concentration'):.0%} of the "
        f"portfolio in one pathway and carries toxicity {mean_of('efficacy','mean_toxicity'):.2f}. "
        f"The selective+hedged strategy ({cap2}) cuts toxicity to "
        f"{mean_of(cap2,'mean_toxicity'):.2f} and concentration to "
        f"{mean_of(cap2,'concentration'):.0%}, while keeping max-efficacy "
        f"{mean_of(cap2,'max_efficacy'):.2f} vs {mean_of('efficacy','max_efficacy'):.2f} "
        f"(mean efficacy {mean_of('efficacy','mean_efficacy'):.2f}→{mean_of(cap2,'mean_efficacy'):.2f})."
    ) if cap2 else "risk-aware nomination."

    # progression table
    prog_cols = ["mean eff", "max eff", "toxicity↓", "concentration↓", "robustness↑"]
    prog_keys = ["mean_efficacy", "max_efficacy", "mean_toxicity", "concentration", "robustness"]
    prog_rows = [{"m": meth, "cells": [f"{mean_of(meth,k):.3f}" for k in prog_keys]}
                 for meth in order]

    metrics = []
    for col, label, hib, desc in _METRICS:
        rows = [{"method": m, "cell": f"{_ci(df[df.method==m][col])[0]:.3f} ± {_ci(df[df.method==m][col])[1]:.3f}"}
                for m in order]
        metrics.append({"label": label, "desc": desc,
                        "plot": _curve(df, col, label, order, hib), "rows": rows})

    html = Environment(loader=BaseLoader()).from_string(_TEMPLATE).render(
        title=title, n_lines=df.cell_line.nunique(), n_seeds=df.seed.nunique(),
        K=K, headline=headline, glossary=_GLOSSARY, scatter_div=_scatter(scatter),
        tau_div=_tau_frontier_div(tau_frontier),
        prog_cols=prog_cols, prog_rows=prog_rows, metrics=metrics)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path


if __name__ == "__main__":
    import sys
    df = pd.read_parquet(sys.argv[1])
    base = Path(sys.argv[1]).parent
    sc = pd.read_parquet(base / "scatter.parquet") if (base / "scatter.parquet").exists() else None
    tf = pd.read_parquet(base / "tau_frontier.parquet") if (base / "tau_frontier.parquet").exists() else None
    p = build_risk_report(df, sys.argv[2] if len(sys.argv) > 2 else "risk_report.html",
                          scatter=sc, tau_frontier=tf)
    print(f"report -> {p}")
