# src/geneal/report/ablation_report.py
"""Default two-analysis ablation report (geneal Plan 7, revision v2).

Faceted by toxicity definition (contrast line = primary; aggregate common-
essential = baseline). For each:

  Section A -- safety vs efficacy: 7 methods (greedy / trunc_known / trunc_pred /
    ehvi + random / farthest / cluster) as points on the efficacy-toxicity
    tradeoff, with an aggregate panel AND a per-cell-line subplot grid (manuscript
    subfigures), plus a method table.

  Section B -- diversity / robustness: operators (none/cap/kdpp) on three bases
    (greedy / truncation / ehvi): metric grid, concentration/robustness bar,
    efficacy-vs-concentration.

Publication-grade plotly styling; key figures also exported as vector PDF + PNG
to <fig_dir> (via kaleido, if available)."""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jinja2 import Environment, BaseLoader

# ---- publication palette + typography ----------------------------------------
FONT = "Inter, 'Helvetica Neue', Arial, sans-serif"
INK = "#1f2933"
GRID = "#e8ebef"
COLORS = {
    "greedy": "#d1495b", "trunc_known": "#2e6f95", "trunc_pred": "#7eb6d9",
    "ehvi": "#1b9e77", "random": "#9aa0a6", "farthest": "#8e6fb0",
    "cluster": "#e0a32e", "info_div": "#c2548a",
}
LABELS = {
    "greedy": "greedy (no safety)", "trunc_known": "truncation · known tox",
    "trunc_pred": "truncation · predicted tox", "ehvi": "EHVI (learned)",
    "random": "random", "farthest": "farthest (coverage)", "cluster": "cluster (density)",
    "info_div": "info-diverse (IterPert-like)",
}
A_ORDER = ["greedy", "trunc_known", "trunc_pred", "ehvi", "random", "farthest",
           "cluster", "info_div"]
OP_ORDER = ["none", "cap", "kdpp"]
OP_COLOR = {"none": "#9aa0a6", "cap": "#2e6f95", "kdpp": "#1b9e77"}
B_BASE_ORDER = ["greedy", "truncation", "ehvi"]
B_BASE_COLOR = {"greedy": "#d1495b", "truncation": "#2e6f95", "ehvi": "#1b9e77"}

_A_METRICS = [("mean_efficacy", "Mean efficacy"), ("max_efficacy", "Max efficacy"),
              ("mean_toxicity", "Mean toxicity")]
_B_METRICS = [("concentration", "Concentration↓"), ("robustness", "Robustness↑"),
              ("n_pathways", "Distinct pathways↑"), ("alpha_ndcg", "α-NDCG↑"),
              ("mean_efficacy", "Mean efficacy↑")]


def _ci(x):
    x = np.asarray(x, float); n = len(x); m = float(x.mean()) if n else float("nan")
    return m, (0.0 if n < 2 else 1.96 * float(x.std(ddof=1)) / np.sqrt(n))


def _style(fig, height=480, legend_bottom=True):
    fig.update_layout(
        font=dict(family=FONT, size=14, color=INK),
        plot_bgcolor="white", paper_bgcolor="white",
        height=height, margin=dict(l=72, r=28, t=30, b=84 if legend_bottom else 56),
        colorway=list(COLORS.values()))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     linecolor="#c7ccd1", ticks="outside", tickcolor="#c7ccd1")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     linecolor="#c7ccd1", ticks="outside", tickcolor="#c7ccd1")
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.16,
                                      xanchor="left", x=0, font=dict(size=12)))
    return fig


def _export(fig, fig_dir, name):
    """Write a vector PDF + PNG for manuscript use (kaleido). Silently skip if
    kaleido is unavailable."""
    if fig_dir is None:
        return
    try:
        fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(fig_dir / f"{name}.pdf"))
        fig.write_image(str(fig_dir / f"{name}.png"), scale=2)
    except Exception:
        pass


def _html(fig):
    return fig.to_html(full_html=False, include_plotlyjs="cdn",
                       config={"displayModeBar": False})


# ---- Section A ---------------------------------------------------------------
def _tradeoff_points(d, src, fig_dir):
    """Aggregate efficacy-toxicity tradeoff: 7 methods as points (mean over lines
    x seeds) with 95% CI bars. Up-left = safer & more lethal."""
    methods = [m for m in A_ORDER if m in set(d.method)]
    fig = go.Figure()
    for m in methods:
        g = d[d.method == m]
        ex, exc = _ci(g["mean_toxicity"]); ey, eyc = _ci(g["mean_efficacy"])
        fig.add_trace(go.Scatter(
            x=[ex], y=[ey], mode="markers", name=LABELS.get(m, m),
            error_x=dict(type="data", array=[exc], thickness=1.3, width=5, color=COLORS.get(m)),
            error_y=dict(type="data", array=[eyc], thickness=1.3, width=5, color=COLORS.get(m)),
            marker=dict(size=15, color=COLORS.get(m, "#444"),
                        line=dict(width=1.4, color="white"))))
    fig.add_annotation(x=0.01, y=0.99, xref="paper", yref="paper",
                       text="◤ safer &amp; more lethal", showarrow=False,
                       font=dict(color="#127a4f", size=13), xanchor="left", yanchor="top")
    _style(fig, height=520)
    fig.update_layout(xaxis_title="mean toxicity of nominees  (← safer)",
                      yaxis_title="mean efficacy (lethality)  (↑ more potent)")
    _export(fig, fig_dir, f"tradeoff_{src}")
    return _html(fig)


def _tradeoff_per_line(d, src, lines, fig_dir):
    """Per-cell-line tradeoff subplot grid (manuscript subfigures): one panel per
    line, the 7 methods as points (mean over seeds)."""
    lines = [l for l in lines if l in set(d.cell_line)]
    if not lines:
        return None
    ncol = min(3, len(lines)); nrow = math.ceil(len(lines) / ncol)
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=lines,
                        horizontal_spacing=0.07, vertical_spacing=0.12)
    methods = [m for m in A_ORDER if m in set(d.method)]
    for k, cl in enumerate(lines):
        r, c = k // ncol + 1, k % ncol + 1
        dl = d[d.cell_line == cl]
        for m in methods:
            g = dl[dl.method == m]
            if not len(g):
                continue
            ex, exc = _ci(g["mean_toxicity"]); ey, eyc = _ci(g["mean_efficacy"])
            fig.add_trace(go.Scatter(
                x=[ex], y=[ey], mode="markers", name=LABELS.get(m, m),
                legendgroup=m, showlegend=(k == 0),
                error_x=dict(type="data", array=[exc], thickness=1, width=3, color=COLORS.get(m)),
                error_y=dict(type="data", array=[eyc], thickness=1, width=3, color=COLORS.get(m)),
                marker=dict(size=11, color=COLORS.get(m, "#444"),
                            line=dict(width=1, color="white"))), row=r, col=c)
    _style(fig, height=300 * nrow)
    fig.update_xaxes(title_text="toxicity →", title_font=dict(size=11))
    fig.update_yaxes(title_text="efficacy ↑", title_font=dict(size=11))
    for ann in fig.layout.annotations:
        ann.font = dict(size=13, family=FONT, color=INK)
    _export(fig, fig_dir, f"tradeoff_perline_{src}")
    return _html(fig)


def _table_A(d):
    methods = [m for m in A_ORDER if m in set(d.method)]
    rows = []
    for m in methods:
        cells = []
        for col, _ in _A_METRICS:
            mn, ci = _ci(d[d.method == m][col]); cells.append(f"{mn:.3f} ± {ci:.3f}")
        rows.append({"m": LABELS.get(m, m), "cells": cells})
    return [lbl for _, lbl in _A_METRICS], rows


# ---- Section B ---------------------------------------------------------------
def _b_table(d):
    cols = [lbl for _, lbl in _B_METRICS]
    rows = []
    for base in B_BASE_ORDER:
        for op in OP_ORDER:
            g = d[(d.base == base) & (d.operator == op)]
            if not len(g):
                continue
            cells = [f"{_ci(g[col])[0]:.3f} ± {_ci(g[col])[1]:.3f}" for col, _ in _B_METRICS]
            rows.append({"label": f"{base} + {op}", "cells": cells})
    return cols, rows


def _b_bar(d, src, fig_dir):
    bases = [b for b in B_BASE_ORDER if b in set(d.base)]
    labels, conc, rob = [], [], []
    for b in bases:
        for op in OP_ORDER:
            g = d[(d.base == b) & (d.operator == op)]
            if len(g):
                labels.append(f"{b}+{op}"); conc.append(_ci(g["concentration"])[0])
                rob.append(_ci(g["robustness"])[0])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=conc, name="concentration ↓", marker_color="#d1495b"))
    fig.add_trace(go.Bar(x=labels, y=rob, name="robustness ↑", marker_color="#1b9e77"))
    _style(fig, height=430)
    fig.update_layout(barmode="group", yaxis_title="metric value")
    fig.update_xaxes(tickangle=-30)
    _export(fig, fig_dir, f"diversity_bar_{src}")
    return _html(fig)


def _b_eff_conc(d, src, fig_dir):
    bases = [b for b in B_BASE_ORDER if b in set(d.base)]
    sym = {"none": "circle", "cap": "square", "kdpp": "diamond"}
    fig = go.Figure()
    for b in bases:
        xs, ys, txt = [], [], []
        for op in OP_ORDER:
            g = d[(d.base == b) & (d.operator == op)]
            if not len(g):
                continue
            xs.append(_ci(g["concentration"])[0]); ys.append(_ci(g["mean_efficacy"])[0]); txt.append(op)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers+text", text=txt,
                      textposition="top center", name=b,
                      line=dict(color=B_BASE_COLOR.get(b, "#444"), dash="dot", width=1.5),
                      marker=dict(size=13, color=B_BASE_COLOR.get(b, "#444"),
                                  symbol=[sym.get(t, "circle") for t in txt],
                                  line=dict(width=1, color="white"))))
    _style(fig, height=470)
    fig.update_layout(xaxis_title="pathway concentration  (← better hedged)",
                      yaxis_title="mean efficacy  (↑ more potent)")
    _export(fig, fig_dir, f"eff_vs_conc_{src}")
    return _html(fig)


# ---- headlines ---------------------------------------------------------------
def _headline_A(d):
    def mof(m, c):
        g = d[d.method == m]
        return _ci(g[c])[0] if len(g) else float("nan")
    return (
        f"Naive <b>greedy</b> carries toxicity {mof('greedy','mean_toxicity'):.2f} at "
        f"efficacy {mof('greedy','mean_efficacy'):.2f}. The <b>known-toxicity ceiling</b> "
        f"(trunc_known) reaches toxicity {mof('trunc_known','mean_toxicity'):.2f} / efficacy "
        f"{mof('trunc_known','mean_efficacy'):.2f}; the <b>learned</b> ceilings "
        f"(trunc_pred {mof('trunc_pred','mean_toxicity'):.2f}, EHVI "
        f"{mof('ehvi','mean_toxicity'):.2f}) sit between — the gap to trunc_known is the "
        f"price of learning safety from the embedding."
    )


def _headline_B(d):
    bits = []
    for base in [b for b in B_BASE_ORDER if b in set(d.base)]:
        g0 = d[(d.base == base) & (d.operator == "none")]
        if not len(g0):
            continue
        c0, e0 = _ci(g0["concentration"])[0], _ci(g0["mean_efficacy"])[0]
        best_op, best_c, best_e = "none", c0, e0
        for op in ("cap", "kdpp"):
            g = d[(d.base == base) & (d.operator == op)]
            if len(g) and _ci(g["concentration"])[0] < best_c:
                best_op, best_c, best_e = op, _ci(g["concentration"])[0], _ci(g["mean_efficacy"])[0]
        if best_op != "none":
            bits.append(f"<b>{base}</b>: {best_op} cuts concentration "
                        f"{c0:.0%}→{best_c:.0%} (efficacy {e0:.2f}→{best_e:.2f})")
    return ("Diversity operators de-concentrate the portfolio — " + "; ".join(bits) + "."
            ) if bits else "Diversity operators applied per base."


_FACET_TMPL = """
<h2>Toxicity definition: <span style="color:#2e6f95">{{ src }}</span>{{ primary }}</h2>

<h3>A &middot; Safety vs efficacy — all methods</h3>
<div class="note">Each point a method (mean over lines × seeds, 95% CI bars). Up = more lethal, left = safer; the up-left corner is the goal.</div>
<div class="card">{{ tradeoff }}</div>
{% if perline %}
<h3>A &middot; Per-cell-line tradeoff (manuscript subfigures)</h3>
<div class="note">One panel per cell line; the same 7 methods. Shows the safety ordering is consistent across lines, not an averaging artifact.</div>
<div class="card">{{ perline }}</div>
{% endif %}
<h3>A &middot; Method table</h3>
<div class="card"><table>
<thead><tr><th>method</th>{% for h in a_cols %}<th>{{ h }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in a_rows %}<tr><td>{{ r.m }}</td>{% for c in r.cells %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table></div>

<h3>B &middot; Diversity / robustness</h3>
<div class="key">{{ headline_b|safe }}</div>
<div class="note">Operators none → cap → kdpp on three bases (greedy / truncation / ehvi). k-DPP similarity = {{ kdpp_sim }} (mechanism space, never outcome).</div>
<div class="card"><table>
<thead><tr><th>base + operator</th>{% for h in b_cols %}<th>{{ h }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in b_rows %}<tr><td>{{ r.label }}</td>{% for c in r.cells %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table></div>
<div class="card">{{ b_bar }}</div>
<h3>B &middot; Efficacy vs concentration</h3>
<div class="note">Each line a base; markers are operators none→cap→kdpp. Left = better hedged; high = efficacy retained.</div>
<div class="card">{{ b_eff_conc }}</div>
"""

_TEMPLATE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<title>{{ title }}</title><style>
 body{font-family:Inter,'Helvetica Neue',Arial,sans-serif;color:#1f2933;margin:0;padding:2.6rem 3.2rem;background:#fbfcfd;line-height:1.6;max-width:1100px}
 h1{font-weight:750;letter-spacing:-.02em;margin-bottom:.15rem;font-size:1.7rem}
 .sub{color:#6b7280;margin-bottom:1.3rem;font-size:.95rem}
 h2{margin-top:2.8rem;font-weight:700;font-size:1.3rem;border-bottom:2.5px solid #2e6f95;padding-bottom:4px}
 h3{margin-top:1.6rem;font-weight:650;color:#374151;font-size:1.05rem}
 .card{background:#fff;border:1px solid #e8ebef;border-radius:14px;padding:1.1rem 1.4rem;margin:.9rem 0;box-shadow:0 1px 4px rgba(20,30,45,.05)}
 table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.92rem}
 th,td{border-bottom:1px solid #eef1f4;padding:7px 12px;text-align:right}
 th:first-child,td:first-child{text-align:left;font-weight:600}
 thead th{color:#6b7280;border-bottom:2px solid #e8ebef}
 .note{color:#6b7280;font-size:.9rem;margin:.2rem 0 .6rem}
 .key{background:#eef5fa;border-left:3px solid #2e6f95;padding:.6rem 1rem;border-radius:7px;margin:.5rem 0;font-size:.95rem}
 .gloss{background:#fff;border:1px solid #e8ebef;border-radius:11px;padding:.9rem 1.2rem;font-size:.92rem}
</style></head><body>
<h1>{{ title }}</h1>
<div class="sub">{{ meta.n_genes }} genes ({{ panel_label }}) &middot; {{ n_lines }} cell lines &middot; {{ n_seeds }} seeds &middot; K={{ meta.K }} &middot; AL {{ meta.n_rounds }}×{{ meta.batch }} &middot; τ-quantile={{ meta.tau }} &middot; contrast line {{ meta.contrast_line }} &middot; acq={{ meta.acq_score }}</div>

<h2>How to read this</h2>
<div class="gloss">{{ glossary|safe }}</div>

{% for f in facets %}
<div class="key">{{ f.headline_a|safe }}</div>
{{ f.block|safe }}
{% endfor %}
</body></html>"""

_GLOSSARY = """
<b>Two analyses, two questions, two toxicity definitions.</b><br><br>
<b>Safety rules (Analysis A).</b>
&bull; <b>greedy</b>: top-K predicted efficacy, no safety. &bull; <b>truncation · known</b>: keep the safest τ-fraction by the <i>known</i> toxicity (oracle ceiling). &bull; <b>truncation · predicted</b>: same, but toxicity is <i>learned</i> by a GP from revealed labels (greedy acquisition). &bull; <b>EHVI</b>: learned toxicity, with a dual-objective EHVI <i>acquisition</i>. &bull; <b>random / farthest / cluster</b>: diversity-first baselines (no safety). <i>trunc_known is the upper limit the learned rules chase — their gap is the cost of learning safety.</i><br><br>
<b>Diversity operators (Analysis B).</b> &bull; <b>none</b>: top-K by quality. &bull; <b>cap</b>: ≤ c per CORUM pathway. &bull; <b>kdpp</b>: quality-weighted k-DPP with a dense embedding-cosine (mechanism) similarity — <i>never outcome similarity</i>. Layered on three bases (greedy / truncation / ehvi).<br><br>
<b>Toxicity definitions.</b> &bull; <b>contrast</b> (primary): lethality in one fixed contrast cell line (normal-tissue stand-in). &bull; <b>aggregate</b>: common-essential fraction, excluding the target line (leakage-safe). τ is a quantile (the safest fraction kept), so it is comparable across both.<br><br>
<b>Metrics</b> (true values of the nominated portfolio): mean/max efficacy, mean toxicity; pathway concentration↓, dropout robustness↑, distinct pathways↑, α-NDCG↑.
"""


def build_ablation_report(df: pd.DataFrame, out_path, scatter=None, meta=None,
                          fig_dir=None,
                          title="geneal — default ablation (safety + diversity)") -> Path:
    meta = meta or {}
    lines = meta.get("lines") or sorted(df.cell_line.unique())
    kdpp_sim = meta.get("kdpp_sim", "embedding")
    tox_sources = [s for s in (meta.get("tox_sources") or ["contrast", "aggregate"])
                   if s in set(df.tox_source)] or sorted(df.tox_source.unique())

    facets = []
    for src in tox_sources:
        d = df[df.tox_source == src]
        dA, dB = d[d.analysis == "A"], d[d.analysis == "B"]
        a_cols, a_rows = _table_A(dA); b_cols, b_rows = _b_table(dB)
        block = Environment(loader=BaseLoader()).from_string(_FACET_TMPL).render(
            src=src, primary=" (primary)" if src == "contrast" else "",
            tradeoff=_tradeoff_points(dA, src, fig_dir),
            perline=_tradeoff_per_line(dA, src, lines, fig_dir),
            a_cols=a_cols, a_rows=a_rows, headline_b=_headline_B(dB),
            kdpp_sim=kdpp_sim, b_cols=b_cols, b_rows=b_rows,
            b_bar=_b_bar(dB, src, fig_dir), b_eff_conc=_b_eff_conc(dB, src, fig_dir))
        facets.append({"block": block, "headline_a": _headline_A(dA)})

    panel = meta.get("panel") or ""
    panel_label = panel if (panel and panel.lower() != "none") else "full genome"
    html = Environment(loader=BaseLoader()).from_string(_TEMPLATE).render(
        title=title, meta=meta, panel_label=panel_label,
        n_lines=df.cell_line.nunique(), n_seeds=df.seed.nunique(),
        glossary=_GLOSSARY, facets=facets)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path


if __name__ == "__main__":
    import sys, json
    df = pd.read_parquet(sys.argv[1])
    base = Path(sys.argv[1]).parent
    sc = pd.read_parquet(base / "scatter.parquet") if (base / "scatter.parquet").exists() else None
    mp = base / "meta.json"
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    p = build_ablation_report(df, sys.argv[2] if len(sys.argv) > 2 else "ablation_report.html",
                              scatter=sc, meta=meta, fig_dir=base / "figs")
    print(f"report -> {p}")
