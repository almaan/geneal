# src/geneal/report/ablation_report.py
"""Default two-analysis ablation report (geneal Plan 7).

Section A -- safety vs efficacy: the 6 methods (none/truncation/ehvi +
random/coreset/typiclust) as points on the efficacy-toxicity tradeoff (Pareto-
better = up-left), plus a per-gene cloud with each method's picks overlaid, plus
a method table.

Section B -- diversity / robustness: the operators (none/cap/kdpp) layered on two
bases (greedy + best-safety), shown as a 2x3 metric grid, a grouped bar of
concentration / robustness, and an efficacy-vs-concentration scatter (operators
must de-concentrate without tanking efficacy).

PubMedBERT predicts; CORUM gives pathways (cap / concentration); STRING gives the
k-DPP similarity. STRING is a similarity, never a prediction embedding."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, BaseLoader

# stable colors per Analysis-A method
_COLORS = {
    "greedy": "#ef4444", "truncation": "#2563eb", "ehvi": "#0e9f6e",
    "random": "#9ca3af", "coreset": "#a855f7", "typiclust": "#f59e0b",
}
_A_ORDER = ["greedy", "truncation", "ehvi", "random", "coreset", "typiclust"]
_OP_ORDER = ["none", "cap", "kdpp"]
_OP_COLOR = {"none": "#9ca3af", "cap": "#2563eb", "kdpp": "#0e9f6e"}

_GLOSSARY = """
<b>Two separate questions, two analyses.</b> We do NOT entangle safety and diversity into one grid.<br><br>
<b>Analysis A — safety vs efficacy.</b> One axis, the <i>safety rule</i> applied when nominating the final K targets:<br>
&bull; <b>greedy</b> — top-K by predicted lethality, NO safety constraint (naive baseline).<br>
&bull; <b>truncation</b> — max predicted lethality s.t. <i>known</i> toxicity ≤ τ (common-essential ceiling, a priori).<br>
&bull; <b>ehvi</b> — a dual-objective active-learning acquisition (Expected Hypervolume Improvement over efficacy+toxicity GPs) that <i>learns</i> toxicity; nomination filters on the predicted-toxicity ceiling.<br>
&bull; <b>random / coreset / typiclust</b> — naive + prior-work (IterPert-style) acquisition baselines.<br><br>
<b>Analysis B — diversity / robustness.</b> Operators applied at nomination, layered on a base:<br>
&bull; <b>none</b> — top-K by quality (concentration-blind).<br>
&bull; <b>cap</b> — at most <i>c</i> genes per CORUM pathway/complex (hard hedge; bolts onto any method).<br>
&bull; <b>kdpp</b> — quality-weighted k-DPP with a STRING-derived similarity S (soft diversity).<br><br>
<b>Representations.</b> PubMedBERT embeddings predict efficacy/toxicity. STRING is a <i>network → a similarity</i> (not a per-gene feature vector), so it powers the k-DPP S; CORUM gives pathway membership for capping. They structure the diversity step; they never predict.
"""

_A_METRICS = [
    ("mean_efficacy", "Mean efficacy", True),
    ("max_efficacy", "Max efficacy", True),
    ("mean_toxicity", "Mean toxicity", False),
]
_B_METRICS = [
    ("concentration", "Concentration↓", False),
    ("robustness", "Robustness↑", True),
    ("n_pathways", "Distinct pathways↑", True),
    ("alpha_ndcg", "α-NDCG↑", True),
    ("mean_efficacy", "Mean efficacy↑", True),
]


def _ci(x):
    x = np.asarray(x, float); n = len(x); m = float(x.mean())
    return m, (0.0 if n < 2 else 1.96 * float(x.std(ddof=1)) / np.sqrt(n))


def _agg(df, col):
    """method -> (mean, ci) for a column."""
    return {m: _ci(g[col]) for m, g in df.groupby("method")}


# --------------------------------------------------------------------------- #
# Section A                                                                    #
# --------------------------------------------------------------------------- #
def _tradeoff_points(dfA):
    """Each safety method a point at (mean toxicity, mean efficacy) with x/y 95%CI
    error bars. Pareto-better = UP (more lethal) and LEFT (safer)."""
    methods = [m for m in _A_ORDER if m in set(dfA.method)]
    fig = go.Figure()
    for m in methods:
        g = dfA[dfA.method == m]
        ex, exc = _ci(g["mean_toxicity"]); ey, eyc = _ci(g["mean_efficacy"])
        fig.add_trace(go.Scatter(
            x=[ex], y=[ey], mode="markers+text", name=m, text=[m],
            textposition="top center",
            error_x=dict(type="data", array=[exc], thickness=1.2, width=4),
            error_y=dict(type="data", array=[eyc], thickness=1.2, width=4),
            marker=dict(size=13, color=_COLORS.get(m, "#444"),
                        symbol="circle", line=dict(width=1, color="white"))))
    fig.add_annotation(x=0, y=1, xref="paper", yref="paper",
                       text="◤ safer & more lethal", showarrow=False,
                       font=dict(color="#16a34a", size=12), xanchor="left")
    fig.update_layout(template="simple_white",
                      xaxis_title="mean toxicity (common-essential)  (← safer)",
                      yaxis_title="mean efficacy (lethality)  (↑ more potent)",
                      height=500, margin=dict(l=70, r=30, t=20, b=60),
                      showlegend=False)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _gene_cloud(scatter, methods=("greedy", "truncation", "ehvi")):
    """Per-gene cloud (one representative line/seed): all genes grey; each safety
    method's picks overlaid."""
    if scatter is None or scatter.empty:
        return None
    cl = scatter["cell_line"].iloc[0]
    d = scatter[scatter["cell_line"] == cl]
    eff, tox = d["efficacy"].to_numpy(), d["toxicity"].to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tox, y=eff, mode="markers", name="all genes",
                  marker=dict(size=4, color="#d1d5db"), hoverinfo="skip"))
    symbols = {"greedy": "x", "truncation": "circle-open", "ehvi": "diamond-open"}
    for m in methods:
        col = f"pick_{m}"
        if col not in d.columns:
            continue
        pk = d[col].to_numpy(dtype=bool)
        fig.add_trace(go.Scatter(x=tox[pk], y=eff[pk], mode="markers",
                      name=f"picked: {m}",
                      marker=dict(size=10, color=_COLORS.get(m, "#444"),
                                  symbol=symbols.get(m, "circle"),
                                  line=dict(width=2))))
    fig.update_layout(template="simple_white",
                      xaxis_title="toxicity: common-essential  (← safer)",
                      yaxis_title="efficacy: lethality in this line  (↑ more potent)",
                      height=540, margin=dict(l=70, r=30, t=20, b=110),
                      legend=dict(orientation="h", yanchor="top", y=-0.2,
                                  xanchor="left", x=0))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _table_A(dfA):
    methods = [m for m in _A_ORDER if m in set(dfA.method)]
    aggs = {col: _agg(dfA, col) for col, _, _ in _A_METRICS}
    cols = [label for _, label, _ in _A_METRICS]
    rows = []
    for m in methods:
        cells = []
        for col, _, _ in _A_METRICS:
            mn, ci = aggs[col][m]
            cells.append(f"{mn:.3f} ± {ci:.3f}")
        rows.append({"m": m, "cells": cells})
    return cols, rows


# --------------------------------------------------------------------------- #
# Section B                                                                    #
# --------------------------------------------------------------------------- #
def _b_table(dfB):
    """rows = (base, operator); cols = the diversity metrics, mean±CI."""
    bases = list(dict.fromkeys(dfB["base"]))
    cols = [label for _, label, _ in _B_METRICS]
    rows = []
    for base in bases:
        for op in _OP_ORDER:
            g = dfB[(dfB.base == base) & (dfB.operator == op)]
            if not len(g):
                continue
            cells = []
            for col, _, _ in _B_METRICS:
                mn, ci = _ci(g[col])
                cells.append(f"{mn:.3f} ± {ci:.3f}")
            rows.append({"label": f"{base} + {op}", "cells": cells})
    return cols, rows


def _b_bar(dfB):
    """Grouped bar: concentration (↓ good) and robustness (↑ good) per base×op."""
    bases = list(dict.fromkeys(dfB["base"]))
    labels = [f"{b}+{op}" for b in bases for op in _OP_ORDER
              if len(dfB[(dfB.base == b) & (dfB.operator == op)])]
    def series(col):
        out = []
        for b in bases:
            for op in _OP_ORDER:
                g = dfB[(dfB.base == b) & (dfB.operator == op)]
                if len(g):
                    out.append(_ci(g[col])[0])
        return out
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=series("concentration"),
                  name="concentration ↓ (lower better)", marker_color="#ef4444"))
    fig.add_trace(go.Bar(x=labels, y=series("robustness"),
                  name="robustness ↑ (higher better)", marker_color="#0e9f6e"))
    fig.update_layout(template="simple_white", barmode="group", height=420,
                      margin=dict(l=60, r=20, t=10, b=110),
                      yaxis_title="metric value",
                      legend=dict(orientation="h", yanchor="top", y=-0.25,
                                  xanchor="left", x=0))
    fig.update_xaxes(tickangle=-30)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _b_eff_conc(dfB):
    """Efficacy (y) vs concentration (x) per base×op: operators should move LEFT
    (de-concentrate) with little drop in efficacy. Same base = same color, marker
    by operator; arrows none→cap→kdpp implied by labels."""
    bases = list(dict.fromkeys(dfB["base"]))
    base_color = {b: c for b, c in zip(bases, ["#ef4444", "#2563eb", "#0e9f6e"])}
    sym = {"none": "circle", "cap": "square", "kdpp": "diamond"}
    fig = go.Figure()
    for b in bases:
        xs, ys, txt = [], [], []
        for op in _OP_ORDER:
            g = dfB[(dfB.base == b) & (dfB.operator == op)]
            if not len(g):
                continue
            xs.append(_ci(g["concentration"])[0]); ys.append(_ci(g["mean_efficacy"])[0])
            txt.append(op)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers+text", text=txt,
                      textposition="top center", name=b,
                      line=dict(color=base_color.get(b, "#444"), dash="dot"),
                      marker=dict(size=12, color=base_color.get(b, "#444"))))
    fig.update_layout(template="simple_white",
                      xaxis_title="pathway concentration  (← better hedged)",
                      yaxis_title="mean efficacy  (↑ more potent)",
                      height=460, margin=dict(l=60, r=20, t=20, b=70),
                      legend=dict(orientation="h", yanchor="top", y=-0.18,
                                  xanchor="left", x=0))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


# --------------------------------------------------------------------------- #
# Headlines                                                                    #
# --------------------------------------------------------------------------- #
def _headline_A(dfA, best_safety):
    def mof(m, c):
        g = dfA[dfA.method == m]
        return _ci(g[c])[0] if len(g) else float("nan")
    return (
        f"Naive <b>greedy</b> nomination carries mean toxicity "
        f"{mof('greedy','mean_toxicity'):.2f} at efficacy {mof('greedy','mean_efficacy'):.2f}. "
        f"The best safety rule (<b>{best_safety}</b>) reaches toxicity "
        f"{mof(best_safety,'mean_toxicity'):.2f} at efficacy "
        f"{mof(best_safety,'mean_efficacy'):.2f} — a better point on the "
        f"efficacy–toxicity tradeoff (up-left is better)."
    )


def _headline_B(dfB):
    bits = []
    for base in list(dict.fromkeys(dfB["base"])):
        g0 = dfB[(dfB.base == base) & (dfB.operator == "none")]
        if not len(g0):
            continue
        c0 = _ci(g0["concentration"])[0]; e0 = _ci(g0["mean_efficacy"])[0]
        best_op, best_c, best_e = "none", c0, e0
        for op in ("cap", "kdpp"):
            g = dfB[(dfB.base == base) & (dfB.operator == op)]
            if len(g) and _ci(g["concentration"])[0] < best_c:
                best_op, best_c, best_e = op, _ci(g["concentration"])[0], _ci(g["mean_efficacy"])[0]
        if best_op != "none":
            bits.append(f"on <b>{base}</b>, <b>{best_op}</b> cuts concentration "
                        f"{c0:.0%}→{best_c:.0%} (efficacy {e0:.2f}→{best_e:.2f})")
    return ("Diversity operators de-concentrate the portfolio: " + "; ".join(bits) + "."
            ) if bits else "Diversity operators applied on each base."


_TEMPLATE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<title>{{ title }}</title><style>
 body{font-family:'Inter',system-ui,sans-serif;color:#1a1a2e;margin:0;padding:2.5rem 3rem;background:#fafafb;line-height:1.55}
 h1{font-weight:700;letter-spacing:-.02em;margin-bottom:.2rem}
 .sub{color:#6b7280;margin-bottom:1.2rem}
 h2{margin-top:2.4rem;font-weight:650;border-bottom:2px solid #3b4cca;display:inline-block;padding-bottom:2px}
 h3{margin-top:1.4rem;font-weight:600;color:#374151}
 .card{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:1.1rem 1.4rem;margin:1rem 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}
 table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
 th,td{border-bottom:1px solid #e5e7eb;padding:7px 12px;text-align:right}
 th:first-child,td:first-child{text-align:left;font-weight:600}
 thead th{color:#6b7280;border-bottom:2px solid #e5e7eb}
 .note{color:#6b7280;font-size:.9rem;margin:.2rem 0 .7rem}
 .key{background:#eef2ff;border-left:3px solid #3b4cca;padding:.6rem 1rem;border-radius:6px;margin:.5rem 0}
 .gloss{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:.8rem 1.1rem;font-size:.92rem}
 .banner{background:#0f172a;color:#fff;border-radius:10px;padding:.5rem 1rem;font-weight:600;display:inline-block;margin-top:.4rem}
</style></head><body>
<h1>{{ title }}</h1>
<div class="sub">{{ meta.n_genes }} genes ({{ panel_label }}) &middot; {{ n_lines }} cell lines &middot; {{ n_seeds }} seeds &middot; K={{ meta.K }} targets &middot; AL {{ meta.n_rounds }}×{{ meta.batch }} &middot; τ={{ meta.tau }} &middot; cap={{ meta.cap }}</div>

<h2>How to read this</h2>
<div class="gloss">{{ glossary|safe }}</div>

<h2>Analysis A — safety vs efficacy</h2>
<div class="key">{{ headline_a|safe }}</div>
<h3>Efficacy–toxicity tradeoff (all methods)</h3>
<div class="note">Each point a method, averaged over lines×seeds, with 95% CI error bars. Up = more lethal, left = safer; the up-left corner is the goal. Baselines (random/coreset/typiclust) and naive greedy vs our safety rules (truncation, EHVI).</div>
<div class="card">{{ tradeoff_div|safe }}</div>
{% if cloud_div %}
<h3>Per-gene cloud (one representative line)</h3>
<div class="note">Every grey point a candidate gene. Each safety method's K picks overlaid. Naive greedy reaches into the toxic right; truncation/EHVI stay lethal-but-safe (upper-left).</div>
<div class="card">{{ cloud_div|safe }}</div>
{% endif %}
<h3>Method table</h3>
<div class="card"><table>
<thead><tr><th>method</th>{% for h in a_cols %}<th>{{ h }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in a_rows %}<tr><td>{{ r.m }}</td>{% for c in r.cells %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table></div>

<h2>Analysis B — diversity / robustness</h2>
<div class="key">{{ headline_b|safe }}</div>
<div class="note">Operators (none → cap → kdpp) layered on two bases: <b>greedy</b> (naive) and <b>{{ meta.best_safety }}</b> (the winner of Analysis A). Capping is a bolt-on-any-method hedge (graph-only).</div>
<h3>Diversity metric grid</h3>
<div class="card"><table>
<thead><tr><th>base + operator</th>{% for h in b_cols %}<th>{{ h }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in b_rows %}<tr><td>{{ r.label }}</td>{% for c in r.cells %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table></div>
<h3>Concentration & robustness</h3>
<div class="card">{{ b_bar_div|safe }}</div>
<h3>Efficacy vs concentration — de-concentrate at little efficacy cost</h3>
<div class="note">Each line a base; markers are operators none→cap→kdpp. Moving left = better hedged; staying high = efficacy retained.</div>
<div class="card">{{ b_eff_conc_div|safe }}</div>
</body></html>"""


def build_ablation_report(df: pd.DataFrame, out_path, scatter=None, meta=None,
                          title="geneal — default ablation (safety + diversity)") -> Path:
    meta = meta or {}
    dfA = df[df.analysis == "A"].copy()
    dfB = df[df.analysis == "B"].copy()
    best_safety = meta.get("best_safety", "truncation")

    a_cols, a_rows = _table_A(dfA)
    b_cols, b_rows = _b_table(dfB)
    panel = meta.get("panel") or ""
    panel_label = panel if (panel and panel.lower() != "none") else "full genome"

    html = Environment(loader=BaseLoader()).from_string(_TEMPLATE).render(
        title=title, meta=meta, panel_label=panel_label,
        n_lines=df.cell_line.nunique(), n_seeds=df.seed.nunique(),
        glossary=_GLOSSARY,
        headline_a=_headline_A(dfA, best_safety), headline_b=_headline_B(dfB),
        tradeoff_div=_tradeoff_points(dfA), cloud_div=_gene_cloud(scatter),
        a_cols=a_cols, a_rows=a_rows, b_cols=b_cols, b_rows=b_rows,
        b_bar_div=_b_bar(dfB), b_eff_conc_div=_b_eff_conc(dfB))
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
                              scatter=sc, meta=meta)
    print(f"report -> {p}")
