# src/geneal/report/dual_plot.py
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go


def dual_scatter_div(aux: dict, target: np.ndarray, top_k: int,
                     selected_by_method: dict[str, list[int]]) -> str:
    """2D scatter of lethality_A (x) vs lethality_B (y). The true selective
    top-k (highest target = lethality_A - lethality_B) is highlighted; each
    method's selected genes are overlaid as distinct markers."""
    la = np.asarray(aux["lethality_a"]); lb = np.asarray(aux["lethality_b"])
    target = np.asarray(target)
    k = min(top_k, len(target))
    true_top = set(np.argsort(target)[::-1][:k].tolist())
    is_top = np.array([i in true_top for i in range(len(target))])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=la[~is_top], y=lb[~is_top], mode="markers", name="genes",
        marker=dict(size=5, color="#cbd5e1"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=la[is_top], y=lb[is_top], mode="markers",
        name=f"true selective top-{k}",
        marker=dict(size=9, color="#ef4444", symbol="star",
                    line=dict(width=0.5, color="#7f1d1d"))))
    symbols = ["circle-open", "diamond-open", "square-open", "x", "cross"]
    for s, (mname, idx) in zip(symbols, selected_by_method.items()):
        idx = list(idx)
        fig.add_trace(go.Scatter(
            x=la[idx], y=lb[idx], mode="markers", name=f"selected: {mname}",
            marker=dict(size=12, symbol=s, line=dict(width=1.5))))
    # diagonal: equally lethal in both (no selectivity)
    lo = float(min(la.min(), lb.min())); hi = float(max(la.max(), lb.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(dash="dot", color="#9ca3af"),
                             name="no selectivity", hoverinfo="skip"))
    fig.update_layout(template="simple_white",
                      xaxis_title=f"lethality in {aux.get('line_a','A')} (efficacy)",
                      yaxis_title=f"lethality in {aux.get('line_b','B')} (toxicity)",
                      legend_title="", height=560,
                      margin=dict(l=60, r=20, t=10, b=50))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")
