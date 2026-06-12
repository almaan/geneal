# src/geneal/report/report.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATES = Path(__file__).parent / "templates"


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% normal-approx CI of the metric per (method, round)."""
    g = df.groupby(["method", "round"])["metric"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    se = out["std"].fillna(0.0) / np.sqrt(out["count"].clip(lower=1))
    out["ci_low"] = out["mean"] - 1.96 * se
    out["ci_high"] = out["mean"] + 1.96 * se
    return out[["method", "round", "mean", "ci_low", "ci_high"]]


def to_latex_table(agg: pd.DataFrame) -> str:
    """LaTeX tabular of mean metric per method per round (manuscript-ready)."""
    pivot = agg.pivot(index="round", columns="method", values="mean").round(3)
    return pivot.to_latex(index=True, caption="Mean metric by round",
                          label="tab:recall", escape=False)


def _plot_div(agg: pd.DataFrame, metric_name: str) -> str:
    fig = go.Figure()
    for method, sub in agg.groupby("method"):
        sub = sub.sort_values("round")
        fig.add_trace(go.Scatter(x=sub["round"], y=sub["mean"], mode="lines+markers",
                                 name=method))
        fig.add_trace(go.Scatter(
            x=list(sub["round"]) + list(sub["round"][::-1]),
            y=list(sub["ci_high"]) + list(sub["ci_low"][::-1]),
            fill="toself", line=dict(width=0), showlegend=False, opacity=0.2,
            hoverinfo="skip", name=f"{method} CI"))
    fig.update_layout(xaxis_title="round", yaxis_title=metric_name,
                      template="simple_white")
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def build_report(df: pd.DataFrame, out_path) -> Path:
    """Render the full HTML report from a runner DataFrame."""
    metric_name = df["metric_name"].iloc[0] if len(df) else "metric"
    agg = aggregate(df)
    env = Environment(loader=FileSystemLoader(str(_TEMPLATES)),
                      autoescape=select_autoescape(["html"]))
    template = env.get_template("report.html.j2")
    html = template.render(
        metric_name=metric_name,
        methods=sorted(df["method"].unique().tolist()),
        plot_div=_plot_div(agg, metric_name),
        html_table=agg.round(3).to_html(index=False),
        latex_table=to_latex_table(agg),
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
