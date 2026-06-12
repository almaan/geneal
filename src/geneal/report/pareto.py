# src/geneal/report/pareto.py
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def pareto_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per method: mean batch quality, mean batch diversity, final recall.

    Quality/diversity averaged over all acquired batches (round >= 1, NaN round-0
    rows dropped); final_recall is the mean metric at the last round across seeds.
    """
    acq = df[df["round"] >= 1]
    g = acq.groupby("method").agg(
        mean_quality=("batch_quality", "mean"),
        mean_diversity=("batch_diversity", "mean"),
    ).reset_index()
    last = df["round"].max()
    fin = (df[df["round"] == last].groupby("method")["metric"].mean()
           .reset_index().rename(columns={"metric": "final_recall"}))
    return g.merge(fin, on="method")


def pareto_plot_div(summary: pd.DataFrame) -> str:
    """Scatter of mean diversity (x) vs mean quality (y); marker size = final recall."""
    fig = go.Figure()
    sizes = summary["final_recall"].to_numpy()
    sizes = 10 + 30 * (sizes - sizes.min()) / (np.ptp(sizes) + 1e-9)
    for _, row in summary.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["mean_diversity"]], y=[row["mean_quality"]],
            mode="markers+text", text=[row["method"]], textposition="top center",
            marker=dict(size=float(10 + 30 * row["final_recall"])),
            name=f"{row['method']} (recall={row['final_recall']:.2f})"))
    fig.update_layout(xaxis_title="batch diversity (mean pairwise dist)",
                      yaxis_title="batch quality (mean true effect)",
                      template="simple_white",
                      title="Quality–diversity Pareto (marker size = final recall@k)")
    return fig.to_html(full_html=False, include_plotlyjs="cdn")
