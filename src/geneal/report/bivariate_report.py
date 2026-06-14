# src/geneal/report/bivariate_report.py
"""Report for bivariate efficacy-toxicity active learning (Plan 6).

Hypervolume-over-rounds per condition (learned vs known toxicity, contrast vs
population reference), with error bars over seeds; and a known-vs-learned final
gap table = the cost of having to LEARN toxicity vs knowing it a priori."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, BaseLoader

_GLOSS = """
<b>Setup.</b> Both efficacy (lethality in the target line) and toxicity are unknown
and explored jointly by an active-learning loop using <b>EHVI</b> (Expected
Hypervolume Improvement) — each round acquires the batch expected to most expand the
dominated region of the (efficacy↑, toxicity↓) objective space.<br>
<b>Toxicity reference</b> — <b>contrast</b>: lethality in a fixed proxy cell line (a
different-domain stand-in for normal tissue); <b>population</b>: mean lethality across
all cell lines.<br>
<b>Regime</b> — <b>learned</b>: toxicity is predicted by a 2nd surrogate and explored;
<b>known</b>: toxicity is an oracle (known a priori) — the ceiling. The learned-vs-known
gap = the cost of not knowing toxicity up front.<br>
<b>Hypervolume</b>: dominated area of the revealed set in objective space — higher =
better coverage of high-efficacy / low-toxicity targets.
"""


def _ci(x):
    x = np.asarray(x, float); n = len(x); m = float(x.mean())
    return m, (0.0 if n < 2 else 1.96 * float(x.std(ddof=1)) / np.sqrt(n))


def _curve(df):
    fig = go.Figure()
    conds = ["contrast_known", "contrast_learned", "population_known", "population_learned"]
    dash = {"known": "dot", "learned": "solid"}
    for cond in conds:
        sub = df[df.condition == cond]
        if not len(sub):
            continue
        rounds = sorted(sub["round"].unique())
        means, cis = [], []
        for r in rounds:
            mn, ci = _ci(sub[sub["round"] == r]["hypervolume"]); means.append(mn); cis.append(ci)
        regime = "known" if "known" in cond else "learned"
        fig.add_trace(go.Scatter(x=rounds, y=means, mode="lines+markers", name=cond,
                      line=dict(dash=dash[regime]),
                      error_y=dict(type="data", array=cis, visible=True, thickness=1, width=3)))
    fig.update_layout(template="simple_white", xaxis_title="active-learning round",
                      yaxis_title="dominated hypervolume (↑ better)",
                      height=460, margin=dict(l=60, r=20, t=10, b=50),
                      legend=dict(orientation="h", y=-0.18))
    fig.update_xaxes(dtick=1)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


_TEMPLATE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<title>{{ title }}</title><style>
 body{font-family:'Inter',system-ui,sans-serif;color:#1a1a2e;margin:0;padding:2.5rem 3rem;background:#fafafb;line-height:1.55}
 h1{font-weight:700;margin-bottom:.2rem} .sub{color:#6b7280;margin-bottom:1.2rem}
 h2{margin-top:2rem;font-weight:650;border-bottom:2px solid #3b4cca;display:inline-block;padding-bottom:2px}
 .card{background:#fff;border:1px solid #e5e7eb;border-radius:14px;padding:1.1rem 1.4rem;margin:1rem 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}
 .gloss{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:.8rem 1.1rem;font-size:.92rem}
 .key{background:#eef2ff;border-left:3px solid #3b4cca;padding:.6rem 1rem;border-radius:6px;margin:.5rem 0}
 table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
 th,td{border-bottom:1px solid #e5e7eb;padding:7px 12px;text-align:right} th:first-child,td:first-child{text-align:left;font-weight:600}
 thead th{color:#6b7280}
</style></head><body>
<h1>{{ title }}</h1>
<div class="sub">{{ n_lines }} target lines &middot; {{ n_seeds }} seeds &middot; toxicity proxy line = {{ contrast }}</div>
<div class="key">{{ headline }}</div>
<h2>How to read this</h2><div class="gloss">{{ gloss|safe }}</div>
<h2>Hypervolume over rounds (joint efficacy–toxicity discovery)</h2>
<div class="card">{{ curve|safe }}</div>
<h2>Cost of learning toxicity (final-round, mean ± 95% CI)</h2>
<div class="card"><table>
<thead><tr><th>reference</th><th>known (oracle, ceiling)</th><th>learned</th><th>gap</th></tr></thead>
<tbody>{% for r in rows %}<tr><td>{{ r.ref }}</td><td>{{ r.known }}</td><td>{{ r.learned }}</td><td>{{ r.gap }}</td></tr>{% endfor %}</tbody>
</table></div>
</body></html>"""


def build_bivariate_report(df, out_path, contrast="?",
                           title="geneal — bivariate efficacy-toxicity active learning") -> Path:
    last = df["round"].max()
    def fin(cond):
        return _ci(df[(df.condition == cond) & (df["round"] == last)]["hypervolume"])
    rows = []
    for ref in ["contrast", "population"]:
        k, kc = fin(f"{ref}_known"); l, lc = fin(f"{ref}_learned")
        rows.append({"ref": ref, "known": f"{k:.3f} ± {kc:.3f}",
                     "learned": f"{l:.3f} ± {lc:.3f}",
                     "gap": f"{k - l:+.3f} ({100*(k-l)/k:.0f}%)" if k else "—"})
    ck, _ = fin("contrast_known"); cl, _ = fin("contrast_learned")
    headline = (f"Learning toxicity (vs knowing it a priori) recovers "
                f"{100*cl/ck:.0f}% of the oracle hypervolume on the contrast reference "
                f"by round {last} — the cost of joint efficacy-toxicity exploration."
                if ck else "bivariate efficacy-toxicity active learning.")
    html = Environment(loader=BaseLoader()).from_string(_TEMPLATE).render(
        title=title, n_lines=df.cell_line.nunique(), n_seeds=df.seed.nunique(),
        contrast=contrast, headline=headline, gloss=_GLOSS, curve=_curve(df), rows=rows)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path


if __name__ == "__main__":
    import sys
    build_bivariate_report(pd.read_parquet(sys.argv[1]),
                           sys.argv[2] if len(sys.argv) > 2 else "bivariate_report.html")
