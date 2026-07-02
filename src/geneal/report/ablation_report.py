# src/geneal/report/ablation_report.py
"""Default two-analysis ablation report (geneal Plan 7).

Static, publication-style figures (matplotlib, no gridlines, no interactivity),
embedded as PNG in a self-contained HTML and optionally saved as vector PDF+PNG.
Faceted by non-target reference (contrast line = primary; aggregate = baseline).

  Section A -- safety vs efficacy: 8 methods on the efficacy-non-target efficacy tradeoff
    (aggregate + per-cell-line subfigures), each plot showing the dashed tau
    TOXICITY CEILING; plus per-method gene-landscape clouds (aggregate + per line)
    and a method table.
  Section B -- diversity / robustness: operators (none/cap/kdpp) on three bases
    (greedy/truncation/ehvi): metric table, concentration/robustness bar, and
    efficacy-vs-concentration.

The tau ceiling is a QUANTILE of the candidate non-target efficacy distribution (tau=0.5 =
the safest half); for a single contrast reference line it is
quantile({-effect in the contrast line}, tau)."""
from __future__ import annotations
import base64
import io
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Environment, BaseLoader

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.edgecolor": "#444a52", "axes.linewidth": 0.9,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 120, "savefig.dpi": 120, "svg.fonttype": "none",
})

# Resolution of the PNGs embedded in the HTML report. Higher = sharper but a
# larger HTML file (size grows ~ (dpi/120)^2). Override with GENEAL_REPORT_DPI.
_PNG_DPI = int(os.environ.get("GENEAL_REPORT_DPI", "200"))

COLORS = {
    "greedy": "#d1495b", "trunc_known": "#2e6f95", "trunc_pred": "#7eb6d9",
    "ehvi": "#ff7f0e", "ehvi_trunc": "#1b9e77", "random": "#9aa0a6",
    "farthest": "#8e6fb0", "cluster": "#e0a32e", "info_div": "#c2548a",
    "ehvi_pareto": "#0e7c66",
    "llm_nom": "#5b3a8e", "llm_loop": "#9a6dd7",
}
# Labels: acquisition spelled out (greedy / EHVI); filter codes nom=nomination-only,
# RT=per-round; K=known tox, P=predicted tox (spelled out in the glossary).
LABELS = {
    "greedy": "greedy", "trunc_known": "greedy · nom · known",
    "trunc_pred": "greedy · nom", "greedy_safe": "greedy · per-round",
    "known_safe": "greedy · per-round · known (upper bd)",
    "ehvi": "EHVI", "ehvi_pareto": "EHVI-EHVI",
    "ehvi_trunc": "EHVI · nom", "ehvi_safe": "EHVI · per-round",
    "random": "random", "farthest": "farthest", "cluster": "cluster",
    "info_div": "info-diverse",
    "llm_nom": "LLM (nom)", "llm_loop": "LLM (loop)",
}
# ordered: baselines, then all greedy variants, then all EHVI variants.
# 'known'/oracle methods (trunc_known, known_safe) are intentionally excluded
# from the report (still computed + persisted, just not shown).
A_ORDER = ["random", "farthest", "cluster", "info_div", "llm_nom", "llm_loop",
           "greedy", "trunc_pred", "greedy_safe",
           "ehvi", "ehvi_pareto", "ehvi_trunc", "ehvi_safe"]
COLORS.update({"greedy_safe": "#b23a55", "ehvi_safe": "#11806080", "known_safe": "#0b3d91"})
# acquisitions (for the assayed-set panel + per-round curves)
ACQ_ORDER = ["random", "greedy", "farthest", "cluster", "info_div", "ehvi",
             "greedy_safe", "ehvi_safe"]
ACQ_LABELS = {"random": "random", "greedy": "greedy/UCB", "farthest": "farthest",
              "cluster": "cluster", "info_div": "info-diverse", "ehvi": "EHVI",
              "greedy_safe": "greedy+safe", "ehvi_safe": "EHVI+safe",
              "known_safe": "known-safe (upper bd)"}
OP_ORDER = ["none", "kdpp_string", "kdpp_corum"]   # embedding-S k-DPP & 'cap' excluded from B
OP_LABELS = {"none": "none", "cap": "cap (CORUM)", "kdpp_emb": "embedding",
             "kdpp_string": "STRING", "kdpp_corum": "CORUM"}
OP_SHORT = {"none": "none", "cap": "cap", "kdpp_emb": "kdpp·emb",
            "kdpp_string": "kdpp·STRING", "kdpp_corum": "kdpp·CORUM"}
OP_MARK = {"none": "o", "cap": "s", "kdpp_emb": "D", "kdpp_string": "v", "kdpp_corum": "^"}
OP_COLOR = {"none": "#9aa0a6", "cap": "#2e6f95", "kdpp_emb": "#1b9e77",
            "kdpp_string": "#a855f7", "kdpp_corum": "#e0a32e"}
# B bases = unfiltered greedy/EHVI + the 4 safety-aware methods (keys -> A labels).
# greedy variants first, then EHVI variants
B_BASE_ORDER = ["greedy", "trunc_pred", "greedy_safe", "ehvi", "ehvi_trunc", "ehvi_safe"]
B_BASE_COLOR = {"greedy": "#d1495b", "ehvi": "#ff7f0e", "trunc_pred": "#2e6f95",
                "greedy_safe": "#7eb6d9", "ehvi_trunc": "#1b9e77", "ehvi_safe": "#13634a"}
B_BASE_SHORT = {"greedy": "greedy", "ehvi": "EHVI", "trunc_pred": "greedy·nom",
                "greedy_safe": "greedy·RT", "ehvi_trunc": "EHVI·nom",
                "ehvi_safe": "EHVI·RT"}

_A_MAIN_METRICS = [("mean_efficacy", "Target Efficacy ↑"), ("mean_toxicity", "Non-Target Efficacy ↓"),
                   ("sel_ratio", "T/NT ratio ↑"),
                   ("useful_efficacy", "Realizable target efficacy↑"), ("n_safe", "# safe (of K)↑")]
_A_DIAG_METRICS = [("max_efficacy", "Max efficacy↑"), ("n_novel", "# novel (of K)↑"),
                   ("hypervolume", "Hypervolume (target, −non-target) ↑"),
                   ("pareto_recall_norm", "Pareto recall (norm)↑")]
_B_METRICS = [("concentration", "Concentration↓"), ("robustness", "Robustness↑"),
              ("n_pathways", "Distinct pathways↑"), ("mean_efficacy", "Target Efficacy ↑"),
              ("mean_toxicity", "Non-Target Efficacy ↓")]
# Merged Section-B tables: identical columns, evaluated on CORUM vs on STRING.
# Both tables use the k-DPP·CORUM hedge (vs the none baseline), q=efficacy.
_B_EVAL_CORUM = [("concentration", "Concentration↓"), ("robustness", "Robustness↑"),
                 ("n_pathways", "Distinct pathways↑"), ("risk_corum", "Risk↓"),
                 ("neff_corum", "N_eff↑"), ("mean_efficacy", "Target Efficacy ↑"),
                 ("mean_toxicity", "Non-Target Efficacy ↓")]
_B_EVAL_STRING = [("concentration_string", "Concentration↓"), ("robustness_string", "Robustness↑"),
                  ("n_pathways_string", "Distinct pathways↑"), ("risk_string", "Risk↓"),
                  ("neff_string", "N_eff↑"), ("mean_efficacy", "Target Efficacy ↑"),
                  ("mean_toxicity", "Non-Target Efficacy ↓")]
_B_HEDGE_OPS = ["none", "kdpp_corum"]   # baseline vs the CORUM hedge
# compact column headers for the LaTeX export (the HTML keeps the full names)
_B_SHORT_COLS = ["Conc. ↓", "Robust. ↑", "Avg. Units ↑", "Risk ↓", "Neff ↑",
                 "T. Eff. ↑", "NT. Eff. ↓"]

# Self-contained captions for the Section B (diversity/hedging) tables: spell out
# every abbreviation so the table reads standalone. {src}=non-target reference,
# {spread}=dispersion shown.
_B_CAP_COMMON = (
    "Each row is a base nomination rule combined with a diversity operator. "
    "Bases: \\emph{{greedy}} (top-$K$ predicted efficacy, no safety), "
    "\\emph{{greedy$\\cdot$nom}} / \\emph{{EHVI$\\cdot$nom}} (greedy or EHVI acquisition with the "
    "predicted-non-target efficacy filter applied to the final shortlist only), and "
    "\\emph{{greedy$\\cdot$RT}} / \\emph{{EHVI$\\cdot$RT}} (filter applied every round). "
    "Operators: \\emph{{none}} = top-$K$ by quality; "
    "\\emph{{k-DPP}} = a quality-weighted $k$-determinantal point process that trades a "
    "little efficacy for mechanistic spread, using the {sim} similarity. "
    "Cells are mean $\\pm$ {spread} over cell lines $\\times$ seeds; $K{{=}}30$ nominees, "
    "{src} non-target reference.")
_B_CAP_CORUM = (
    "Diversity / hedging over CORUM protein complexes ({src} non-target efficacy). "
    + _B_CAP_COMMON.replace("{sim}", "CORUM complex co-membership (Jaccard)")
    + " \\emph{{Concentration}} $\\downarrow$ = fraction of the portfolio in its most "
    "common complex; \\emph{{robustness}} $\\uparrow$ = value retained if a random "
    "complex fails; \\emph{{distinct complexes}} $\\uparrow$; with target efficacy and "
    "non-target efficacy of the nominated set.")
_B_CAP_STRING = (
    "Diversity / hedging over the STRING functional network ({src} non-target efficacy). "
    + _B_CAP_COMMON.replace("{sim}", "STRING combined-score network")
    + " \\emph{{Mean / max pairwise STRING similarity}} $\\downarrow$ = how "
    "functionally redundant the nominated set is (lower = more spread); with mean "
    "efficacy and non-target efficacy of the nominated set.")


def _stat(x, kind="ci"):
    """(mean, spread). kind='ci' -> 95% CI half-width (1.96*sem); 'sem' -> standard
    error of the mean (std/sqrt(n)); 'std' -> sample std."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    m = float(x.mean()) if n else float("nan")
    if n < 2:
        return m, 0.0
    sd = float(x.std(ddof=1)); sem = sd / np.sqrt(n)
    return m, {"std": sd, "sem": sem}.get(kind, 1.96 * sem)


def _ci(x):
    return _stat(x, "ci")


def _cell(mn, ci):
    """Table cell; '—' when a metric is undefined for every fold (e.g. permissible
    efficacy when a method nominates zero safe genes)."""
    return "—" if np.isnan(mn) else f"{mn:.3f} ± {ci:.3f}"


_TEX = [("±", r"$\pm$"), ("↑", r"$\uparrow$"), ("↓", r"$\downarrow$"),
        ("α", r"$\alpha$"), ("·", r"$\cdot$"), ("τ", r"$\tau$"),
        ("Δ", r"$\Delta$"), ("→", r"$\rightarrow$"),
        ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]


def _tex(s):
    s = str(s)
    for a, b in _TEX:
        s = s.replace(a, b)
    return s


_REF_ROWCOLOR = "EAF2FB"   # reference-row (unmodified greedy/EHVI) background; needs [table]{xcolor}


def _latex_table(row_header, cols, rows, caption, label):
    """booktabs LaTeX tabular with the same data as an HTML table. Rows may carry
    'biggroup' (double rule on change), 'group' (single rule on change) and
    'highlight' (\\rowcolor reference row). Caption is placed at the BOTTOM."""
    out = [r"\begin{table}[t]", r"\centering",
           r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
           _tex(row_header) + " & " + " & ".join(_tex(c) for c in cols) + r" \\",
           r"\midrule"]
    prev_group = prev_big = None
    for r in rows:
        group, big = r.get("group"), r.get("biggroup")
        if prev_big is not None and big is not None and big != prev_big:
            out.append(r"\midrule\midrule")        # double rule between sections (baseline/greedy/EHVI)
        elif prev_group is not None and group is not None and group != prev_group:
            out.append(r"\midrule")                # single rule between sub-groups (e.g. B bases)
        prev_group, prev_big = group, big
        name = r.get("m") or r.get("label") or ""
        line = _tex(name) + " & " + " & ".join(_tex(c) for c in r["cells"]) + r" \\"
        if r.get("highlight"):
            line = f"\\rowcolor[HTML]{{{_REF_ROWCOLOR}}} " + line
        out.append(line)
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{" + caption + "}", r"\label{tab:" + label + "}", r"\end{table}"]
    return "\n".join(out)


def _despine(ax):
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _legend_out(ax, fontsize=8, **kw):
    """Place the legend OUTSIDE the axes (upper-left of the right margin) so it
    never overlaps data. _emit saves with bbox_inches='tight' so it's captured."""
    ax.legend(frameon=False, fontsize=fontsize, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, **kw)


def _ceiling(scatter, src, tau, tau_mode="absolute", cell_line=None):
    """The non-target efficacy ceiling for the dashed line. 'absolute': the bar IS tau (a
    Chronos-scale value). 'quantile': tau-quantile of the candidate true-non-target efficacy
    distribution. Returns None if no scatter data."""
    if tau_mode == "absolute":
        return float(tau)
    if scatter is None or scatter.empty:
        return None
    d = scatter[scatter.tox_source == src] if "tox_source" in scatter.columns else scatter
    if cell_line is not None:
        d = d[d.cell_line == cell_line]
    if d.empty:
        return None
    return float(np.quantile(d["toxicity"].to_numpy(), tau))


def _emit(fig, fig_dir, name):
    """Embed the figure as a base64 PNG <img>; also save PDF+PNG when fig_dir set."""
    if fig_dir is not None:
        fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")               # vector
        fig.savefig(fig_dir / f"{name}.png", bbox_inches="tight", dpi=_PNG_DPI)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=_PNG_DPI)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto"/>'


# ---- Section A ---------------------------------------------------------------
def _draw_tradeoff(ax, d, methods, ceiling, tau, legend=True):
    for m in methods:
        g = d[d.method == m]
        if not len(g):
            continue
        ex, exc = _ci(g["mean_toxicity"]); ey, eyc = _ci(g["mean_efficacy"])
        ax.errorbar(ex, ey, xerr=exc, yerr=eyc, fmt="o", ms=9, color=COLORS.get(m, "#444"),
                    ecolor=COLORS.get(m, "#444"), elinewidth=1.1, capsize=2.5,
                    mec="white", mew=1.0, label=LABELS.get(m, m), zorder=3)
    if ceiling is not None:
        x0 = ax.get_xlim()[0]
        ax.axvspan(x0, ceiling, color="#1b9e77", alpha=0.07, zorder=0)  # permissible region
        ax.axvline(ceiling, ls="--", lw=1.1, color="#6b7280", zorder=1)
        ax.text(ceiling, ax.get_ylim()[1], f" τ ceiling ({tau:g})", color="#6b7280",
                fontsize=8, va="top", ha="left")
    _despine(ax)
    ax.set_xlabel("non-target efficacy ↓")
    ax.set_ylabel("target target efficacy ↑")
    if legend:
        _legend_out(ax, fontsize=8, ncol=1)


def _tradeoff_points(dA, src, scatter, tau, tau_mode, fig_dir):
    methods = [m for m in A_ORDER if m in set(dA.method)]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    _draw_tradeoff(ax, dA, methods, _ceiling(scatter, src, tau, tau_mode), tau)
    fig.tight_layout()
    return _emit(fig, fig_dir, f"tradeoff_{src}")


def _consistency_strip(dA, src, fig_dir, a="greedy", b="greedy_safe"):
    """Per-cell-line non-target efficacy reduction from the per-round predicted-tox filter,
    tox(a) - tox(b). All-positive bars = the filter is safer than greedy in every
    line (the paired, common-random-numbers robustness claim)."""
    if not {a, b}.issubset(set(dA.method)):
        return None
    ga = dA[dA.method == a].groupby("cell_line")["mean_toxicity"].mean()
    gb = dA[dA.method == b].groupby("cell_line")["mean_toxicity"].mean()
    j = pd.concat([ga, gb], axis=1, keys=["a", "b"]).dropna()
    if j.empty:
        return None
    delta = (j["a"] - j["b"]).sort_values()
    fig, ax = plt.subplots(figsize=(6.2, max(2.6, 0.30 * len(delta))))
    ax.barh(range(len(delta)), delta.values,
            color=["#1b9e77" if x > 0 else "#d1495b" for x in delta.values])
    ax.axvline(0, color="#444a52", lw=0.8)
    ax.set_yticks(range(len(delta)))
    ax.set_yticklabels(list(delta.index), fontsize=6)
    ax.set_xlabel("non-target efficacy reduction:  greedy − greedy·per-round   (→ filter safer)")
    n_pos = int((delta > 0).sum())
    ax.set_title(f"predicted-tox filter safer than greedy in {n_pos}/{len(delta)} lines",
                 fontsize=10)
    _despine(ax)
    fig.tight_layout()
    return _emit(fig, fig_dir, f"consistency_{src}")


def _tradeoff_per_line(dA, src, lines, scatter, tau, tau_mode, fig_dir):
    lines = [l for l in (lines or []) if l in set(dA.cell_line)] or sorted(dA.cell_line.unique())
    if not lines:
        return None
    methods = [m for m in A_ORDER if m in set(dA.method)]
    ncol = min(3, len(lines)); nrow = math.ceil(len(lines) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.6 * nrow), squeeze=False)
    for k, cl in enumerate(lines):
        ax = axes[k // ncol][k % ncol]
        _draw_tradeoff(ax, dA[dA.cell_line == cl], methods,
                       _ceiling(scatter, src, tau, tau_mode, cell_line=cl), tau, legend=False)
        ax.set_title(cl)
    for k in range(len(lines), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    # one shared legend
    handles = [plt.Line2D([0], [0], marker="o", ls="", ms=8, color=COLORS.get(m), label=LABELS.get(m, m))
               for m in methods]
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center",
               ncol=min(4, len(methods)), bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    return _emit(fig, fig_dir, f"tradeoff_perline_{src}")


def _cloud_fig(d, methods, ceiling, fig_dir, name):
    """One panel per method: faint all-gene background + a KDE density of that
    method's pooled picks (method-colored filled contours), so it's clear WHERE
    most nominations land rather than over-plotting hundreds of markers. Falls
    back to open markers when too few picks to estimate a density."""
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap
    tox, eff = d["toxicity"].to_numpy(), d["efficacy"].to_numpy()
    xmin, xmax = float(np.nanmin(tox)), float(np.nanmax(tox))
    ymin, ymax = float(np.nanmin(eff)), float(np.nanmax(eff))
    xx, yy = np.mgrid[xmin:xmax:120j, ymin:ymax:120j]
    grid = np.vstack([xx.ravel(), yy.ravel()])
    ncol = min(4, len(methods)); nrow = math.ceil(len(methods) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.9 * nrow), squeeze=False)
    for k, m in enumerate(methods):
        ax = axes[k // ncol][k % ncol]
        ax.scatter(tox, eff, s=2, c="#e8ebef", linewidths=0, rasterized=True, zorder=1)
        col = COLORS.get(m, "#444")
        pk = d[f"pick_{m}"].to_numpy(dtype=bool)
        px, py = tox[pk], eff[pk]
        if len(px) >= 10 and np.ptp(px) > 1e-9 and np.ptp(py) > 1e-9:
            zz = gaussian_kde(np.vstack([px, py]))(grid).reshape(xx.shape)
            cmap = LinearSegmentedColormap.from_list(f"c_{m}", ["white", col])
            levels = np.linspace(zz.max() * 0.10, zz.max(), 7)   # skip near-zero band so bg shows
            ax.contourf(xx, yy, zz, levels=levels, cmap=cmap, alpha=0.72, zorder=2)
            ax.contour(xx, yy, zz, levels=levels, colors=col, linewidths=0.35, alpha=0.5, zorder=3)
        else:
            ax.scatter(px, py, s=22, facecolors="none", edgecolors=col, linewidths=1.2, zorder=3)
        if ceiling is not None:
            ax.axvline(ceiling, ls="--", lw=1.0, color="#6b7280", zorder=4)
        _despine(ax)
        ax.set_title(LABELS.get(m, m), fontsize=9)
        if k // ncol == nrow - 1:
            ax.set_xlabel("non-target efficacy →", fontsize=8)
        if k % ncol == 0:
            ax.set_ylabel("target efficacy ↑", fontsize=8)
    for k in range(len(methods), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.tight_layout()
    return _emit(fig, fig_dir, name)


def _gene_cloud_section(scatter, src, lines, tau, tau_mode, fig_dir):
    if scatter is None or scatter.empty:
        return None
    d = scatter[scatter.tox_source == src] if "tox_source" in scatter.columns else scatter
    if d.empty:
        return None
    methods = [m for m in A_ORDER if f"pick_{m}" in d.columns]
    if not methods:
        return None
    agg = _cloud_fig(d, methods, _ceiling(scatter, src, tau, tau_mode), fig_dir, f"gene_cloud_{src}")
    cl_order = [c for c in (lines or []) if c in set(d.cell_line)] or sorted(d.cell_line.unique())
    per = [(cl, _cloud_fig(d[d.cell_line == cl], methods,
                           _ceiling(scatter, src, tau, tau_mode, cell_line=cl),
                           fig_dir, f"gene_cloud_{src}_{cl}"))
           for cl in cl_order]
    return {"agg": agg, "per": per}


_A_FAMILY = {**{m: "baseline" for m in ("random", "farthest", "cluster", "info_div",
                                         "llm_nom", "llm_loop")},
             **{m: "greedy" for m in ("greedy", "trunc_pred", "greedy_safe",
                                      "trunc_known", "known_safe")},
             **{m: "ehvi" for m in ("ehvi", "ehvi_pareto", "ehvi_trunc", "ehvi_safe")}}


def _table_A(d, metric_set, kind="ci"):
    methods = [m for m in A_ORDER if m in set(d.method)]
    metrics = [(c, lbl) for c, lbl in metric_set if c in d.columns]
    rows = []
    for m in methods:
        cells = [_cell(*_stat(d[d.method == m][c], kind)) for c, _ in metrics]
        rows.append({"m": LABELS.get(m, m), "cells": cells,
                     "biggroup": _A_FAMILY.get(m),                # double rule baseline|greedy|EHVI
                     "highlight": m in ("greedy", "ehvi")})        # unmodified reference methods
    return [lbl for _, lbl in metrics], rows


def _safety_bar(dA, src, fig_dir):
    """Per method: # of the K nominees that are SAFE (true tox <= ceiling) vs
    TOXIC (above), stacked. Mean over lines x seeds."""
    if "n_safe" not in dA.columns:
        return None
    methods = [m for m in A_ORDER if m in set(dA.method)]
    safe = [_ci(dA[dA.method == m]["n_safe"])[0] for m in methods]
    toxic = [_ci(dA[dA.method == m]["n_toxic"])[0] for m in methods]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(max(6, 0.95 * len(methods)), 4.2))
    ax.bar(x, safe, 0.62, color="#1b9e77", label="permissible (non-target eff ≤ τ)")
    ax.bar(x, toxic, 0.62, bottom=safe, color="#d1495b", label="over threshold (non-target eff > τ)")
    ax.set_xticks(x); ax.set_xticklabels([LABELS.get(m, m) for m in methods],
                                          rotation=55, ha="right", fontsize=7)
    _despine(ax); ax.set_ylabel("nominees (count of K)")
    _legend_out(ax, fontsize=9)
    fig.tight_layout()
    return _emit(fig, fig_dir, f"safety_count_{src}")


def _final_k_bars(dA, src, fig_dir):
    """Per-method bars of the FINAL nominated K-set metrics (mean over lines x
    seeds, 95% CI): realizable target efficacy, non-target efficacy, # safe, concentration."""
    methods = [m for m in A_ORDER if m in set(dA.method)]
    specs = [("useful_efficacy", "realizable target target efficacy ↑"),
             ("mean_toxicity", "non-target efficacy ↓"),
             ("n_safe", "# safe (of K) ↑"),
             ("n_novel", "# novel/unassayed (of K)"),
             ("concentration", "concentration ↓")]
    specs = [(c, l) for c, l in specs if c in dA.columns]
    fig, axes = plt.subplots(1, len(specs), figsize=(4.0 * len(specs), 4.2))
    if len(specs) == 1:
        axes = [axes]
    cols = [COLORS.get(m, "#888") for m in methods]
    for ax, (col, lab) in zip(axes, specs):
        means = [_ci(dA[dA.method == m][col])[0] for m in methods]
        errs = [_ci(dA[dA.method == m][col])[1] for m in methods]
        ax.bar(range(len(methods)), means, yerr=errs, color=cols, capsize=2,
               error_kw=dict(lw=1, ecolor="#444"))
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([LABELS.get(m, m) for m in methods], rotation=55, ha="right", fontsize=7)
        ax.set_title(lab, fontsize=10); _despine(ax)
    fig.tight_layout()
    return _emit(fig, fig_dir, f"final_k_{src}")


def _admission_curve(scatter, src, tau, tau_mode, fig_dir):
    """For the NON-filtered methods: fraction of their K nominees that would pass
    at each hypothetical non-target threshold τ (the empirical CDF of nominee
    non-target efficacy, pooled over lines). Shows how badly unconstrained methods fare
    against a range of safety bars. Dashed line = the τ used in the main results."""
    if scatter is None or scatter.empty:
        return None
    d = scatter[scatter.tox_source == src] if "tox_source" in scatter.columns else scatter
    if d.empty:
        return None
    nofilter = ["greedy", "ehvi", "ehvi_pareto", "random", "farthest", "cluster", "info_div"]
    methods = [m for m in nofilter if f"pick_{m}" in d.columns]
    if not methods:
        return None
    grid = np.linspace(0.0, 1.0, 80)            # non-target threshold τ over [0,1]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for m in methods:
        col = f"pick_{m}"
        # per-cell-line admission curve, averaged across lines (no envelope)
        curves = []
        for _, gg in d.groupby("cell_line"):
            t = gg.loc[gg[col].astype(bool), "toxicity"].to_numpy()
            if len(t):
                curves.append([np.mean(t <= g) for g in grid])
        if not curves:
            continue
        mu = np.asarray(curves).mean(0)
        auc = float(np.trapezoid(mu, grid)) if hasattr(np, "trapezoid") else float(np.trapz(mu, grid))
        ax.plot(grid, mu, "-", lw=1.7, color=COLORS.get(m, "#888"),
                label=f"{LABELS.get(m, m)} (AUC {auc:.2f})")
    _despine(ax)
    ax.set_xlabel("non-target threshold τ  (non-target efficacy ceiling)")
    ax.set_ylabel("fraction of K nominees admitted (non-target eff ≤ τ)")
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.02); _legend_out(ax, fontsize=8)
    return _emit(fig, fig_dir, f"admission_{src}")


def _admission_efficacy_curve(scatter, src, tau, tau_mode, fig_dir):
    """Companion to the admission curve: y = REALIZABLE (permissible) efficacy vs
    threshold τ. For each non-filtered method, at each τ the realizable target efficacy is
    the mean over its K picks of efficacy with over-threshold (tox > τ) picks scored
    0 -- i.e. the efficacy you can actually act on once the threshold is imposed.
    Per-cell-line then averaged across lines (no envelope)."""
    if scatter is None or scatter.empty:
        return None
    d = scatter[scatter.tox_source == src]
    if d.empty:
        return None
    nofilter = ["greedy", "ehvi", "ehvi_pareto", "random", "farthest", "cluster", "info_div"]
    methods = [m for m in nofilter if f"pick_{m}" in d.columns]
    if not methods:
        return None
    grid = np.linspace(0.0, 1.0, 80)            # non-target threshold τ over [0,1]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for m in methods:
        col = f"pick_{m}"
        curves = []
        for _, gg in d.groupby("cell_line"):
            sub = gg.loc[gg[col].astype(bool), ["efficacy", "toxicity"]]
            if not len(sub):
                continue
            e = sub["efficacy"].to_numpy(); t = sub["toxicity"].to_numpy()
            curves.append([float(np.mean(np.where(t <= g, e, 0.0))) for g in grid])
        if not curves:
            continue
        mu = np.asarray(curves).mean(0)
        auc = float(np.trapezoid(mu, grid)) if hasattr(np, "trapezoid") else float(np.trapz(mu, grid))
        ax.plot(grid, mu, "-", lw=1.7, color=COLORS.get(m, "#888"),
                label=f"{LABELS.get(m, m)} (AUC {auc:.2f})")
    _despine(ax)
    ax.set_xlabel("non-target threshold τ  (non-target efficacy ceiling)")
    ax.set_ylabel("realizable target efficacy of K nominees (non-target eff > τ scored 0)")
    ax.set_xlim(0, 1); _legend_out(ax, fontsize=8)
    return _emit(fig, fig_dir, f"admission_eff_{src}")


def _assayed_panel(assayed, src, fig_dir):
    """Per-ACQUISITION quality of the assayed set (the ~120 genes actually
    measured): #lethal-&-safe of the assayed, non-target efficacy. Shows whether
    constrained / EHVI acquisition collects safer data than greedy."""
    if assayed is None or assayed.empty:
        return None, ([], [])
    d = assayed[assayed.tox_source == src]
    accs = [a for a in ACQ_ORDER if a in set(d.acq)]
    nsafe = [_ci(d[d.acq == a]["n_safe"])[0] for a in accs]
    mtox = [_ci(d[d.acq == a]["mean_toxicity"])[0] for a in accs]
    meff = [_ci(d[d.acq == a]["mean_efficacy"])[0] for a in accs]
    nass = [_ci(d[d.acq == a]["n_assayed"])[0] for a in accs]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.bar(range(len(accs)), nsafe, color="#1b9e77")
    a1.set_xticks(range(len(accs))); a1.set_xticklabels([ACQ_LABELS.get(a, a) for a in accs],
                                                        rotation=55, ha="right", fontsize=7)
    a1.set_ylabel(f"# assayed that are safe (of ~{int(np.nanmean(nass)) if nass else 0})")
    a1.set_title("safe genes collected", fontsize=10); _despine(a1)
    a2.bar(range(len(accs)), mtox, color="#d1495b")
    a2.set_xticks(range(len(accs))); a2.set_xticklabels([ACQ_LABELS.get(a, a) for a in accs],
                                                        rotation=55, ha="right", fontsize=7)
    a2.set_ylabel("non-target efficacy of assayed set"); a2.set_title("non-target efficacy of collected data", fontsize=10)
    _despine(a2)
    fig.tight_layout()
    img = _emit(fig, fig_dir, f"assayed_{src}")
    cols = ["target efficacy", "non-target efficacy", "# safe", "# assayed"]
    rows = [{"m": ACQ_LABELS.get(a, a),
             "cells": [f"{meff[i]:.3f}", f"{mtox[i]:.3f}", f"{nsafe[i]:.1f}", f"{nass[i]:.0f}"]}
            for i, a in enumerate(accs)]
    return img, (cols, rows)


def _round_curves(rounds, src, fig_dir):
    """Per-round learning curves. Two figures (nomination quality, assayed-set
    quality), each 1x3 panels; one line per method, mean over lines x seeds."""
    if rounds is None or rounds.empty:
        return None, None
    d = rounds[rounds.tox_source == src]
    methods = [m for m in A_ORDER if m in set(d.method)]

    def panel(specs, name):
        fig, axes = plt.subplots(1, len(specs), figsize=(5.0 * len(specs), 4.0))
        if len(specs) == 1:
            axes = [axes]
        for ax, (col, ylab) in zip(axes, specs):
            for m in methods:
                g = d[d.method == m]
                agg = g.groupby("round")[col].mean()
                ax.plot(agg.index, agg.values, "-o", ms=3, lw=1.4,
                        color=COLORS.get(m, "#888"), label=LABELS.get(m, m))
            _despine(ax); ax.set_xlabel("AL round"); ax.set_ylabel(ylab)
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, frameon=False, fontsize=7, loc="lower center",
                   ncol=min(5, len(lab)), bbox_to_anchor=(0.5, -0.06))
        fig.tight_layout()
        return _emit(fig, fig_dir, name)

    nom = panel([("nom_mean_efficacy_safe", "permissible target target efficacy ↑"),
                 ("nom_n_safe", "# safe nominees ↑"),
                 ("nom_mean_toxicity", "nominee non-target efficacy ↓")], f"rounds_nom_{src}")
    asy = panel([("assayed_mean_efficacy", "assayed target efficacy"),
                 ("assayed_mean_toxicity", "assayed non-target efficacy ↓"),
                 ("assayed_n_safe", "# safe assayed ↑")], f"rounds_assayed_{src}")
    return nom, asy


def _realizable_round_table(rounds, src, K=30.0):
    """Section A: realizable target efficacy of the nominated set per AL round.
    Realizable = permissible efficacy x yield = nom_mean_efficacy_safe * n_safe / K
    (toxic picks scored 0) -- the un-gameable combination of potency and how many
    of the K picks are actually permissible. One row per method, one column per
    round shown, plus the r0->last gain. Cells mean +/- 95% CI over lines x seeds."""
    if rounds is None or rounds.empty or "nom_mean_efficacy_safe" not in rounds.columns:
        return None, None
    d = rounds[rounds.tox_source == src].copy()
    d["realizable"] = d["nom_mean_efficacy_safe"].fillna(0.0) * d["nom_n_safe"] / K
    methods = [m for m in A_ORDER if m in set(d.method)]
    all_rounds = sorted(d["round"].unique())
    show = [r for r in all_rounds if r % 2 == 0] or all_rounds   # every other round
    r0, rL = all_rounds[0], all_rounds[-1]
    cols = [f"r{int(r)}" for r in show] + [f"Δ r{int(r0)}→r{int(rL)}"]
    rows = []
    for m in methods:
        g = d[d.method == m]
        cells = []
        for r in show:
            cells.append(_cell(*_stat(g[g["round"] == r]["realizable"].to_numpy(float), "ci")))
        # paired gain per (line, seed)
        piv = g.pivot_table(index=["cell_line", "seed"], columns="round", values="realizable")
        gain = (piv[rL] - piv[r0]).to_numpy(float) if {r0, rL} <= set(piv.columns) else np.array([np.nan])
        cells.append(_cell(*_stat(gain, "ci")))
        rows.append({"m": LABELS.get(m, m), "cells": cells,
                     "highlight": m in ("greedy", "ehvi")})
    return cols, rows


def _ratio_round_table(rounds, src):
    """Section A: selectivity ratio (target / non-target efficacy) of the nominated
    set per AL round, reported as the PAIRED change from round 0 (per line x seed,
    then averaged) so every method starts at 0 and the columns show how much each
    acquisition improves selectivity over the loop. ratio = nom_mean_efficacy /
    max(nom_mean_toxicity, 0.05) (same floor as the main table). One row per method,
    one column per round shown; cells mean +/- 95% CI."""
    if rounds is None or rounds.empty or "nom_mean_toxicity" not in rounds.columns:
        return None, None
    d = rounds[rounds.tox_source == src].copy()
    d["ratio"] = d["nom_mean_efficacy"] / d["nom_mean_toxicity"].clip(lower=0.05)
    methods = [m for m in A_ORDER if m in set(d.method)]
    all_rounds = sorted(d["round"].unique())
    show = [r for r in all_rounds if r % 2 == 0] or all_rounds
    r0 = all_rounds[0]
    cols = [f"Δ r{int(r)}" for r in show]
    rows = []
    for m in methods:
        g = d[d.method == m]
        piv = g.pivot_table(index=["cell_line", "seed"], columns="round", values="ratio")
        cells = []
        for r in show:
            delta = (piv[r] - piv[r0]).to_numpy(float) if {r, r0} <= set(piv.columns) else np.array([np.nan])
            cells.append(_cell(*_stat(delta, "ci")))
        rows.append({"m": LABELS.get(m, m), "cells": cells,
                     "highlight": m in ("greedy", "ehvi")})
    return cols, rows


# ---- Section B ---------------------------------------------------------------
def _auc_hv(g):
    """Normalized area under a per-(line,seed) hypervolume-vs-round curve:
    mean over (line, seed) of trapz(hv, round) / round-span. Returns (mean, ci)."""
    vals = []
    for _, sub in g.groupby(["cell_line", "seed"]):
        sub = sub.sort_values("round")
        rr = sub["round"].to_numpy(float); hv = sub["hv"].to_numpy(float)
        span = (rr.max() - rr.min()) or 1.0
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))   # numpy>=2 renamed
        vals.append(float(_trapz(hv, rr) / span) if len(rr) > 1 else float(hv[0]))
    return _stat(np.array(vals), "ci")


def _auc_hv_table(rounds, src):
    """Section A separate table: AUC of the hypervolume-over-rounds curve, in BOTH
    senses - revealed/assayed front (exploration) and the nominated set (decision
    quality). One row per method; cells mean +/- 95% CI over lines x seeds."""
    if rounds is None or rounds.empty or "assayed_hypervolume" not in rounds.columns:
        return None, None
    d = rounds[rounds.tox_source == src]
    methods = [m for m in A_ORDER if m in set(d.method)]
    cols = ["revealed-front AUC-HV", "nominated AUC-HV"]
    rows = []
    for m in methods:
        g = d[d.method == m]
        a = g.rename(columns={"assayed_hypervolume": "hv"})
        n = g.rename(columns={"nom_hypervolume": "hv"})
        rows.append({"m": LABELS.get(m, m),
                     "cells": [_cell(*_auc_hv(a)), _cell(*_auc_hv(n))],
                     "highlight": m in ("greedy", "ehvi")})
    return cols, rows


def _hv_round_curve(rounds, src, fig_dir):
    """HV-over-rounds curve (revealed/assayed front), one line per method."""
    if rounds is None or rounds.empty or "assayed_hypervolume" not in rounds.columns:
        return None
    d = rounds[rounds.tox_source == src]
    methods = [m for m in A_ORDER if m in set(d.method)]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m in methods:
        g = d[d.method == m].groupby("round")["assayed_hypervolume"].mean()
        ax.plot(g.index, g.values, "-o", ms=3, lw=1.4,
                color=COLORS.get(m, "#888"), label=LABELS.get(m, m))
    _despine(ax); ax.set_xlabel("AL round")
    ax.set_ylabel("revealed-front hypervolume (eff, -tox) ↑")
    _legend_out(ax, fontsize=8)
    fig.tight_layout()
    return _emit(fig, fig_dir, f"hv_rounds_{src}")


_RISK_QUALITIES = ("eff", "sel")
_RISK_QLABEL = {"eff": "q=efficacy", "sel": "q=selectivity"}


def _risk_table(dB, src, kind="ci"):
    """Section B portfolio-risk table: R (equal-weight portfolio variance, lower =
    better hedged) and N_eff (effective independent bets) on BOTH the hedge's own
    graph (CORUM) and the held-out graph (STRING), per (base, operator, quality).
    The CORUM-vs-STRING comparison tests whether hedging on CORUM generalizes."""
    if "risk_corum" not in dB.columns:
        return None, None
    cols = ["R · CORUM ↓", "N_eff · CORUM ↑", "R · STRING ↓", "N_eff · STRING ↑"]
    has_q = "quality" in dB.columns
    rows = []
    for base in B_BASE_ORDER:
        for op in OP_ORDER:
            for q in (_RISK_QUALITIES if has_q else ("eff",)):
                g = dB[(dB.base == base) & (dB.operator == op)]
                if has_q:
                    g = g[g.quality == q]
                if not len(g):
                    continue
                cells = [_cell(*_stat(g[c], kind)) for c in
                         ("risk_corum", "neff_corum", "risk_string", "neff_string")]
                base_lbl = B_BASE_SHORT.get(base, base)
                lab = base_lbl if op == "none" else f"{base_lbl} + {OP_LABELS.get(op, op)}"
                if has_q:
                    lab += f" ({_RISK_QLABEL[q]})"
                rows.append({"label": lab, "cells": cells,
                             "group": base, "biggroup": _A_FAMILY.get(base),
                             "highlight": op == "none" and base in ("greedy", "ehvi") and q == "eff"})
    return cols, rows


def _diversity_barplot(group_counts, src, fig_dir):
    """Per-complex gene counts of the headline base's K picks: none vs k-DPP·CORUM
    (q=efficacy). none spikes into a few complexes; k-DPP spreads. Top-15 complexes
    by mean count shown."""
    if group_counts is None or group_counts.empty:
        return None
    g = group_counts[(group_counts.tox_source == src) & (group_counts.quality == "eff")]
    if g.empty:
        return None
    # mean genes per (operator, complex) over line x seed; complex_id -1 = unannotated.
    agg = g.groupby(["operator", "complex_id"])["count"].mean().reset_index()
    ops = [o for o in ("none", "kdpp_corum") if o in set(agg.operator)]
    if not ops:
        return None
    # top complexes by max mean count across ops, EXCLUDING the unannotated bucket
    real = agg[agg.complex_id >= 0]
    top = (real.groupby("complex_id")["count"].max().sort_values(ascending=False)
           .head(15).index.tolist())
    labels = [str(c) for c in top] + ["rest"]
    x = np.arange(len(labels)); w = 0.8 / max(1, len(ops))
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(labels)), 4.2))
    for j, op in enumerate(ops):
        sub = agg[agg.operator == op].set_index("complex_id")["count"]
        total = float(sub.sum())                              # = mean K (single-assignment)
        top_vals = [float(sub.get(c, 0.0)) for c in top]
        rest = max(0.0, total - sum(top_vals))                # other complexes + unannotated
        ax.bar(x + (j - (len(ops) - 1) / 2) * w, top_vals + [rest], w,
               color=OP_COLOR.get(op, "#888"), label=OP_LABELS.get(op, op))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    _despine(ax); ax.set_xlabel("CORUM complex id  (each gene counted once; 'rest' = other + unannotated)")
    ax.set_ylabel("mean # genes (bars sum to K)")
    _legend_out(ax, fontsize=9)
    fig.tight_layout()
    return _emit(fig, fig_dir, f"diversity_barplot_{src}")


_CV_AXIS_LABEL = {"lethality": "target-line lethality (per cell line)",
                  "target_pop": "target population (efficacy)",
                  "nontarget_contrast": "non-target population — contrast line",
                  "nontarget_aggregate": "non-target population — population mean",
                  "nontarget_population": "non-target population — population mean"}


def _cv_section(cv, fig_dir):
    """Initial GP-fit-quality section: ONE pooled 5-fold CV of the surrogate per
    objective axis (genes pooled across all target lines). Method-independent
    (cached). Returns (table_cols, table_rows)."""
    if cv is None or (hasattr(cv, "empty") and cv.empty):
        return None, None
    cols = ["R^2", "Spearman", "RMSE"]
    rows = []
    for axis, g in cv.groupby("axis"):
        rows.append({"m": _CV_AXIS_LABEL.get(axis, axis),
                     "cells": [_cell(*_stat(g["r2"], "ci")),
                               _cell(*_stat(g["spearman"], "ci")),
                               _cell(*_stat(g["rmse"], "ci"))],
                     "highlight": False})
    return cols, rows


_MC_ORDER = ["greedy", "random", "cluster", "info_div", "ehvi", "ehvi_pareto"]
_MC_LABEL = {"greedy": "greedy (target-efficacy)", "random": "random", "cluster": "cluster",
             "info_div": "info-diverse", "ehvi": "EHVI (N-D)", "ehvi_pareto": "EHVI · Pareto-nom"}


def _multicontrast_tables(mc):
    """Two tables for the multi-contrast (joint-GP N-D EHVI) experiment: (1) target
    efficacy + non-target efficacy in EACH contrast + average; (2) condensed =
    target efficacy + average non-target efficacy. Cells mean +/- 95% CI over
    lines x seeds. Returns (per_cols, per_rows, cond_cols, cond_rows, ncon)."""
    if mc is None or (hasattr(mc, "empty") and mc.empty):
        return None, None, None, None, 0
    tcols = sorted([c for c in mc.columns if c.startswith("tox_c")])
    ncon = len(tcols)
    methods = [m for m in _MC_ORDER if m in set(mc.method)]
    per_cols = (["Target Efficacy ↑"] +
                [f"Non-Target Efficacy · c{j+1} ↓" for j in range(ncon)] +
                ["Non-Target Efficacy · avg ↓"])
    cond_cols = ["Target Efficacy ↑", "Non-Target Efficacy (avg) ↓"]
    per_rows, cond_rows = [], []
    for m in methods:
        g = mc[mc.method == m]
        hl = m in ("ehvi", "ehvi_pareto")
        per_rows.append({"m": _MC_LABEL.get(m, m), "highlight": hl,
                         "cells": [_cell(*_stat(g["mean_efficacy"], "ci"))]
                         + [_cell(*_stat(g[c], "ci")) for c in tcols]
                         + [_cell(*_stat(g["tox_avg"], "ci"))]})
        cond_rows.append({"m": _MC_LABEL.get(m, m), "highlight": hl,
                          "cells": [_cell(*_stat(g["mean_efficacy"], "ci")),
                                    _cell(*_stat(g["tox_avg"], "ci"))]})
    return per_cols, per_rows, cond_cols, cond_rows, ncon


def _tex_escape(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("—", "--")]:
        s = s.replace(a, b)
    return s


def _celllines_table(meta, model_csv="data/raw/depmap/Model.csv"):
    """Appendix table of the target + contrast cell lines: (cell line, role,
    description) from DepMap Model.csv. Returns (rows, latex) or (None, None)."""
    from pathlib import Path
    lines = meta.get("lines") or []
    contrasts = meta.get("contrast_lines") or ([meta["contrast_line"]]
                                               if meta.get("contrast_line") else [])
    if not (lines or contrasts) or not Path(model_csv).exists():
        return None, None
    m = pd.read_csv(model_csv).set_index("ModelID")
    def info(mid):
        if mid not in m.index:
            return mid, "—", "—"
        r = m.loc[mid]
        return (str(r.get("CellLineName", mid) or mid),
                str(r.get("OncotreeLineage", "—") or "—"),
                str(r.get("OncotreePrimaryDisease", "—") or "—"))
    rows, tex = [], []
    for role, mids in [("contrast", contrasts), ("target", lines)]:
        for mid in mids:
            name, lin, dis = info(mid)
            desc = f"{lin} — {dis}"
            rows.append({"cells": [f"{name} ({mid})", role, desc]})
            tex.append(f"{_tex_escape(name)} ({mid}) & {role} & {_tex_escape(desc)} \\\\")
    latex = ("\\begin{table}[t]\\centering\\small\n\\begin{tabular}{lll}\n\\toprule\n"
             "Cell line & Role & Description (lineage --- primary disease) \\\\\n\\midrule\n"
             + "\n".join(tex) +
             "\n\\bottomrule\n\\end{tabular}\n"
             "\\caption{Target and contrast (non-target reference) cell lines used in the "
             "experiments.}\\label{tab:celllines}\n\\end{table}")
    return rows, latex


def _b_table(d, ops, metrics, kind="ci"):
    # family-major (greedy block, then EHVI block, double-ruled); within each family,
    # the no-diversity rows (base / *nom / *RT) come before the k-DPP rows.
    cols = [lbl for _, lbl in metrics]
    rows = []
    for fam in ("greedy", "ehvi"):
        fam_bases = [b for b in B_BASE_ORDER if _A_FAMILY.get(b) == fam]
        for op in ops:
            for base in fam_bases:
                g = d[(d.base == base) & (d.operator == op)]
                if not len(g) or metrics[0][0] not in g.columns:
                    continue
                cells = [_cell(*_stat(g[c], kind)) for c, _ in metrics]
                base_lbl = B_BASE_SHORT.get(base, base)
                # 'none' = the base alone; otherwise append the (short) graph name
                label = base_lbl if op == "none" else f"{base_lbl} + {OP_LABELS.get(op, op)}"
                rows.append({"label": label, "cells": cells,
                             "group": f"{fam}:{op}",            # single rule: none|k-DPP within family
                             "biggroup": fam,                   # double rule between greedy|EHVI
                             "highlight": op == "none" and base in ("greedy", "ehvi")})
    return cols, rows


def _b_bar(d, src, fig_dir):
    bases = [b for b in B_BASE_ORDER if b in set(d.base)]
    labels, conc, rob = [], [], []
    for b in bases:
        for op in OP_ORDER:
            g = d[(d.base == b) & (d.operator == op)]
            if len(g):
                labels.append(f"{B_BASE_SHORT.get(b, b)}+{OP_SHORT.get(op, op)}"); conc.append(_ci(g["concentration"])[0])
                rob.append(_ci(g["robustness"])[0])
    x = np.arange(len(labels)); w = 0.4
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(labels)), 4.2))
    ax.bar(x - w / 2, conc, w, color="#d1495b", label="concentration ↓")
    ax.bar(x + w / 2, rob, w, color="#1b9e77", label="robustness ↑")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    _despine(ax); ax.set_ylabel("metric value")
    _legend_out(ax, fontsize=9)
    fig.tight_layout()
    return _emit(fig, fig_dir, f"diversity_bar_{src}")


def _b_eff_conc(d, src, fig_dir):
    bases = [b for b in B_BASE_ORDER if b in set(d.base)]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for b in bases:
        xs, ys = [], []
        for op in OP_ORDER:
            g = d[(d.base == b) & (d.operator == op)]
            if not len(g):
                continue
            xc, yc = _ci(g["concentration"])[0], _ci(g["mean_efficacy"])[0]
            xs.append(xc); ys.append(yc)
            ax.scatter(xc, yc, s=72, marker=OP_MARK.get(op, "o"),
                       color=B_BASE_COLOR.get(b, "#444"), edgecolors="white", linewidths=1, zorder=3)
            ax.annotate(OP_SHORT.get(op, op), (xc, yc), fontsize=7, xytext=(0, 6),
                        textcoords="offset points", ha="center")
        ax.plot(xs, ys, ls=":", lw=1.2, color=B_BASE_COLOR.get(b, "#444"),
                label=B_BASE_SHORT.get(b, b), zorder=2)
    _despine(ax)
    ax.set_xlabel("pathway concentration  (← better hedged)")
    ax.set_ylabel("target target efficacy ↑")
    _legend_out(ax, fontsize=9, title="base")
    fig.tight_layout()
    return _emit(fig, fig_dir, f"eff_vs_conc_{src}")


def _filter_timing_curve(rounds, src, fig_dir):
    """Per-round filtering vs end-stage filtering, permissible target efficacy vs round,
    for the greedy and EHVI pairs. The gap (per-round − nomination) should widen
    with rounds, since per-round filtering avoids spending budget on toxic genes."""
    if rounds is None or rounds.empty:
        return None
    d = rounds[rounds.tox_source == src]
    pairs = [("trunc_pred", "greedy_safe", "greedy"), ("ehvi_trunc", "ehvi_safe", "EHVI")]
    pairs = [(a, b, lab) for a, b, lab in pairs if a in set(d.method) and b in set(d.method)]
    if not pairs:
        return None
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 4.0), squeeze=False)
    for ax, (nom, rt, lab) in zip(axes[0], pairs):
        for m, sty, nm in [(nom, "--", "end-stage filter"), (rt, "-", "per-round filter")]:
            agg = d[d.method == m].groupby("round")["nom_mean_efficacy_safe"].mean()
            ax.plot(agg.index, agg.values, sty + "o", ms=4, lw=1.6,
                    color=("#2e6f95" if "per-round" in nm else "#9aa0a6"), label=nm)
        _despine(ax); ax.set_title(f"{lab} acquisition", fontsize=10)
        ax.set_xlabel("AL round"); ax.set_ylabel("permissible target target efficacy ↑")
        _legend_out(ax, fontsize=8)
    fig.tight_layout()
    return _emit(fig, fig_dir, f"filter_timing_{src}")


def _failure_curve(dB, src, fig_dir):
    """Value-of-diversity: portfolio value retained as the d most-valuable
    pathways fail, one line per operator (on a representative base). Diversified
    portfolios (cap/kdpp) should sit above the concentrated 'none'."""
    dropcols = sorted([c for c in dB.columns if c.startswith("drop_")],
                      key=lambda c: int(c.split("_")[1]))
    if not dropcols:
        return None
    base = "ehvi_trunc" if "ehvi_trunc" in set(dB.base) else sorted(set(dB.base))[0]
    d = dB[dB.base == base]
    xs = list(range(len(dropcols)))
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for op in OP_ORDER:
        g = d[d.operator == op]
        if not len(g):
            continue
        ys = [g[c].mean() for c in dropcols]
        ax.plot(xs, ys, "-o", ms=4, lw=1.6, color=OP_COLOR.get(op, "#888"),
                label=OP_LABELS.get(op, op))
    _despine(ax)
    ax.set_xlabel("# most-valuable pathways eliminated")
    ax.set_ylabel("portfolio value retained")
    ax.set_title(f"base = {B_BASE_SHORT.get(base, base)}", fontsize=10)
    _legend_out(ax, fontsize=8)
    fig.tight_layout()
    return _emit(fig, fig_dir, f"failure_sim_{src}")


# ---- headlines ---------------------------------------------------------------
def _headline_A(d):
    def mof(m, c):
        g = d[d.method == m]
        return _ci(g[c])[0] if len(g) else float("nan")
    return (
        f"Naive <b>greedy</b> carries non-target efficacy {mof('greedy','mean_toxicity'):.2f} at "
        f"efficacy {mof('greedy','mean_efficacy'):.2f}. The <b>learned</b> safety rules "
        f"(trunc_pred tox {mof('trunc_pred','mean_toxicity'):.2f}, EHVI+trunc "
        f"{mof('ehvi_trunc','mean_toxicity'):.2f}, EHVI-Pareto "
        f"{mof('ehvi_pareto','mean_toxicity'):.2f}) drive non-target efficacy to the threshold while "
        f"keeping realizable target efficacy. (Untruncated EHVI baseline: "
        f"tox {mof('ehvi','mean_toxicity'):.2f}.)"
    )


def _headline_B(d):
    bits = []
    for base in [b for b in B_BASE_ORDER if b in set(d.base)]:
        g0 = d[(d.base == base) & (d.operator == "none")]
        if not len(g0):
            continue
        c0, e0 = _ci(g0["concentration"])[0], _ci(g0["mean_efficacy"])[0]
        best_op, best_c, best_e = "none", c0, e0
        for op in ("cap", "kdpp_emb", "kdpp_corum"):
            g = d[(d.base == base) & (d.operator == op)]
            if len(g) and _ci(g["concentration"])[0] < best_c:
                best_op, best_c, best_e = op, _ci(g["concentration"])[0], _ci(g["mean_efficacy"])[0]
        if best_op != "none":
            bits.append(f"<b>{B_BASE_SHORT.get(base, base)}</b>: {OP_LABELS.get(best_op, best_op)} cuts concentration "
                        f"{c0:.0%}→{best_c:.0%} (efficacy {e0:.2f}→{best_e:.2f})")
    return ("Diversity operators de-concentrate the portfolio — " + "; ".join(bits) + "."
            ) if bits else "Diversity operators applied per base."


_FACET_TMPL = """
<h2>Non-target reference: <span style="color:#2e6f95">{{ src }}</span>{{ primary }}</h2>

<h3>A &middot; Target vs Non-Target Efficacy — all methods</h3>
<div class="note">Each point a method (mean over lines × seeds, 95% CI bars). Up = more lethal, left = safer. Dashed line = the τ non-target efficacy ceiling; the shaded green band left of it is the permissible region.</div>
<div class="card">{{ tradeoff|safe }}</div>
{% if consistency %}
<h3>A &middot; Filter robustness across cell lines</h3>
<div class="note">Per-line non-target efficacy reduction from the per-round filter (greedy minus greedy·per-round), paired under common random numbers. All-positive bars = the filter is safer than greedy in every line.</div>
<div class="card">{{ consistency|safe }}</div>
{% endif %}
{% if safety_bar %}
<h3>A &middot; Permissible vs over-threshold targets</h3>
<div class="note">For each method, how many of the K nominees have TRUE non-target efficacy below (permissible) vs above the τ ceiling. The naive/diversity baselines nominate many over-threshold targets; the safety rules keep them permissible.</div>
<div class="card">{{ safety_bar|safe }}</div>
{% endif %}
{% if admission %}
<h3>A &middot; Admission curve — non-filtered methods vs the non-target threshold</h3>
<div class="note">For the methods with no safety filter: the fraction of their K nominees that would pass at each hypothetical threshold τ (empirical CDF of nominee non-target efficacy, pooled over lines). Curves that stay low until high τ are nominating over-threshold targets. Dashed line = the τ used in the main results.</div>
<div class="card">{{ admission|safe }}</div>
{% endif %}
{% if admission_eff %}
<h3>A &middot; Admission curve — realizable target efficacy vs the non-target threshold</h3>
<div class="note">Same as above but the y-axis is REALIZABLE target efficacy: at each hypothetical threshold τ, the mean over a method's K nominees of efficacy with over-threshold picks scored 0. Shows how much usable potency survives the threshold, not just how many picks pass.</div>
<div class="card">{{ admission_eff|safe }}</div>
{% endif %}
{% if perline %}
<h3>A &middot; Per-cell-line tradeoff</h3>
<div class="note">One panel per cell line; the same methods; each with its own τ ceiling. The safety ordering holds across lines.</div>
<div class="card">{{ perline|safe }}</div>
{% endif %}
{% if cloud %}
<h3>A &middot; Gene landscape — nominated targets per method (all lines pooled)</h3>
<div class="note">Each panel a method. Grey = all candidate genes (pooled over {{ n_lines }} lines); open circles = that method's {{ cloudK }} targets; dashed line = τ ceiling. Safety rules pull picks left of the ceiling.</div>
<div class="card">{{ cloud.agg|safe }}</div>
<details><summary style="cursor:pointer;color:#2e6f95;font-weight:600;margin:.4rem 0">▸ per-cell-line gene landscapes ({{ cloud.per|length }})</summary>
{% for cl, c in cloud.per %}
<div class="note" style="margin-top:.8rem"><b>{{ cl }}</b></div>
<div class="card">{{ c|safe }}</div>
{% endfor %}
</details>
{% endif %}
{% if final_k %}
<h3>A &middot; Final K-set performance (per method)</h3>
<div class="note">The nominated top-{{ cloudK }} shortlist scored on its true values (mean over lines × seeds, 95% CI). Realizable target efficacy = target efficacy over the K picks with over-threshold (non-permissible) picks scored 0.</div>
<div class="card">{{ final_k|safe }}</div>
{% endif %}
<h3>A &middot; Method table (main)</h3>
<div class="note">Rows grouped: baselines, then greedy variants, then EHVI variants. Cells = mean ± 95% CI. The unmodified reference methods (greedy, EHVI) are shaded. LaTeX export uses double rules between groups + <code>\\rowcolor</code> for the reference rows (needs <code>\\usepackage[table]{xcolor}</code>).</div>
<div class="card"><table>
<thead><tr><th>method</th>{% for h in a_cols %}<th>{{ h }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in a_rows %}<tr{% if r.highlight %} style="background:#eaf2fb"{% endif %}><td>{{ r.m }}</td>{% for c in r.cells %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table><details class="tex"><summary>LaTeX</summary><pre><code>{{ a_latex }}</code></pre></details>
<details><summary style="cursor:pointer;color:#2e6f95;font-weight:600;margin:.4rem 0">▸ same table, mean ± SEM</summary>
<table><thead><tr><th>method</th>{% for h in a_cols %}<th>{{ h }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in a_rows_std %}<tr{% if r.highlight %} style="background:#eaf2fb"{% endif %}><td>{{ r.m }}</td>{% for c in r.cells %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table><details class="tex"><summary>LaTeX</summary><pre><code>{{ a_latex_std }}</code></pre></details></details></div>
<h3>A &middot; Diagnostics</h3>
<div class="note">Secondary / diagnostic quantities — # novel (unassayed) nominees, nominee hypervolume, and normalized Pareto recall. Cells = mean ± 95% CI.</div>
<div class="card"><table>
<thead><tr><th>method</th>{% for h in ad_cols %}<th>{{ h }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in ad_rows %}<tr{% if r.highlight %} style="background:#eaf2fb"{% endif %}><td>{{ r.m }}</td>{% for c in r.cells %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table><details class="tex"><summary>LaTeX</summary><pre><code>{{ ad_latex }}</code></pre></details>
<details><summary style="cursor:pointer;color:#2e6f95;font-weight:600;margin:.4rem 0">▸ same table, mean ± SEM</summary>
<table><thead><tr><th>method</th>{% for h in ad_cols %}<th>{{ h }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in ad_rows_std %}<tr{% if r.highlight %} style="background:#eaf2fb"{% endif %}><td>{{ r.m }}</td>{% for c in r.cells %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table><details class="tex"><summary>LaTeX</summary><pre><code>{{ ad_latex_std }}</code></pre></details></details></div>
{% if assayed_img %}
<h3>A &middot; Assayed set — quality of the genes actually measured</h3>
<div class="note">Per <b>acquisition</b> (not nomination): of the ~120 genes measured during the loop, how many are safe and how high the collected data's non-target efficacy is. Tests whether EHVI / constrained ("+safe") acquisition gathers safer data than greedy.</div>
<div class="card">{{ assayed_img|safe }}</div>
<div class="card"><table>
<thead><tr><th>acquisition</th>{% for h in asy_cols %}<th>{{ h }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in asy_rows %}<tr{% if r.highlight %} style="background:#eaf2fb"{% endif %}><td>{{ r.m }}</td>{% for c in r.cells %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table><details class="tex"><summary>LaTeX</summary><pre><code>{{ asy_latex }}</code></pre></details></div>
{% endif %}
{% if rounds_nom %}
<h3>A &middot; Per-round learning curves</h3>
<div class="note">Performance vs AL round (mean over lines × seeds). Top — nomination quality (the final shortlist if we stopped at round r). Bottom — assayed-set quality (what's been measured so far).</div>
<div class="card">{{ rounds_nom|safe }}</div>
<div class="card">{{ rounds_assayed|safe }}</div>
{% endif %}
{% if rz_rows %}
<h3>A &middot; Realizable target efficacy per round <span style="color:#2e6f95">[{{ src }}]</span></h3>
<div class="note">The nominated set's <b>realizable</b> target efficacy if we stopped at round r — permissible efficacy × yield = (mean efficacy of safe nominees) × (# safe) / K, i.e. mean efficacy over all K picks with non-permissible picks scored 0. Combines the two top-panel curves into the single un-gameable number (inert picks score ~0). Selectivity-aware acquisition keeps climbing; random plateaus. Final column = round-0→final gain.</div>
<div class="card"><table>
<thead><tr><th>method</th>{% for h in rz_cols %}<th>{{ h }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in rz_rows %}<tr{% if r.highlight %} style="background:#eaf2fb"{% endif %}><td>{{ r.m }}</td>{% for c in r.cells %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table><details class="tex"><summary>LaTeX</summary><pre><code>{{ rz_latex }}</code></pre></details></div>
{% endif %}
{% if rr_rows %}
<h3>A &middot; Selectivity ratio per round <span style="color:#2e6f95">[{{ src }}]</span></h3>
<div class="note">Change from round 0 in the nominated set's <b>selectivity ratio</b> = target efficacy / non-target efficacy (denominator floored at 0.05), paired per line × seed. Every method starts at 0; larger = acquisition improves selectivity more over the loop.</div>
<div class="card"><table>
<thead><tr><th>method</th>{% for h in rr_cols %}<th>{{ h }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in rr_rows %}<tr{% if r.highlight %} style="background:#eaf2fb"{% endif %}><td>{{ r.m }}</td>{% for c in r.cells %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table><details class="tex"><summary>LaTeX</summary><pre><code>{{ rr_latex }}</code></pre></details></div>
{% endif %}
{% if filter_timing %}
<h3>A &middot; Per-round vs end-stage filtering (does filter timing matter?)</h3>
<div class="note">Permissible target efficacy vs AL round for per-round filtering (solid) vs a single end-stage filter (dashed). The gap widens with rounds — per-round filtering increasingly pays off as more budget would otherwise be spent on genes that get discarded.</div>
<div class="card">{{ filter_timing|safe }}</div>
{% endif %}
{% if auc_rows %}
<h3>A &middot; Hypervolume accumulation over rounds (AUC-HV) <span style="color:#2e6f95">[{{ src }}]</span></h3>
<div class="note">Area under the hypervolume-over-rounds curve in the (target population, −non-target population) plane — how fast each acquisition accumulates dominated hypervolume. Two senses: revealed-front (exploration) and nominated-set (decision quality). EHVI is expected to lead by construction. Separate from the main table's candidate-set HV.</div>
<div class="card"><table>
<thead><tr><th>method</th>{% for h in auc_cols %}<th>{{ h }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in auc_rows %}<tr{% if r.highlight %} style="background:#eaf2fb"{% endif %}><td>{{ r.m }}</td>{% for c in r.cells %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table><details class="tex"><summary>LaTeX</summary><pre><code>{{ auc_latex }}</code></pre></details></div>
{% if hv_curve %}<div class="card">{{ hv_curve|safe }}</div>{% endif %}
{% endif %}

<h3>B &middot; Diversity &amp; portfolio risk <span style="color:#2e6f95">[{{ src }}]</span></h3>
<div class="key">{{ headline_b|safe }}</div>
<div class="note">The k-DPP&middot;CORUM hedge (q=efficacy) vs the none baseline, layered on each base. Identical columns evaluated two ways: on CORUM (the graph the hedge selects on) and on the held-out STRING network. Concentration↓ / Robustness↑ / Distinct groups↑ over the respective graph's groups; Risk = equal-weight portfolio variance wᵀSw↓; N_eff = effective independent bets↑. Lower Risk / higher N_eff on STRING (which the hedge never optimizes) = the hedge generalizes.</div>
{% if bc_rows %}
<h4>B.1 &middot; Evaluated on CORUM <span style="color:#2e6f95">[{{ src }}]</span></h4>
<div class="card"><table>
<thead><tr><th>base + operator</th>{% for h in bc_cols %}<th>{{ h }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in bc_rows %}<tr{% if r.highlight %} style="background:#eaf2fb"{% endif %}><td>{{ r.label }}</td>{% for c in r.cells %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table><details class="tex"><summary>LaTeX</summary><pre><code>{{ bc_latex }}</code></pre></details></div>
{% endif %}
{% if bstr_rows %}
<h4>B.2 &middot; Evaluated on held-out STRING <span style="color:#2e6f95">[{{ src }}]</span></h4>
<div class="card"><table>
<thead><tr><th>base + operator</th>{% for h in bstr_cols %}<th>{{ h }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in bstr_rows %}<tr{% if r.highlight %} style="background:#eaf2fb"{% endif %}><td>{{ r.label }}</td>{% for c in r.cells %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table><details class="tex"><summary>LaTeX</summary><pre><code>{{ bstr_latex }}</code></pre></details></div>
{% endif %}
{% if barplot %}
<h4>B.3 &middot; Mechanism diversity — genes per CORUM complex <span style="color:#2e6f95">[{{ src }}]</span></h4>
<div class="note">Per-complex gene counts of the headline (EHVI) base's K picks, none vs k-DPP·CORUM (q=efficacy). Each gene assigned to one complex (smallest id; "rest" = other complexes + unannotated), so both bars sum to K. The unhedged portfolio spikes into a few complexes; k-DPP spreads across mechanisms.</div>
<div class="card">{{ barplot|safe }}</div>
{% endif %}
<div class="card">{{ b_bar|safe }}</div>
<h3>B &middot; Target Efficacy vs concentration (CORUM) <span style="color:#2e6f95">[{{ src }} non-target efficacy]</span></h3>
<div class="note">Each line a base; markers are operators (none / k-DPP·STRING / k-DPP·CORUM). Left = better hedged (lower CORUM concentration); high = efficacy retained.</div>
<div class="card">{{ b_eff_conc|safe }}</div>
{% if failure_sim %}
<h3>B &middot; Value of diversity — pathway-failure simulation <span style="color:#2e6f95">[{{ src }} non-target efficacy]</span></h3>
<div class="note">Portfolio value retained as the most-valuable pathways are eliminated one by one (a pathway proving non-viable). A diversified shortlist (cap / k-DPP) loses less than the concentrated baseline.</div>
<div class="card">{{ failure_sim|safe }}</div>
{% endif %}
"""

def _r2(y, yh):
    y = np.asarray(y, float); yh = np.asarray(yh, float)
    ss = float(np.sum((y - y.mean()) ** 2))
    return float(1 - np.sum((y - yh) ** 2) / ss) if ss > 0 else float("nan")


def _joint_eff_tox(scatter, src, tau, tau_mode, fig_dir):
    """Efficacy-vs-non-target efficacy scatter with marginal histograms; Pearson r. Pooled
    over all cell lines for one non-target reference."""
    if scatter is None or scatter.empty:
        return None
    d = scatter[scatter.tox_source == src] if "tox_source" in scatter.columns else scatter
    if d.empty:
        return None
    tox = d["toxicity"].to_numpy(float); eff = d["efficacy"].to_numpy(float)
    m = np.isfinite(tox) & np.isfinite(eff); tox, eff = tox[m], eff[m]
    if len(tox) < 3:
        return None
    r = float(np.corrcoef(tox, eff)[0, 1])
    ceil = _ceiling(scatter, src, tau, tau_mode)
    fig = plt.figure(figsize=(5.6, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)
    ax.scatter(tox, eff, s=3, c="#7aa6c2", linewidths=0, alpha=0.45, rasterized=True)
    if ceil is not None:
        ax.axvline(ceil, ls="--", lw=1, color="#d1495b")
        axt.axvline(ceil, ls="--", lw=1, color="#d1495b")
    ax.set_xlabel(f"non-target efficacy ({src})"); ax.set_ylabel("efficacy (target-line lethality)")
    ax.annotate(f"Pearson r = {r:.2f}\nn = {len(tox):,}", xy=(0.04, 0.96),
                xycoords="axes fraction", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
    axt.hist(tox, bins=60, color="#7aa6c2")
    axr.hist(eff, bins=60, orientation="horizontal", color="#7aa6c2")
    for a in (axt, axr):
        a.axis("off")
    _despine(ax)
    return _emit(fig, fig_dir, f"ds_jointeff_{src}")


def _embed_tox_fig(scatter, src, fig_dir):
    """Predicted (5-fold OOF) vs true non-target efficacy from the embedding: R^2 + Spearman.
    kNN-cosine probe (GP-smoothness proxy); ridge R^2 reported alongside."""
    if scatter is None or "pred_toxicity" not in getattr(scatter, "columns", []):
        return None
    d = scatter[scatter.tox_source == src] if "tox_source" in scatter.columns else scatter
    if d.empty:
        return None
    from scipy.stats import spearmanr
    t = d["toxicity"].to_numpy(float); p = d["pred_toxicity"].to_numpy(float)
    m = np.isfinite(t) & np.isfinite(p); t, p = t[m], p[m]
    if len(t) < 3:
        return None
    txt = f"kNN  R² = {_r2(t, p):.2f}\nSpearman ρ = {float(spearmanr(t, p).statistic):.2f}"
    if "pred_toxicity_lin" in d.columns:
        pl = d["pred_toxicity_lin"].to_numpy(float)[m]
        txt += f"\nridge R² = {_r2(t, pl):.2f}"
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(t, p, s=3, c="#7aa6c2", linewidths=0, alpha=0.45, rasterized=True)
    lo = float(min(t.min(), p.min())); hi = float(max(t.max(), p.max()))
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="#999999")
    ax.set_xlabel(f"true non-target efficacy ({src})")
    ax.set_ylabel("predicted non-target efficacy (5-fold OOF)")
    ax.annotate(txt, xy=(0.04, 0.96), xycoords="axes fraction", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
    _despine(ax)
    return _emit(fig, fig_dir, f"ds_embedtox_{src}")


def _dataset_section(scatter, tau, tau_mode, tox_sources, fig_dir):
    """Section 0: data characteristics — eff/tox coupling + embedding predictivity."""
    if scatter is None or scatter.empty:
        return ""
    blocks = []
    for src in tox_sources:
        j = _joint_eff_tox(scatter, src, tau, tau_mode, fig_dir)
        e = _embed_tox_fig(scatter, src, fig_dir)
        if not j and not e:
            continue
        prim = " (primary)" if src == "contrast" else ""
        blocks.append(
            f'<h3>Non-target reference: <span style="color:#2e6f95">{src}</span>{prim}</h3>'
            '<div style="display:flex;gap:1.2rem;flex-wrap:wrap;align-items:flex-start">'
            '<div style="flex:1;min-width:340px"><div class="note">Target efficacy vs non-target efficacy '
            'over all candidate genes (pooled across lines), with marginal histograms. '
            'Dashed = τ ceiling. Positive r ⇒ the most lethal knockouts also tend to have '
            'high non-target efficacy — the tension the safety rules must resolve.</div>'
            f'<div class="card">{j or "—"}</div></div>'
            '<div style="flex:1;min-width:320px"><div class="note">Can the embedding '
            'predict non-target efficacy? 5-fold out-of-fold kNN-cosine (a local-smoothness proxy '
            'for the GP surrogate) + ridge (linear probe). Tight diagonal ⇒ embedding '
            'carries non-target efficacy signal; diffuse ⇒ representation is the bottleneck.</div>'
            f'<div class="card">{e or "—"}</div></div></div>')
    if not blocks:
        return ""
    return ('<h2>Dataset characteristics</h2>\n'
            '<div class="note">General properties of the candidate space, before any '
            'method is run.</div>\n' + "\n".join(blocks))


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
 details.tex{margin-top:.6rem}
 details.tex summary{cursor:pointer;color:#2e6f95;font-weight:600;font-size:.85rem}
 details.tex pre{background:#0f172a;color:#e2e8f0;border-radius:8px;padding:.8rem 1rem;overflow-x:auto;font-size:.8rem;margin-top:.4rem}
 details.tex code{font-family:ui-monospace,'SF Mono',Menlo,monospace}
</style></head><body>
<h1>{{ title }}</h1>
<div class="sub">A: {{ meta.n_genes_a }} genes &middot; B: {{ meta.n_genes_b }} genes &middot; {{ n_lines }} cell lines &middot; {{ n_seeds }} seeds &middot; K={{ meta.K }} &middot; AL {{ meta.n_rounds }}×{{ meta.batch }} &middot; τ={{ meta.tau }} ({{ meta.tau_mode }}) &middot; {% if meta.contrast_lines and meta.contrast_lines|length > 1 %}contrasts ({{ meta.contrast_lines|length }}) {{ meta.contrast_lines|join(", ") }}{% else %}contrast {{ meta.contrast_line }}{% endif %} &middot; acq={{ meta.acq_score }}{% if meta.joint_gp %} &middot; JOINT GP{% endif %}</div>
{% if launch_info %}<div class="note">Reproduce — launched via: {% for name in launch_info %}<code>{{ name }}</code>{% if not loop.last %}, {% endif %}{% endfor %}{% for name, txt in launch_info.items() %}<details class="tex"><summary>{{ name }}</summary><pre><code>{{ txt }}</code></pre></details>{% endfor %}</div>{% endif %}

<h2>How to read this</h2>
<div class="gloss">{{ glossary|safe }}</div>

{% if dataset %}{{ dataset|safe }}{% endif %}

{% if cv_rows %}
<h2>0 &middot; Surrogate fit quality (5-fold CV)</h2>
<div class="note">How well the GP surrogate predicts target-line <b>lethality</b> from the embeddings — a 5-fold cross-validation within each of the target cell lines, averaged over the lines (full-genome pooling across lines is infeasible for an exact GP, so each line is CV'd then averaged). Method-independent: never changes when the acquisition/diversity method changes, so it is cached per line. Cells = mean ± 95% CI over lines × folds. Values are modest by design — the pipeline works with a deliberately weak surrogate; the contribution is portfolio-risk hedging, which does not need an accurate surrogate.</div>
<div class="note"><b>Target cell lines ({{ meta.lines|length }}):</b> {{ meta.lines|join(", ") }}. &nbsp; <b>Contrast (non-target reference) line:</b> {{ meta.contrast_line }}.</div>
<div class="card"><table>
<thead><tr><th>objective axis</th>{% for h in cv_cols %}<th>{{ h }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in cv_rows %}<tr><td>{{ r.m }}</td>{% for c in r.cells %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table></div>
{% endif %}


{% for f in facets %}
<div class="key">{{ f.headline_a|safe }}</div>
{{ f.block|safe }}
{% endfor %}

{% if cl_rows %}
<h2>Appendix &middot; Cell lines</h2>
<div class="note">Target cell lines and contrast (non-target reference) cell lines used in the experiments, with DepMap lineage and primary disease.</div>
<div class="card"><table>
<thead><tr><th>cell line</th><th>role</th><th>description</th></tr></thead>
<tbody>{% for r in cl_rows %}<tr><td>{{ r.cells[0] }}</td><td>{{ r.cells[1] }}</td><td>{{ r.cells[2] }}</td></tr>{% endfor %}</tbody>
</table><details class="tex"><summary>LaTeX</summary><pre><code>{{ cl_latex }}</code></pre></details></div>
{% endif %}
</body></html>"""

_GLOSSARY = """
<b>Two analyses, two questions, two non-target references.</b><br><br>
<b>Reading method labels</b> (compact <code>[acq]·[filtering]</code>): <b>greedy</b>=greedy acquisition (UCB on efficacy), <b>EHVI</b>=dual-objective acquisition; <b>nom</b>=filter the final shortlist only, <b>per-round</b>=filter the candidate pool every round. All filtering uses the GP-<b>predicted</b> non-target efficacy (the non-target efficacy code is dropped from labels since no oracle/known variants are shown). So <code>greedy·per-round</code> = greedy acquisition + per-round predicted-tox filter; <code>EHVI·per-round</code> = EHVI + per-round predicted-tox filter. <code>random/farthest/cluster/info-div</code> are the no-safety baselines.<br><br>
<b>Safety rules (Analysis A).</b>
&bull; <b>greedy</b>: top-K predicted efficacy, no safety. &bull; <b>filter · predicted</b>: keep genes below the τ non-target efficacy ceiling using the GP-<i>learned</i> non-target efficacy. &bull; <b>EHVI</b>: dual-objective EHVI <i>acquisition</i> (collects data balancing efficacy and learned non-target efficacy), then nominates <b>top-K by predicted efficacy with no filter</b> — the max-efficacy shortlist. &bull; <b>EHVI-EHVI</b>: <i>identical EHVI acquisition</i>; differs only at nomination — instead of max-efficacy it nominates the K genes on the <i>predicted</i> (efficacy↑, non-target efficacy↓) Pareto front, the balanced <b>threshold-free</b> nomination (no τ): a gene is picked because nothing dominates it on both axes, trading potency against selectivity. (The τ-<i>filtered</i> EHVI variants are <code>EHVI·nom</code> = end-stage filter and <code>EHVI·per-round</code>.) &bull; <b>random / farthest / cluster / info_div</b>: prior-work / naive baselines (info_div = informativeness+diversity, IterPert-like; greedy = quality-only, NAIAD-like).<br><br>
<b>Filter timing (end-stage vs per-round).</b> The filter can be applied only to the final shortlist (<i>end-stage</i>, <code>nom</code>) or at <i>every acquisition round</i> (<code>RT</code>), restricting each round to genes believed safe — so the assay budget isn't spent on genes we think are over-threshold. Acquisition and nomination remain distinct stages; per-round filtering constrains both.<br><br>
<b>Diversity operators (Analysis B).</b> &bull; <b>none</b>: top-K by quality. &bull; <b>k-DPP · embedding</b>: quality-weighted k-DPP with the <i>learned</i> embedding-cosine similarity. &bull; <b>k-DPP · STRING</b>: external STRING combined-score network similarity. &bull; <b>k-DPP · CORUM</b>: external CORUM protein-complex (Jaccard co-membership) similarity. None is outcome similarity — the embedding-vs-STRING-vs-CORUM comparison is the similarity-source ablation (learned vs two external knowledge graphs, dense → sparse). Layered on the four safety-aware bases (G·nom·P / G·RT·P / E·nom·P / E·RT·P). A pathway-failure simulation quantifies the value of the resulting diversity.<br><br>
<b>Non-target efficacy & τ.</b> &bull; <b>contrast</b> (primary): lethality in one fixed contrast line (normal-tissue stand-in). &bull; <b>aggregate</b>: common-essential fraction, excluding the target line. <b>τ is a quantile</b>: the dashed line on each plot is the τ-quantile of the candidate non-target efficacy (τ=0.5 = the safest half) — for the contrast definition, quantile(−effect in the contrast line, τ).<br><br>
<b>Pareto recall (normalized).</b> Fraction of the true (efficacy, −non-target efficacy) Pareto front recovered by the K nominees, normalized by min(K, |front|) — the most front genes K picks <i>could</i> recover. Genome-wide the front can be hundreds of genes, so the raw fraction is capped near K/|front|; normalizing makes 1.0 attainable and the score discriminating. Threshold-free (a front gene is non-dominated by definition; no efficacy cutoff).<br><br>
<b>Spread.</b> Tables show mean ± 95% CI (1.96·sem over lines × seeds); each has a "mean ± SEM" dropdown with the standard error of the mean (std/√n) instead.
"""


def build_ablation_report(df: pd.DataFrame, out_path, scatter=None, meta=None,
                          assayed=None, rounds=None, fig_dir=None,
                          group_counts=None, cv=None, multicontrast=None, launch_info=None,
                          title="geneal — default ablation (safety + diversity)") -> Path:
    meta = meta or {}
    df = df.copy()

    # Multi-contrast runs label sources contrast_1/2/3; POOL them into a single
    # "contrast" source so every table/plot aggregates over contrasts x lines x
    # seeds (no per-contrast sections). Single-contrast runs already use "contrast".
    import re as _re
    def _pool_contrasts(frame):
        if frame is None or "tox_source" not in getattr(frame, "columns", []):
            return frame
        f = frame.copy()
        f["tox_source"] = f["tox_source"].astype(str).str.replace(r"^contrast_\d+$",
                                                                   "contrast", regex=True)
        return f
    df = _pool_contrasts(df)
    scatter = _pool_contrasts(scatter)
    assayed = _pool_contrasts(assayed)
    rounds = _pool_contrasts(rounds)
    group_counts = _pool_contrasts(group_counts)
    n_contrasts = sum(1 for s in (meta.get("tox_sources") or []) if _re.match(r"contrast_\d+$", str(s)))

    # useful efficacy = total efficacy of the safe picks / K (toxic picks scored 0).
    # Derive it for older runs that predate the metric; per-row this equals
    # mean_efficacy_safe * n_safe / K exactly (sum_safe_eff / K).
    if "useful_efficacy" not in df.columns and {"mean_efficacy_safe", "n_safe"} <= set(df.columns):
        _K = float(meta.get("K", 30))
        df["useful_efficacy"] = df["mean_efficacy_safe"].fillna(0.0) * df["n_safe"] / _K
    # selectivity ratio = target / non-target efficacy; floor denom at 0.05 so the
    # ~1% of rows with non-positive non-target efficacy (genuinely non-toxic picks)
    # read as highly selective instead of blowing up / flipping sign. Higher = better.
    if {"mean_efficacy", "mean_toxicity"} <= set(df.columns):
        df["sel_ratio"] = df["mean_efficacy"] / df["mean_toxicity"].clip(lower=0.05)
    lines = meta.get("lines") or sorted(df.cell_line.unique())
    kdpp_sim = meta.get("kdpp_sim", "embedding")
    tau = float(meta.get("tau", 0.5))
    tau_mode = meta.get("tau_mode", "absolute")
    tox_sources = [s for s in ["contrast", "aggregate"] if s in set(df.tox_source)] \
        or sorted(df.tox_source.unique())

    facets = []
    for src in tox_sources:
        d = df[df.tox_source == src]
        dA, dB = d[d.analysis == "A"], d[d.analysis == "B"]
        # existing Section B tables/figures describe the efficacy-quality nomination;
        # filter to q=efficacy so the new selectivity-quality rows don't double them.
        dB_eff = dB[dB["quality"] == "eff"] if "quality" in dB.columns else dB
        a_cols, a_rows = _table_A(dA, _A_MAIN_METRICS, "ci")
        ad_cols, ad_rows = _table_A(dA, _A_DIAG_METRICS, "ci")
        _, a_rows_std = _table_A(dA, _A_MAIN_METRICS, "sem")
        _, ad_rows_std = _table_A(dA, _A_DIAG_METRICS, "sem")
        # two merged Section-B tables, identical columns, k-DPP·CORUM hedge vs none:
        # diversity metrics + R/N_eff evaluated on CORUM and (held-out) STRING.
        bc_cols, bc_rows = _b_table(dB_eff, _B_HEDGE_OPS, _B_EVAL_CORUM)     # eval on CORUM
        bstr_cols, bstr_rows = _b_table(dB_eff, _B_HEDGE_OPS, _B_EVAL_STRING)  # eval on STRING
        auc_cols, auc_rows = _auc_hv_table(rounds, src)                   # AUC-HV (separate table)
        assayed_img, (asy_cols, asy_rows) = _assayed_panel(assayed, src, fig_dir)
        rounds_nom, rounds_assayed = _round_curves(rounds, src, fig_dir)
        rz_cols, rz_rows = _realizable_round_table(rounds, src, float(meta.get("K", 30)))
        rr_cols, rr_rows = _ratio_round_table(rounds, src)
        block = Environment(loader=BaseLoader()).from_string(_FACET_TMPL).render(
            src=src,
            primary=((" (primary)" if src == "contrast" else "")
                     + (f" — pooled over {n_contrasts} contrast lines: "
                        + ", ".join(meta.get('contrast_lines', []))
                        if src == "contrast" and n_contrasts > 1 else "")), tau=tau,
            tradeoff=_tradeoff_points(dA, src, scatter, tau, tau_mode, fig_dir),
            consistency=_consistency_strip(dA, src, fig_dir),
            safety_bar=_safety_bar(dA, src, fig_dir),
            admission=_admission_curve(scatter, src, tau, tau_mode, fig_dir),
            admission_eff=_admission_efficacy_curve(scatter, src, tau, tau_mode, fig_dir),
            final_k=_final_k_bars(dA, src, fig_dir),
            perline=_tradeoff_per_line(dA, src, lines, scatter, tau, tau_mode, fig_dir),
            cloud=_gene_cloud_section(scatter, src, lines, tau, tau_mode, fig_dir),
            cloudK=meta.get("K", ""), n_lines=df.cell_line.nunique(),
            a_cols=a_cols, a_rows=a_rows, a_rows_std=a_rows_std, ad_rows_std=ad_rows_std,
            a_latex_std=_latex_table("method", a_cols, a_rows_std,
                                     f"Target vs Non-Target Efficacy ({src} non-target efficacy); cells are mean $\\pm$ sem over cell lines $\\times$ seeds. \\emph{{Realizable target efficacy}} (mean over the $K$ nominees of efficacy with each non-permissible nominee scored 0 -- equivalently the total efficacy of the safe picks divided by $K$) and \\emph{{\\# safe}} (count of the $K$ nominees with non-target efficacy at or below the $\\tau$ ceiling) are defined only when a non-target efficacy threshold is known, and quantify the threshold-known regime; \\emph{{target efficacy}} and \\emph{{non-target efficacy}} require no threshold.",
                                     f"A_{src}_sem"),
            ad_latex_std=_latex_table("method", ad_cols, ad_rows_std,
                                      f"Diagnostics ({src} non-target efficacy), mean $\\pm$ sem.",
                                      f"Adiag_{src}_sem"),
            headline_b=_headline_B(dB_eff),
            bc_cols=bc_cols, bc_rows=bc_rows, bstr_cols=bstr_cols, bstr_rows=bstr_rows,
            bc_latex=_latex_table("base + op", _B_SHORT_COLS, bc_rows,
                                  f"Section B, evaluated on CORUM ({src} non-target population): "
                                  f"diversity + portfolio risk (R $\\downarrow$, $N_{{eff}}\\uparrow$) "
                                  f"of the k-DPP$\\cdot$CORUM hedge vs none. "
                                  f"Conc.=concentration, Robust.=robustness, Avg. Units=distinct "
                                  f"units covered, T./NT. Eff.=target/non-target efficacy, "
                                  f"Neff=effective independent bets.", f"Bcorum_{src}") if bc_rows else "",
            bstr_latex=_latex_table("base + op", _B_SHORT_COLS, bstr_rows,
                                    f"Section B, evaluated on held-out STRING ({src} non-target "
                                    f"population): same columns; the hedge selects on CORUM, so gains "
                                    f"here show generalization.", f"Bstring_{src}") if bstr_rows else "",
            barplot=_diversity_barplot(group_counts, src, fig_dir),
            auc_cols=auc_cols, auc_rows=auc_rows, hv_curve=_hv_round_curve(rounds, src, fig_dir),
            auc_latex=_latex_table("method", auc_cols, auc_rows,
                                   f"AUC of hypervolume over rounds ({src} non-target population): "
                                   f"revealed-front (exploration) and nominated-set (decision) senses.",
                                   f"auchv_{src}") if auc_rows else "",
            assayed_img=assayed_img, asy_cols=asy_cols, asy_rows=asy_rows,
            rounds_nom=rounds_nom, rounds_assayed=rounds_assayed,
            rz_cols=rz_cols, rz_rows=rz_rows,
            rz_latex=_latex_table("method", rz_cols, rz_rows,
                                  f"Realizable target efficacy of the nominated set per AL round "
                                  f"({src} non-target population): permissible efficacy $\\times$ "
                                  f"yield $=$ (mean efficacy of safe nominees) $\\times$ (\\# safe) "
                                  f"$/ K$, i.e. mean efficacy over the $K$ picks with non-permissible "
                                  f"picks scored 0. Last column is the round-0 to final gain. "
                                  f"Cells mean $\\pm$ 95\\% CI over lines $\\times$ seeds.",
                                  f"rzround_{src}") if rz_rows else "",
            rr_cols=rr_cols, rr_rows=rr_rows,
            rr_latex=_latex_table("method", rr_cols, rr_rows,
                                  f"Change from round 0 in the nominated set's selectivity ratio "
                                  f"(target / non-target efficacy) per AL round ({src} non-target "
                                  f"population): ratio $=$ mean target efficacy / max(mean non-target "
                                  f"efficacy, 0.05), paired per line $\\times$ seed so every method "
                                  f"starts at 0. Larger $=$ acquisition improves selectivity more; "
                                  f"cells mean $\\pm$ 95\\% CI over lines $\\times$ seeds.",
                                  f"rrround_{src}") if rr_rows else "",
            filter_timing=_filter_timing_curve(rounds, src, fig_dir),
            ad_cols=ad_cols, ad_rows=ad_rows,
            ad_latex=_latex_table("method", ad_cols, ad_rows,
                                  f"Diagnostics ({src} non-target efficacy).", f"Adiag_{src}"),
            a_latex=_latex_table("method", a_cols, a_rows,
                                 f"Target vs Non-Target Efficacy ({src} non-target efficacy); cells are mean $\\pm$ 95\\% CI over cell lines $\\times$ seeds. \\emph{{Realizable target efficacy}} (mean over the $K$ nominees of efficacy with each non-permissible nominee scored 0 -- equivalently the total efficacy of the safe picks divided by $K$) and \\emph{{\\# safe}} (count of the $K$ nominees with non-target efficacy at or below the $\\tau$ ceiling) are defined only when a non-target efficacy threshold is known, and quantify the threshold-known regime; \\emph{{target efficacy}} and \\emph{{non-target efficacy}} require no threshold.", f"A_{src}"),
            asy_latex=_latex_table("acquisition", asy_cols, asy_rows,
                                   f"Assayed-set quality ({src} non-target efficacy).", f"assayed_{src}")
                       if asy_rows else "",
            kdpp_sim=kdpp_sim,
            b_bar=_b_bar(dB_eff, src, fig_dir), b_eff_conc=_b_eff_conc(dB_eff, src, fig_dir),
            failure_sim=_failure_curve(dB_eff, src, fig_dir))
        facets.append({"block": block, "headline_a": _headline_A(dA)})

    dataset = _dataset_section(scatter, tau, tau_mode, tox_sources, fig_dir)
    cv_cols, cv_rows = _cv_section(cv, fig_dir)
    cl_rows, cl_latex = _celllines_table(meta)
    html = Environment(loader=BaseLoader()).from_string(_TEMPLATE).render(
        title=title, meta=meta,
        n_lines=df.cell_line.nunique(), n_seeds=df.seed.nunique(),
        glossary=_GLOSSARY, dataset=dataset, facets=facets,
        cv_cols=cv_cols, cv_rows=cv_rows,
        cl_rows=cl_rows, cl_latex=cl_latex, launch_info=launch_info or {})
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path


if __name__ == "__main__":
    import sys, json
    df = pd.read_parquet(sys.argv[1])
    base = Path(sys.argv[1]).parent
    def _opt(name):
        return pd.read_parquet(base / name) if (base / name).exists() else None
    sc = _opt("scatter.parquet"); asy = _opt("assayed.parquet"); rnd = _opt("rounds.parquet")
    gc = _opt("group_counts.parquet"); cvd = _opt("gp_cv.parquet")
    launch_info = {f.name: f.read_text() for f in
                   [base / "launch.sh", base / "launch_multicontrast.sh"] if f.exists()}
    mp = base / "meta.json"
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    fd = (base / "figs") if (len(sys.argv) > 3 and sys.argv[3] == "--figs") else None
    p = build_ablation_report(df, sys.argv[2] if len(sys.argv) > 2 else "ablation_report.html",
                              scatter=sc, meta=meta, assayed=asy, rounds=rnd, fig_dir=fd,
                              group_counts=gc, cv=cvd)
    print(f"report -> {p}")
