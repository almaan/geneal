# scripts/merge_llm_report.py
"""Merge a standalone LLM-baseline run (scripts/run_llm_baseline.py) into an
existing ablation report and rebuild report.html IN PLACE.

The GP sweep parquets are NOT overwritten — LLM rows are appended in memory and
the report is re-rendered, so the GP outputs stay pristine and re-runnable. The
LLM methods (llm_nom / llm_loop) appear in every Section-A table/plot because they
were registered in ablation_report (A_ORDER/LABELS/COLORS/_A_FAMILY).

Usage:
  python scripts/merge_llm_report.py --run-dir <sweep_dir> --llm res/runs_llm/<run>/llm.parquet
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from geneal.report.ablation_report import build_ablation_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="sweep dir with ablation/scatter/rounds parquets")
    ap.add_argument("--llm", required=True, help="llm.parquet from run_llm_baseline.py")
    ap.add_argument("--out", default=None, help="report path (default <run-dir>/report.html)")
    args = ap.parse_args()
    rd = Path(args.run_dir)

    ab = pd.read_parquet(rd / "ablation.parquet")
    llm = pd.read_parquet(args.llm)
    meta = json.loads((rd / "meta.json").read_text()) if (rd / "meta.json").exists() else {}
    scatter = pd.read_parquet(rd / "scatter.parquet") if (rd / "scatter.parquet").exists() else None
    rounds = pd.read_parquet(rd / "rounds.parquet") if (rd / "rounds.parquet").exists() else None
    assayed = pd.read_parquet(rd / "assayed.parquet") if (rd / "assayed.parquet").exists() else None
    group_counts = pd.read_parquet(rd / "group_counts.parquet") if (rd / "group_counts.parquet").exists() else None
    cv = pd.read_parquet(rd / "gp_cv.parquet") if (rd / "gp_cv.parquet").exists() else None

    # ---- build ablation-schema rows for the LLM methods -------------------------
    acq_of = {"llm_nom": "random", "llm_loop": "llm"}      # llm_nom hosts on the random 120-set
    llm_rows = llm.copy()
    llm_rows["analysis"] = "A"
    llm_rows["base"] = llm_rows["method"]
    llm_rows["operator"] = "none"
    llm_rows["safety"] = "llm"
    llm_rows["quality"] = "eff"
    llm_rows["acq"] = llm_rows["method"].map(acq_of).fillna("llm")
    for c in ab.columns:
        if c not in llm_rows.columns:
            llm_rows[c] = np.nan
    picks_by = None
    if "picks" in llm_rows.columns:
        picks_by = {(r.cell_line, r.seed, r.tox_source, r.method): list(r.picks)
                    for r in llm.itertuples() if isinstance(getattr(r, "picks", None), (list, np.ndarray))}
    llm_rows = llm_rows[[c for c in ab.columns]]            # exact column order
    merged_ab = pd.concat([ab, llm_rows], ignore_index=True)

    # ---- add pick_<method> columns to scatter (seed-0 picks, per cell_line/src) -
    if scatter is not None and picks_by is not None:
        seed0 = min(int(s) for s in llm.seed.unique())
        for method in sorted(llm.method.unique()):
            col = f"pick_{method}"
            flag = np.zeros(len(scatter), dtype=bool)
            # scatter rows are positional in build order per (cell_line, tox_source)
            for (cl, src), grp in scatter.groupby(["cell_line", "tox_source"]):
                key = (cl, seed0, src, method)
                pk = picks_by.get(key)
                if not pk:
                    continue
                pos = grp.index.to_numpy()
                valid = [p for p in pk if 0 <= p < len(pos)]
                flag[pos[valid]] = True
            scatter[col] = flag

    n_llm = merged_ab.method.isin(llm.method.unique()).sum()
    print(f"merged {n_llm} LLM rows into {len(ab)} ablation rows; methods={sorted(llm.method.unique())}")
    out = Path(args.out) if args.out else (rd / "report.html")
    build_ablation_report(merged_ab, out, scatter=scatter, meta=meta, assayed=assayed,
                          rounds=rounds, group_counts=group_counts, cv=cv)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
