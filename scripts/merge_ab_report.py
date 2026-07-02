# scripts/merge_ab_report.py
"""Merge a prior Analysis-A run with a fresh Analysis-B run, then build ONE report.

Use case: Analysis A (Table 1, genome-wide) is expensive and already ran for the
manuscript; only Analysis B (Tables 2 / A.2, the CORUM/STRING hedging) needed to be
re-run on the full genome. This stitches:

  A side  (kept from --a-run): ablation rows analysis=='A', assayed / rounds /
          scatter / gp_cv / multicontrast (all A-only outputs), and the A meta.
  B side  (taken from --b-run): ablation rows analysis=='B' and group_counts
          (the B diversity barplot).

The merged parquet + report land in --out. Nothing is recomputed; this is a pure
concatenation + report build, so the manuscript's Table 1 is byte-identical while
Tables 2/A.2 reflect the new (genome-wide) B run.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def _load_ablation(run: Path) -> pd.DataFrame:
    """Aggregated ablation.parquet for a run; fall back to concatenating shards."""
    top = run / "ablation.parquet"
    if top.exists():
        return pd.read_parquet(top)
    shards = sorted(run.glob("shard_*/ablation.parquet"))
    if not shards:
        raise FileNotFoundError(f"no ablation.parquet or shards under {run}")
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


def _load_opt(run: Path, name: str):
    top = run / name
    if top.exists():
        return pd.read_parquet(top)
    shards = sorted(run.glob(f"shard_*/{name}"))
    if not shards:
        return None
    parts = [pd.read_parquet(s) for s in shards]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-run", required=True, help="prior run dir supplying Analysis A")
    ap.add_argument("--b-run", required=True, help="new run dir supplying Analysis B")
    ap.add_argument("--out", required=True, help="output dir for the merged parquet + report")
    args = ap.parse_args()
    a_run, b_run, out = Path(args.a_run), Path(args.b_run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    a_df = _load_ablation(a_run)
    b_df = _load_ablation(b_run)
    a_rows = a_df[a_df.analysis == "A"].copy()
    b_rows = b_df[b_df.analysis == "B"].copy()
    if not len(a_rows):
        raise ValueError(f"--a-run {a_run} has no analysis=='A' rows")
    if not len(b_rows):
        raise ValueError(f"--b-run {b_run} has no analysis=='B' rows")
    merged = pd.concat([a_rows, b_rows], ignore_index=True)
    merged.to_parquet(out / "ablation.parquet")

    # A-side artifacts kept from the prior run; B barplot from the new run.
    assayed = _load_opt(a_run, "assayed.parquet")
    rounds = _load_opt(a_run, "rounds.parquet")
    scatter = _load_opt(a_run, "scatter.parquet")
    cv = _load_opt(a_run, "gp_cv.parquet")
    multicontrast = _load_opt(a_run, "multicontrast.parquet")
    group_counts = _load_opt(b_run, "group_counts.parquet")
    for name, frame in [("assayed", assayed), ("rounds", rounds), ("scatter", scatter),
                        ("gp_cv", cv), ("group_counts", group_counts),
                        ("multicontrast", multicontrast)]:
        if frame is not None:
            frame.to_parquet(out / f"{name}.parquet")

    # merged meta: A geometry from a-run, B geometry (panel/n_genes) from b-run.
    a_meta = json.loads((a_run / "meta.json").read_text())
    b_meta = json.loads((b_run / "meta.json").read_text())
    meta = dict(a_meta)
    meta["analyses"] = "both"
    meta["panel_b"] = b_meta.get("panel_b", meta.get("panel_b"))
    meta["n_genes_b"] = b_meta.get("n_genes_b", meta.get("n_genes_b"))
    meta["merged_from"] = {"a_run": str(a_run), "b_run": str(b_run)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"merged: A rows={len(a_rows)} (from {a_run.name}), "
          f"B rows={len(b_rows)} (from {b_run.name}) -> {out/'ablation.parquet'}")
    print(f"A panel n_genes_a={meta.get('n_genes_a')}, "
          f"B panel n_genes_b={meta.get('n_genes_b')} ({meta.get('panel_b')})")

    from geneal.report.ablation_report import build_ablation_report
    build_ablation_report(merged, out / "report.html", scatter=scatter, meta=meta,
                          assayed=assayed, rounds=rounds,
                          group_counts=group_counts, cv=cv,
                          multicontrast=multicontrast, fig_dir=None)
    print(f"report -> {out / 'report.html'}")


if __name__ == "__main__":
    main()
