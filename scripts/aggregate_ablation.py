# scripts/aggregate_ablation.py
"""Aggregate per-line ablation shards (from launch_ablation_sweep.sh) into one
combined result + report. Concatenates each shard's ablation/scatter/assayed/
rounds/group_counts/gp_cv parquet and builds the report once over the combined
frames."""
from __future__ import annotations
import sys, glob, json
from pathlib import Path
import pandas as pd


def main():
    root = Path(sys.argv[1])
    parts = sorted(glob.glob(str(root / "shard_*")))
    if not parts:
        print(f"no shards under {root}"); sys.exit(1)

    def concat(name):
        fs = [Path(p) / name for p in parts if (Path(p) / name).exists()]
        return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else None

    df = concat("ablation.parquet")
    if df is None:
        print("no ablation.parquet in shards"); sys.exit(1)
    scatter = concat("scatter.parquet")
    assayed = concat("assayed.parquet")
    rounds = concat("rounds.parquet")
    group_counts = concat("group_counts.parquet")    # diversity barplot
    cv = concat("gp_cv.parquet")                      # method-independent GP-fit CV
    if cv is not None:
        cv = cv.drop_duplicates(subset=["cell_line", "axis", "fold"])
    multicontrast = concat("multicontrast.parquet")   # multi-contrast EHVI (native)
    df.to_parquet(root / "ablation.parquet")
    if scatter is not None: scatter.to_parquet(root / "scatter.parquet")
    if assayed is not None: assayed.to_parquet(root / "assayed.parquet")
    if rounds is not None: rounds.to_parquet(root / "rounds.parquet")
    if group_counts is not None: group_counts.to_parquet(root / "group_counts.parquet")
    if cv is not None: cv.to_parquet(root / "gp_cv.parquet")
    if multicontrast is not None: multicontrast.to_parquet(root / "multicontrast.parquet")

    # meta: take any shard's, fix the lines list to the full set actually present
    metas = sorted(glob.glob(str(root / "shard_*/meta.json")))
    meta = json.loads(Path(metas[0]).read_text()) if metas else {}
    meta["lines"] = sorted(df.cell_line.unique().tolist())
    (root / "meta.json").write_text(json.dumps(meta, indent=2))

    n_lines = df.cell_line.nunique(); n_seeds = df.seed.nunique()
    print(f"aggregated {len(parts)} shards -> {n_lines} lines x {n_seeds} seeds, "
          f"{len(df)} rows")
    from geneal.report.ablation_report import build_ablation_report
    launch_info = {f: (root / f).read_text() for f in ("launch.sh", "launch_multicontrast.sh")
                   if (root / f).exists()}
    build_ablation_report(df, root / "report.html", scatter=scatter, meta=meta,
                          assayed=assayed, rounds=rounds,
                          group_counts=group_counts, cv=cv, multicontrast=multicontrast,
                          launch_info=launch_info,
                          fig_dir=(root / "figs") if len(sys.argv) > 2 and sys.argv[2] == "--figs" else None)
    print(f"report -> {root / 'report.html'}")


if __name__ == "__main__":
    main()
