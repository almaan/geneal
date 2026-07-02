#!/bin/bash
#SBATCH --job-name=geneal_mc_agg
#SBATCH --partition=braid
#SBATCH --account=braid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Concatenate the per-line multi-contrast shards, copy into the target ablation
# run dir, and rebuild that report (adds the 2 MC tables). Driven by ROOT, SWEEP_DIR.
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export MAMBA_EXE=/cv/home/andera29/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/cv/scratch/u/andera29/micromamba/
eval "$($MAMBA_EXE shell hook --shell bash)"; micromamba activate geneal

SWEEP_DIR="${SWEEP_DIR:?set SWEEP_DIR}"
python -u - "$ROOT" "$SWEEP_DIR" <<'PY'
import sys, glob
import pandas as pd
root, sweep = sys.argv[1:3]
fs = sorted(glob.glob(f"{root}/shard_*/multicontrast.parquet"))
assert fs, f"no shard parquets under {root}"
df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
df.to_parquet(f"{root}/multicontrast.parquet")
df.to_parquet(f"{sweep}/multicontrast.parquet")
print(f"aggregated {len(fs)} shards -> {df.cell_line.nunique()} lines, {len(df)} rows")
PY
python -u -m geneal.report.ablation_report "$SWEEP_DIR/ablation.parquet" "$SWEEP_DIR/report.html"
echo "MC AGG DONE -> $SWEEP_DIR/report.html"
