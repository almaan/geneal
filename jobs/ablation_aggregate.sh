#!/bin/bash
#SBATCH --job-name=geneal_abl_agg
#SBATCH --partition=braid
#SBATCH --account=braid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Aggregate all shards under $ROOT into one combined result + report.
# Submitted by launch_ablation_sweep.sh with --dependency=afterok on the array.
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export MAMBA_EXE=/cv/home/andera29/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/cv/scratch/u/andera29/micromamba/
eval "$($MAMBA_EXE shell hook --shell bash)"; micromamba activate geneal
echo "aggregating $ROOT"
python -u scripts/aggregate_ablation.py "$ROOT" ${EXPORTFIGS:+--figs}
echo "AGG DONE"
