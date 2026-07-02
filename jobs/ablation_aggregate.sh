#!/bin/bash
#SBATCH --job-name=geneal_abl_agg
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
# MAMBA_EXE / MAMBA_ROOT_PREFIX / GENEAL_ENV arrive via sbatch --export=ALL.
: "${MAMBA_EXE:?MAMBA_EXE not set — source slurm_env.sh before launching}"
eval "$("$MAMBA_EXE" shell hook --shell bash)"; micromamba activate "${GENEAL_ENV:-geneal}"
echo "aggregating $ROOT"
python -u scripts/aggregate_ablation.py "$ROOT" ${EXPORTFIGS:+--figs}
echo "AGG DONE"
