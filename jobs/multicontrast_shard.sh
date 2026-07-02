#!/bin/bash
#SBATCH --job-name=geneal_mc_shard
#SBATCH --partition=braid
#SBATCH --account=braid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
# One multi-contrast shard = one target cell line (array task). Full genome.
# Driven by launch_multicontrast.sh (sets ROOT, CONTRAST, SEEDS).
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export MAMBA_EXE=/cv/home/andera29/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/cv/scratch/u/andera29/micromamba/
eval "$($MAMBA_EXE shell hook --shell bash)"; micromamba activate geneal
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

SEEDS="${SEEDS:-0 1}"; K="${K:-30}"; NINIT="${NINIT:-300}"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ROOT/lines.txt")
echo "mc shard ${SLURM_ARRAY_TASK_ID}: line=$LINE contrasts=$CONTRAST node=$(hostname)"
python -u scripts/run_multicontrast_ehvi.py --panel none \
    --cell-lines "$LINE" --contrast-lines $CONTRAST --seeds $SEEDS --K "$K" \
    --n-initial "$NINIT" --out-root "$ROOT" --run-name "shard_${SLURM_ARRAY_TASK_ID}"
echo "MC SHARD DONE"
