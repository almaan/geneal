#!/bin/bash
#SBATCH --job-name=geneal_abl_shard
#SBATCH --partition=braid
#SBATCH --account=braid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
# One ablation shard = one target cell line (array task). Writes its own
# shard_<id>/ outputs (no report); aggregation builds the combined report.
# Driven by launch_ablation_sweep.sh (sets ROOT, SEEDS, JOINT, CONTRAST).
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export MAMBA_EXE=/cv/home/andera29/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/cv/scratch/u/andera29/micromamba/
eval "$($MAMBA_EXE shell hook --shell bash)"; micromamba activate geneal
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

EMB="${EMB:-data/processed/embeddings/pubmedbert_all.parquet}"
PANEL_A="${PANEL_A:-none}"; PANEL_B="${PANEL_B:-data/processed/depmap/panel_5k.txt}"
SEEDS="${SEEDS:-0 1}"; K="${K:-30}"; ROUNDS="${ROUNDS:-8}"; BATCH="${BATCH:-10}"
NINIT="${NINIT:-40}"; TAU="${TAU:-0.5}"
JOINT_ARG=""; [ -n "${JOINT:-}" ] && JOINT_ARG="--joint-gp"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ROOT/lines.txt")
echo "shard ${SLURM_ARRAY_TASK_ID}: line=$LINE contrast=$CONTRAST node=$(hostname)"
python -u scripts/run_ablation.py \
    --embeddings "$EMB" --panel-a "$PANEL_A" --panel-b "$PANEL_B" $JOINT_ARG --no-report \
    --cell-lines "$LINE" --contrast-line "$CONTRAST" --seeds $SEEDS --K "$K" \
    --n-initial "$NINIT" --n-rounds "$ROUNDS" --batch "$BATCH" --tau "$TAU" \
    --out-root "$ROOT" --run-name "shard_${SLURM_ARRAY_TASK_ID}"
echo "SHARD DONE"
