#!/bin/bash
#SBATCH --job-name=geneal_ablation
#SBATCH --partition=braid
#SBATCH --account=braid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# DEFAULT geneal output: the two-analysis ablation (safety-vs-efficacy +
# diversity/robustness). One process runs every (line, seed) sequentially; EHVI
# is the only expensive acquisition, so we let BLAS use all the cores rather than
# sharding. 5k panel by default; full genome via PANEL= empty.
#   sbatch jobs/ablation.sh
#   sbatch --export=ALL,PANEL=,NLINES=8 jobs/ablation.sh         # full genome, 8 lines
#   sbatch --export=ALL,NLINES=8,SEEDS="0 1 2 3" jobs/ablation.sh

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export MAMBA_EXE=/cv/home/andera29/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/cv/scratch/u/andera29/micromamba/
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate geneal
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

EMB="${EMB:-data/processed/embeddings/pubmedbert_all.parquet}"
# PANEL defaults to the 5k panel; set PANEL= (empty) for the full genome.
PANEL="${PANEL-data/processed/depmap/panel_5k.txt}"
PANEL_ARG=""; [ -n "$PANEL" ] && PANEL_ARG="--panel $PANEL"
NLINES="${NLINES:-6}"; SEEDS="${SEEDS:-0 1 2}"; K="${K:-30}"
ROUNDS="${ROUNDS:-8}"; BATCH="${BATCH:-10}"; NINIT="${NINIT:-40}"; TAU="${TAU:-0.5}"
# JOINT=1 -> learned-safety methods use the joint multitask GP. Empty = independent (default).
JOINT_ARG=""; [ -n "${JOINT:-}" ] && JOINT_ARG="--joint-gp"
# EXPORTFIGS=1 -> also write vector PDF+PNG (slow; off by default so the run never stalls).
EXPORT_ARG=""; [ -n "${EXPORTFIGS:-}" ] && EXPORT_ARG="--export-figs"

echo "node=$(hostname) cpus=${SLURM_CPUS_PER_TASK} job=${SLURM_JOB_ID}"
python -c "import geneal" || { echo "geneal import failed"; exit 1; }
echo "ablation: panel='${PANEL:-FULL GENOME}' ${NLINES} lines x [${SEEDS}] seeds, K=${K}, AL ${ROUNDS}x${BATCH}"

python -u scripts/run_ablation.py \
    --embeddings "$EMB" $PANEL_ARG $JOINT_ARG $EXPORT_ARG \
    --n-cell-lines "$NLINES" --seeds $SEEDS --K "$K" \
    --n-initial "$NINIT" --n-rounds "$ROUNDS" --batch "$BATCH" --tau "$TAU" \
    --out-root res/runs_ablation
echo "DONE"
