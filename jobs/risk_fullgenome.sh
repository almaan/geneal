#!/bin/bash
#SBATCH --job-name=geneal_risk_fullgenome
#SBATCH --partition=braid
#SBATCH --account=braid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Full efficacy-toxicity nomination analysis (with the efficacy-vs-toxicity SCATTER
# and tau-frontier) on the FULL GENOME (~18.5k genes). Uses run_risk_nomination
# directly (not the sharded launcher) so the per-gene scatter + tau sweep are
# produced. Single process; let BLAS use the cores.
#   sbatch jobs/risk_fullgenome.sh
#   sbatch --export=ALL,NLINES=12,SEEDS="0 1 2 3" jobs/risk_fullgenome.sh

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export MAMBA_EXE=/cv/home/andera29/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/cv/scratch/u/andera29/micromamba/
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate geneal
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

NLINES="${NLINES:-8}"; SEEDS="${SEEDS:-0 1 2}"; K="${K:-30}"
echo "node=$(hostname) cpus=${SLURM_CPUS_PER_TASK} job=${SLURM_JOB_ID}"
python -c "import geneal" || { echo "geneal import failed"; exit 1; }
echo "full-genome efficacy-toxicity analysis: ${NLINES} lines x [${SEEDS}] seeds"

# No --panel => full ~18.5k-gene genome from the genome-wide PubMedBERT cache.
python scripts/run_risk_nomination.py \
    --embeddings data/processed/embeddings/pubmedbert_all.parquet \
    --n-cell-lines "$NLINES" --seeds $SEEDS --K "$K" \
    --tau 0.5 --caps 2 3 --taus 0.2 0.4 0.6 0.8 1.0 \
    --out-root res/runs_risk_fullgenome
echo "DONE"
