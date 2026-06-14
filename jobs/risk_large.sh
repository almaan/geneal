#!/bin/bash
#SBATCH --job-name=geneal_risk_large
#SBATCH --partition=braid
#SBATCH --account=braid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=6G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Full-genome risk-aware nomination sweep (CPU). The launcher fans cell-line
# shards across cores; we pin BLAS to 1 thread/process so the shards parallelize
# cleanly instead of oversubscribing.
#   sbatch jobs/risk_large.sh
#   sbatch --export=ALL,N_LINES=60,N_SHARDS=24 jobs/risk_large.sh

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export MAMBA_EXE=/cv/home/andera29/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/cv/scratch/u/andera29/micromamba/
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate geneal

# one BLAS thread per process -> shards parallelize across the 32 cores
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

N_LINES="${N_LINES:-40}"; N_SEEDS="${N_SEEDS:-6}"; N_SHARDS="${N_SHARDS:-24}"; K="${K:-30}"
echo "node=$(hostname) cpus=${SLURM_CPUS_PER_TASK} job=${SLURM_JOB_ID}"
echo "python: $(which python)"
python -c "import geneal" || { echo "geneal import failed"; exit 1; }
echo "risk-large full-genome: ${N_LINES} lines x ${N_SEEDS} seeds x ${N_SHARDS} shards, K=${K}"

# EMB defaults to the genome-wide PubMedBERT cache (no panel => full ~18.5k genes)
bash scripts/launch_risk_sweep.sh "$N_LINES" "$N_SEEDS" "$N_SHARDS" "$K"
echo "DONE"
