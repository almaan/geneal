#!/bin/bash
#SBATCH --job-name=geneal_smoke_bfull
#SBATCH --partition=braid
#SBATCH --account=braid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export MAMBA_EXE=/cv/home/andera29/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/cv/scratch/u/andera29/micromamba/
eval "$($MAMBA_EXE shell hook --shell bash)"; micromamba activate geneal
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
echo "START $(date +%s)"
/usr/bin/time -v python -u scripts/run_ablation.py \
  --embeddings data/processed/embeddings/pubmedbert_all.parquet \
  --panel-a none --panel-b none --joint-gp --no-report \
  --cell-lines ACH-000696 --contrast-line ACH-001360 --seeds 0 \
  --K 30 --n-initial 40 --n-rounds 8 --batch 10 --tau 0.5 \
  --out-root res/runs_ablation/smoke_bfull --run-name shard_0
echo "END $(date +%s)"
