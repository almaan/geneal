#!/bin/bash
#SBATCH --job-name=geneal_mc
#SBATCH --partition=braid
#SBATCH --account=braid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Full-genome multi-contrast EHVI -> writes multicontrast.parquet, copies it into
# the target ablation run dir, and rebuilds that report (adds the 2 MC tables).
# Driven by: SWEEP_DIR (the ablation run to update), CONTRAST_LINES (space-sep).
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export MAMBA_EXE=/cv/home/andera29/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/cv/scratch/u/andera29/micromamba/
eval "$($MAMBA_EXE shell hook --shell bash)"; micromamba activate geneal
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

SWEEP_DIR="${SWEEP_DIR:?set SWEEP_DIR}"
CONTRAST_LINES="${CONTRAST_LINES:?set CONTRAST_LINES}"
SEEDS="${SEEDS:-0 1}"; K="${K:-30}"; NINIT="${NINIT:-300}"; NLINES="${NLINES:-12}"

python -u scripts/run_multicontrast_ehvi.py --panel none --n-cell-lines "$NLINES" \
    --contrast-lines $CONTRAST_LINES --seeds $SEEDS --K "$K" --n-initial "$NINIT" \
    --out-root res/runs_multicontrast --run-name full_genome
cp res/runs_multicontrast/full_genome/multicontrast.parquet "$SWEEP_DIR/multicontrast.parquet"
python -u -m geneal.report.ablation_report "$SWEEP_DIR/ablation.parquet" "$SWEEP_DIR/report.html"
echo "MC DONE -> $SWEEP_DIR/report.html"
