#!/usr/bin/env bash
# SLURM + conda config for the sweep launcher. NOT sourced automatically.
#
# Setup:
#   cp slurm_env.template.sh slurm_env.sh     # slurm_env.sh is gitignored
#   # edit slurm_env.sh with your cluster + micromamba paths
#   source slurm_env.sh
#   bash scripts/launch_ablation_sweep.sh 12
#
# launch_ablation_sweep.sh sources slurm_env.sh automatically if present
# (override the path with GENEAL_SLURM_ENV=/path/to/env.sh).

# --- SLURM scheduler ---
export GENEAL_PARTITION="<your_partition>"     # sbatch --partition
export GENEAL_ACCOUNT="<your_account>"         # sbatch --account (leave "" if none)
export GENEAL_SHARD_TIME="04:00:00"            # per-line array task walltime
export GENEAL_SHARD_CPUS="16"                  # cpus-per-task (BLAS threads)
export GENEAL_SHARD_MEM="8G"                   # mem-per-cpu
export GENEAL_AGG_TIME="00:40:00"              # aggregation walltime
export GENEAL_AGG_CPUS="8"
export GENEAL_AGG_MEM="8G"

# --- micromamba / conda env ---
export MAMBA_EXE="<path/to/micromamba>"                 # e.g. $HOME/.local/bin/micromamba
export MAMBA_ROOT_PREFIX="<path/to/micromamba/root>"    # e.g. /scratch/$USER/micromamba
export GENEAL_ENV="geneal"                              # env name (pip install -e .)
