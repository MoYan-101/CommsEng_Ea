#!/bin/bash
#SBATCH --job-name=MY-CE1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00

set -euo pipefail

# Default to the submit directory; override with:
#   sbatch --export=ALL,PROJECT_DIR=/path/to/case_alpha_0_0_colab ...
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"

# Optional module loading (uncomment and adapt for your cluster).
# module purge
# module load anaconda/2024.02

# Prefer conda env; fallback to local .venv.
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  CONDA_ENV="${CONDA_ENV:-comms310}"
  conda activate "$CONDA_ENV"
elif [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

# IMPORTANT: run.sh is interactive if OVERFIT_ALPHA_LIST is unset.
export OVERFIT_ALPHA_LIST="${OVERFIT_ALPHA_LIST:-0.0}"

# Map SLURM resources to your pipeline runtime controls.
CPU_TOTAL="${SLURM_CPUS_PER_TASK:-8}"
export CPU_TOTAL
export MODEL_N_JOBS="${MODEL_N_JOBS:-$CPU_TOTAL}"
export RF_N_JOBS="${RF_N_JOBS:-$CPU_TOTAL}"
export XGB_N_JOBS="${XGB_N_JOBS:-$CPU_TOTAL}"
export CATBOOST_THREAD_COUNT="${CATBOOST_THREAD_COUNT:-$CPU_TOTAL}"
export SVM_N_JOBS="${SVM_N_JOBS:-$CPU_TOTAL}"

# Trial-level parallelism for Optuna (usually less than model threads).
if [ -z "${OPTUNA_N_JOBS:-}" ]; then
  if [ "$CPU_TOTAL" -ge 32 ]; then
    export OPTUNA_N_JOBS=4
  elif [ "$CPU_TOTAL" -ge 16 ]; then
    export OPTUNA_N_JOBS=2
  else
    export OPTUNA_N_JOBS=1
  fi
fi

# Avoid nested BLAS/OpenMP oversubscription.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

# Torch CPU threading.
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$CPU_TOTAL}"
export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-4}"
export PYTHONUNBUFFERED=1

echo "[INFO] JOB_ID=$SLURM_JOB_ID HOST=$(hostname)"
echo "[INFO] PROJECT_DIR=$PROJECT_DIR"
echo "[INFO] OVERFIT_ALPHA_LIST=$OVERFIT_ALPHA_LIST"
echo "[INFO] CPU_TOTAL=$CPU_TOTAL OPTUNA_N_JOBS=$OPTUNA_N_JOBS MODEL_N_JOBS=$MODEL_N_JOBS"

# run.sh executes: train.py -> inference.py -> visualization.py
bash run.sh
