#!/bin/bash
#SBATCH --job-name=MY-CE1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=120:00:00
#SBATCH --array=0-4%4

set -euo pipefail

# Default alpha list (requested):
#   0.0, 0.01, 0.03, 0.05, 0.07
# You may override with ALPHA_LIST (supports comma / semicolon / space separators).
ALPHA_LIST="${ALPHA_LIST:-0.0,0.01,0.03,0.05,0.07}"

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"

# Optional module loading (adapt for your cluster).
# module purge
# module load anaconda/2024.02

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  CONDA_ENV="${CONDA_ENV:-comms310}"
  conda activate "$CONDA_ENV"
elif [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

# Parse alpha list.
ALPHA_CLEAN="$(echo "$ALPHA_LIST" | tr ',;' '  ')"
read -r -a ALPHAS <<< "$ALPHA_CLEAN"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#ALPHAS[@]}" ]; then
  echo "[ERROR] SLURM_ARRAY_TASK_ID=$TASK_ID out of range for ALPHA_LIST='$ALPHA_LIST'"
  echo "[HINT] Adjust --array range to 0-$((${#ALPHAS[@]} - 1))."
  exit 1
fi

ALPHA="${ALPHAS[$TASK_ID]}"
if [ -z "$ALPHA" ]; then
  echo "[ERROR] Empty alpha at index $TASK_ID from ALPHA_LIST='$ALPHA_LIST'"
  exit 1
fi

# Required for concurrent array tasks in the same project directory.
export ALLOW_CONCURRENT_PIPELINE=1
export ALLOW_CONCURRENT_TRAIN=1

# Each task runs one alpha via run.sh.
export OVERFIT_ALPHA_LIST="$ALPHA"

CPU_TOTAL="${SLURM_CPUS_PER_TASK:-8}"
export CPU_TOTAL
export MODEL_N_JOBS="${MODEL_N_JOBS:-$CPU_TOTAL}"
export RF_N_JOBS="${RF_N_JOBS:-$CPU_TOTAL}"
export XGB_N_JOBS="${XGB_N_JOBS:-$CPU_TOTAL}"
export CATBOOST_THREAD_COUNT="${CATBOOST_THREAD_COUNT:-$CPU_TOTAL}"
export SVM_N_JOBS="${SVM_N_JOBS:-$CPU_TOTAL}"

if [ -z "${OPTUNA_N_JOBS:-}" ]; then
  if [ "$CPU_TOTAL" -ge 32 ]; then
    export OPTUNA_N_JOBS=4
  elif [ "$CPU_TOTAL" -ge 16 ]; then
    export OPTUNA_N_JOBS=2
  else
    export OPTUNA_N_JOBS=1
  fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-$CPU_TOTAL}"
export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-4}"
export PYTHONUNBUFFERED=1

echo "[INFO] JOB_ID=$SLURM_JOB_ID ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID HOST=$(hostname)"
echo "[INFO] PROJECT_DIR=$PROJECT_DIR"
echo "[INFO] ALPHA_LIST='$ALPHA_LIST' -> selected alpha=$ALPHA"
echo "[INFO] CPU_TOTAL=$CPU_TOTAL OPTUNA_N_JOBS=$OPTUNA_N_JOBS MODEL_N_JOBS=$MODEL_N_JOBS"

# Optional explicit RUN_ID (usually not needed; run.sh auto-generates).
# export RUN_ID="slurm_${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}_a${ALPHA}"

bash run.sh
