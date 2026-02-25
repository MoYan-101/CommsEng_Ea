#!/bin/bash
#SBATCH --job-name=MY-CE1
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00

set -euo pipefail

# Default to the submit directory; override with:
#   sbatch --export=ALL,PROJECT_DIR=/path/to/CommsEng ...
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"
mkdir -p logs

# Optional module loading (uncomment and adapt for your cluster).
# module purge
# module load anaconda/2024.02

# Prefer conda/micromamba env; fallback to local .venv.
# Default env for this project.
CONDA_ENV="${CONDA_ENV:-mlcpu}"
ENV_MANAGER=""
MICROMAMBA_BIN=""

# Prefer micromamba first.
if command -v micromamba >/dev/null 2>&1; then
  MICROMAMBA_BIN="$(command -v micromamba)"
elif [ -x "$HOME/bin/micromamba" ]; then
  MICROMAMBA_BIN="$HOME/bin/micromamba"
elif [ -x "$HOME/.local/bin/micromamba" ]; then
  MICROMAMBA_BIN="$HOME/.local/bin/micromamba"
fi

if [ -n "$MICROMAMBA_BIN" ]; then
  eval "$("$MICROMAMBA_BIN" shell hook --shell=bash)"
  if micromamba activate "$CONDA_ENV" >/dev/null 2>&1; then
    ENV_MANAGER="micromamba"
  else
    echo "[WARN] micromamba activate '$CONDA_ENV' failed; fallback to conda/.venv."
  fi
fi

if [ -z "$ENV_MANAGER" ]; then
  # Fallback to conda for non-interactive shells.
  if ! command -v conda >/dev/null 2>&1; then
    for conda_sh in \
      "$HOME/miniconda3/etc/profile.d/conda.sh" \
      "$HOME/anaconda3/etc/profile.d/conda.sh" \
      "/opt/conda/etc/profile.d/conda.sh"
    do
      if [ -f "$conda_sh" ]; then
        # shellcheck source=/dev/null
        source "$conda_sh"
        break
      fi
    done
  fi

  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    ENV_MANAGER="conda"
  elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    ENV_MANAGER=".venv"
  else
    echo "[ERROR] Python env not found: env='$CONDA_ENV' unavailable via micromamba/conda and .venv missing."
    exit 1
  fi
fi
echo "[INFO] ENV_MANAGER=$ENV_MANAGER CONDA_ENV=$CONDA_ENV PYTHON=$(command -v python)"

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
