#!/bin/bash
#SBATCH --job-name=skuld_run
#SBATCH -p ampere
#SBATCH -A gvr
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1

#SBATCH --mem=16G
#SBATCH -o /dev/null

if [ -z "$1" ]; then
    echo "Error: Python script not specified!"
    echo "Usage: sbatch research/run_scripts/run.sh path/to/script.py [args...]"
    exit 1
fi

SCRIPT_PATH="$1"

# Load miniconda and initialize conda for non-interactive bash session
module add miniconda
eval "$(conda shell.bash hook)"
conda activate jasher-lustre

PROJECT_ROOT="$(pwd)"
export PYTHONPATH="${PROJECT_ROOT}/research:${PROJECT_ROOT}:${PYTHONPATH}"

LOG_DIR=$(echo "$SCRIPT_PATH" | sed -E 's|research/experiments/|research/results/|' | xargs dirname)
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/slurm-${SLURM_JOB_ID}.log"

exec > >(tee -i "$LOG_FILE") 2>&1

echo "=================================================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Script:     $SCRIPT_PATH"
echo "Arguments:  ${*:2}"
echo "Log file:   $LOG_FILE"
echo "Python:     $(which python)"
echo "=================================================================="

python -u "$@"

