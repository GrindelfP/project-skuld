#!/bin/bash
#SBATCH --job-name=sechiren_m1
#SBATCH -p ampere
#SBATCH -A gvr
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1

PY_BIN="/zfs/store1.hydra.local/user/g/gshipunv/.conda/envs/jasher-lustre/bin/python"

export PYTHONPATH="${PYTHONPATH}:$(pwd)/research:$(pwd)"

mkdir -p research/results/grid/8-sechiren-sweep-m1

$PY_BIN -u research/experiments/grid/8-sechiren-sweep-m1/super_test_sechiren_m1.py

