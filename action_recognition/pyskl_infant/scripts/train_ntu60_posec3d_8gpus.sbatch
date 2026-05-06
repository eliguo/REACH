#!/bin/bash
#SBATCH --job-name=ntu60_posec3d
#SBATCH --partition=a100_short
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

# IMPORTANT: ensure you are using the conda env python, not .venv
# Example:
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyskl

# Go to your pyskl repo root (adjust this!)
cd /gpfs/scratch/yg3030/Video-Based-Infant-Action-Recognition/pyskl

export OMP_NUM_THREADS=1
export MKL_SERVICE_FORCE_INTEL=1
export PYTHONPATH="$(pwd)":$PYTHONPATH

# Master settings for torch.distributed
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((12000 + RANDOM % 20000))

# Total GPUs = nodes * gpus_per_node = 2 * 4 = 8
GPUS_PER_NODE=4

# Use torchrun (more future-proof than torch.distributed.launch)
torchrun \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --nproc_per_node=$GPUS_PER_NODE \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  tools/train.py \
  configs/posec3d/c3d_light_ntu60_xsub/joint.py \
  --launcher pytorch