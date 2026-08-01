#!/bin/bash

# ============================================
# Auto-logging setup
# ============================================
export PYTHONUNBUFFERED=1
mkdir -p logs
SCRIPT_NAME=$(basename "$0" .sh)
LOG_FILE="logs/${SCRIPT_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "==============================================="
echo "Log file: $LOG_FILE"
echo "Started:  $(date)"
echo "Host:     $(hostname)"
echo "User:     $(whoami)"
echo "Conda:    $CONDA_DEFAULT_ENV"
echo "Python:   $(which python)"
echo "==============================================="

# ============================================
# GPU check
# ============================================
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    NUM_GPU=$(nvidia-smi --list-gpus | wc -l)
fi

echo "NUM_GPU=$NUM_GPU"
nvidia-smi | head -30
echo ""

# ============================================
# Environment
# ============================================
export PYTHONPATH="${PWD}"
export WANDB_ENTITY="amiteee12"
export WANDB_PROJECT="drivemoe"
export WANDB_MODE=online

export WANDB_DIR=/data2/mah20012/drivemoe/wandb_logs
export RAY_TMPDIR=/data2/mah20012/drivemoe/ray_tmp
export TMPDIR=/data2/mah20012/tmp
mkdir -p $WANDB_DIR $RAY_TMPDIR $TMPDIR

echo "=== WANDB env check ==="
echo "WANDB_MODE=$WANDB_MODE"
echo "WANDB_DIR=$WANDB_DIR"
echo "WANDB_ENTITY=$WANDB_ENTITY"
echo "WANDB_PROJECT=$WANDB_PROJECT"

# ============================================
# Training
# ============================================
echo "==============================================="
echo "Launching training..."
echo "==============================================="

HYDRA_FULL_ERROR=1 torchrun \
  --nproc_per_node=$NUM_GPU \
  --standalone \
  scripts/run.py \
  --config-path=../config/train/DrivePi0 \
  --config-name=closed_loop

# ============================================
# End
# ============================================
echo ""
echo "==============================================="
echo "Completed: $(date)"
echo "==============================================="