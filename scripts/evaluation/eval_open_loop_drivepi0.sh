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
echo "Conda:    $CONDA_DEFAULT_ENV"
echo "==============================================="

# ============================================
# GPU check
# ============================================
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    NUM_GPU=1
    echo "Warning: CUDA_VISIBLE_DEVICES not set. Defaulting to 1 GPU."
fi

echo "NUM_GPU=$NUM_GPU"
nvidia-smi | head -30
echo ""

# ============================================
# Config
# ============================================
CKPT_PATH=/home/mah20012/drivemoe/log/train/drive-pi0/2026-07-31_19-31_42/checkpoint/step2640.pt

echo "Checkpoint: $CKPT_PATH"
echo ""

# ============================================
# Environment
# ============================================
export PYTHONPATH="${PWD}"

# ============================================
# Evaluation
# ============================================
echo "==============================================="
echo "Launching open-loop evaluation..."
echo "==============================================="

HYDRA_FULL_ERROR=1 torchrun \
    --nproc_per_node=$NUM_GPU \
    --standalone \
    scripts/run.py \
    --config-name=open_loop \
    --config-path=../config/eval/DrivePi0 \
    checkpoint_path=${CKPT_PATH} \
    data.work_dir=${PWD}/exp/b2d_action \
    data.statistics_path=${PWD}/config/statistics/b2d_statistics.json \
    log_dir=${PWD}/log/eval/drive-pi0/$(date +%Y-%m-%d_%H-%M-%S)

# ============================================
# End
# ============================================
echo ""
echo "==============================================="
echo "Completed: $(date)"
echo "==============================================="