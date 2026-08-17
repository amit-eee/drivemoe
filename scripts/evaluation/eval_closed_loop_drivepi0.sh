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
nvidia-smi | head -30
echo ""

# ============================================
# Config
# ============================================
BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive220
TEAM_AGENT=/home/mah20012/drivemoe/src/agent/team_code/drivepi0_carla_agent.py
TEAM_CONFIG=/home/mah20012/drivemoe/config/eval/DrivePi0/closed_loop.yaml
BASE_CHECKPOINT_ENDPOINT=eval_bench2drive220
PLANNER_TYPE=traj
ALGO=drivepi0

# Your trained checkpoint (override in yaml or command-line)
CKPT_PATH=/home/mah20012/drivemoe/log/train/drive-pi0/2026-07-31_19-31_42/checkpoint/step2640.pt
echo "Checkpoint: $CKPT_PATH"

SAVE_PATH=./eval_bench2drive220_${ALGO}_${PLANNER_TYPE}
export PYTHONPATH=$PYTHONPATH:/home/mah20012/drivemoe
export DRIVEMOE_REPO_DIR=/home/mah20012/drivemoe
export REPO_DIR=/home/mah20012/drivemoe

RUN_DIR="/home/mah20012/drivemoe/logs/carla_$(date +%Y%m%d_%H%M%S)_${ALGO}_${PLANNER_TYPE}"
mkdir -p "$RUN_DIR"
echo "Log folder for this run: $RUN_DIR"
# ============================================
# Environment - CARLA and Bench2Drive
# ============================================
export CARLA_ROOT=/home/mah20012/carla
export SCENARIO_RUNNER_ROOT=/home/mah20012/Bench2Drive/scenario_runner
export LEADERBOARD_ROOT=/home/mah20012/Bench2Drive/leaderboard

# Python path setup
export PYTHONPATH=""
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$SCENARIO_RUNNER_ROOT
export PYTHONPATH=$PYTHONPATH:$LEADERBOARD_ROOT
export PYTHONPATH=$PYTHONPATH:/home/mah20012/Bench2Drive
export PYTHONPATH=$PYTHONPATH:/home/mah20012/drivemoe

# DriveMoE-specific env vars from their scripts
export DRIVEMOE_REPO_DIR=/home/mah20012/drivemoe
export REPO_DIR=/home/mah20012/drivemoe

# CARLA server config
export SAVE_PATH  # will be used by the CARLA agent

echo "=== Environment ==="
echo "CARLA_ROOT: $CARLA_ROOT"
echo "SCENARIO_RUNNER_ROOT: $SCENARIO_RUNNER_ROOT"
echo "LEADERBOARD_ROOT: $LEADERBOARD_ROOT"
echo "DRIVEMOE_REPO_DIR: $DRIVEMOE_REPO_DIR"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# Verify CARLA can be imported
echo "=== Verifying CARLA import ==="
python -c "import carla; print('CARLA imported successfully')" || {
    echo "ERROR: CARLA import failed"
    exit 1
}
echo ""
# ============================================
# Setup task directories
# ============================================
if [ ! -d "${ALGO}_b2d_${PLANNER_TYPE}" ]; then
    mkdir ${ALGO}_b2d_${PLANNER_TYPE}
    echo "Created ${ALGO}_b2d_${PLANNER_TYPE}"
else
    echo "${ALGO}_b2d_${PLANNER_TYPE} already exists"
fi

# ============================================
# XML splitting (24 sub-routes as designed)
# ============================================
if [ ! -f "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag" ]; then
    echo "Running split_xml.py..."
    TASK_NUM=24
    python tools/split_xml.py $BASE_ROUTES $TASK_NUM $ALGO $PLANNER_TYPE
    touch "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag"
    echo "Splitting complete."
else
    echo "Splitting already done."
fi

# ============================================
# GPU allocation for 3 GPUs, 2 tasks per GPU
# ============================================
# Adapt: 3 GPUs (0, 2, 3) × 2 tasks each = 6 concurrent tasks
# Total 24 sub-routes → 4 sequential batches of 6

# Physical GPU IDs (matches your CUDA_VISIBLE_DEVICES)
PHYSICAL_GPUS=(3)

# Physical GPU -> CARLA graphicsadapter mapping
# Unreal Engine graphicsadapter IDs are reversed relative to CUDA physical GPU IDs.
#
# Physical GPU : CARLA graphicsadapter
#      0       : 3
#      1       : 2
#      2       : 1
#      3       : 0
declare -A PHYSICALGPU_TO_CARLAADAPTER=(
    [0]=3
    [1]=2
    [2]=1
    [3]=0
)
# ============================================
# Tasks to run
# ============================================
# Failed tasks from the previous run
TASK_LIST=(11)

TOTAL_TASKS=${#TASK_LIST[@]}

TASKS_PER_GPU=1
TASKS_PER_BATCH=$((TASKS_PER_GPU * ${#PHYSICAL_GPUS[@]}))

# Ceiling division so a partial final batch is included
NUM_BATCHES=$(((TOTAL_TASKS + TASKS_PER_BATCH - 1) / TASKS_PER_BATCH))

echo "==============================================="
echo "Tasks to run:      ${TASK_LIST[*]}"
echo "Total tasks:       $TOTAL_TASKS"
echo "Tasks per batch:   $TASKS_PER_BATCH"
echo "Number of batches: $NUM_BATCHES"
echo "==============================================="

# ============================================
# Ctrl+C cleanup for the current batch
# ============================================
interrupt_cleanup() {
    echo ""
    echo "Interrupt received. Stopping current batch..."

    for task_pgid in "${BATCH_PGIDS[@]:-}"; do
        kill -TERM -- "-$task_pgid" 2>/dev/null || true
    done

    sleep 3

    for port in "${BATCH_PORTS[@]:-}"; do
        pkill -TERM -u "$USER" -f "carla-rpc-port=${port}" \
            2>/dev/null || true
    done

    sleep 2

    for port in "${BATCH_PORTS[@]:-}"; do
        pkill -KILL -u "$USER" -f "carla-rpc-port=${port}" \
            2>/dev/null || true
    done

    echo "Remaining matching processes:"
    pgrep -af \
        "CarlaUE4|leaderboard_evaluator|run_evaluation.sh" \
        || echo "None"

    exit 130
}

trap interrupt_cleanup INT TERM

# ============================================
# Run batches sequentially,
# tasks within each batch in parallel
# ============================================
for ((batch=0; batch<NUM_BATCHES; batch++)); do
    echo "==============================================="
    echo "Starting batch $((batch + 1)) of $NUM_BATCHES"
    echo "Time: $(date)"
    echo "==============================================="

    BATCH_PGIDS=()
    BATCH_TASK_IDS=()
    BATCH_PORTS=()
    BATCH_FAILED_TASKS=()

    for ((slot=0; slot<TASKS_PER_BATCH; slot++)); do
        # Position inside TASK_LIST
        task_index=$((batch * TASKS_PER_BATCH + slot))

        # Final batch may contain fewer than six tasks
        if [ "$task_index" -ge "$TOTAL_TASKS" ]; then
            break
        fi

        # Retrieve the actual Bench2Drive task ID
        task_id=${TASK_LIST[$task_index]}

        # Two tasks per physical GPU
        gpu_idx=$((slot / TASKS_PER_GPU))
        physical_gpu=${PHYSICAL_GPUS[$gpu_idx]}
        carla_adapter=${PHYSICALGPU_TO_CARLAADAPTER[$physical_gpu]}

        PORT=$((BASE_PORT + task_id * 150))
        TM_PORT=$((BASE_TM_PORT + task_id * 150))

        ROUTES="${BASE_ROUTES}_${task_id}_${ALGO}_${PLANNER_TYPE}.xml"
        CHECKPOINT_ENDPOINT="${ALGO}_b2d_${PLANNER_TYPE}/${BASE_CHECKPOINT_ENDPOINT}_${task_id}.json"
        TASK_LOG="${RUN_DIR}/task_${task_id}.log"

        echo "-----------------------------------------------"
        echo "Task-list index:    $task_index"
        echo "Bench2Drive task:   $task_id"
        echo "Model physical GPU: $physical_gpu"
        echo "CARLA adapter:      $carla_adapter"
        echo "PORT:               $PORT"
        echo "TM_PORT:            $TM_PORT"
        echo "Routes:             $ROUTES"
        echo "Result JSON:        $CHECKPOINT_ENDPOINT"
        echo "Log:                $TASK_LOG"
        echo "-----------------------------------------------"

        setsid bash -e leaderboard/scripts/run_evaluation.sh \
            "$PORT" \
            "$TM_PORT" \
            "$IS_BENCH2DRIVE" \
            "$ROUTES" \
            "$TEAM_AGENT" \
            "$TEAM_CONFIG" \
            "$CHECKPOINT_ENDPOINT" \
            "$SAVE_PATH" \
            "$PLANNER_TYPE" \
            "$physical_gpu" \
            "$carla_adapter" \
            > "$TASK_LOG" 2>&1 &

        task_pgid=$!

        BATCH_PGIDS+=("$task_pgid")
        BATCH_TASK_IDS+=("$task_id")
        BATCH_PORTS+=("$PORT")

        echo "Task $task_id started with process-group ID $task_pgid"

        sleep 10
    done
    echo ""
    echo "Batch $((batch + 1)) launched ${#BATCH_PGIDS[@]} tasks."
    echo "Waiting for task completion..."

    # Wait for each task separately.
    for ((i=0; i<${#BATCH_PGIDS[@]}; i++)); do
        task_pgid=${BATCH_PGIDS[$i]}
        task_id=${BATCH_TASK_IDS[$i]}

        if wait "$task_pgid"; then
            echo "Task $task_id completed successfully."
        else
            exit_code=$?
            echo "Task $task_id failed with exit code $exit_code."
            BATCH_FAILED_TASKS+=("$task_id")
        fi
    done

    echo ""
    echo "All wrapper processes for batch $batch have exited."
    echo "Checking for residual CARLA processes..."

    sleep 5

    # Clean residual CARLA processes associated with this batch.
    for port in "${BATCH_PORTS[@]}"; do
        if pgrep -af "carla-rpc-port=${port}" >/dev/null; then
            echo "Stopping residual CARLA process on port $port"
            pkill -TERM -u "$USER" -f "carla-rpc-port=${port}" \
                2>/dev/null || true
        fi
    done

    sleep 3

    for port in "${BATCH_PORTS[@]}"; do
        if pgrep -af "carla-rpc-port=${port}" >/dev/null; then
            echo "Force-stopping CARLA process on port $port"
            pkill -KILL -u "$USER" -f "carla-rpc-port=${port}" \
                2>/dev/null || true
        fi
    done

    # Clean any remaining process groups.
    for task_pgid in "${BATCH_PGIDS[@]}"; do
        if kill -0 "$task_pgid" 2>/dev/null; then
            echo "Stopping remaining process group $task_pgid"
            kill -TERM -- "-$task_pgid" 2>/dev/null || true
            sleep 1
            kill -KILL -- "-$task_pgid" 2>/dev/null || true
        fi
    done

    echo ""
    echo "Remaining evaluation processes after batch cleanup:"
    pgrep -af \
        "CarlaUE4|leaderboard_evaluator|run_evaluation.sh" \
        || echo "None"

    if [ ${#BATCH_FAILED_TASKS[@]} -gt 0 ]; then
        echo "Failed tasks in batch $batch:"
        echo "${BATCH_FAILED_TASKS[*]}"
    else
        echo "All tasks in batch $batch completed successfully."
    fi

    echo "Batch $((batch + 1)) completed at $(date)"
done

# ============================================
# End
# ============================================
echo ""
echo "==============================================="
echo "All batches completed: $(date)"
echo "==============================================="