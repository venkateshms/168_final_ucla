#!/bin/bash

# Create directories for experiment results
mkdir -p results/link_prediction
mkdir -p results/regression
mkdir -p models
mkdir -p logs

# Create a timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG_FILE="logs/baseline_run_${TIMESTAMP}.log"

# Start logging all output to the main log file
exec > >(tee -a "${MAIN_LOG_FILE}") 2>&1

echo "==================================================================="
echo "Starting baseline experiments at $(date)"
echo "This run's main log file: ${MAIN_LOG_FILE}"
echo "==================================================================="

# Set common parameters
SOURCE_DATA="SCING_GRN.csv"
BASE_DATA_DIR="processed_data"
SPECTRAL="/u/home/m/mven/project-spellman/168_gpu/processed_data/spectral"  # For link prediction with log normalization
MINMAX_DATA_DIR="${BASE_DATA_DIR}/minmax_normalized_data"  # For regression with minmax normalization

BATCH_SIZE=128
EPOCHS=40
PATIENCE=999  # Disable early stopping
WEIGHT_DECAY=0.01

# Use a consistent learning rate for all models
LR=0.0001


echo "===== Using preprocessed data for experiments ====="

# Log system info
echo "===== System Information ====="
echo "Hostname: $(hostname)"
echo "CPU Info: $(lscpu | grep 'Model name' | sed 's/Model name:[[:space:]]*//')"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "GPU Info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU available')"
echo "Python version: $(python --version)"
echo "Torch version: $(python -c 'import torch; print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")')"

echo "===== Setting up model experiments with global step logging ====="

# Function to run a baseline experiment for both tasks
run_baseline() {
    MODEL=$1
    MODEL_TYPE=$2
    HIDDEN=$3
    LAYERS=$4
    DROPOUT=$5
    EXTRA_ARGS=$6
    TAG=$7
    
    # Loop through both tasks with appropriate data directories
    for TASK in "link_prediction" "regression"; do
        # Select appropriate data directory based on task
        if [ "$TASK" = "link_prediction" ]; then
            DATA_DIR=${SPECTRAL}
        else
            DATA_DIR=${MINMAX_DATA_DIR}
        fi

        # Create a distinctive name for this run
        RUN_NAME="${TAG}_${MODEL}_${MODEL_TYPE}_${TASK}"
        LOG_DIR="logs/${RUN_NAME}"
        RESULT_DIR="results/${TASK}/${RUN_NAME}"
        EXP_LOG_FILE="${RESULT_DIR}/output.log"
        
        mkdir -p ${LOG_DIR}
        mkdir -p ${RESULT_DIR}
        
        echo ""
        echo "===== Running experiment: $RUN_NAME ====="
        echo "Start time: $(date)"
        echo "Log file: ${EXP_LOG_FILE}"
        echo "Using data from: ${DATA_DIR}"
        
        # Build the command
        CMD="python train.py \
            --model $MODEL \
            --task ${TASK} \
            --data-dir $DATA_DIR \
            --batch-size $BATCH_SIZE \
            --epochs $EPOCHS \
            --patience $PATIENCE \
            --lr $LR \
            --weight-decay $WEIGHT_DECAY \
            --hidden-channels $HIDDEN \
            --num-layers $LAYERS \
            --dropout $DROPOUT \
            --model-dir models/${RUN_NAME} \
            --log-dir ${LOG_DIR} \
            --log-steps \
            --log-global-steps \
            ${EXTRA_ARGS}"
        
        # Log the command
        echo "Command: $CMD" | tee ${RESULT_DIR}/command.txt
        
        # Execute the command
        eval "$CMD" 2>&1 | tee ${EXP_LOG_FILE}
        
        # Capture the exit status
        EXIT_STATUS=${PIPESTATUS[0]}
        
        # Copy the best model and logs to results directory
        cp -r ${LOG_DIR}/* ${RESULT_DIR}/
        
        echo "End time: $(date)"
        if [ $EXIT_STATUS -eq 0 ]; then
            echo "Experiment $RUN_NAME completed successfully"
        else
            echo "Experiment $RUN_NAME failed with exit status $EXIT_STATUS"
        fi
        echo "==============================================="
    done
}

# Run GNN models
echo "===== Running GNN models (link prediction and regression) ====="
run_baseline "gnn" "GCN" 64 4 0.6 "--gnn-type GCN" "baseline"
run_baseline "gnn" "GAT" 64 4 0.6 "--gnn-type GAT" "baseline"
run_baseline "gnn" "SAGE" 64 4 0.6 "--gnn-type SAGE" "baseline"

# CNN models
echo "===== Running CNN models (link prediction and regression) ====="
run_baseline "cnn" "CNN" 64 4 0.6 "--embedding-dim 64 --num-filters 64 --kernel-size 3" "baseline"

# Transformer models
echo "===== Running transformer models (link prediction and regression) ====="
run_baseline "transformer" "transformer" 64 4 0.6 "--nhead 8" "baseline"

# Transformer models with different sizes
echo "===== Running transformer models with different sizes (link prediction and regression) ====="

# XSmall transformer
run_baseline "transformer" "transformer" 16 2 0.6 "--nhead 2" "xsmall"
# Small transformer
run_baseline "transformer" "transformer" 32 3 0.6 "--nhead 4" "small"
# Medium transformer
run_baseline "transformer" "transformer" 64 4 0.6 "--nhead 8" "medium"
# Large transformer
run_baseline "transformer" "transformer" 128 6 0.6 "--nhead 8" "large"
# XLarge transformer
run_baseline "transformer" "transformer" 256 8 0.6 "--nhead 16" "xlarge"
# XXLarge transformer
run_baseline "transformer" "transformer" 512 10 0.6 "--nhead 16" "xxlarge"

# Graph Transformer models
echo "===== Running graph transformer models (link prediction and regression) ====="
run_baseline "graph_transformer" "graph_transformer" 64 4 0.6 "--gnn-type GCN --nhead 8 --gnn-layers 2 --transformer-layers 3" "baseline"

# Graph Transformer models with different sizes
echo "===== Running graph transformer models with different sizes (link prediction and regression) ====="

# XSmall graph transformer
run_baseline "graph_transformer" "graph_transformer" 16 2 0.6 "--gnn-type GCN --nhead 2 --gnn-layers 1 --transformer-layers 1" "xsmall"
# Small graph transformer
run_baseline "graph_transformer" "graph_transformer" 32 3 0.6 "--gnn-type GCN --nhead 4 --gnn-layers 1 --transformer-layers 2" "small"
# Medium graph transformer
run_baseline "graph_transformer" "graph_transformer" 64 4 0.6 "--gnn-type GCN --nhead 8 --gnn-layers 2 --transformer-layers 3" "medium"
# Large graph transformer
run_baseline "graph_transformer" "graph_transformer" 128 6 0.6 "--gnn-type GCN --nhead 8 --gnn-layers 2 --transformer-layers 4" "large"
# XLarge graph transformer
run_baseline "graph_transformer" "graph_transformer" 256 8 0.6 "--gnn-type GCN --nhead 16 --gnn-layers 3 --transformer-layers 5" "xlarge"
# XXLarge graph transformer
run_baseline "graph_transformer" "graph_transformer" 512 10 0.6 "--gnn-type GCN --nhead 16 --gnn-layers 4 --transformer-layers 6" "xxlarge"

echo "All baseline experiments completed!"
echo "Results saved in the 'results/link_prediction' and 'results/regression' directories"
echo "To analyze results, run: python analyze_compute.py --task link_prediction"
echo "To analyze regression results, run: python analyze_compute.py --task regression" 