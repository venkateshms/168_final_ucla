#!/bin/bash
#
# Run a series of experiments for link prediction on GRN data.
# This script trains multiple models (GNN variants, CNN, Transformer, Graph Transformer).
# Logs and results are saved under ./logs and ./results/link_prediction.

# 1. Base path setup
BASE_PATH="$(pwd)"

# 2. Create a timestamped log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG_FILE="$BASE_PATH/experiment_run_${TIMESTAMP}.log"

# 3. Start logging to both console and file
exec > >(tee -a "$MAIN_LOG_FILE") 2>&1

echo "====================================================" 
echo "            EXPERIMENTS SCRIPT START"
echo "====================================================" 
echo "Working in directory: $(pwd)"
echo "All output is being logged to: $MAIN_LOG_FILE"

# 4. Create directories if they don't exist
mkdir -p "$BASE_PATH/models"
mkdir -p "$BASE_PATH/logs"
mkdir -p "$BASE_PATH/results/link_prediction"

# 5. Set common parameters
DATA_DIR="$BASE_PATH/processed_data"
BATCH_SIZE="64"
EPOCHS="30"
PATIENCE="999"         # Effectively disable early stopping
WEIGHT_DECAY="0.01"

# 6. Verify data file presence
if [ ! -f "$DATA_DIR/processed_data.pt" ]; then
    echo "ERROR: Preprocessed data file not found at $DATA_DIR/processed_data.pt"
    exit 1
fi

echo "Verified data file at: $DATA_DIR/processed_data.pt"
ls -lh "$DATA_DIR/processed_data.pt"

########################################
# run_experiment function
########################################
# Arguments:
#   1) MODEL:        (gnn, cnn, transformer, graph_transformer)
#   2) MODEL_TYPE:   (GCN, GAT, GIN, SAGE, CNN, Transformer, etc.)
#   3) LR:           Learning rate
#   4) HIDDEN:       Hidden dimension
#   5) LAYERS:       Number of layers
#   6) DROPOUT:      Dropout rate
#   7) TAG:          Label/tag for the experiment (e.g. 'baseline')
#   8) WARMUP:       Warmup ratio (optional, default 0.10)
#   9) DATA:         Data directory override (optional, default DATA_DIR)
########################################
run_experiment() {
    MODEL=$1
    MODEL_TYPE=$2
    LR=$3
    HIDDEN=$4
    LAYERS=$5
    DROPOUT=$6
    TAG=$7
    WARMUP=${8:-0.10}
    DATA=${9:-"$DATA_DIR"}

    echo "----------------------------------------------------" 
    echo " RUNNING EXPERIMENT: $TAG-$MODEL-$MODEL_TYPE"
    echo "----------------------------------------------------"
    
    # Experiment naming and directories
    EXPERIMENT_NAME="${TAG}_${MODEL}_${MODEL_TYPE}"
    MODEL_DIR="$BASE_PATH/models/${EXPERIMENT_NAME}"
    LOG_DIR="$BASE_PATH/logs/${EXPERIMENT_NAME}"
    RESULT_DIR="$BASE_PATH/results/link_prediction/${EXPERIMENT_NAME}"
    mkdir -p "$MODEL_DIR" "$LOG_DIR" "$RESULT_DIR"
    
    CMD="python train.py \
        --model $MODEL \
        --task link_prediction \
        --data-dir $DATA \
        --model-dir $MODEL_DIR \
        --log-dir $LOG_DIR \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --patience $PATIENCE \
        --weight-decay $WEIGHT_DECAY \
        --hidden-channels $HIDDEN \
        --num-layers $LAYERS \
        --dropout $DROPOUT \
        --log-steps \
        --log-global-steps \
        --lr $LR \
        --warmup-ratio $WARMUP"
    
    # Model-specific args
    if [ "$MODEL" == "gnn" ]; then
        CMD="$CMD --gnn-type $MODEL_TYPE"
    elif [ "$MODEL" == "cnn" ]; then
        CMD="$CMD --embedding-dim $HIDDEN --num-filters $HIDDEN --kernel-size 3"
    elif [ "$MODEL" == "transformer" ] || [ "$MODEL" == "graph_transformer" ]; then
        CMD="$CMD --nhead 8"
    fi
    
    # Logging setup
    LOG_FILE="$RESULT_DIR/output.log"
    echo "Start time: $(date)" > "$LOG_FILE"
    echo "Experiment Name: $EXPERIMENT_NAME" | tee -a "$LOG_FILE"
    echo "Data File: $DATA/processed_data.pt" | tee -a "$LOG_FILE"
    echo "Command: $CMD" | tee -a "$LOG_FILE"
    
    # Run experiment
    echo "STARTING EXPERIMENT - $(date)" | tee -a "$LOG_FILE"
    eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
    EXPERIMENT_STATUS=${PIPESTATUS[0]}
    
    echo "Experiment finished at: $(date)" | tee -a "$LOG_FILE"
    echo "Exit status: $EXPERIMENT_STATUS" | tee -a "$LOG_FILE"
    
    # Find checkpoint
    LATEST_CHECKPOINT=$(find "$MODEL_DIR" -name "*_checkpoint_*.pt" | sort -V | tail -n 1)
    if [ -z "$LATEST_CHECKPOINT" ]; then
        LATEST_CHECKPOINT=$(find "$MODEL_DIR" -name "*.pt" | sort -V | tail -n 1)
    fi
    
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "No model checkpoint found, skipping evaluation." | tee -a "$LOG_FILE"
    else
        echo "Found checkpoint: $LATEST_CHECKPOINT" | tee -a "$LOG_FILE"
        
        # Evaluate threshold
        THRESHOLD_CMD="python evaluate_thresholds.py \
            --model $LATEST_CHECKPOINT \
            --data $DATA/processed_data.pt \
            --output $RESULT_DIR/threshold_eval"
        eval "$THRESHOLD_CMD" 2>&1 | tee -a "$LOG_FILE"
        
        OPTIMAL_THRESHOLD=$(grep "RECOMMENDATION: Use a threshold of" "$LOG_FILE" | tail -n 1 | awk '{print $5}')
        if [ -z "$OPTIMAL_THRESHOLD" ]; then
            OPTIMAL_THRESHOLD="0.2"
            echo "Could not determine threshold; defaulting to $OPTIMAL_THRESHOLD" | tee -a "$LOG_FILE"
        fi
        
        # Evaluate the model
        EVAL_CMD="python evaluate_model.py \
            --model $LATEST_CHECKPOINT \
            --data $DATA/processed_data.pt \
            --output $RESULT_DIR/evaluation \
            --threshold $OPTIMAL_THRESHOLD"
        echo "Evaluating with threshold=$OPTIMAL_THRESHOLD" | tee -a "$LOG_FILE"
        eval "$EVAL_CMD" 2>&1 | tee -a "$LOG_FILE"
        
        EVAL_STATUS=$?
        if [ $EVAL_STATUS -eq 0 ]; then
            echo "Evaluation completed." | tee -a "$LOG_FILE"
            if [ -f "$RESULT_DIR/evaluation/evaluation_results.txt" ]; then
                echo "Test Results:" | tee -a "$LOG_FILE"
                cat "$RESULT_DIR/evaluation/evaluation_results.txt" | tee -a "$LOG_FILE"
            fi
        else
            echo "Evaluation failed with status $EVAL_STATUS" | tee -a "$LOG_FILE"
        fi
    fi
    
    # Completion logging
    if [ $EXPERIMENT_STATUS -eq 0 ]; then
        echo "✅ Experiment success: $EXPERIMENT_NAME"
    else
        echo "❌ Experiment failed: $EXPERIMENT_NAME"
    fi
    
    cp -r "$LOG_DIR"/* "$RESULT_DIR"/ 2>/dev/null || true
    echo "----------------------------------------------------" 
    return $EXPERIMENT_STATUS
}

########################################
# Run multiple experiments
########################################
echo "====================================================" 
echo "           STARTING ALL EXPERIMENTS"
echo "====================================================" 
echo "Patience effectively disabled (patience=$PATIENCE)"

# 1. Baseline GNN models
echo "Running baseline GNN models..."
run_experiment "gnn" "GCN"  "0.0001" "64"  "4" "0.6" "baseline"
run_experiment "gnn" "GAT"  "0.0001" "64"  "4" "0.6" "baseline"
run_experiment "gnn" "SAGE" "0.0001" "64"  "4" "0.6" "baseline"
run_experiment "gnn" "GIN"  "0.0001" "64"  "4" "0.6" "baseline"

# 2. Complex GNN models (deeper)
echo "Running complex GNN models..."
run_experiment "gnn" "GCN" "0.0001" "128" "6" "0.6" "complex"
run_experiment "gnn" "GAT" "0.0001" "128" "6" "0.6" "complex"

# 3. CNN models
echo "Running CNN models..."
run_experiment "cnn" "CNN" "0.0001" "64"  "3" "0.5" "baseline"
run_experiment "cnn" "CNN" "0.0001" "128" "4" "0.5" "complex"

# 4. Transformer model
echo "Running Transformer model..."
run_experiment "transformer" "Transformer" "0.0001" "128" "4" "0.4" "baseline" "0.2"

# 5. Graph Transformer model
echo "Running Graph Transformer model..."
run_experiment "graph_transformer" "GCN" "0.0001" "128" "3" "0.4" "baseline" "0.2"

# 6. Learning rate study (GCN)
echo "Running LR study on GCN..."
run_experiment "gnn" "GCN" "0.001"   "64" "4" "0.6" "lr_high"
run_experiment "gnn" "GCN" "0.0001"  "64" "4" "0.6" "lr_medium"
run_experiment "gnn" "GCN" "0.00001" "64" "4" "0.6" "lr_low"

echo "====================================================" 
echo "           ALL EXPERIMENTS COMPLETED"
echo "====================================================" 
echo "Results: ./results/link_prediction/"
echo "Full log: $MAIN_LOG_FILE"