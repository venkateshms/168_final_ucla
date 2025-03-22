#!/bin/bash

# Create a timestamp for the log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="preprocess_log_${TIMESTAMP}.txt"

# Path to the input data file (adjust as needed)
INPUT_FILE="SCING_GRN.csv"

# Output directory
OUTPUT_DIR="processed_data/spectral"

# Create the output directory
mkdir -p "$OUTPUT_DIR"

# Run preprocessing with special focus on class balancing
echo "Running preprocessing with output to $LOG_FILE"
python preprocess.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_DIR" \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --use-node-features \
    --feature-engineering spectral \
    --normalize minmax \
    --edge-weighting continuous \
    --class-balancing \
    --node-embedding-dim 64 \
    2>&1 | tee "$LOG_FILE"

# Check if preprocessing completed successfully
if [ $? -eq 0 ]; then
    echo "Preprocessing completed successfully."
    echo "Log saved to $LOG_FILE"
    
    # Print the class distribution from the processed data
    echo -e "\nAnalyzing processed data class distribution:"
    python -c "
import torch
import numpy as np
data = torch.load('$OUTPUT_DIR/processed_data.pt')
print(f\"Train labels: {data['train_labels'].sum().item()} positive, {len(data['train_labels'])-data['train_labels'].sum().item()} negative\")
print(f\"Val labels: {data['val_labels'].sum().item()} positive, {len(data['val_labels'])-data['val_labels'].sum().item()} negative\")
print(f\"Test labels: {data['test_labels'].sum().item()} positive, {len(data['test_labels'])-data['test_labels'].sum().item()} negative\")
print(f\"Label distribution summary:\")
print(f\"  Train: {data['train_labels'].sum().item()/len(data['train_labels']):.2%} positive\")
print(f\"  Val: {data['val_labels'].sum().item()/len(data['val_labels']):.2%} positive\")
print(f\"  Test: {data['test_labels'].sum().item()/len(data['test_labels']):.2%} positive\")
" 2>&1 | tee -a "$LOG_FILE"
else
    echo "Preprocessing failed. Check $LOG_FILE for errors."
fi

echo "Analysis complete. Check $LOG_FILE for detailed output." 