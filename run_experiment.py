import os
import argparse
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

def run_preprocess(args):
    """
    Run preprocessing step
    """
    print("\n=== Running Preprocessing ===")
    preprocess_cmd = [
        "python", "preprocess.py",
        "--input", args.input,
        "--output", args.output_dir,
        "--val-ratio", str(args.val_ratio),
        "--test-ratio", str(args.test_ratio)
    ]
    
    subprocess.run(preprocess_cmd)
    print("Preprocessing completed.")

def run_model(model_type, args):
    """
    Run training for a specific model
    """
    print(f"\n=== Training {model_type} Model ===")
    
    # Create base command
    train_cmd = [
        "python", "train.py",
        "--data-dir", args.output_dir,
        "--model", model_type,
        "--hidden-dim", str(args.hidden_dim),
        "--task", args.task,
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--patience", str(args.patience),
        "--log-dir", os.path.join(args.log_dir, model_type),
        "--model-dir", os.path.join(args.model_dir, model_type)
    ]
    
    # Add model-specific arguments
    if model_type == "gnn":
        train_cmd.extend([
            "--num-layers", str(args.gnn_layers),
            "--dropout", str(args.gnn_dropout),
            "--gnn-type", args.gnn_type
        ])
    elif model_type == "cnn":
        train_cmd.extend([
            "--dropout", str(args.cnn_dropout)
        ])
    elif model_type == "transformer":
        train_cmd.extend([
            "--num-layers", str(args.transformer_layers),
            "--dropout", str(args.transformer_dropout),
            "--nhead", str(args.transformer_heads)
        ])
    elif model_type == "graph_transformer":
        train_cmd.extend([
            "--gnn-layers", str(args.graph_transformer_gnn_layers),
            "--transformer-layers", str(args.graph_transformer_transformer_layers),
            "--dropout", str(args.graph_transformer_dropout),
            "--nhead", str(args.graph_transformer_heads),
            "--gnn-type", args.graph_transformer_gnn_type
        ])
    
    # Run training
    subprocess.run(train_cmd)
    print(f"Training {model_type} model completed.")

def collect_results(model_types, args):
    """
    Collect and summarize results from all models
    """
    print("\n=== Collecting Results ===")
    
    # Create results directory
    results_dir = os.path.join(args.log_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Collect metrics
    results = {}
    
    for model_type in model_types:
        log_dir = os.path.join(args.log_dir, model_type)
        
        # Find most recent log file
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        if not log_files:
            print(f"No log files found for {model_type}")
            continue
        
        log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
        most_recent_log = os.path.join(log_dir, log_files[0])
        
        # Extract test metrics
        metrics = {}
        with open(most_recent_log, 'r') as f:
            for line in f:
                if "TEST Detailed" in line:
                    # Extract metrics from the line
                    parts = line.split("TEST Detailed:")[1].strip().split(", ")
                    for part in parts:
                        key, value = part.split("=")
                        metrics[key.strip()] = float(value)
        
        if metrics:
            results[model_type] = metrics
    
    # Save results to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(results_dir, f"results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {json_path}")
    
    return results

def plot_comparison(results, args):
    """
    Plot comparison of different models
    """
    print("\n=== Plotting Comparison ===")
    
    # Define metrics to plot
    if args.task == "link_prediction":
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    else:
        metrics = ['mse', 'rmse', 'mae', 'r2']
    
    available_metrics = set()
    for model_metrics in results.values():
        available_metrics.update(model_metrics.keys())
    
    metrics = [m for m in metrics if m in available_metrics]
    
    # Create dataframe
    df = pd.DataFrame.from_dict(results, orient='index')
    
    # Create plots directory
    plots_dir = os.path.join(args.log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Bar chart for each metric
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        ax = df[metric].plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.title(f'Comparison of {metric.upper()}')
        plt.ylabel(metric.upper())
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        
        # Add values on top of bars
        for i, v in enumerate(df[metric]):
            ax.text(i, v * 1.01, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'comparison_{metric}.png'))
        plt.close()
    
    # Create table of all metrics
    plt.figure(figsize=(12, len(metrics) * 0.5 + 2))
    plt.axis('off')
    table_data = df[metrics].round(4)
    table = plt.table(
        cellText=table_data.values,
        rowLabels=table_data.index,
        colLabels=table_data.columns,
        cellLoc='center',
        loc='center',
        bbox=[0.2, 0.2, 0.7, 0.6]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    plt.title('Model Comparison Summary', fontsize=16, pad=20)
    plt.savefig(os.path.join(plots_dir, 'comparison_table.png'), bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {plots_dir}")

def run_experiment(args):
    """
    Run full experiment pipeline
    """
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Define models to run
    if args.models == 'all':
        model_types = ["gnn", "cnn", "transformer", "graph_transformer"]
    else:
        model_types = args.models.split(',')
    
    print(f"Running experiment with models: {', '.join(model_types)}")
    
    # Run preprocessing
    if args.preprocess:
        run_preprocess(args)
    
    # Train models
    for model_type in model_types:
        run_model(model_type, args)
    
    # Collect and compare results
    results = collect_results(model_types, args)
    if results:
        plot_comparison(results, args)
    
    print("\n=== Experiment Completed ===")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run experiments with different model architectures')
    parser.add_argument('--preprocess', action='store_true',
                       help='Run data preprocessing')
    parser.add_argument('--models', type=str, default='all',
                       help='Comma-separated list of models to run (gnn,cnn,transformer,graph_transformer)')
    parser.add_argument('--data-dir', type=str, default='processed_data',
                       help='Directory containing processed data')
    parser.add_argument('--output-dir', type=str, default='experiment_results',
                       help='Directory to save experiment results')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--collect-only', action='store_true',
                       help='Only collect and plot results without running models')
    args = parser.parse_args()
    
    # Run experiment with parsed arguments
    run_experiment(args)

if __name__ == "__main__":
    main()