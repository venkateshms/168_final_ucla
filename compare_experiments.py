#!/usr/bin/env python
"""
Compare experiment results from different model runs.
This script parses log files from different experiments and generates comparison plots.
"""

import os
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict


def parse_log_file(log_file):
    """Parse a log file to extract metrics"""
    results = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_roc_auc': [],
        'test_accuracy': None,
        'test_roc_auc': None,
        'test_pr_auc': None,
    }
    
    # Pattern to match epoch metrics
    epoch_pattern = re.compile(
        r'Epoch (\d+)/\d+ - '
        r'Train Loss: ([0-9.]+), '
        r'Val Loss: ([0-9.]+), '
        r'Val Accuracy: ([0-9.]+), '
        r'Val ROC-AUC: ([0-9.]+)'
    )
    
    # Pattern to match test metrics
    test_accuracy_pattern = re.compile(r'Test Accuracy: ([0-9.]+)')
    test_roc_auc_pattern = re.compile(r'Test ROC-AUC: ([0-9.]+)')
    test_pr_auc_pattern = re.compile(r'Test PR-AUC: ([0-9.]+)')
    
    with open(log_file, 'r') as f:
        for line in f:
            # Match epoch metrics
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                epoch, train_loss, val_loss, val_acc, val_roc = epoch_match.groups()
                results['epochs'].append(int(epoch))
                results['train_loss'].append(float(train_loss))
                results['val_loss'].append(float(val_loss))
                results['val_accuracy'].append(float(val_acc))
                results['val_roc_auc'].append(float(val_roc))
            
            # Match test metrics
            test_acc_match = test_accuracy_pattern.search(line)
            if test_acc_match:
                results['test_accuracy'] = float(test_acc_match.group(1))
                
            test_roc_match = test_roc_auc_pattern.search(line)
            if test_roc_match:
                results['test_roc_auc'] = float(test_roc_match.group(1))
                
            test_pr_match = test_pr_auc_pattern.search(line)
            if test_pr_match:
                results['test_pr_auc'] = float(test_pr_match.group(1))
    
    return results


def extract_experiment_info(filepath):
    """Extract experiment information from filepath"""
    filename = os.path.basename(filepath)
    
    # Extract tag, model, and model type
    pattern = r'([^_]+)_([^_]+)_([^_.]+)'
    match = re.match(pattern, filename)
    
    if match:
        tag, model, model_type = match.groups()
    else:
        tag = "unknown"
        model = "unknown"
        model_type = "unknown"
    
    # Extract task from directory
    task = os.path.basename(os.path.dirname(filepath))
    
    return {
        'tag': tag,
        'model': model,
        'model_type': model_type,
        'task': task,
        'filepath': filepath
    }


def plot_comparison(experiments_data, metric, title, save_path=None):
    """Plot comparison of a specific metric across experiments"""
    plt.figure(figsize=(12, 8))
    
    for exp_name, exp_data in experiments_data.items():
        if metric in exp_data and len(exp_data[metric]) > 0:
            plt.plot(exp_data['epochs'], exp_data[metric], label=exp_name)
    
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def create_summary_table(experiments_data):
    """Create a summary table of test metrics"""
    data = []
    
    for exp_name, exp_data in experiments_data.items():
        exp_info = extract_experiment_info(exp_data['filepath'])
        
        row = {
            'Experiment': exp_name,
            'Task': exp_info['task'],
            'Model': exp_info['model'],
            'Type': exp_info['model_type'],
            'Test Accuracy': exp_data.get('test_accuracy', '-'),
            'Test ROC-AUC': exp_data.get('test_roc_auc', '-'),
            'Test PR-AUC': exp_data.get('test_pr_auc', '-'),
            'Best Val Epoch': exp_data.get('best_val_epoch', '-'),
            'Total Epochs': len(exp_data.get('epochs', [])),
        }
        data.append(row)
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(data)
    return df


def main(args):
    # Find all log files
    if args.task:
        log_pattern = f"experiments/{args.task}/*.log"
    else:
        log_pattern = "experiments/**/*.log"
    
    log_files = glob.glob(log_pattern, recursive=True)
    
    print(f"Found {len(log_files)} log files")
    
    # Parse log files
    experiments_data = {}
    
    for log_file in log_files:
        exp_info = extract_experiment_info(log_file)
        exp_name = f"{exp_info['tag']}_{exp_info['model']}_{exp_info['model_type']}"
        
        print(f"Parsing {exp_name}...")
        exp_data = parse_log_file(log_file)
        exp_data['filepath'] = log_file
        
        experiments_data[exp_name] = exp_data
    
    # Create summary table
    summary = create_summary_table(experiments_data)
    print("\nExperiment Summary:")
    print(summary.to_string(index=False))
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Save summary to CSV
    summary_file = 'results/experiments_summary.csv'
    summary.to_csv(summary_file, index=False)
    print(f"\nSaved summary to {summary_file}")
    
    # Plot metrics
    if args.task:
        task_suffix = f"_{args.task}"
    else:
        task_suffix = "_all"
    
    # Plot training loss
    plot_comparison(
        experiments_data, 
        'train_loss', 
        'Training Loss Comparison',
        f'results/train_loss_comparison{task_suffix}.png'
    )
    
    # Plot validation loss
    plot_comparison(
        experiments_data, 
        'val_loss', 
        'Validation Loss Comparison',
        f'results/val_loss_comparison{task_suffix}.png'
    )
    
    # Plot validation accuracy
    plot_comparison(
        experiments_data, 
        'val_accuracy', 
        'Validation Accuracy Comparison',
        f'results/val_accuracy_comparison{task_suffix}.png'
    )
    
    # Plot validation ROC-AUC
    plot_comparison(
        experiments_data, 
        'val_roc_auc', 
        'Validation ROC-AUC Comparison',
        f'results/val_roc_auc_comparison{task_suffix}.png'
    )
    
    # Create a bar plot for test metrics
    test_metrics = ['test_accuracy', 'test_roc_auc', 'test_pr_auc']
    plt.figure(figsize=(14, 8))
    
    bar_width = 0.25
    index = np.arange(len(experiments_data))
    
    for i, metric in enumerate(test_metrics):
        values = [exp_data.get(metric, 0) for exp_data in experiments_data.values()]
        plt.bar(index + i*bar_width, values, bar_width, label=metric.replace('test_', '').replace('_', ' ').title())
    
    plt.xlabel('Experiment')
    plt.ylabel('Score')
    plt.title('Test Metrics Comparison')
    plt.xticks(index + bar_width, experiments_data.keys(), rotation=45, ha='right')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'results/test_metrics_comparison{task_suffix}.png', dpi=300, bbox_inches='tight')
    
    print("\nAnalysis completed. Results saved to 'results' directory.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare experiment results')
    parser.add_argument('--task', type=str, help='Filter by task (link_prediction or regression)')
    
    args = parser.parse_args()
    main(args) 