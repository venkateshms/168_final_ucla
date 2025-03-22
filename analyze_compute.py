#!/usr/bin/env python3
"""
analyze_compute.py - Analyzes the relationship between model size, computational complexity,
and performance for link prediction tasks.

This script:
1. Extracts model parameters, FLOPs, and performance metrics from logs
2. Fits scaling laws to the data
3. Plots the relationships between model size and performance
"""

import os
import re
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from scipy.stats import linregress
import argparse
import scipy.stats as stats
from collections import defaultdict

def extract_model_params(log_file):
    """Extract model parameter count from the log file"""
    params = None
    with open(log_file, 'r') as f:
        for line in f:
            if "Model parameters:" in line:
                params = int(line.split("Model parameters:")[1].strip().replace(',', ''))
                break
    return params

def extract_flops(log_file):
    """Extract estimated FLOPs from the log file"""
    flops = None
    with open(log_file, 'r') as f:
        for line in f:
            if "Total estimated training FLOPs:" in line:
                flops_str = line.split("Total estimated training FLOPs:")[1].strip()
                # Handle scientific notation like 1.23e+10
                if 'e' in flops_str:
                    base, exp = flops_str.split('e')
                    flops = float(base) * (10 ** int(exp.replace('+', '')))
                else:
                    flops = float(flops_str)
                break
    return flops

def extract_performance(log_file, task='link_prediction'):
    """Extract model performance metrics from log file"""
    metrics = {}
    with open(log_file, 'r') as f:
        content = f.read()
        
        # Extract test metrics
        if task == 'link_prediction':
            acc_match = re.search(r'Test Accuracy:\s+([\d\.]+)', content)
            roc_match = re.search(r'Test ROC-AUC:\s+([\d\.]+)', content)
            pr_match = re.search(r'Test PR-AUC:\s+([\d\.]+)', content)
            
            if acc_match:
                metrics['accuracy'] = float(acc_match.group(1))
            if roc_match:
                metrics['roc_auc'] = float(roc_match.group(1))
            if pr_match:
                metrics['pr_auc'] = float(pr_match.group(1))
        else:  # regression
            loss_match = re.search(r'Test Loss:\s+([\d\.]+)', content)
            if loss_match:
                metrics['mse'] = float(loss_match.group(1))
    
    return metrics

def extract_model_info(result_dir):
    """Extract all model information from the result directory"""
    log_files = glob.glob(os.path.join(result_dir, '*.log'))
    
    if not log_files:
        return None
    
    log_file = log_files[0]  # Take the first log file
    
    # Extract model type and size from directory name
    dir_name = os.path.basename(result_dir)
    parts = dir_name.split('_')
    
    # Default values
    model_info = {
        'tag': parts[0] if len(parts) > 0 else 'unknown',
        'model': parts[1] if len(parts) > 1 else 'unknown',
        'type': parts[2] if len(parts) > 2 else 'unknown',
        'task': parts[3] if len(parts) > 3 else 'unknown',
    }
    
    # Extract parameters and FLOPs
    model_info['params'] = extract_model_params(log_file)
    model_info['flops'] = extract_flops(log_file)
    
    # Extract performance metrics
    performance = extract_performance(log_file, model_info['task'])
    model_info.update(performance)
    
    return model_info

def fit_scaling_law(data, x_key='params', y_key='accuracy'):
    """Fit power law scaling curve to the data"""
    # Extract x and y values
    x = np.array([d[x_key] for d in data if x_key in d and y_key in d])
    y = np.array([d[y_key] for d in data if x_key in d and y_key in d])
    
    if len(x) < 2:
        return None
    
    # Log transformation for power law fitting
    log_x = np.log(x)
    log_y = np.log(y)
    
    # Linear regression on log-transformed data
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    
    # Power law parameters
    a = np.exp(intercept)
    b = slope
    
    return {
        'a': a,
        'b': b,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }

def plot_flops_performance(data, task='link_prediction', output_dir='plots'):
    """
    Plot model performance against FLOPs with log scale axes
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group data by model type
    models = defaultdict(list)
    for item in data:
        if 'model' in item and 'flops' in item:
            models[item['model']].append(item)
    
    # Create plots
    plt.figure(figsize=(12, 8))
    
    # Define colors and markers for different model types
    colors = {'gnn': 'blue', 'cnn': 'green', 'transformer': 'red', 'graph_transformer': 'purple'}
    markers = {'gnn': 'o', 'cnn': 's', 'transformer': '^', 'graph_transformer': 'D'}
    
    # Select performance metric based on task
    if task == 'link_prediction':
        metrics = [('accuracy', 'Accuracy'), ('roc_auc', 'ROC-AUC'), ('pr_auc', 'PR-AUC')]
    else:  # regression
        metrics = [('mse', 'Mean Squared Error')]
    
    # Create a plot for each metric
    for metric, metric_name in metrics:
        plt.figure(figsize=(12, 8))
        
        for model_type, model_data in models.items():
            # Filter data that has this metric
            filtered_data = [d for d in model_data if metric in d and 'flops' in d]
            
            if filtered_data:
                x = [d['flops'] for d in filtered_data]
                y = [d[metric] for d in filtered_data]
                
                # Get sizes for visualization (based on parameter count)
                sizes = [np.sqrt(d.get('params', 1e5)) for d in filtered_data]
                
                # Plot points
                plt.scatter(x, y, 
                          s=sizes, 
                          color=colors.get(model_type, 'gray'),
                          marker=markers.get(model_type, 'x'),
                          alpha=0.7,
                          label=f"{model_type}")
                
                # Add labels for each point
                for i, d in enumerate(filtered_data):
                    plt.annotate(f"{d['tag']}",
                               (x[i], y[i]),
                               textcoords="offset points",
                               xytext=(0, 5),
                               ha='center',
                               fontsize=8)
        
        # Set plot properties based on metric (log scale for MSE, linear for others)
        plt.xscale('log')
        if metric == 'mse':
            plt.yscale('log')
            plt.title(f"Model {metric_name} vs. Computational Cost (Lower is Better)")
        else:
            plt.title(f"Model {metric_name} vs. Computational Cost (Higher is Better)")
        
        plt.xlabel("Training FLOPs (log scale)")
        plt.ylabel(metric_name + (" (log scale)" if metric == 'mse' else ""))
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"flops_vs_{metric}_{task}.png"), dpi=300)
        plt.close()

def plot_scaling_laws(data, task='link_prediction', output_dir='plots'):
    """Plot scaling laws for model performance vs parameter count"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group data by model type
    models = defaultdict(list)
    for item in data:
        if 'model' in item and 'params' in item:
            models[item['model']].append(item)
    
    # Define performance metrics to plot
    if task == 'link_prediction':
        metrics = [('accuracy', 'Accuracy'), ('roc_auc', 'ROC-AUC'), ('pr_auc', 'PR-AUC')]
    else:  # regression
        metrics = [('mse', 'Mean Squared Error')]
    
    # Create a plot for each metric
    for metric, metric_name in metrics:
        plt.figure(figsize=(12, 8))
        
        # Define colors and markers for different model types
        colors = {'gnn': 'blue', 'cnn': 'green', 'transformer': 'red', 'graph_transformer': 'purple'}
        markers = {'gnn': 'o', 'cnn': 's', 'transformer': '^', 'graph_transformer': 'D'}
        
        for model_type, model_data in models.items():
            # Filter data that has this metric
            filtered_data = [d for d in model_data if metric in d and 'params' in d]
            
            if filtered_data:
                x = [d['params'] for d in filtered_data]
                y = [d[metric] for d in filtered_data]
                
                # Sort by parameter count
                sorted_idx = np.argsort(x)
                x = [x[i] for i in sorted_idx]
                y = [y[i] for i in sorted_idx]
                
                # Get sizes for visualization based on parameter count
                sizes = [np.sqrt(params) for params in x]
                
                # Plot points
                plt.scatter(x, y, 
                          s=sizes, 
                          color=colors.get(model_type, 'gray'),
                          marker=markers.get(model_type, 'x'),
                          alpha=0.7,
                          label=f"{model_type}")
                
                # Add labels for each point
                for i, d in enumerate([filtered_data[i] for i in sorted_idx]):
                    plt.annotate(f"{d['tag']}",
                               (x[i], y[i]),
                               textcoords="offset points",
                               xytext=(0, 5),
                               ha='center',
                               fontsize=8)
                
                # Try to fit scaling law if we have enough points
                if len(x) >= 3:
                    try:
                        # For MSE in regression, we expect performance to improve (decrease) with more parameters
                        # For classification metrics, we expect performance to improve (increase) with more parameters
                        fit_data = filtered_data
                        scaling = fit_scaling_law(fit_data, 'params', metric)
                        
                        if scaling:
                            # Generate points for the fitted curve
                            x_curve = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)
                            
                            if metric == 'mse':
                                # For MSE, we expect a power law where MSE decreases with more parameters
                                # Typically something like MSE ∝ N^(-alpha)
                                y_curve = scaling['a'] * x_curve ** scaling['b']
                            else:
                                # For accuracy, ROC-AUC, etc., we expect improvement with more parameters
                                # Typically something like Acc = 1 - a*N^(-b)
                                # or simpler approximation: Acc ∝ N^b
                                y_curve = scaling['a'] * x_curve ** scaling['b']
                            
                            plt.plot(x_curve, y_curve, '--', color=colors.get(model_type, 'gray'), 
                                   alpha=0.5, linewidth=1)
                            
                            # Add scaling law formula to legend
                            if metric == 'mse':
                                label = f"{model_type} fit: MSE = {scaling['a']:.2e} × N^({scaling['b']:.2f})"
                            else:
                                label = f"{model_type} fit: {metric_name} = {scaling['a']:.2e} × N^({scaling['b']:.2f})"
                            
                            plt.plot([], [], '--', color=colors.get(model_type, 'gray'), label=label)
                    except Exception as e:
                        print(f"Error fitting scaling law for {model_type}: {e}")
        
        # Set plot properties
        plt.xscale('log')
        if metric == 'mse':
            plt.yscale('log')
            plt.title(f"Model {metric_name} vs. Parameter Count (Lower is Better)")
        else:
            plt.title(f"Model {metric_name} vs. Parameter Count (Higher is Better)")
        
        plt.xlabel("Number of Parameters (log scale)")
        plt.ylabel(metric_name + (" (log scale)" if metric == 'mse' else ""))
        plt.legend(loc='best')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"params_vs_{metric}_{task}.png"), dpi=300)
        plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze compute requirements for different models')
    parser.add_argument('--task', type=str, default='link_prediction',
                       help='Task type (link_prediction or regression)')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing model results')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Directory to save analysis outputs')
    parser.add_argument('--include-models', type=str, default='all',
                       help='Comma-separated list of models to include (gnn,cnn,transformer,graph_transformer)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process model types
    if args.include_models.lower() == 'all':
        include_models = None  # Include all models
    else:
        include_models = [m.strip() for m in args.include_models.split(',')]
    
    # Find result directories
    task_dir = os.path.join(args.results_dir, args.task)
    if not os.path.exists(task_dir):
        print(f"Error: Results directory for task '{args.task}' not found at {task_dir}")
        return
    
    # Collect data for all model types
    data = []
    for result_dir in sorted(glob.glob(f"{task_dir}/*")):
        if os.path.isdir(result_dir):
            model_info = extract_model_info(result_dir)
            
            # Filter by model type if specified
            if include_models and model_info.get('model') not in include_models:
                continue
                
            data.append(model_info)
    
    # Plot results
    if data:
        plot_flops_performance(data, args.task, args.output_dir)
        plot_scaling_laws(data, args.task, args.output_dir)
    else:
        print("No data found for analysis")

if __name__ == "__main__":
    main() 