import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy import optimize
from scipy import stats
import pandas as pd
from tabulate import tabulate
import datetime
import math
import argparse
import torch

# Base path to the results directory
base_path = "/u/home/m/mven/project-spellman/168_gpu/results/link_prediction"

# Output directory for plots
output_dir = "scaling_plots"
os.makedirs(output_dir, exist_ok=True)

# Define model variants with their specifications from scaling_exp.sh
transformer_models = {
    "scaling_small_transformer_Transformer": {"hidden_dim": 128, "layers": 4},
    "scaling_medium_transformer_Transformer": {"hidden_dim": 256, "layers": 6},
    "scaling_large_transformer_Transformer": {"hidden_dim": 512, "layers": 8}
}

graph_transformer_models = {
    "scaling_small_graph_transformer_GCN": {"hidden_dim": 128, "layers": 3},
    "scaling_medium_graph_transformer_GCN": {"hidden_dim": 256, "layers": 4},
    "scaling_large_graph_transformer_GCN": {"hidden_dim": 512, "layers": 5}
}

# Color palette with stronger blues for better visibility
blue_colors = {
    "small": "#1f77b4",    # darker blue
    "medium": "#4292c6",   # medium blue (slightly darker than before)
    "large": "#9ecae1"     # light blue (slightly darker than before)
}

# New color palette with reds and oranges for better visibility
orange_red_colors = {
    "small": "#e6550d",    # dark orange
    "medium": "#fd8d3c",   # medium orange
    "large": "#fdbe85"     # light orange/peach
}

# Alternative red palette
red_colors = {
    "small": "#a50f15",    # dark red
    "medium": "#de2d26",   # medium red
    "large": "#fb6a4a"     # light red/coral
}

# Use red_colors as our color scheme
color_palette = red_colors  # Change this line to switch between color palettes

# Function to extract loss values from log file
def extract_train_loss_from_log(log_file):
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return None, None
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Extract epochs and losses
        steps = []
        losses = []
        
        for line in lines:
            # Look for "loss: X.XXXX" pattern
            match = re.search(r'Epoch (\d+).*Step (\d+).*loss: (\d+\.\d+)', line)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                loss = float(match.group(3))
                
                steps.append(step)
                losses.append(loss)
        
        if not steps:
            # Try alternative patterns
            for line in lines:
                match = re.search(r'Step: (\d+), Loss: (\d+\.\d+)', line)
                if match:
                    step = int(match.group(1))
                    loss = float(match.group(2))
                    steps.append(step)
                    losses.append(loss)
        
        if steps:
            return steps, losses
        else:
            print(f"No loss information found in {log_file}")
            return None, None
    
    except Exception as e:
        print(f"Error extracting loss from {log_file}: {e}")
        return None, None

# Function to estimate parameters based on model architecture
def estimate_params(hidden_dim, layers, model_type="transformer"):
    if model_type == "transformer":
        # Transformer: ~4 * h^2 * L
        return 4 * (hidden_dim ** 2) * layers / 1000000  # in millions
    else:  # graph_transformer
        # Graph transformer: ~6 * h^2 * L (includes GNN layers)
        return 6 * (hidden_dim ** 2) * layers / 1000000  # in millions

# Function to calculate FLOPs for a model
def calculate_flops(params, steps, seq_length=100, batch_size=64):
    """
    Calculate approximate FLOPs for transformer forward pass.
    Based on Kaplan et al. scaling laws paper approach.
    """
    # Convert to actual number
    params_actual = params * 1e6
    
    # Estimate FLOPs per forward pass as ~6x parameter count
    flops_per_step = 6 * params_actual * seq_length * batch_size
    
    # Calculate cumulative FLOPs
    return np.array(steps) * flops_per_step

# Function to fit power law (y = a * x^b)
def power_law(x, a, b):
    return a * np.power(x, b)

def fit_power_law(x_data, y_data):
    # Make sure we have positive values for log-log fitting
    valid_indices = [i for i, (x, y) in enumerate(zip(x_data, y_data)) if x > 0 and y > 0]
    x_filtered = [x_data[i] for i in valid_indices]
    y_filtered = [y_data[i] for i in valid_indices]
    
    if len(x_filtered) < 2:
        return None, None, None, None
    
    try:
        # Fit in log space for better numerics
        log_x = np.log(x_filtered)
        log_y = np.log(y_filtered)
        slope, intercept = np.polyfit(log_x, log_y, 1)
        
        a = np.exp(intercept)
        b = slope
        
        # Calculate R-squared
        y_pred = power_law(np.array(x_filtered), a, b)
        log_y_pred = np.log(y_pred)
        ss_total = np.sum((log_y - np.mean(log_y))**2)
        ss_residual = np.sum((log_y - log_y_pred)**2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        
        # Calculate p-value
        n = len(x_filtered)
        p = 2  # Two parameters in the model
        if n > p:
            f_statistic = (ss_total - ss_residual) / p / (ss_residual / (n - p))
            p_value = 1 - stats.f.cdf(f_statistic, p, n - p)
        else:
            p_value = 1.0
        
        return a, b, r_squared, p_value
    
    except Exception as e:
        print(f"Error fitting power law: {e}")
        return None, None, None, None

# Apply stronger smoothing than before
def apply_smoothing(values, window_size=50):
    """Apply moving average smoothing with window size parameter"""
    if len(values) < window_size:
        window_size = max(3, len(values) // 5)  # Use a smaller window if data is small
    
    smoothed = np.zeros_like(values, dtype=float)
    
    # Apply moving average
    for i in range(len(values)):
        start = max(0, i - window_size // 2)
        end = min(len(values), i + window_size // 2 + 1)
        smoothed[i] = np.mean(values[start:end])
    
    return smoothed

# Function to calculate training time from log
def calculate_training_time(log_file):
    """Extract the total training time from the log file."""
    if not os.path.exists(log_file):
        return None
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Try to find training time patterns
        # Pattern 1: "Total training time: X hours Y minutes Z seconds"
        match = re.search(r'Total training time: (\d+) hours (\d+) minutes (\d+) seconds', content)
        if match:
            hours, minutes, seconds = map(int, match.groups())
            return hours * 3600 + minutes * 60 + seconds
            
        # Pattern 2: "Training completed in X seconds"
        match = re.search(r'Training completed in (\d+) seconds', content)
        if match:
            return int(match.group(1))
            
        # Pattern 3: Try to find the start and end timestamps
        start_match = re.search(r'Training started at: ([\d\-: ]+)', content)
        end_match = re.search(r'Training completed at: ([\d\-: ]+)', content)
        
        if start_match and end_match:
            try:
                start_time = datetime.datetime.strptime(start_match.group(1).strip(), '%Y-%m-%d %H:%M:%S')
                end_time = datetime.datetime.strptime(end_match.group(1).strip(), '%Y-%m-%d %H:%M:%S')
                delta = end_time - start_time
                return delta.total_seconds()
            except Exception as e:
                print(f"Error parsing timestamps: {e}")
                
        # If we have steps data, estimate from the step times
        steps_pattern = re.findall(r'Step.*?time: (\d+\.\d+)s', content)
        if steps_pattern:
            step_times = [float(t) for t in steps_pattern]
            return sum(step_times)
            
        return None
    except Exception as e:
        print(f"Error calculating training time: {e}")
        return None

def format_time(seconds):
    """Format seconds into hours:minutes:seconds"""
    if seconds is None:
        return "N/A"
    
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

# For smoothing the loss curves
def smooth(data, window_size=100):
    """
    Simple moving average to smooth out noisy time-series data.
    """
    if window_size < 2:
        return data
    smoothed = []
    cumsum = 0.0
    buffer_list = []
    for i, val in enumerate(data):
        buffer_list.append(val)
        cumsum += val
        if i >= window_size:
            cumsum -= buffer_list.pop(0)
        smoothed.append(cumsum / min(window_size, i+1))
    return np.array(smoothed)

def load_and_process_logs(log_dir_pattern, skip_ratio=0.15, smoothing_window=100):
    """
    Gathers log directories matching the provided pattern, 
    extracts FLOPs, loss, hidden dims, and param counts.
    
    skip_ratio: fraction of data steps to skip (e.g. remove warmup)
    smoothing_window: window size for moving average on the loss curve
    """
    results = []
    log_dirs = glob.glob(log_dir_pattern)
    if not log_dirs:
        print(f"No directories found for pattern: {log_dir_pattern}")
        return None
    
    for dpath in log_dirs:
        # Parse hidden dim from directory name if possible
        if '128' in dpath:
            hidden_dim = 128
        elif '256' in dpath:
            hidden_dim = 256
        elif '512' in dpath:
            hidden_dim = 512
        else:
            hidden_dim = 64  # fallback or parse more thoroughly if needed
        
        # Parse param count from log
        param_file = os.path.join(dpath, 'output.log')
        param_count = None
        if os.path.isfile(param_file):
            with open(param_file, 'r') as f:
                for line in f:
                    if "Model parameters:" in line:
                        try:
                            param_count = int(line.split(':')[-1].strip())
                            break
                        except:
                            pass
        
        flops_file = os.path.join(dpath, 'training_flops.csv')
        loss_file = os.path.join(dpath, 'training_loss.csv')
        if not os.path.isfile(flops_file) or not os.path.isfile(loss_file):
            continue
        
        flops = np.loadtxt(flops_file, delimiter=',')
        loss = np.loadtxt(loss_file, delimiter=',')
        
        # Ensure they're numpy arrays
        flops = np.array(flops)
        loss = np.array(loss)
        
        # Sort by FLOPs (if out of order)
        sort_idx = np.argsort(flops)
        flops = flops[sort_idx]
        loss = loss[sort_idx]
        
        # Skip an initial percentage
        skip_n = int(len(flops) * skip_ratio)
        flops = flops[skip_n:]
        loss = loss[skip_n:]
        
        # Smooth the loss
        loss_smooth = smooth(loss, window_size=smoothing_window)
        
        # Param count in millions
        param_count_m = param_count / 1e6 if param_count else np.nan
        
        results.append({
            'dir': dpath,
            'hidden_dim': hidden_dim,
            'params': param_count_m,
            'flops': flops,
            'loss': loss_smooth
        })
    
    if not results:
        print(f"No valid logs found under pattern: {log_dir_pattern}")
        return None
    
    return results

def plot_scaling_laws(
    data_list,
    title="Transformer Pre-training",
    output_file="transformer_flops_scaling.png",
    separate_legend_file="transformer_legend.png",
    color_list=None
):
    """
    Plot scaling laws on a log-log axis (FLOPs vs. training loss).
    Skips a legend in the main figure to avoid overlap, 
    then saves a separate legend image.
    
    Args:
        data_list: list of dicts with keys 'hidden_dim', 'params', 'flops', 'loss'
        title: plot title
        output_file: main figure's filename
        separate_legend_file: separate legend figure's filename
        color_list: optional list of colors for lines
    """
    if not data_list:
        print("No data to plot.")
        return
    
    # Reds/Oranges for better visibility
    if color_list is None:
        color_list = ["#ff0000", "#ff5e00", "#ff9a00", "#ffce61"]
    
    plt.figure(figsize=(6, 4))
    
    for i, item in enumerate(data_list):
        color = color_list[i % len(color_list)]
        flops = item['flops']
        loss = item['loss']
        hidden_dim = item['hidden_dim']
        params = item['params']
        
        plt.plot(flops, loss, color=color, linewidth=2,
                 label=f"{hidden_dim}d / {params:.1f}M")
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Cumulative Training FLOPs (log scale)")
    plt.ylabel("Training Loss")
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Save main plot (no legend displayed)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    # Build separate legend figure
    fig_legend = plt.figure(figsize=(4, 1))
    ax_legend = fig_legend.add_subplot(111)
    lines = []
    labels = []
    for i, item in enumerate(data_list):
        color = color_list[i % len(color_list)]
        line, = ax_legend.plot([], [], color=color,
                               label=f"{item['hidden_dim']}d / {item['params']:.1f}M")
        lines.append(line)
        labels.append(f"{item['hidden_dim']}d / {item['params']:.1f}M")
    ax_legend.legend(lines, labels, loc='center', frameon=False, ncol=len(data_list))
    ax_legend.axis('off')
    fig_legend.savefig(separate_legend_file, bbox_inches='tight')
    plt.close(fig_legend)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze model scaling and compute requirements')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory containing training logs')
    parser.add_argument('--output-dir', type=str, default='scaling_plots',
                       help='Directory to save scaling analysis outputs')
    parser.add_argument('--model-types', type=str, default='all',
                       help='Comma-separated list of model types to analyze (gnn,cnn,transformer,graph_transformer)')
    parser.add_argument('--skip-ratio', type=float, default=0.15,
                       help='Fraction of initial training steps to skip in analysis')
    parser.add_argument('--smoothing-window', type=int, default=100,
                       help='Window size for smoothing learning curves')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process model types
    if args.model_types.lower() == 'all':
        model_types = ['gnn', 'cnn', 'transformer', 'graph_transformer']
    else:
        model_types = [m.strip() for m in args.model_types.split(',')]
    
    # Generate log directory patterns
    log_patterns = []
    for model in model_types:
        log_patterns.append(f"{args.log_dir}/{model}_*")
    
    # Load and process logs
    data_list = []
    color_list = []
    
    colors = {
        'gnn': '#1f77b4',
        'cnn': '#ff7f0e',
        'transformer': '#2ca02c',
        'graph_transformer': '#d62728'
    }
    
    for pattern, model in zip(log_patterns, model_types):
        model_data = load_and_process_logs(pattern, args.skip_ratio, args.smoothing_window)
        if model_data:
            data_list.extend(model_data)
            color_list.extend([colors.get(model, 'black')] * len(model_data))
    
    # Plot scaling laws
    if data_list:
        plot_scaling_laws(
            data_list,
            title="Model Scaling Analysis",
            output_file=f"{args.output_dir}/flops_scaling.png",
            separate_legend_file=f"{args.output_dir}/legend.png",
            color_list=color_list
        )
    else:
        print("No data found for analysis")

if __name__ == "__main__":
    main()