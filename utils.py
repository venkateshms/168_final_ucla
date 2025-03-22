import torch
import numpy as np
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, average_precision_score
)
import matplotlib.pyplot as plt
from thop import profile, clever_format
import argparse

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, delta=0, save_path='checkpoint.pt', verbose=False):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

def evaluate_link_prediction(model, pos_edge_index, neg_edge_index, edge_attr=None):
    """
    Evaluate model on link prediction task
    
    Args:
        model: Trained model
        pos_edge_index: Positive edge indices [2, num_pos_edges]
        neg_edge_index: Negative edge indices [2, num_neg_edges]
        edge_attr: Edge attributes (optional)
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    with torch.no_grad():
        # Predict on positive edges
        if edge_attr is not None:
            # If we have edge attributes for positive edges
            pos_attr = edge_attr[:pos_edge_index.size(1)]
            pos_pred = model(pos_edge_index, pos_attr)
        else:
            pos_pred = model(pos_edge_index)
        
        # Predict on negative edges
        neg_pred = model(neg_edge_index)
        
        # Create true labels: 1 for positive edges, 0 for negative edges
        y_true = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))]).cpu().numpy()
        
        # Combine predictions
        y_pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()
        
        # Convert to binary predictions with threshold 0.5
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
        roc_auc = roc_auc_score(y_true, y_pred)
        average_precision = average_precision_score(y_true, y_pred)
        
        # Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall_curve, precision_curve)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'average_precision': average_precision
        }
        
        return metrics

def evaluate_edge_regression(model, edge_index, edge_attr, edge_labels):
    """
    Evaluate model on edge regression task
    
    Args:
        model: Trained model
        edge_index: Edge indices
        edge_attr: Edge attributes
        edge_labels: True edge labels/values
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    with torch.no_grad():
        # Predict edge values
        pred = model(edge_index, edge_attr)
        
        # Convert to numpy for metrics calculation
        y_true = edge_labels.cpu().numpy()
        y_pred = pred.cpu().numpy()
        
        # Calculate metrics
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2
        }
        
        return metrics

def plot_metrics(train_metrics, val_metrics, metric_name, save_path=None):
    """
    Plot training and validation metrics
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics, label=f'Train {metric_name}')
    plt.plot(val_metrics, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Training and Validation {metric_name}')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def count_parameters(model):
    """
    Count number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model, data):
    """
    Estimate FLOPs for a model forward pass with improved error handling
    and more detailed output
    """
    try:
        if hasattr(data, 'x') and hasattr(data, 'edge_index'):
            # For GNN models
            macs, params = profile(model, inputs=(data.x, data.edge_index))
        elif isinstance(data, tuple) and len(data) == 2:
            # For models with edge_index and edge_attr
            edge_index, edge_attr = data
            if edge_attr is not None:
                macs, params = profile(model, inputs=(edge_index, edge_attr))
            else:
                macs, params = profile(model, inputs=(edge_index, None))
        else:
            # For other models
            macs, params = profile(model, inputs=(data,))
        
        # Convert from string format to actual numbers
        if isinstance(macs, str):
            # Handle units (K, M, G, etc.)
            unit_multipliers = {'K': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12}
            for unit, multiplier in unit_multipliers.items():
                if unit in macs:
                    macs = float(macs.replace(unit, '')) * multiplier
                    break
            else:
                # No unit found, try direct conversion
                macs = float(macs)
        
        # Return raw MACs value and params
        return macs, params
    
    except Exception as e:
        print(f"Error estimating FLOPS: {e}")
        # Return reasonable defaults
        return 0, 0

def time_inference(model, data, num_runs=100):
    """
    Measure inference time for a model
    """
    # Warm-up
    for _ in range(10):
        if hasattr(data, 'x') and hasattr(data, 'edge_index'):
            # For GNN models
            _ = model(data.x, data.edge_index)
        elif isinstance(data, tuple) and len(data) == 2:
            # For models with edge_index and edge_attr
            edge_index, edge_attr = data
            _ = model(edge_index, edge_attr)
        else:
            # For other models
            _ = model(data)
    
    # Measure inference time
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            if hasattr(data, 'x') and hasattr(data, 'edge_index'):
                # For GNN models
                _ = model(data.x, data.edge_index)
            elif isinstance(data, tuple) and len(data) == 2:
                # For models with edge_index and edge_attr
                edge_index, edge_attr = data
                _ = model(edge_index, edge_attr)
            else:
                # For other models
                _ = model(data)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time

def log_metrics(metrics, epoch, prefix='', logger=None):
    """
    Log metrics, either to a logger or print to console
    """
    log_str = f"{prefix} Epoch {epoch}: "
    log_str += ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
    
    if logger:
        logger.info(log_str)
    else:
        print(log_str)
    
    return log_str

def log_model_info(model, input_data, logger=None):
    """
    Log model information including parameter count and FLOPs
    with improved reporting
    """
    # Count parameters
    param_count = count_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Estimate FLOPS
    macs, _ = estimate_flops(model, input_data)
    flops = macs * 2  # Approximate FLOPS as 2x MACs
    
    # Measure inference time
    inference_time = time_inference(model, input_data)
    
    # Format values for better readability
    def format_with_si_prefix(value):
        if value >= 1e12:
            return f"{value/1e12:.2f} T"
        elif value >= 1e9:
            return f"{value/1e9:.2f} G"
        elif value >= 1e6:
            return f"{value/1e6:.2f} M"
        elif value >= 1e3:
            return f"{value/1e3:.2f} K"
        else:
            return f"{value:.2f}"
    
    # Prepare info dict with both raw and formatted values
    info = {
        "model_name": model.__class__.__name__,
        "trainable_parameters": param_count,
        "total_parameters": total_params,
        "trainable_parameters_formatted": format_with_si_prefix(param_count),
        "total_parameters_formatted": format_with_si_prefix(total_params),
        "macs_raw": macs,
        "flops_raw": flops,
        "macs_formatted": format_with_si_prefix(macs),
        "flops_formatted": format_with_si_prefix(flops),
        "inference_time_ms": inference_time * 1000
    }
    
    # Create detailed log string
    log_str = [
        f"MODEL INFORMATION SUMMARY:",
        f"  Model: {info['model_name']}",
        f"  Trainable Parameters: {info['trainable_parameters_formatted']} ({info['trainable_parameters']:,})",
        f"  Total Parameters: {info['total_parameters_formatted']} ({info['total_parameters']:,})",
        f"  MACs per forward pass: {info['macs_formatted']}",
        f"  FLOPs per forward pass: {info['flops_formatted']}",
        f"  Inference time: {info['inference_time_ms']:.2f} ms"
    ]
    
    # Print or log the information
    if logger:
        for line in log_str:
            logger.info(line)
    else:
        for line in log_str:
            print(line)
    
    return info

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Utility functions for GRN models')
    parser.add_argument('--test-plotting', action='store_true',
                       help='Test the plotting functionality')
    parser.add_argument('--test-metrics', action='store_true',
                       help='Test the metrics calculation functionality')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model for testing model info functions')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data for testing model info functions')
    parser.add_argument('--output-dir', type=str, default='util_tests',
                       help='Directory to save test outputs')
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.test_plotting or args.test_metrics or args.model_path:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Test plotting functionality
    if args.test_plotting:
        print("Testing plotting functionality...")
        train_metrics = {'loss': [0.5, 0.4, 0.3, 0.25, 0.2], 'accuracy': [0.6, 0.7, 0.75, 0.8, 0.85]}
        val_metrics = {'loss': [0.55, 0.45, 0.35, 0.3, 0.28], 'accuracy': [0.55, 0.65, 0.7, 0.75, 0.78]}
        
        plot_metrics(train_metrics, val_metrics, 'loss', f"{args.output_dir}/test_loss_plot.png")
        plot_metrics(train_metrics, val_metrics, 'accuracy', f"{args.output_dir}/test_accuracy_plot.png")
        print("Plots saved to output directory")
    
    # Test metrics functions
    if args.test_metrics and args.model_path and args.data_path:
        print("Testing model info functions...")
        # This would require loading a model and data, skipping for the template
        print("To test model info, provide both --model-path and --data-path")

if __name__ == "__main__":
    main() 