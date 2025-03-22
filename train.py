import os
import torch
import argparse
import logging
import datetime
import numpy as np
import random
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix
from torch import nn, optim
import torch.nn.functional as F

from models import create_model
from utils import count_parameters, estimate_flops, time_inference

def seed_everything(seed):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file):
    """
    Set up logging to file and console
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def log_model_info(model, test_input):
    """
    Log model information including number of parameters and memory usage
    """
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Model Parameters: {num_params:,} (Trainable: {num_trainable_params:,})")
    
    # Estimate memory usage
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    
    # Convert bytes to MB
    param_size_mb = param_size / 1024**2
    buffer_size_mb = buffer_size / 1024**2
    
    logging.info(f"Model Memory: {param_size_mb:.2f} MB (Parameters) + {buffer_size_mb:.2f} MB (Buffers)")
    
    # Log model forward/backward pass time
    try:
        with torch.no_grad():
            # Log forward pass time
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            _ = model(*test_input)
            end_time.record()
            
            torch.cuda.synchronize()
            forward_time = start_time.elapsed_time(end_time)
            
            logging.info(f"Forward Pass Time: {forward_time:.4f} ms")
    except Exception as e:
        logging.warning(f"Could not measure forward/backward pass time: {e}")


def train_epoch(model, optimizer, criterion, edge_index, edge_attr, labels, batch_size, log_steps=False, log_global_steps=False, epoch=0, global_step=0):
    """
    Train the model for one epoch and return the average training loss.
    Also logs step-level metrics if log_steps is True
    Uses global step counting if log_global_steps is True
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    total_correct = 0
    total_samples = 0
    
    # Check data for NaN values first
    if torch.isnan(edge_index).any():
        logging.warning(f"NaN values detected in edge_index during training")
        edge_index = torch.nan_to_num(edge_index)
    
    if edge_attr is not None and torch.isnan(edge_attr).any():
        logging.warning(f"NaN values detected in edge_attr during training")
        edge_attr = torch.nan_to_num(edge_attr)
    
    if torch.isnan(labels).any():
        logging.warning(f"NaN values detected in labels during training")
        labels = torch.nan_to_num(labels)
    
    # Shuffle indices
    indices = torch.randperm(edge_index.size(1))
    n_samples = len(indices)
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        
        # Create batch data
        batch_edge_index = edge_index[:, batch_indices]
        batch_labels = labels[batch_indices]
        
        if edge_attr is not None and edge_attr.shape[0] > 0:
            batch_edge_attr = edge_attr[batch_indices]
        else:
            batch_edge_attr = None
        
        # Forward pass
        optimizer.zero_grad()
        
        try:
            logits = model(batch_edge_index, batch_edge_attr)
            
            # Fix dimensional mismatch: squeeze extra dimensions if necessary
            if logits.dim() > 1 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            
            # Check for NaN values in logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logging.warning(f"NaN or Inf values detected in model outputs during training at epoch {epoch}, step {global_step + (i // batch_size) + 1 if log_global_steps else (i // batch_size) + 1}")
                # Replace NaNs and Infs with safe values
                logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # Handle outputs differently depending on loss function
            if criterion.__class__.__name__ == 'BCEWithLogitsLoss':
                # BCEWithLogitsLoss expects raw logits (no sigmoid)
                loss = criterion(logits, batch_labels)
                # Also store sigmoid of outputs for metrics
                outputs_prob = torch.sigmoid(logits)
            elif criterion.__class__.__name__ == 'BCELoss':
                # Ensure outputs are in the right range for BCE loss
                logits = torch.clamp(logits, 0.0, 1.0)
                loss = criterion(logits, batch_labels)
                outputs_prob = logits  # Already probabilities
            else:
                loss = criterion(logits, batch_labels)
                outputs_prob = logits
            
            # Check for NaN loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logging.warning(f"NaN or Inf loss detected during training at epoch {epoch}, step {global_step + (i // batch_size) + 1 if log_global_steps else (i // batch_size) + 1}")
                # Skip this batch
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            # Calculate batch accuracy for link prediction
            if criterion.__class__.__name__ == 'BCELoss' or criterion.__class__.__name__ == 'BCEWithLogitsLoss':
                # Binary predictions using threshold
                binary_preds = (logits >= 0.5).float()
                n_correct = (binary_preds == batch_labels).sum().item()
                total_correct += n_correct
                total_samples += len(batch_labels)
            
            # Log step-level metrics
            if log_steps:
                current_step = global_step + (i // batch_size) + 1 if log_global_steps else (i // batch_size) + 1
                
                # Calculate step metrics
                if criterion.__class__.__name__ == 'BCELoss' or criterion.__class__.__name__ == 'BCEWithLogitsLoss':
                    # Binary classification metrics
                    batch_accuracy = n_correct / len(batch_labels)
                    logging.info(f"Epoch {epoch}, Step {current_step}, loss: {loss.item():.4f}, accuracy: {batch_accuracy:.4f}")
                else:
                    # Regression metrics
                    mse = F.mse_loss(outputs_prob, batch_labels).item()
                    logging.info(f"Epoch {epoch}, Step {current_step}, loss: {loss.item():.4f}, mse: {mse:.4f}")
        
        except Exception as e:
            logging.error(f"Error during training: {e}")
            # Skip this batch
            continue
    
    # Calculate overall training accuracy
    if criterion.__class__.__name__ == 'BCELoss' or criterion.__class__.__name__ == 'BCEWithLogitsLoss':
        if total_samples > 0:
            train_accuracy = total_correct / total_samples
            logging.info(f"Epoch {epoch} training accuracy: {train_accuracy:.4f} ({total_correct}/{total_samples})")
    
    if log_global_steps:
        # Return the updated global step count
        return total_loss / n_batches if n_batches > 0 else float('inf'), global_step + (n_samples // batch_size) + (1 if n_samples % batch_size > 0 else 0)
    else:
        return total_loss / n_batches if n_batches > 0 else float('inf')


def evaluate(model, criterion, edge_index, edge_attr, labels, batch_size):
    """
    Evaluate the model on the given data
    """
    model.eval()
    all_outputs = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0
    
    # First, check data for issues
    if torch.isnan(edge_index).any():
        logging.warning(f"NaN values detected in edge_index during evaluation")
        edge_index = torch.nan_to_num(edge_index)
    
    if edge_attr is not None and torch.isnan(edge_attr).any():
        logging.warning(f"NaN values detected in edge_attr during evaluation")
        edge_attr = torch.nan_to_num(edge_attr)
    
    if torch.isnan(labels).any():
        logging.warning(f"NaN values detected in labels during evaluation")
        labels = torch.nan_to_num(labels)
    
    with torch.no_grad():
        # Process in batches to avoid OOM
        n_samples = edge_index.size(1)
        
        # Add more detailed logging for debugging
        logging.info(f"Evaluation on {n_samples} samples, batch_size={batch_size}")
        logging.info(f"Label distribution: {labels.sum().item()} positive, {(1 - labels).sum().item()} negative")
        
        for i in range(0, n_samples, batch_size):
            batch_edge_index = edge_index[:, i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            if edge_attr is not None and edge_attr.shape[0] > 0:
                batch_edge_attr = edge_attr[i:i+batch_size]
            else:
                batch_edge_attr = None
            
            # Forward pass
            try:
                outputs = model(batch_edge_index, batch_edge_attr)
                
                # Fix dimensional mismatch: squeeze extra dimensions if necessary
                if outputs.dim() > 1 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
                
                # DIAGNOSTIC: Log statistics about model outputs
                if i == 0:  # Only log for first batch
                    logging.info(f"MODEL OUTPUT DIAGNOSTICS:")
                    logging.info(f"Model type: {model.__class__.__name__}")
                    logging.info(f"Output shape: {outputs.shape}")
                    logging.info(f"Output mean: {outputs.mean().item():.6f}, std: {outputs.std().item():.6f}")
                    logging.info(f"Output min: {outputs.min().item():.6f}, max: {outputs.max().item():.6f}")
                    logging.info(f"Output histogram: 0-0.1: {((outputs >= 0) & (outputs < 0.1)).sum().item()}, " +
                               f"0.1-0.2: {((outputs >= 0.1) & (outputs < 0.2)).sum().item()}, " +
                               f"0.2-0.3: {((outputs >= 0.2) & (outputs < 0.3)).sum().item()}, " +
                               f"0.3-0.4: {((outputs >= 0.3) & (outputs < 0.4)).sum().item()}, " +
                               f"0.4-0.5: {((outputs >= 0.4) & (outputs < 0.5)).sum().item()}, " +
                               f"0.5-0.6: {((outputs >= 0.5) & (outputs < 0.6)).sum().item()}, " +
                               f"0.6-0.7: {((outputs >= 0.6) & (outputs < 0.7)).sum().item()}, " +
                               f"0.7-0.8: {((outputs >= 0.7) & (outputs < 0.8)).sum().item()}, " +
                               f"0.8-0.9: {((outputs >= 0.8) & (outputs < 0.9)).sum().item()}, " +
                               f"0.9-1.0: {((outputs >= 0.9) & (outputs <= 1.0)).sum().item()}")
                    logging.info(f"Label shape: {batch_labels.shape}")
                
                # Check for NaN values
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    logging.warning(f"NaN or Inf values in model outputs: {torch.isnan(outputs).sum()} NaNs, {torch.isinf(outputs).sum()} Infs")
                    outputs = torch.nan_to_num(outputs, nan=0.5, posinf=1.0, neginf=0.0)
                
                # Handle outputs differently depending on loss function
                if criterion.__class__.__name__ == 'BCEWithLogitsLoss':
                    # BCEWithLogitsLoss expects raw logits (no sigmoid)
                    loss = criterion(outputs, batch_labels)
                    # Also store sigmoid of outputs for metrics
                    outputs_prob = torch.sigmoid(outputs)
                elif criterion.__class__.__name__ == 'BCELoss':
                    # Ensure outputs are in the right range for BCE loss
                    outputs = torch.clamp(outputs, 0.0, 1.0)
                    loss = criterion(outputs, batch_labels)
                    outputs_prob = outputs  # Already probabilities
                else:
                    loss = criterion(outputs, batch_labels)
                    outputs_prob = outputs
                
                total_loss += loss.item()
                n_batches += 1
                
                # Store outputs and labels for metrics calculation
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(batch_labels.cpu().numpy())
            
            except Exception as e:
                logging.error(f"Error during evaluation: {e}")
                # Continue with next batch
                continue
    
    # Check if we have any outputs
    if not all_outputs:
        logging.error("No valid outputs during evaluation")
        return {
            'loss': float('inf'),
            'accuracy': 0.0,
            'roc_auc': 0.5,
            'pr_auc': 0.0
        }
    
    # Concatenate all batches
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    
    # Convert logits to probabilities if using BCEWithLogitsLoss
    if criterion.__class__.__name__ == 'BCEWithLogitsLoss':
        # Apply sigmoid to convert logits to probabilities
        logging.info(f"LOGITS DIAGNOSTICS (before sigmoid):")
        logging.info(f"Logits mean: {np.mean(all_outputs):.6f}, std: {np.std(all_outputs):.6f}")
        logging.info(f"Logits min: {np.min(all_outputs):.6f}, max: {np.max(all_outputs):.6f}")
        
        # Apply sigmoid: p = 1/(1+exp(-x))
        all_probs = 1.0 / (1.0 + np.exp(-np.clip(all_outputs, -88.0, 88.0)))  # Clip to avoid overflow
        logging.info(f"PROBABILITY DIAGNOSTICS (after sigmoid):")
        logging.info(f"Probabilities mean: {np.mean(all_probs):.6f}, std: {np.std(all_probs):.6f}")
        logging.info(f"Probabilities min: {np.min(all_probs):.6f}, max: {np.max(all_probs):.6f}")
        all_outputs = all_probs
    else:
        # DIAGNOSTIC: Log statistics about all outputs
        logging.info(f"ALL OUTPUTS DIAGNOSTICS:")
        logging.info(f"All outputs mean: {np.mean(all_outputs):.6f}, std: {np.std(all_outputs):.6f}")
        logging.info(f"All outputs min: {np.min(all_outputs):.6f}, max: {np.max(all_outputs):.6f}")
    
    # Check for NaN values in outputs and handle them
    if np.isnan(all_outputs).any() or np.isinf(all_outputs).any():
        logging.warning(f"NaN or Inf values in final outputs: {np.isnan(all_outputs).sum()} NaNs, {np.isinf(all_outputs).sum()} Infs")
        all_outputs = np.nan_to_num(all_outputs, nan=0.5, posinf=1.0, neginf=0.0)
    
    # Double check that outputs are in valid range 
    all_outputs = np.clip(all_outputs, 0.0, 1.0)
    
    # Calculate metrics
    if criterion.__class__.__name__ == 'BCELoss' or criterion.__class__.__name__ == 'BCEWithLogitsLoss':
        # Binary classification metrics - for link prediction
        # Use stable threshold
        binary_outputs = (all_outputs >= 0.5).astype(int)
        
        # Super verbose debugging of accuracy calculation
        n_correct = np.sum(binary_outputs == all_labels)
        n_total = len(all_labels)
        accuracy = n_correct / n_total
        
        # Detailed breakdown of predictions
        n_true_positive = np.sum((binary_outputs == 1) & (all_labels == 1))
        n_true_negative = np.sum((binary_outputs == 0) & (all_labels == 0))
        n_false_positive = np.sum((binary_outputs == 1) & (all_labels == 0))
        n_false_negative = np.sum((binary_outputs == 0) & (all_labels == 1))
        
        logging.info(f"Accuracy calculation: {n_correct} correct out of {n_total} ({accuracy:.4f})")
        logging.info(f"True Positives: {n_true_positive}")
        logging.info(f"True Negatives: {n_true_negative}")
        logging.info(f"False Positives: {n_false_positive}")
        logging.info(f"False Negatives: {n_false_negative}")
        
        logging.info(f"Label distribution: {np.sum(all_labels == 1)} positive, {np.sum(all_labels == 0)} negative")
        logging.info(f"Prediction distribution: {np.sum(binary_outputs == 1)} positive, {np.sum(binary_outputs == 0)} negative")
        
        # Calculate additional metrics
        try:
            roc_auc = roc_auc_score(all_labels, all_outputs)
            pr_auc = average_precision_score(all_labels, all_outputs)
        except Exception as e:
            logging.warning(f"Error calculating ROC-AUC or PR-AUC: {e}")
            # Fallback values if calculation fails
            roc_auc = 0.5
            pr_auc = np.mean(all_labels)
            
    else:
        # Regression metrics
        accuracy = 0  # Not applicable for regression
        roc_auc = 0   # Not applicable for regression
        pr_auc = 0    # Not applicable for regression
    
    avg_loss = total_loss / n_batches if n_batches > 0 else float('inf')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }


def main(args):
    """
    Main training function.
    """
    # Seed for reproducibility
    seed_everything(args.seed)
    
    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model name
    model_name = f"{args.model}_{args.gnn_type if args.model == 'gnn' else ''}_{args.task}_{timestamp}"
    
    # Create log directory for this run
    run_log_dir = os.path.join(args.log_dir, model_name)
    os.makedirs(run_log_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(run_log_dir, f"{model_name}.log")
    setup_logging(log_file)
    
    # Log arguments
    logging.info(f"Arguments: {args}")
    
    # Load data
    logging.info(f"Loading data from {args.data_dir}")
    data_file = os.path.join(args.data_dir, 'processed_data.pt')
    processed_data = torch.load(data_file)
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Number of nodes
    num_nodes = processed_data['num_nodes']
    logging.info(f"Number of nodes: {num_nodes}")
    
    # Set up model
    logging.info(f"Creating {args.model} model for {args.task} task")
    
    # Pass node features to model if they exist
    model_kwargs = {
        'hidden_channels': args.hidden_channels,
        'dropout': args.dropout,
    }
    
    # Fix for node_features - check if exists and is not None
    if 'node_features' in processed_data and processed_data['node_features'] is not None:
        logging.info(f"Using pre-computed node features of shape {processed_data['node_features'].shape}")
        model_kwargs['node_features'] = processed_data['node_features']
    else:
        logging.info("No pre-computed node features found, model will create its own embeddings")
    
    # Add GNN-specific arguments
    if args.model == 'gnn':
        model_kwargs['gnn_type'] = args.gnn_type
        model_kwargs['num_layers'] = args.num_layers
    
    # Add CNN-specific arguments
    if args.model == 'cnn':
        model_kwargs['embedding_dim'] = args.embedding_dim
        model_kwargs['num_filters'] = args.num_filters
        model_kwargs['kernel_size'] = args.kernel_size
        model_kwargs['num_layers'] = args.num_layers
    
    # Add transformer-specific arguments
    if args.model == 'transformer' or args.model == 'graph_transformer':
        if 'nhead' in args:
            model_kwargs['nhead'] = args.nhead
    
    # Add graph_transformer-specific arguments
    if args.model == 'graph_transformer':
        if 'gnn_layers' in args:
            model_kwargs['gnn_layers'] = args.gnn_layers
        if 'transformer_layers' in args:
            model_kwargs['transformer_layers'] = args.transformer_layers
        if 'gnn_type' in args:
            model_kwargs['gnn_type'] = args.gnn_type
    
    # Create model
    model = create_model(
        model_type=args.model,
        num_nodes=num_nodes,
        task=args.task,
        **model_kwargs
    )
    model = model.to(device)
    
    # Apply custom weight initialization to help training
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # Use Xavier/Glorot initialization for linear layers
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            # Initialize embeddings with small normal values
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    # Apply the initialization
    model.apply(init_weights)
    logging.info("Applied custom weight initialization to model")
    
    # DIAGNOSTIC: Check model parameter initialization
    with torch.no_grad():
        param_stats = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_stats[name] = {
                    'mean': param.mean().item(),
                    'std': param.std().item(),
                    'min': param.min().item(),
                    'max': param.max().item(),
                }
        
        # Log parameter statistics
        logging.info(f"MODEL PARAMETER INITIALIZATION STATISTICS:")
        for name, stats in param_stats.items():
            logging.info(f"  {name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, min={stats['min']:.6f}, max={stats['max']:.6f}")
    
    # Log model architecture
    logging.info(f"Model architecture:\n{model}")
    
    # Set up optimizer with a higher learning rate
    initial_lr = args.lr
    if args.model == 'gnn' and args.task == 'link_prediction':
        # For GNN link prediction, sometimes higher learning rates work better
        initial_lr = max(args.lr, 0.005)
        logging.info(f"Using adjusted learning rate for GNN link prediction: {initial_lr}")
    
    # Set up optimizer - Using AdamW instead of Adam for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=args.weight_decay, amsgrad=True)
    
    # Set up learning rate scheduler with warmup and longer decay
    def get_lr_lambda(current_step, warmup_steps, total_steps):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay after warmup
            return 0.5 * (1.0 + np.cos(np.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))
    
    # Calculate total steps and warmup steps - use longer warmup for better convergence
    steps_per_epoch = (processed_data['train_edge_index'].size(1) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_ratio = max(args.warmup_ratio, 0.15)  # Use at least 15% warmup
    warmup_steps = int(total_steps * warmup_ratio)
    
    logging.info(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps} ({warmup_ratio*100:.1f}%)")
    
    # Create lambda function for LambdaLR scheduler
    lr_lambda = lambda step: get_lr_lambda(step, warmup_steps, total_steps)
    
    # Set up scheduler
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Define loss function based on task
    if args.task == 'link_prediction':
        # Calculate positive sample weight based on actual class distribution in training data
        # Count number of positive and negative samples
        num_pos = processed_data['train_labels'].sum().item()
        num_neg = len(processed_data['train_labels']) - num_pos
        # Calculate weight as ratio of negative to positive samples
        pos_weight_value = num_neg / max(num_pos, 1)  # Avoid division by zero
        
        # Use a higher weight to emphasize learning the positive class
        pos_weight_value = max(pos_weight_value, 3.0)  # Use at least 3.0 as weight
        
        pos_weight = torch.tensor([pos_weight_value]).to(device)
        
        # Use BCEWithLogitsLoss with class weights
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logging.info(f"Using BCEWithLogitsLoss with positive class weight: {pos_weight.item()}")
        logging.info(f"Class distribution in training data: {num_pos} positive, {num_neg} negative")
    else:
        criterion = nn.MSELoss()
    
    # Train model
    logging.info("Starting training with adaptive learning rate monitoring")
    
    # History dictionary to store metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_roc_auc': [],
        'val_pr_auc': [],
        'test_loss': [],
        'test_accuracy': [],
        'test_roc_auc': [],
        'test_pr_auc': [],
        'best_val_accuracy': 0.0,
        'best_val_roc_auc': 0.0,
        'best_val_epoch': 0,
        'model_path': os.path.join(args.model_dir, f"{model_name}.pt"),
        'run_log_dir': run_log_dir,
        'log_steps': args.log_steps,
        'log_global_steps': args.log_global_steps,
        'epochs': [],
        'weight_decay': args.weight_decay,
        'warmup_ratio': args.warmup_ratio,
        'learning_rates': [],
        'flops_per_batch': [],
        'total_flops': []
    }
    
    # Initialize global step counter
    global_step = 0
    
    # Early stopping (but still run all epochs if patience is very high)
    early_stopping_counter = 0
    early_stopping_patience = args.patience
    disable_early_stopping = (early_stopping_patience > 500)  # If patience is very high, we're effectively disabling early stopping
    
    if disable_early_stopping:
        logging.info("Early stopping effectively disabled (patience set very high)")
    
    # Initialize best metric tracking based on task
    if args.task == 'regression':
        # For regression, we want to minimize loss
        best_val_metric = float('inf')
        is_better = lambda current, best: current < best
    else:
        # For link prediction, we want to maximize ROC-AUC
        best_val_metric = 0.0
        is_better = lambda current, best: current > best
    
    # Save initial model to ensure a model file exists
    torch.save(model.state_dict(), history['model_path'])
    logging.info(f"Initial model saved to {history['model_path']}")
    
    # Estimate FLOPS per batch
    try:
        # Sample batch for FLOPS estimation
        sample_size = min(args.batch_size, processed_data['train_edge_index'].size(1))
        sample_indices = torch.randperm(processed_data['train_edge_index'].size(1))[:sample_size]
        
        sample_edge_index = processed_data['train_edge_index'][:, sample_indices].to(device)
        sample_edge_attr = processed_data['test_edge_attr'][:sample_size].to(device) if 'test_edge_attr' in processed_data else None
        
        batch_data = (sample_edge_index, sample_edge_attr)
        
        # Estimate FLOPS
        macs, params = estimate_flops(model, batch_data)
        flops_per_sample = macs * 2  # Approximate FLOPS as 2 * MACs
        flops_per_batch = flops_per_sample * args.batch_size
        
        logging.info(f"Estimated MACs per sample: {macs}")
        logging.info(f"Estimated FLOPS per sample: {flops_per_sample}")
        logging.info(f"Estimated FLOPS per batch: {flops_per_batch}")
        
        history['flops_per_batch'] = flops_per_batch
    except Exception as e:
        logging.error(f"Error estimating FLOPS: {e}")
        history['flops_per_batch'] = None
    
    # Metrics for tracking potential convergence issues
    plateau_counter = 0
    last_val_roc_auc = 0.0
    non_learning_counter = 0
    lr_increase_counter = 0
    
    logging.info("Starting training with adaptive learning rate monitoring")
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        if args.log_global_steps:
            train_loss, global_step = train_epoch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                edge_index=processed_data['train_edge_index'].to(device),
                edge_attr=processed_data['train_edge_attr'].to(device) if 'train_edge_attr' in processed_data and processed_data['train_edge_attr'] is not None else None,
                labels=processed_data['train_labels'].to(device),
                batch_size=args.batch_size,
                log_steps=args.log_steps,
                log_global_steps=args.log_global_steps,
                epoch=epoch,
                global_step=global_step
            )
        else:
            train_loss = train_epoch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                edge_index=processed_data['train_edge_index'].to(device),
                edge_attr=processed_data['train_edge_attr'].to(device) if 'train_edge_attr' in processed_data and processed_data['train_edge_attr'] is not None else None,
                labels=processed_data['train_labels'].to(device),
                batch_size=args.batch_size,
                log_steps=args.log_steps,
                log_global_steps=args.log_global_steps,
                epoch=epoch
            )
        
        # Validate
        model.eval()
        val_metrics = evaluate(
            model=model,
            criterion=criterion,
            edge_index=processed_data['val_edge_index'].to(device),
            edge_attr=processed_data['val_edge_attr'].to(device) if 'val_edge_attr' in processed_data and processed_data['val_edge_attr'] is not None else None,
            labels=processed_data['val_labels'].to(device),
            batch_size=args.batch_size
        )
        
        # Track current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        # Record cumulative FLOPS
        if history['flops_per_batch'] is not None:
            if epoch == 0:
                history['total_flops'] = [history['flops_per_batch'] * (processed_data['train_edge_index'].size(1) // args.batch_size + 1)]
            else:
                prev_flops = history['total_flops'][-1]
                batch_flops = history['flops_per_batch'] * (processed_data['train_edge_index'].size(1) // args.batch_size + 1)
                history['total_flops'].append(prev_flops + batch_flops)
        
        # Check for learning issues - see if ROC-AUC is improving
        current_val_roc_auc = val_metrics['roc_auc']
        roc_auc_improvement = current_val_roc_auc - last_val_roc_auc
        last_val_roc_auc = current_val_roc_auc
        
        # If ROC-AUC is close to 0.5, model might not be learning class differentiation
        if epoch > 3 and abs(current_val_roc_auc - 0.5) < 0.05:
            non_learning_counter += 1
            if non_learning_counter >= 3:
                # Model seems stuck near random performance
                logging.warning(f"Model not learning meaningful class differentiation (ROC-AUC: {current_val_roc_auc:.4f})")
                
                # Increase learning rate to try to escape local minimum
                if lr_increase_counter < 3:  # Only try this a few times
                    old_lr = optimizer.param_groups[0]['lr']
                    new_lr = old_lr * 5.0  # Try a much higher learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    logging.warning(f"Increasing learning rate from {old_lr:.6f} to {new_lr:.6f} to escape local minimum")
                    lr_increase_counter += 1
                    non_learning_counter = 0  # Reset counter
        else:
            non_learning_counter = 0  # Reset if we're learning well
        
        # Check for plateaus in ROC-AUC improvement
        if abs(roc_auc_improvement) < 0.001 and epoch > 5:
            plateau_counter += 1
            if plateau_counter >= 3:
                # We're on a plateau, adjust learning rate
                old_lr = optimizer.param_groups[0]['lr']
                new_lr = old_lr * 0.5  # Try a lower learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                logging.info(f"ROC-AUC plateau detected, reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")
                plateau_counter = 0
        else:
            plateau_counter = 0  # Reset if we're improving
        
        # Now update learning rate scheduler as usual
        if args.log_global_steps:
            for _ in range(processed_data['train_edge_index'].size(1) // args.batch_size + 1):
                scheduler.step()
        else:
            scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        history['val_pr_auc'].append(val_metrics['pr_auc'])
        
        # Log metrics
        logging.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Accuracy: {val_metrics['accuracy']:.4f}, "
            f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}, "
            f"Val PR-AUC: {val_metrics['pr_auc']:.4f}, "
            f"LR: {current_lr:.6f}"
        )
        
        # Check if this is the best model
        current_metric = val_metrics['loss'] if args.task == 'regression' else val_metrics['roc_auc']
        
        if is_better(current_metric, best_val_metric):
            if args.task == 'regression':
                logging.info(f"New best model with Val Loss: {current_metric:.4f}")
            else:
                logging.info(f"New best model with Val ROC-AUC: {current_metric:.4f}")
                
            best_val_metric = current_metric
            history['best_val_roc_auc'] = val_metrics['roc_auc']
            history['best_val_accuracy'] = val_metrics['accuracy']
            history['best_val_epoch'] = epoch
            
            # Save model
            torch.save(model.state_dict(), history['model_path'])
            logging.info(f"Model saved to {history['model_path']}")
            
            # Reset early stopping counter
            early_stopping_counter = 0
        else:
            # Increment early stopping counter
            early_stopping_counter += 1
            
            if not disable_early_stopping:
                logging.info(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
                
                if early_stopping_counter >= early_stopping_patience:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.model_dir, f"{model_name}_checkpoint_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_metrics': val_metrics,
                'global_step': global_step if args.log_global_steps else None
            }, checkpoint_path)
            logging.info(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")
        
        # Update global step counter
        history['epochs'].append(epoch)
    
    # Instead of loading the best model, use the latest model
    logging.info(f"Using the latest model (from epoch {args.epochs})")
    
    # Save the latest model if needed
    latest_model_path = os.path.join(args.model_dir, f"{model_name}_latest.pt")
    torch.save(model.state_dict(), latest_model_path)
    logging.info(f"Latest model saved to {latest_model_path}")
    
    # Evaluate on test set
    logging.info("Evaluating on test set...")
    test_metrics = evaluate(
        model=model,
        criterion=criterion,
        edge_index=processed_data['test_edge_index'].to(device),
        edge_attr=processed_data['test_edge_attr'].to(device) if 'test_edge_attr' in processed_data else None,
        labels=processed_data['test_labels'].to(device),
        batch_size=args.batch_size
    )
    
    # Update history
    history['test_loss'] = test_metrics['loss']
    history['test_accuracy'] = test_metrics['accuracy']
    history['test_roc_auc'] = test_metrics['roc_auc']
    history['test_pr_auc'] = test_metrics['pr_auc']
    
    # Log test metrics
    logging.info(
        f"Test Loss: {test_metrics['loss']:.4f}, "
        f"Test Accuracy: {test_metrics['accuracy']:.4f}, "
        f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}, "
        f"Test PR-AUC: {test_metrics['pr_auc']:.4f}"
    )
    
    # DIAGNOSTIC: Analyze predictions vs. labels in more detail
    if args.task == 'link_prediction':
        try:
            with torch.no_grad():
                # Get a sample of test predictions for detailed analysis
                sample_size = min(1000, processed_data['test_edge_index'].size(1))
                sample_indices = torch.randperm(processed_data['test_edge_index'].size(1))[:sample_size]
                
                sample_edge_index = processed_data['test_edge_index'][:, sample_indices].to(device)
                sample_edge_attr = processed_data['test_edge_attr'][sample_indices].to(device) if 'test_edge_attr' in processed_data else None
                sample_labels = processed_data['test_labels'][sample_indices].to(device)
                
                # Forward pass
                sample_outputs = model(sample_edge_index, sample_edge_attr)
                
                # Convert to numpy
                sample_outputs = sample_outputs.cpu().numpy()
                sample_labels = sample_labels.cpu().numpy()
                
                # Analyze predictions
                logging.info("DETAILED PREDICTION ANALYSIS:")
                logging.info(f"Sample size: {sample_size}")
                
                # Distribution of predictions
                pred_bins = np.arange(0, 1.1, 0.1)
                pred_hist = np.histogram(sample_outputs, bins=pred_bins)[0]
                logging.info(f"Prediction distribution:")
                for i in range(len(pred_bins) - 1):
                    logging.info(f"  {pred_bins[i]:.1f}-{pred_bins[i+1]:.1f}: {pred_hist[i]}")
                
                # Prediction stats by true label
                positive_preds = sample_outputs[sample_labels == 1]
                negative_preds = sample_outputs[sample_labels == 0]
                
                logging.info(f"Predictions for positive examples (true=1): mean={np.mean(positive_preds):.4f}, std={np.std(positive_preds):.4f}")
                logging.info(f"Predictions for negative examples (true=0): mean={np.mean(negative_preds):.4f}, std={np.std(negative_preds):.4f}")
                
                # Check if the model is discriminating between classes at all
                class_diff = np.mean(positive_preds) - np.mean(negative_preds)
                logging.info(f"Class differentiation: {class_diff:.4f}")
                
                if abs(class_diff) < 0.05:
                    logging.warning("MODEL IS NOT EFFECTIVELY DISTINGUISHING BETWEEN CLASSES")
                    logging.warning("This suggests the model is not learning to discriminate")
                
                # Check prediction entropy (measure of uncertainty/confidence)
                # For binary classification, entropy is -p*log(p) - (1-p)*log(1-p)
                epsilon = 1e-10  # Avoid log(0)
                entropy = -sample_outputs * np.log(sample_outputs + epsilon) - (1 - sample_outputs) * np.log(1 - sample_outputs + epsilon)
                
                logging.info(f"Prediction entropy: mean={np.mean(entropy):.4f}, std={np.std(entropy):.4f}")
                logging.info(f"Prediction entropy range: min={np.min(entropy):.4f}, max={np.max(entropy):.4f}")
                
                # Near-random predictions would have high entropy (close to 0.693 = -log(0.5))
                if np.mean(entropy) > 0.6:
                    logging.warning("HIGH PREDICTION ENTROPY - MODEL IS PRODUCING UNCERTAIN PREDICTIONS")
                    
                # Create confusion matrix visualization
                from sklearn.metrics import confusion_matrix
                import seaborn as sns
                
                plt.figure(figsize=(10, 8))
                
                # Get binary predictions using the optimal threshold
                # First try to find a threshold that balances precision and recall
                from sklearn.metrics import precision_recall_curve
                precision, recall, thresholds = precision_recall_curve(test_labels, test_preds)
                
                # Find the threshold that maximizes F1 score
                f1_scores = np.zeros_like(thresholds)
                for i, threshold in enumerate(thresholds):
                    if precision[i] + recall[i] > 0:  # Avoid division by zero
                        f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
                    
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx]
                
                logging.info(f"Optimal threshold for F1 score: {optimal_threshold:.4f}")
                
                # Use the threshold to get binary predictions
                test_binary_preds = (test_preds >= optimal_threshold).astype(int)
                
                # Calculate confusion matrix
                cm = confusion_matrix(test_labels, test_binary_preds)
                
                # Get counts for annotations
                tn, fp, fn, tp = cm.ravel()
                total = tn + fp + fn + tp
                
                # Calculate metrics for annotations
                accuracy = (tp + tn) / total
                precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
                
                # Plot confusion matrix with better styling
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                          xticklabels=['Negative', 'Positive'],
                          yticklabels=['Negative', 'Positive'])
                
                plt.xlabel('Predicted Label', fontweight='bold', fontsize=14)
                plt.ylabel('True Label', fontweight='bold', fontsize=14)
                plt.title('Confusion Matrix', fontweight='bold', fontsize=16)
                
                # Add metrics as text annotation
                plt.figtext(0.5, 0.01, 
                           f"Accuracy: {accuracy:.4f} | Precision: {precision_score:.4f} | Recall: {recall_score:.4f} | F1: {f1:.4f}",
                           ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for annotation
                plt.savefig(os.path.join(run_log_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
                logging.info(f"Confusion matrix saved to {os.path.join(run_log_dir, 'confusion_matrix.png')}")
                
                # Create precision-recall curve
                plt.figure(figsize=(10, 8))
                
                from sklearn.metrics import precision_recall_curve, average_precision_score
                
                # Calculate precision-recall curves
                val_precision, val_recall, _ = precision_recall_curve(val_labels, val_preds)
                val_avg_precision = average_precision_score(val_labels, val_preds)
                
                test_precision, test_recall, _ = precision_recall_curve(test_labels, test_preds)
                test_avg_precision = average_precision_score(test_labels, test_preds)
                
                # Calculate baseline (random classifier) based on class distribution
                baseline = np.sum(test_labels) / len(test_labels)
                
                # Plot precision-recall curves
                plt.plot(val_recall, val_precision, color='#1f77b4', lw=2,
                       label=f'Validation PR Curve (AP = {val_avg_precision:.4f})')
                plt.plot(test_recall, test_precision, color='#ff7f0e', lw=2,
                       label=f'Test PR Curve (AP = {test_avg_precision:.4f})')
                
                # Plot baseline
                plt.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7,
                          label=f'Random Classifier (AP = {baseline:.4f})')
                
                # Mark the optimal threshold point
                opt_precision = precision[optimal_idx]
                opt_recall = recall[optimal_idx]
                plt.plot(opt_recall, opt_precision, 'ro', ms=8, label=f'Optimal Threshold = {optimal_threshold:.2f}')
                
                # Add annotations
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Recall', fontweight='bold', fontsize=14)
                plt.ylabel('Precision', fontweight='bold', fontsize=14)
                plt.title('Precision-Recall Curve', fontweight='bold', fontsize=16)
                plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
                plt.grid(True, linestyle='--', alpha=0.6)
                
                # Improve appearance
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Save figure
                plt.tight_layout()
                plt.savefig(os.path.join(run_log_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
                logging.info(f"Precision-recall curve saved to {os.path.join(run_log_dir, 'precision_recall_curve.png')}")
                
                # Create a threshold analysis plot to help users select the best threshold
                plt.figure(figsize=(12, 8))
                
                # Calculate metrics at different thresholds
                thresholds = np.linspace(0, 1, 100)
                accuracy_scores = []
                precision_scores = []
                recall_scores = []
                f1_scores = []
                
                for threshold in thresholds:
                    binary_preds = (test_preds >= threshold).astype(int)
                    
                    # Calculate metrics
                    tn, fp, fn, tp = confusion_matrix(test_labels, binary_preds).ravel()
                    
                    # Calculate accuracy
                    acc = (tp + tn) / (tp + tn + fp + fn)
                    accuracy_scores.append(acc)
                    
                    # Calculate precision
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                    precision_scores.append(prec)
                    
                    # Calculate recall
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                    recall_scores.append(rec)
                    
                    # Calculate F1
                    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
                    f1_scores.append(f1)
                
                # Plot all metrics
                plt.plot(thresholds, accuracy_scores, 'b-', lw=2, label='Accuracy')
                plt.plot(thresholds, precision_scores, 'g-', lw=2, label='Precision')
                plt.plot(thresholds, recall_scores, 'r-', lw=2, label='Recall')
                plt.plot(thresholds, f1_scores, 'purple', lw=2, label='F1 Score')
                
                # Mark the F1-optimal threshold
                plt.axvline(x=optimal_threshold, color='black', linestyle='--', alpha=0.7,
                          label=f'Optimal Threshold = {optimal_threshold:.2f}')
                
                # Add annotations
                plt.xlabel('Classification Threshold', fontweight='bold', fontsize=14)
                plt.ylabel('Score', fontweight='bold', fontsize=14)
                plt.title('Performance Metrics vs. Classification Threshold', fontweight='bold', fontsize=16)
                plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
                plt.grid(True, linestyle='--', alpha=0.6)
                
                # Improve appearance
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Save figure
                plt.tight_layout()
                plt.savefig(os.path.join(run_log_dir, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
                logging.info(f"Threshold analysis plot saved to {os.path.join(run_log_dir, 'threshold_analysis.png')}")
        except Exception as e:
            logging.error(f"Error in detailed prediction analysis: {e}")
    
    # Save plots
    logging.info("Saving plots...")
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        
        # Set global plot styling for better readability
        plt.style.use('seaborn-v0_8-whitegrid')
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        mpl.rcParams['font.size'] = 11
        mpl.rcParams['axes.labelsize'] = 12
        mpl.rcParams['axes.titlesize'] = 14
        mpl.rcParams['xtick.labelsize'] = 10
        mpl.rcParams['ytick.labelsize'] = 10
        mpl.rcParams['legend.fontsize'] = 10
        mpl.rcParams['figure.titlesize'] = 16
        mpl.rcParams['axes.linewidth'] = 1.2
        mpl.rcParams['grid.linewidth'] = 0.8
        mpl.rcParams['lines.linewidth'] = 2.0
        mpl.rcParams['lines.markersize'] = 6
        mpl.rcParams['savefig.dpi'] = 300
        mpl.rcParams['savefig.bbox'] = 'tight'
        mpl.rcParams['savefig.pad_inches'] = 0.1
        
        # Create color palette for consistency
        colors = {
            'train': '#1f77b4',    # blue
            'val': '#ff7f0e',      # orange
            'test': '#2ca02c',     # green
            'best_model': '#d62728', # red
            'accuracy': '#9467bd',  # purple
            'roc_auc': '#8c564b',  # brown
            'pr_auc': '#e377c2',   # pink
            'lr': '#7f7f7f'        # gray
        }
        
        # Save learning curves - improved design
        plt.figure(figsize=(14, 10))
        
        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train', color=colors['train'], linewidth=2)
        plt.plot(history['val_loss'], label='Validation', color=colors['val'], linewidth=2)
        plt.axvline(x=history['best_val_epoch'], color=colors['best_model'], linestyle='--', label='Best Model')
        plt.xlabel('Epoch', fontweight='bold')
        plt.ylabel('Loss', fontweight='bold')
        plt.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
        plt.title('Loss Curves', fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Accuracy plot
        plt.subplot(2, 2, 2)
        plt.plot(history['val_accuracy'], label='Validation', color=colors['accuracy'], linewidth=2)
        plt.axvline(x=history['best_val_epoch'], color=colors['best_model'], linestyle='--', label='Best Model')
        plt.axhline(y=history['test_accuracy'], color=colors['test'], linestyle='--', 
                   label=f'Test: {history["test_accuracy"]:.4f}')
        plt.xlabel('Epoch', fontweight='bold')
        plt.ylabel('Accuracy', fontweight='bold')
        plt.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
        plt.title('Accuracy Curves', fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # ROC-AUC plot
        plt.subplot(2, 2, 3)
        plt.plot(history['val_roc_auc'], label='Validation', color=colors['roc_auc'], linewidth=2)
        plt.axvline(x=history['best_val_epoch'], color=colors['best_model'], linestyle='--', label='Best Model')
        plt.axhline(y=history['test_roc_auc'], color=colors['test'], linestyle='--', 
                   label=f'Test: {history["test_roc_auc"]:.4f}')
        plt.xlabel('Epoch', fontweight='bold')
        plt.ylabel('ROC-AUC', fontweight='bold')
        plt.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
        plt.title('ROC-AUC Curves', fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Learning rate plot
        plt.subplot(2, 2, 4)
        plt.plot(history['learning_rates'], color=colors['lr'], linewidth=2)
        plt.xlabel('Epoch', fontweight='bold')
        plt.ylabel('Learning Rate', fontweight='bold')
        plt.title('Learning Rate Schedule', fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_log_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
        logging.info(f"Learning curves saved to {os.path.join(run_log_dir, 'learning_curves.png')}")
        
        # FLOPS vs metrics plot with log scale for FLOPS - improved design
        if history['flops_per_batch'] is not None and history['total_flops']:
            plt.figure(figsize=(18, 6))
            
            # FLOPS vs Loss (log scale for FLOPS)
            plt.subplot(1, 3, 1)
            plt.semilogx(history['total_flops'], history['train_loss'], label='Train Loss', color=colors['train'], linewidth=2)
            plt.semilogx(history['total_flops'], history['val_loss'], label='Val Loss', color=colors['val'], linewidth=2)
            plt.xlabel('Cumulative FLOPS (log scale)', fontweight='bold')
            plt.ylabel('Loss', fontweight='bold')
            plt.grid(True, which="both", ls="--", alpha=0.7)
            plt.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
            plt.title('FLOPS vs Loss', fontweight='bold')
            
            # FLOPS vs Accuracy (log scale for FLOPS)
            plt.subplot(1, 3, 2)
            plt.semilogx(history['total_flops'], history['val_accuracy'], label='Val Accuracy', color=colors['accuracy'], linewidth=2)
            plt.xlabel('Cumulative FLOPS (log scale)', fontweight='bold')
            plt.ylabel('Accuracy', fontweight='bold')
            plt.grid(True, which="both", ls="--", alpha=0.7)
            plt.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
            plt.title('FLOPS vs Accuracy', fontweight='bold')
            
            # FLOPS vs ROC-AUC (log scale for FLOPS)
            plt.subplot(1, 3, 3)
            plt.semilogx(history['total_flops'], history['val_roc_auc'], label='Val ROC-AUC', color=colors['roc_auc'], linewidth=2)
            plt.xlabel('Cumulative FLOPS (log scale)', fontweight='bold')
            plt.ylabel('ROC-AUC', fontweight='bold')
            plt.grid(True, which="both", ls="--", alpha=0.7)
            plt.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
            plt.title('FLOPS vs ROC-AUC', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(run_log_dir, 'flops_metrics.png'), dpi=300, bbox_inches='tight')
            logging.info(f"FLOPS vs metrics plot saved to {os.path.join(run_log_dir, 'flops_metrics.png')}")
            
            # Also create a linear scale version for comparison - improved design
            plt.figure(figsize=(18, 6))
            
            # FLOPS vs Loss (linear scale)
            plt.subplot(1, 3, 1)
            plt.plot(history['total_flops'], history['train_loss'], label='Train Loss', color=colors['train'], linewidth=2)
            plt.plot(history['total_flops'], history['val_loss'], label='Val Loss', color=colors['val'], linewidth=2)
            plt.xlabel('Cumulative FLOPS', fontweight='bold')
            plt.ylabel('Loss', fontweight='bold')
            plt.grid(True, ls="--", alpha=0.7)
            plt.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
            plt.title('FLOPS vs Loss (Linear Scale)', fontweight='bold')
            
            # FLOPS vs Accuracy (linear scale)
            plt.subplot(1, 3, 2)
            plt.plot(history['total_flops'], history['val_accuracy'], label='Val Accuracy', color=colors['accuracy'], linewidth=2)
            plt.xlabel('Cumulative FLOPS', fontweight='bold')
            plt.ylabel('Accuracy', fontweight='bold')
            plt.grid(True, ls="--", alpha=0.7)
            plt.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
            plt.title('FLOPS vs Accuracy (Linear Scale)', fontweight='bold')
            
            # FLOPS vs ROC-AUC (linear scale)
            plt.subplot(1, 3, 3)
            plt.plot(history['total_flops'], history['val_roc_auc'], label='Val ROC-AUC', color=colors['roc_auc'], linewidth=2)
            plt.xlabel('Cumulative FLOPS', fontweight='bold')
            plt.ylabel('ROC-AUC', fontweight='bold')
            plt.grid(True, ls="--", alpha=0.7)
            plt.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
            plt.title('FLOPS vs ROC-AUC (Linear Scale)', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(run_log_dir, 'flops_metrics_linear.png'), dpi=300, bbox_inches='tight')
            logging.info(f"FLOPS vs metrics plot (linear scale) saved to {os.path.join(run_log_dir, 'flops_metrics_linear.png')}")
            
            # Compute efficiency plot (improvement per FLOP) - log scale for x-axis - improved design
            plt.figure(figsize=(10, 6))
            
            # Calculate improvement per FLOP
            flops_arr = np.array(history['total_flops'])
            roc_auc_arr = np.array(history['val_roc_auc'])
            
            # Calculate difference in ROC-AUC between consecutive epochs
            roc_auc_diff = np.diff(roc_auc_arr)
            flops_diff = np.diff(flops_arr)
            
            # Avoid division by zero
            efficiency = np.zeros_like(flops_diff)
            nonzero_mask = flops_diff > 0
            efficiency[nonzero_mask] = roc_auc_diff[nonzero_mask] / flops_diff[nonzero_mask]
            
            # Plot computational efficiency with log scale for x-axis
            plt.semilogx(flops_arr[1:], efficiency * 1e12, marker='o', color='#17becf', markersize=6, 
                       markeredgecolor='white', markeredgewidth=0.8, linewidth=1.5, alpha=0.8)
            plt.xlabel('Cumulative FLOPS (log scale)', fontweight='bold')
            plt.ylabel('ROC-AUC Improvement per Trillion FLOPS', fontweight='bold')
            plt.title('Computational Efficiency', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(run_log_dir, 'computational_efficiency.png'), dpi=300, bbox_inches='tight')
            logging.info(f"Computational efficiency plot saved to {os.path.join(run_log_dir, 'computational_efficiency.png')}")
            
            # Create publication-quality summary plot with logarithmic x-axis
            plt.figure(figsize=(10, 8))
            
            # Set publication-quality styling
            plt.rcParams['axes.labelsize'] = 14
            plt.rcParams['axes.titlesize'] = 16
            plt.rcParams['xtick.labelsize'] = 12
            plt.rcParams['ytick.labelsize'] = 12
            plt.rcParams['legend.fontsize'] = 12
            
            # Normalize FLOPS to the model with the most FLOPS for easy comparison
            max_flops = max(history['total_flops'])
            normalized_flops = [f/max_flops for f in history['total_flops']]
            
            # Plot with enhanced visual style
            plt.semilogx(history['total_flops'], history['val_roc_auc'], 'o-', color='#1f77b4', 
                       linewidth=2.5, label='Val ROC-AUC', markersize=7, markeredgecolor='white', markeredgewidth=0.8)
            plt.semilogx(history['total_flops'], history['val_accuracy'], 's-', color='#ff7f0e', 
                       linewidth=2.5, label='Val Accuracy', markersize=7, markeredgecolor='white', markeredgewidth=0.8)
            
            # Add test performance as horizontal lines with enhanced styling
            if 'test_roc_auc' in history and 'test_accuracy' in history:
                plt.axhline(y=history['test_roc_auc'], color='#1f77b4', linestyle='--', linewidth=2.0,
                          label=f'Test ROC-AUC: {history["test_roc_auc"]:.4f}')
                plt.axhline(y=history['test_accuracy'], color='#ff7f0e', linestyle='--', linewidth=2.0,
                          label=f'Test Accuracy: {history["test_accuracy"]:.4f}')
            
            # Add vertical line at best epoch with enhanced styling
            best_flops = history['total_flops'][history['best_val_epoch']] if history['best_val_epoch'] < len(history['total_flops']) else history['total_flops'][-1]
            plt.axvline(x=best_flops, color='#2ca02c', linestyle='--', linewidth=2.0,
                      label=f'Best Model: {best_flops:.2e} FLOPS')
            
            plt.xlabel('Cumulative FLOPS (log scale)', fontweight='bold')
            plt.ylabel('Performance Metric', fontweight='bold')
            plt.title(f'Performance vs. Computational Cost\n{args.model} ({args.gnn_type if args.model == "gnn" else ""}) Model', 
                   fontweight='bold')
            plt.grid(True, which="both", ls="--", alpha=0.7)
            plt.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray', loc='best')
            
            # Format x-axis with scientific notation and improve tick spacing
            from matplotlib.ticker import ScalarFormatter, LogLocator
            ax = plt.gca()
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
            
            # Set y-axis limits for better visualization
            plt.ylim([min(0.4, min(history['val_accuracy']) * 0.9), 
                    max(1.0, max(history['val_roc_auc']) * 1.1)])
            
            # Improve figure quality
            plt.tight_layout()
            plt.savefig(os.path.join(run_log_dir, 'performance_vs_cost.png'), dpi=300, bbox_inches='tight')
            logging.info(f"Performance vs. cost summary plot saved to {os.path.join(run_log_dir, 'performance_vs_cost.png')}")
            
            # Create a clean, visually appealing metric comparison chart
            plt.figure(figsize=(12, 6))
            
            # Prepare epoch data for x-axis
            epochs = list(range(len(history['val_accuracy'])))
            
            # Create a bar chart comparing metrics across epochs
            x = np.arange(len(epochs))
            width = 0.25  # the width of the bars
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Only plot every Nth epoch to avoid crowding (adjust based on total epochs)
            plot_every = max(1, len(epochs) // 20)
            x_sparse = x[::plot_every]
            
            # Plot bars for each metric
            rects1 = ax.bar(x_sparse - width, 
                           [history['val_accuracy'][i] for i in range(0, len(epochs), plot_every)], 
                           width, label='Accuracy', color='#3274A1', alpha=0.8)
            
            rects2 = ax.bar(x_sparse, 
                           [history['val_roc_auc'][i] for i in range(0, len(epochs), plot_every)], 
                           width, label='ROC-AUC', color='#E1812C', alpha=0.8)
            
            rects3 = ax.bar(x_sparse + width, 
                           [history['val_pr_auc'][i] for i in range(0, len(epochs), plot_every)], 
                           width, label='PR-AUC', color='#3A923A', alpha=0.8)
            
            # Add labels and customize appearance
            ax.set_xlabel('Epoch', fontweight='bold', fontsize=14)
            ax.set_ylabel('Metric Value', fontweight='bold', fontsize=14)
            ax.set_title('Validation Metrics Comparison by Epoch', fontweight='bold', fontsize=16)
            ax.set_xticks(x_sparse)
            ax.set_xticklabels([str(epochs[i]) for i in range(0, len(epochs), plot_every)])
            ax.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
            
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add a subtle background color for better readability
            ax.set_facecolor('#f8f9fa')
            
            fig.tight_layout()
            plt.savefig(os.path.join(run_log_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
            logging.info(f"Metrics comparison chart saved to {os.path.join(run_log_dir, 'metrics_comparison.png')}")
            
            # Create a training dynamics heatmap - useful for publications and presentations
            try:
                plt.figure(figsize=(12, 8))
                
                # Collect all relevant metrics
                metrics_data = {
                    'Train Loss': history['train_loss'],
                    'Val Loss': history['val_loss'],
                    'Val Accuracy': history['val_accuracy'],
                    'Val ROC-AUC': history['val_roc_auc'],
                    'Val PR-AUC': history['val_pr_auc'],
                    'Learning Rate': history['learning_rates']
                }
                
                # Normalize each metric series to 0-1 range for fair comparison
                metrics_normalized = {}
                for metric_name, metric_values in metrics_data.items():
                    if len(metric_values) > 0:
                        min_val = min(metric_values)
                        max_val = max(metric_values)
                        
                        if max_val > min_val:
                            # Normal normalization for metrics we want to maximize
                            if metric_name in ['Val Accuracy', 'Val ROC-AUC', 'Val PR-AUC']:
                                normalized = [(x - min_val) / (max_val - min_val) for x in metric_values]
                            # Inverse normalization for metrics we want to minimize
                            elif metric_name in ['Train Loss', 'Val Loss']:
                                normalized = [1 - ((x - min_val) / (max_val - min_val)) for x in metric_values]
                            # Special case for learning rate
                            else:
                                normalized = [(x - min_val) / (max_val - min_val) for x in metric_values]
                        else:
                            normalized = [0.5] * len(metric_values)
                        
                        metrics_normalized[metric_name] = normalized
                
                # Create heatmap data
                heatmap_data = []
                for metric_name in metrics_normalized:
                    heatmap_data.append(metrics_normalized[metric_name])
                
                # Only show a reasonable number of epochs for readability
                max_epochs_to_show = 50
                if len(history['epochs']) > max_epochs_to_show:
                    stride = len(history['epochs']) // max_epochs_to_show
                    heatmap_data = [data[::stride] for data in heatmap_data]
                    epochs_to_show = history['epochs'][::stride]
                else:
                    epochs_to_show = history['epochs']
                
                # Create heatmap with better coloring
                import seaborn as sns
                ax = sns.heatmap(
                    heatmap_data, 
                    cmap='viridis',
                    vmin=0, 
                    vmax=1,
                    cbar_kws={'label': 'Normalized Value'},
                    linewidths=0.1,
                    linecolor='whitesmoke',
                    xticklabels=[f"{e}" for e in epochs_to_show]
                )
                
                # Set y-axis labels
                ax.set_yticklabels(list(metrics_normalized.keys()), rotation=0)
                
                # Set labels and title
                plt.xlabel('Epoch', fontweight='bold', fontsize=14)
                plt.ylabel('Metric', fontweight='bold', fontsize=14)
                plt.title('Training Dynamics Heatmap', fontweight='bold', fontsize=16)
                
                # Adjust layout and save
                plt.tight_layout()
                plt.savefig(os.path.join(run_log_dir, 'training_dynamics_heatmap.png'), dpi=300, bbox_inches='tight')
                logging.info(f"Training dynamics heatmap saved to {os.path.join(run_log_dir, 'training_dynamics_heatmap.png')}")
                
                # Create a publication-ready ROC curve visualization if it's the final epoch
                if epoch == args.epochs - 1:
                    plt.figure(figsize=(10, 8))
                    
                    # Calculate ROC curve points
                    from sklearn.metrics import roc_curve, auc
                    
                    # Create a helper function to get prediction scores for a set
                    def get_predictions(edge_index, edge_attr, labels, batch_size=1024):
                        all_preds = []
                        all_labels = []
                        
                        with torch.no_grad():
                            for i in range(0, edge_index.size(1), batch_size):
                                batch_edge_index = edge_index[:, i:i+batch_size]
                                batch_edge_attr = edge_attr[i:i+batch_size] if edge_attr is not None else None
                                batch_labels = labels[i:i+batch_size]
                                
                                outputs = model(batch_edge_index, batch_edge_attr)
                                
                                if outputs.dim() > 1 and outputs.size(1) == 1:
                                    outputs = outputs.squeeze(1)
                                    
                                # Apply sigmoid if using BCE with logits
                                if criterion.__class__.__name__ == 'BCEWithLogitsLoss':
                                    outputs = torch.sigmoid(outputs)
                                
                                all_preds.append(outputs.cpu().numpy())
                                all_labels.append(batch_labels.cpu().numpy())
                        
                        return np.concatenate(all_preds), np.concatenate(all_labels)
                    
                    try:
                        # Get predictions for validation set
                        val_preds, val_labels = get_predictions(
                            processed_data['val_edge_index'].to(device),
                            processed_data['val_edge_attr'].to(device) if 'val_edge_attr' in processed_data else None,
                            processed_data['val_labels'].to(device)
                        )
                        
                        # Get predictions for test set
                        test_preds, test_labels = get_predictions(
                            processed_data['test_edge_index'].to(device),
                            processed_data['test_edge_attr'].to(device) if 'test_edge_attr' in processed_data else None,
                            processed_data['test_labels'].to(device)
                        )
                        
                        # Calculate ROC curves
                        val_fpr, val_tpr, _ = roc_curve(val_labels, val_preds)
                        val_auc = auc(val_fpr, val_tpr)
                        
                        test_fpr, test_tpr, _ = roc_curve(test_labels, test_preds)
                        test_auc = auc(test_fpr, test_tpr)
                        
                        # Plot ROC curves
                        plt.plot(val_fpr, val_tpr, color='#1f77b4', lw=2, 
                               label=f'Validation ROC (AUC = {val_auc:.4f})')
                        plt.plot(test_fpr, test_tpr, color='#ff7f0e', lw=2, 
                               label=f'Test ROC (AUC = {test_auc:.4f})')
                        
                        # Plot diagonal line (random classifier)
                        plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', alpha=0.7,
                               label='Random Classifier')
                        
                        # Add annotations for publication-quality visualization
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate', fontweight='bold', fontsize=14)
                        plt.ylabel('True Positive Rate', fontweight='bold', fontsize=14)
                        plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold', fontsize=16)
                        plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
                        
                        # Improve appearance
                        ax = plt.gca()
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        
                        # Save figure
                        plt.tight_layout()
                        plt.savefig(os.path.join(run_log_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
                        logging.info(f"ROC curve saved to {os.path.join(run_log_dir, 'roc_curve.png')}")
                        
                        # Create confusion matrix visualization
                        from sklearn.metrics import confusion_matrix
                        import seaborn as sns
                        
                        plt.figure(figsize=(10, 8))
                        
                        # Get binary predictions using the optimal threshold
                        # First try to find a threshold that balances precision and recall
                        from sklearn.metrics import precision_recall_curve
                        precision, recall, thresholds = precision_recall_curve(test_labels, test_preds)
                        
                        # Find the threshold that maximizes F1 score
                        f1_scores = np.zeros_like(thresholds)
                        for i, threshold in enumerate(thresholds):
                            if precision[i] + recall[i] > 0:  # Avoid division by zero
                                f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
                            
                        optimal_idx = np.argmax(f1_scores)
                        optimal_threshold = thresholds[optimal_idx]
                        
                        logging.info(f"Optimal threshold for F1 score: {optimal_threshold:.4f}")
                        
                        # Use the threshold to get binary predictions
                        test_binary_preds = (test_preds >= optimal_threshold).astype(int)
                        
                        # Calculate confusion matrix
                        cm = confusion_matrix(test_labels, test_binary_preds)
                        
                        # Get counts for annotations
                        tn, fp, fn, tp = cm.ravel()
                        total = tn + fp + fn + tp
                        
                        # Calculate metrics for annotations
                        accuracy = (tp + tn) / total
                        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
                        
                        # Plot confusion matrix with better styling
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                                  xticklabels=['Negative', 'Positive'],
                                  yticklabels=['Negative', 'Positive'])
                        
                        plt.xlabel('Predicted Label', fontweight='bold', fontsize=14)
                        plt.ylabel('True Label', fontweight='bold', fontsize=14)
                        plt.title('Confusion Matrix', fontweight='bold', fontsize=16)
                        
                        # Add metrics as text annotation
                        plt.figtext(0.5, 0.01, 
                                   f"Accuracy: {accuracy:.4f} | Precision: {precision_score:.4f} | Recall: {recall_score:.4f} | F1: {f1:.4f}",
                                   ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
                        
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for annotation
                        plt.savefig(os.path.join(run_log_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
                        logging.info(f"Confusion matrix saved to {os.path.join(run_log_dir, 'confusion_matrix.png')}")
                        
                        # Create precision-recall curve
                        plt.figure(figsize=(10, 8))
                        
                        from sklearn.metrics import precision_recall_curve, average_precision_score
                        
                        # Calculate precision-recall curves
                        val_precision, val_recall, _ = precision_recall_curve(val_labels, val_preds)
                        val_avg_precision = average_precision_score(val_labels, val_preds)
                        
                        test_precision, test_recall, _ = precision_recall_curve(test_labels, test_preds)
                        test_avg_precision = average_precision_score(test_labels, test_preds)
                        
                        # Calculate baseline (random classifier) based on class distribution
                        baseline = np.sum(test_labels) / len(test_labels)
                        
                        # Plot precision-recall curves
                        plt.plot(val_recall, val_precision, color='#1f77b4', lw=2,
                               label=f'Validation PR Curve (AP = {val_avg_precision:.4f})')
                        plt.plot(test_recall, test_precision, color='#ff7f0e', lw=2,
                               label=f'Test PR Curve (AP = {test_avg_precision:.4f})')
                        
                        # Plot baseline
                        plt.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7,
                                  label=f'Random Classifier (AP = {baseline:.4f})')
                        
                        # Mark the optimal threshold point
                        opt_precision = precision[optimal_idx]
                        opt_recall = recall[optimal_idx]
                        plt.plot(opt_recall, opt_precision, 'ro', ms=8, label=f'Optimal Threshold = {optimal_threshold:.2f}')
                        
                        # Add annotations
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('Recall', fontweight='bold', fontsize=14)
                        plt.ylabel('Precision', fontweight='bold', fontsize=14)
                        plt.title('Precision-Recall Curve', fontweight='bold', fontsize=16)
                        plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
                        plt.grid(True, linestyle='--', alpha=0.6)
                        
                        # Improve appearance
                        ax = plt.gca()
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        
                        # Save figure
                        plt.tight_layout()
                        plt.savefig(os.path.join(run_log_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
                        logging.info(f"Precision-recall curve saved to {os.path.join(run_log_dir, 'precision_recall_curve.png')}")
                        
                        # Create a threshold analysis plot to help users select the best threshold
                        plt.figure(figsize=(12, 8))
                        
                        # Calculate metrics at different thresholds
                        thresholds = np.linspace(0, 1, 100)
                        accuracy_scores = []
                        precision_scores = []
                        recall_scores = []
                        f1_scores = []
                        
                        for threshold in thresholds:
                            binary_preds = (test_preds >= threshold).astype(int)
                            
                            # Calculate metrics
                            tn, fp, fn, tp = confusion_matrix(test_labels, binary_preds).ravel()
                            
                            # Calculate accuracy
                            acc = (tp + tn) / (tp + tn + fp + fn)
                            accuracy_scores.append(acc)
                            
                            # Calculate precision
                            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                            precision_scores.append(prec)
                            
                            # Calculate recall
                            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                            recall_scores.append(rec)
                            
                            # Calculate F1
                            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
                            f1_scores.append(f1)
                        
                        # Plot all metrics
                        plt.plot(thresholds, accuracy_scores, 'b-', lw=2, label='Accuracy')
                        plt.plot(thresholds, precision_scores, 'g-', lw=2, label='Precision')
                        plt.plot(thresholds, recall_scores, 'r-', lw=2, label='Recall')
                        plt.plot(thresholds, f1_scores, 'purple', lw=2, label='F1 Score')
                        
                        # Mark the F1-optimal threshold
                        plt.axvline(x=optimal_threshold, color='black', linestyle='--', alpha=0.7,
                                  label=f'Optimal Threshold = {optimal_threshold:.2f}')
                        
                        # Add annotations
                        plt.xlabel('Classification Threshold', fontweight='bold', fontsize=14)
                        plt.ylabel('Score', fontweight='bold', fontsize=14)
                        plt.title('Performance Metrics vs. Classification Threshold', fontweight='bold', fontsize=16)
                        plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
                        plt.grid(True, linestyle='--', alpha=0.6)
                        
                        # Improve appearance
                        ax = plt.gca()
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        
                        # Save figure
                        plt.tight_layout()
                        plt.savefig(os.path.join(run_log_dir, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
                        logging.info(f"Threshold analysis plot saved to {os.path.join(run_log_dir, 'threshold_analysis.png')}")
                    except Exception as e:
                        logging.error(f"Error creating additional visualizations: {e}")
                
            except Exception as e:
                logging.error(f"Error creating training dynamics visualizations: {e}")
    except Exception as e:
        logging.error(f"Error saving plots: {e}")
    
    # Print summary
    logging.info("\n" + "="*50)
    logging.info(f"Training Summary for {args.model} model")
    logging.info("="*50)
    logging.info(f"Total Epochs Trained: {args.epochs}")
    logging.info(f"Best Validation Epoch: {history['best_val_epoch']+1} (for reference)")
    logging.info(f"Latest Model Test Accuracy: {history['test_accuracy']:.4f}")
    logging.info(f"Latest Model Test ROC-AUC: {history['test_roc_auc']:.4f}")
    logging.info(f"Latest Model Test PR-AUC: {history['test_pr_auc']:.4f}")
    
    # Add explicit FLOPS logging to text summary
    if history['flops_per_batch'] is not None:
        total_flops = history['total_flops'][-1] if history['total_flops'] else None
        logging.info(f"FLOPS per batch: {history['flops_per_batch']:.2e}")
        logging.info(f"Total training FLOPS: {total_flops:.2e}")
    else:
        logging.info("FLOPS estimation unavailable")
        
    logging.info("="*50)
    
    # Log model information
    test_input = (
        processed_data['test_edge_index'][:, :100].to(device),
        processed_data['test_edge_attr'][:100].to(device) if 'test_edge_attr' in processed_data else None
    )
    log_model_info(model, test_input)
    
    # Log number of parameters and FLOPs
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {num_params}")
    
    # Additional metrics for model complexity analysis
    total_epochs = len(history['epochs'])
    logging.info(f"Total epochs trained: {total_epochs}")
    
    # Estimate FLOPs per batch (rough calculation)
    if args.model == 'gnn':
        # For GNN, operations scale with number of edges and parameters
        avg_edges_per_batch = processed_data['train_edge_index'].size(1) / (len(processed_data['train_labels']) / args.batch_size)
        flops_per_batch = num_params * avg_edges_per_batch * 2  # Rough estimate: 2 ops per parameter per edge
        logging.info(f"Estimated FLOPs per batch: {flops_per_batch:.2e} (GNN model)")
    else:
        # For other models, simpler calculation
        flops_per_batch = num_params * args.batch_size * 2  # Rough estimate: 2 ops per parameter per sample
        logging.info(f"Estimated FLOPs per batch: {flops_per_batch:.2e}")
    
    # Calculate total batches and steps
    batches_per_epoch = (processed_data['train_edge_index'].size(1) + args.batch_size - 1) // args.batch_size
    total_batches = batches_per_epoch * total_epochs
    logging.info(f"Batches per epoch: {batches_per_epoch}")
    logging.info(f"Total batches processed: {total_batches}")
    
    # Calculate total FLOPs
    total_flops = flops_per_batch * total_batches
    logging.info(f"Total estimated training FLOPs: {total_flops:.2e}")
    
    # For computing efficiency analysis
    if args.log_global_steps:
        logging.info(f"Final global step count: {global_step}")
        flops_per_step = flops_per_batch
        logging.info(f"Estimated FLOPs per step: {flops_per_step:.2e}")
        logging.info(f"Total FLOPs based on steps: {flops_per_step * global_step:.2e}")
    
    print("Training completed.")
    print(f"Model saved to: {history['model_path']}")
    print(f"Logs saved to: {history['run_log_dir']}")
    
    return history


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train GRN model')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model-dir', type=str, default='models', help='Model directory')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='gnn', choices=['gnn', 'cnn', 'transformer', 'graph_transformer'], help='Model type')
    parser.add_argument('--task', type=str, default='link_prediction', choices=['link_prediction'], help='Task type (only link_prediction supported)')
    parser.add_argument('--hidden-channels', type=int, default=64, help='Hidden channels')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # GNN-specific arguments
    parser.add_argument('--gnn-type', type=str, default='GCN', choices=['GCN', 'GAT', 'SAGE', 'GIN'], help='GNN type')
    
    # Transformer-specific arguments
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--gnn-layers', type=int, default=2, help='Number of GNN layers (for graph_transformer)')
    parser.add_argument('--transformer-layers', type=int, default=2, help='Number of transformer layers')
    
    # CNN-specific arguments
    parser.add_argument('--embedding-dim', type=int, default=32, help='Embedding dimension for CNN')
    parser.add_argument('--num-filters', type=int, default=32, help='Number of filters for CNN')
    parser.add_argument('--kernel-size', type=int, default=3, help='Kernel size for CNN')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--warmup-ratio', type=float, default=0.1, help='Ratio of warmup steps (as fraction of total steps)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay for regularization')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Logging arguments
    parser.add_argument('--log-steps', action='store_true', help='Log metrics at each training step')
    parser.add_argument('--log-global-steps', action='store_true', help='Use continuous global step counting (not resetting each epoch)')
    
    args = parser.parse_args()
    
    # Train model
    main(args)