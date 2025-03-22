#!/usr/bin/env python
"""
Evaluate a trained model on test data and generate detailed performance reports.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, 
    accuracy_score, confusion_matrix, classification_report
)

from models import create_model


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt


def evaluate_model(model_path, data_path, output_dir='evaluation', threshold=0.5):
    """Evaluate model and generate reports"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_path}")
    try:
        data = torch.load(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    # Extract model information from filename
    model_filename = os.path.basename(model_path)
    print(f"Evaluating model: {model_filename}")
    
    # Try to determine model type and task from filename
    # First check if it's a checkpoint file (contains "checkpoint")
    if "checkpoint" in model_filename:
        model_parts = model_filename.split('_checkpoint_')[0].split('_')
    else:
        model_parts = model_filename.split('_')
    
    # Default values in case we can't extract from filename
    model_type = "gnn"
    task = "link_prediction"
    gnn_type = "GCN"
    
    # Try to extract model type from filename
    if len(model_parts) >= 1:
        if model_parts[0] in ["gnn", "cnn", "transformer", "graph_transformer"]:
            model_type = model_parts[0]
            
        # Try to extract GNN type if present
        if model_type == "gnn" and len(model_parts) >= 2:
            if model_parts[1] in ["GCN", "GAT", "SAGE", "GIN"]:
                gnn_type = model_parts[1]
    
    print(f"Model type: {model_type}, Task: {task}")
    if model_type == "gnn":
        print(f"GNN type: {gnn_type}")
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model_kwargs = {
        'hidden_channels': 64,  # Default values
        'num_layers': 4,
        'dropout': 0.5,
    }
    
    # Add GNN-specific arguments
    if model_type == 'gnn':
        model_kwargs['gnn_type'] = gnn_type
    
    # Add node features if available
    if 'node_features' in data and data['node_features'] is not None:
        model_kwargs['node_features'] = data['node_features']
        print(f"Using node features of shape {data['node_features'].shape}")
    else:
        print("No node features found, model will create its own embeddings")
    
    # Create the model
    try:
        model = create_model(
            model_type=model_type,
            num_nodes=data['num_nodes'],
            task=task,
            **model_kwargs
        )
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    # Load model weights
    print(f"Loading model weights from {model_path}")
    try:
        # First try direct loading
        state_dict = torch.load(model_path, map_location=device)
        
        # Check if we loaded a checkpoint dict instead of just the state_dict
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        model.load_state_dict(state_dict, strict=False)
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Attempting to continue with potentially incomplete model...")
    
    model = model.to(device)
    model.eval()
    
    # Evaluate on test data
    with torch.no_grad():
        test_edge_index = data['test_edge_index'].to(device)
        test_edge_attr = data['test_edge_attr'].to(device) if 'test_edge_attr' in data else None
        test_labels = data['test_labels'].to(device)
        
        # Get predictions
        predictions = model(test_edge_index, test_edge_attr)
        
        # Move to CPU for numpy operations
        raw_predictions = predictions.cpu().numpy()
        test_labels = test_labels.cpu().numpy()
        
        # Check if predictions look like logits (not constrained to [0,1])
        is_logits = (np.min(raw_predictions) < 0.0 or np.max(raw_predictions) > 1.0 or 
                     np.mean(raw_predictions) < 0.1 or np.mean(raw_predictions) > 0.9)
        
        # Apply sigmoid if the model output looks like logits
        if is_logits:
            print(f"Model outputs appear to be logits. Applying sigmoid function.")
            print(f"Raw predictions - min: {np.min(raw_predictions):.4f}, max: {np.max(raw_predictions):.4f}")
            print(f"Raw predictions - mean: {np.mean(raw_predictions):.4f}, std: {np.std(raw_predictions):.4f}")
            
            # Apply sigmoid: p = 1/(1+exp(-x))
            predictions = 1.0 / (1.0 + np.exp(-np.clip(raw_predictions, -88.0, 88.0)))  # Clip to avoid overflow
            
            print(f"After sigmoid - min: {np.min(predictions):.4f}, max: {np.max(predictions):.4f}")
            print(f"After sigmoid - mean: {np.mean(predictions):.4f}, std: {np.std(predictions):.4f}")
        else:
            predictions = raw_predictions
            print(f"Model outputs appear to be probabilities (already in [0,1] range).")
            print(f"Predictions - min: {np.min(predictions):.4f}, max: {np.max(predictions):.4f}")
            print(f"Predictions - mean: {np.mean(predictions):.4f}, std: {np.std(predictions):.4f}")
    
    # Analyze results
    if task == 'link_prediction':
        # Binary predictions using specified threshold
        binary_predictions = (predictions >= threshold).astype(int)
        print(f"Using classification threshold: {threshold}")
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, binary_predictions)
        cm = confusion_matrix(test_labels, binary_predictions)
        
        # Classification report
        report = classification_report(test_labels, binary_predictions, target_names=['Negative', 'Positive'])
        
        # ROC curve
        fpr, tpr, _ = roc_curve(test_labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(test_labels, predictions)
        pr_auc = auc(recall, precision)
        
        # Print results
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"Test ROC AUC: {roc_auc:.4f}")
        print(f"Test PR AUC: {pr_auc:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Save results to file
        with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
            f.write(f"Model: {model_path}\n")
            f.write(f"Task: {task}\n\n")
            f.write(f"Test Accuracy: {accuracy:.4f}\n")
            f.write(f"Test ROC AUC: {roc_auc:.4f}\n")
            f.write(f"Test PR AUC: {pr_auc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        
        # Plot Precision-Recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.fill_between(recall, precision, alpha=0.2, color='green')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
        
        # Plot confusion matrix
        cm_plot = plot_confusion_matrix(cm, classes=['Negative', 'Positive'])
        cm_plot.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        
        # Plot prediction distribution
        plt.figure(figsize=(10, 6))
        plt.hist(predictions[test_labels == 0], bins=50, alpha=0.5, label='Negative Edges')
        plt.hist(predictions[test_labels == 1], bins=50, alpha=0.5, label='Positive Edges')
        plt.xlabel('Prediction Score')
        plt.ylabel('Count')
        plt.title('Distribution of Prediction Scores')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
        
    else:  # Regression task
        # Calculate metrics
        mse = np.mean((predictions - test_labels) ** 2)
        mae = np.mean(np.abs(predictions - test_labels))
        
        # Print results
        print(f"\nTest MSE: {mse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        
        # Save results to file
        with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
            f.write(f"Model: {model_path}\n")
            f.write(f"Task: {task}\n\n")
            f.write(f"Test MSE: {mse:.4f}\n")
            f.write(f"Test MAE: {mae:.4f}\n")
        
        # Plot predictions vs. actual
        plt.figure(figsize=(10, 6))
        plt.scatter(test_labels, predictions, alpha=0.5)
        plt.plot([min(test_labels), max(test_labels)], [min(test_labels), max(test_labels)], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Regression Performance')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'regression_performance.png'), dpi=300, bbox_inches='tight')
        
        # Plot error distribution
        errors = predictions - test_labels
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title('Distribution of Prediction Errors')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    
    print(f"\nEvaluation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data', type=str, default='processed_data/processed_data.pt', 
                       help='Path to the processed data')
    parser.add_argument('--output', type=str, default='evaluation', 
                       help='Output directory for evaluation results')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Decision threshold for link prediction')
    
    args = parser.parse_args()
    evaluate_model(args.model, args.data, args.output, args.threshold) 