#!/usr/bin/env python
"""
Evaluate a model with different classification thresholds to find the optimal threshold.
"""

import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)

from models import create_model

def evaluate_with_thresholds(model_path, data_path, output_dir='thresholds_eval'):
    """
    Evaluate a model with different classification thresholds
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_path}")
    data = torch.load(data_path)
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Extract model information from filename
    model_filename = os.path.basename(model_path)
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
    
    # Create the model
    model = create_model(
        model_type=model_type,
        num_nodes=data['num_nodes'],
        task=task,
        **model_kwargs
    )
    
    # Load model weights
    print(f"Loading model from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    
    # Check if we loaded a checkpoint dict instead of just the state_dict
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
        
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    # Get test data
    test_edge_index = data['test_edge_index'].to(device)
    test_edge_attr = data['test_edge_attr'].to(device) if 'test_edge_attr' in data else None
    test_labels = data['test_labels'].to(device)
    
    # Generate predictions
    with torch.no_grad():
        preds = model(test_edge_index, test_edge_attr).cpu().numpy()
        labels = test_labels.cpu().numpy()
    
    print(f"Generated {len(preds)} predictions with distribution:")
    print(f"  Min: {preds.min():.4f}, Max: {preds.max():.4f}")
    print(f"  Mean: {preds.mean():.4f}, Std: {preds.std():.4f}")
    print(f"  Predictions histogram:")
    
    # Print histogram of predictions
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(preds, bins=bins)
    for i in range(len(hist)):
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]} ({hist[i]/len(preds):.2%})")
    
    # Evaluate with different thresholds
    thresholds = np.linspace(0.01, 0.99, 99)
    results = []
    
    for threshold in thresholds:
        binary_preds = (preds >= threshold).astype(int)
        
        # Compute metrics
        try:
            accuracy = accuracy_score(labels, binary_preds)
            precision = precision_score(labels, binary_preds, zero_division=0)
            recall = recall_score(labels, binary_preds, zero_division=0)
            f1 = f1_score(labels, binary_preds, zero_division=0)
            
            # Compute confusion matrix
            tn, fp, fn, tp = confusion_matrix(labels, binary_preds, labels=[0, 1]).ravel()
            
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            })
        except Exception as e:
            print(f"Error computing metrics for threshold {threshold}: {e}")
    
    # Find optimal thresholds
    best_accuracy = max(results, key=lambda x: x['accuracy'])
    best_f1 = max(results, key=lambda x: x['f1'])
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    
    # Calculate PR curve and AUC
    precision_vals, recall_vals, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recall_vals, precision_vals)
    
    # Print results
    print("\nROC AUC: {:.4f}".format(roc_auc))
    print("PR AUC: {:.4f}".format(pr_auc))
    print("\nBest threshold for accuracy: {:.4f} (Accuracy: {:.4f})".format(
        best_accuracy['threshold'], best_accuracy['accuracy']))
    print("Best threshold for F1: {:.4f} (F1: {:.4f})".format(
        best_f1['threshold'], best_f1['f1']))
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    
    # Plot PR curve
    plt.subplot(2, 2, 2)
    plt.plot(recall_vals, precision_vals, 'r-', label=f'PR (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    
    # Plot metrics vs threshold
    plt.subplot(2, 2, 3)
    plt.plot([r['threshold'] for r in results], [r['accuracy'] for r in results], 'b-', label='Accuracy')
    plt.plot([r['threshold'] for r in results], [r['precision'] for r in results], 'r-', label='Precision')
    plt.plot([r['threshold'] for r in results], [r['recall'] for r in results], 'g-', label='Recall')
    plt.plot([r['threshold'] for r in results], [r['f1'] for r in results], 'y-', label='F1')
    plt.axvline(x=best_f1['threshold'], color='k', linestyle='--', label=f'Best F1 threshold: {best_f1["threshold"]:.4f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Threshold')
    plt.legend(loc='best')
    
    # Plot confusion matrix metrics
    plt.subplot(2, 2, 4)
    plt.plot([r['threshold'] for r in results], [r['true_positives'] for r in results], 'g-', label='True Positives')
    plt.plot([r['threshold'] for r in results], [r['false_positives'] for r in results], 'r-', label='False Positives')
    plt.plot([r['threshold'] for r in results], [r['false_negatives'] for r in results], 'b-', label='False Negatives')
    plt.axvline(x=best_f1['threshold'], color='k', linestyle='--')
    plt.xlabel('Threshold')
    plt.ylabel('Count')
    plt.title('Confusion Matrix Components vs Threshold')
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'))
    print(f"Saved threshold analysis plot to {os.path.join(output_dir, 'threshold_analysis.png')}")
    
    # Print detailed results for the best F1 threshold
    print("\nDetailed results for best F1 threshold ({:.4f}):".format(best_f1['threshold']))
    print("  Accuracy: {:.4f}".format(best_f1['accuracy']))
    print("  Precision: {:.4f}".format(best_f1['precision']))
    print("  Recall: {:.4f}".format(best_f1['recall']))
    print("  F1: {:.4f}".format(best_f1['f1']))
    print("  True Positives: {}".format(best_f1['true_positives']))
    print("  False Positives: {}".format(best_f1['false_positives']))
    print("  True Negatives: {}".format(best_f1['true_negatives']))
    print("  False Negatives: {}".format(best_f1['false_negatives']))
    
    # Save results to file
    results_df = {
        'threshold': [r['threshold'] for r in results],
        'accuracy': [r['accuracy'] for r in results],
        'precision': [r['precision'] for r in results],
        'recall': [r['recall'] for r in results],
        'f1': [r['f1'] for r in results],
        'true_positives': [r['true_positives'] for r in results],
        'false_positives': [r['false_positives'] for r in results],
        'true_negatives': [r['true_negatives'] for r in results],
        'false_negatives': [r['false_negatives'] for r in results]
    }
    
    # Import pandas only when needed
    import pandas as pd
    pd.DataFrame(results_df).to_csv(os.path.join(output_dir, 'threshold_results.csv'), index=False)
    print(f"Saved detailed results to {os.path.join(output_dir, 'threshold_results.csv')}")
    
    # Return the recommended threshold
    return best_f1['threshold']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model with different thresholds')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data', type=str, default='processed_data/spectral/processed_data.pt', help='Path to the processed data')
    parser.add_argument('--output', type=str, default='thresholds_eval', help='Output directory for evaluation results')
    
    args = parser.parse_args()
    optimal_threshold = evaluate_with_thresholds(args.model, args.data, args.output)
    
    print(f"\nRECOMMENDATION: Use a threshold of {optimal_threshold:.4f} for this model.") 