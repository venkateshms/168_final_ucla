#!/usr/bin/env python
"""
Analyze processed data to help identify issues with class distribution.
"""

import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def analyze_processed_data(data_file):
    """
    Analyze the processed data file and print statistics
    """
    print(f"Analyzing processed data from: {data_file}")
    
    # Load data
    try:
        data = torch.load(data_file)
        print(f"Successfully loaded data")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Print basic statistics
    print("\n==== DATASET STATISTICS ====")
    print(f"Number of nodes: {data['num_nodes']}")
    
    # Edge counts
    train_edges = data['train_edge_index'].shape[1]
    val_edges = data['val_edge_index'].shape[1]
    test_edges = data['test_edge_index'].shape[1]
    total_edges = train_edges + val_edges + test_edges
    
    print(f"Total edges: {total_edges}")
    print(f"Train edges: {train_edges} ({train_edges/total_edges:.2%})")
    print(f"Val edges: {val_edges} ({val_edges/total_edges:.2%})")
    print(f"Test edges: {test_edges} ({test_edges/total_edges:.2%})")
    
    # Positive/negative class distribution
    train_positive = data['train_labels'].sum().item()
    val_positive = data['val_labels'].sum().item()
    test_positive = data['test_labels'].sum().item()
    
    print("\n==== CLASS DISTRIBUTION ====")
    print(f"Train: {train_positive} positive, {train_edges-train_positive} negative ({train_positive/train_edges:.2%} positive)")
    print(f"Val: {val_positive} positive, {val_edges-val_positive} negative ({val_positive/val_edges:.2%} positive)")
    print(f"Test: {test_positive} positive, {test_edges-test_positive} negative ({test_positive/test_edges:.2%} positive)")
    
    # Check for negative samples (if available)
    print("\n==== NEGATIVE SAMPLES ====")
    if 'train_neg_edge_index' in data:
        train_neg = data['train_neg_edge_index'].shape[1]
        print(f"Train negative samples: {train_neg} ({train_neg/(train_neg+train_edges):.2%} of total)")
    else:
        print("No explicit negative samples found in data")
    
    # Check edge attributes
    if 'train_edge_attr' in data and data['train_edge_attr'] is not None:
        print("\n==== EDGE ATTRIBUTES ====")
        train_attr = data['train_edge_attr'].squeeze()
        val_attr = data['val_edge_attr'].squeeze()
        test_attr = data['test_edge_attr'].squeeze()
        
        print(f"Train edge attr - shape: {data['train_edge_attr'].shape}")
        print(f"  min: {train_attr.min().item():.4f}, max: {train_attr.max().item():.4f}")
        print(f"  mean: {train_attr.mean().item():.4f}, std: {train_attr.std().item():.4f}")
        
        # Check correlation between edge attributes and labels
        if train_attr.shape[0] == data['train_labels'].shape[0]:
            pos_attr = train_attr[data['train_labels'] == 1]
            neg_attr = train_attr[data['train_labels'] == 0]
            
            print("\n==== ATTRIBUTE DISTRIBUTION BY CLASS ====")
            print(f"Positive examples - count: {len(pos_attr)}")
            if len(pos_attr) > 0:
                print(f"  min: {pos_attr.min().item():.4f}, max: {pos_attr.max().item():.4f}")
                print(f"  mean: {pos_attr.mean().item():.4f}, std: {pos_attr.std().item():.4f}")
            else:
                print("  No positive examples found!")
                
            print(f"Negative examples - count: {len(neg_attr)}")
            print(f"  min: {neg_attr.min().item():.4f}, max: {neg_attr.max().item():.4f}")
            print(f"  mean: {neg_attr.mean().item():.4f}, std: {neg_attr.std().item():.4f}")
            
            # Plot histograms of attributes for positive and negative examples
            try:
                plt.figure(figsize=(10, 6))
                plt.hist(neg_attr.numpy(), bins=20, alpha=0.5, label='Negative', density=True)
                if len(pos_attr) > 0:
                    plt.hist(pos_attr.numpy(), bins=20, alpha=0.5, label='Positive', density=True)
                plt.xlabel('Edge Attribute Value')
                plt.ylabel('Density')
                plt.title('Distribution of Edge Attributes by Class')
                plt.legend()
                plt.tight_layout()
                
                # Save the plot
                plot_file = os.path.join(os.path.dirname(data_file), 'edge_attr_distribution.png')
                plt.savefig(plot_file)
                print(f"Saved attribute distribution plot to: {plot_file}")
            except Exception as e:
                print(f"Error generating plot: {e}")
    
    # Check node features
    if 'node_features' in data and data['node_features'] is not None:
        print("\n==== NODE FEATURES ====")
        node_features = data['node_features']
        print(f"Node features shape: {node_features.shape}")
        print(f"Node feature statistics:")
        print(f"  min: {node_features.min().item():.4f}, max: {node_features.max().item():.4f}")
        print(f"  mean: {node_features.mean().item():.4f}, std: {node_features.std().item():.4f}")
        print(f"  NaNs: {torch.isnan(node_features).sum().item()}")
        print(f"  Infs: {torch.isinf(node_features).sum().item()}")
    
    print("\n==== CONCLUSION ====")
    test_pos_ratio = test_positive/test_edges
    threshold_recommendation = None
    
    if test_pos_ratio < 0.10:
        print("WARNING: Very imbalanced dataset with few positive examples!")
        print("This may cause the model to predict all negatives.")
        threshold_recommendation = 0.05
    elif test_pos_ratio < 0.20:
        print("CAUTION: Somewhat imbalanced dataset.")
        print("Consider using class weights or lowering the decision threshold.")
        threshold_recommendation = 0.1
    else:
        print("Class distribution is relatively balanced.")
    
    if threshold_recommendation:
        print(f"\nRECOMMENDATION: Try lowering the classification threshold to ~{threshold_recommendation:.2f} instead of 0.5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze processed data')
    parser.add_argument('--data', type=str, default='processed_data/spectral/processed_data.pt', 
                       help='Path to processed data file')
    
    args = parser.parse_args()
    analyze_processed_data(args.data) 