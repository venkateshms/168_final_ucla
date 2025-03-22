#!/usr/bin/env python3
"""
Visualization script for GRN data and model performance
"""

import os
import re
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.manifold import TSNE
import torch
from torch_geometric.utils import to_networkx
import argparse
from pathlib import Path

def load_grn_data(csv_path):
    """Load and analyze the GRN data from CSV file"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} regulatory interactions")
    print(f"Unique TFs: {df['TF'].nunique()}")
    print(f"Unique targets: {df['target'].nunique()}")
    
    # Display importance statistics
    print("\nImportance value statistics:")
    print(f"Min: {df['importance'].min()}")
    print(f"Max: {df['importance'].max()}")
    print(f"Mean: {df['importance'].mean()}")
    print(f"Median: {df['importance'].median()}")
    print(f"Std: {df['importance'].std()}")
    
    return df

def visualize_importance_distribution(df, output_dir="plots"):
    """Visualize the distribution of importance values"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['importance'], bins=50, kde=True)
    plt.title('Distribution of Importance Values')
    plt.xlabel('Importance')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'importance_distribution.png'), dpi=300)
    plt.close()
    
    # Log-scale view
    plt.figure(figsize=(10, 6))
    sns.histplot(np.log1p(df['importance']), bins=50, kde=True)
    plt.title('Log-transformed Distribution of Importance Values')
    plt.xlabel('Log(Importance + 1)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'log_importance_distribution.png'), dpi=300)
    plt.close()
    
    print(f"Importance distribution plots saved to {output_dir}")

def visualize_network_structure(df, max_nodes=1000, output_dir="plots"):
    """Create a visualization of the network structure"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a graph from the data
    G = nx.DiGraph()
    
    # Add edges with importance as weight
    edges = []
    for _, row in df.iterrows():
        G.add_edge(row['TF'], row['target'], weight=row['importance'])
        edges.append((row['TF'], row['target']))
    
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # If graph is too large, visualize a subset
    if G.number_of_nodes() > max_nodes:
        print(f"Graph too large, visualizing top {max_nodes} nodes by degree centrality")
        centrality = nx.degree_centrality(G)
        top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes)
    
    # Calculate node degree (connection count)
    degrees = dict(G.degree())
    
    # Calculate node centrality
    centrality = nx.degree_centrality(G)
    
    # Create plot
    plt.figure(figsize=(12, 12))
    
    # Use spring layout for visualization
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Draw nodes with size based on centrality and color based on in/out degree ratio
    node_size = [v * 500 + 10 for v in centrality.values()]
    
    # Determine node types (TF, target, or both)
    node_types = {}
    for node in G.nodes():
        if G.out_degree(node) > 0 and G.in_degree(node) == 0:
            node_types[node] = 'TF'  # Only a TF
        elif G.out_degree(node) == 0 and G.in_degree(node) > 0:
            node_types[node] = 'Target'  # Only a target
        else:
            node_types[node] = 'Both'  # Both TF and target
    
    # Draw nodes by type
    tf_nodes = [node for node, type_ in node_types.items() if type_ == 'TF']
    target_nodes = [node for node, type_ in node_types.items() if type_ == 'Target']
    both_nodes = [node for node, type_ in node_types.items() if type_ == 'Both']
    
    # Use different colors for different node types
    nx.draw_networkx_nodes(G, pos, nodelist=tf_nodes, node_size=[node_size[i] for i, node in enumerate(G.nodes()) if node in tf_nodes], 
                           node_color='red', alpha=0.7, label='TF only')
    nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, node_size=[node_size[i] for i, node in enumerate(G.nodes()) if node in target_nodes], 
                           node_color='blue', alpha=0.7, label='Target only')
    nx.draw_networkx_nodes(G, pos, nodelist=both_nodes, node_size=[node_size[i] for i, node in enumerate(G.nodes()) if node in both_nodes], 
                           node_color='purple', alpha=0.7, label='TF & Target')
    
    # Draw edges with transparency based on importance
    edge_weights = [G[u][v]['weight'] / df['importance'].max() for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.3, edge_color='gray')
    
    # Don't draw labels if graph is too large
    if G.number_of_nodes() < 100:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(f'Gene Regulatory Network Structure (showing {G.number_of_nodes()} nodes)')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'network_structure.png'), dpi=300)
    plt.close()
    
    print(f"Network structure visualization saved to {output_dir}")
    
    # Create degree distribution plots
    plt.figure(figsize=(12, 6))
    
    # Plot in-degree distribution (how many TFs regulate a gene)
    in_degrees = [d for n, d in G.in_degree()]
    plt.subplot(1, 2, 1)
    plt.hist(in_degrees, bins=20, alpha=0.7, color='blue')
    plt.title('In-Degree Distribution')
    plt.xlabel('Number of regulating TFs')
    plt.ylabel('Number of genes')
    plt.yscale('log')
    
    # Plot out-degree distribution (how many genes a TF regulates)
    out_degrees = [d for n, d in G.out_degree()]
    plt.subplot(1, 2, 2)
    plt.hist(out_degrees, bins=20, alpha=0.7, color='red')
    plt.title('Out-Degree Distribution')
    plt.xlabel('Number of regulated genes')
    plt.ylabel('Number of TFs')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'degree_distribution.png'), dpi=300)
    plt.close()
    
    print(f"Degree distribution plots saved to {output_dir}")

def visualize_preprocessed_data(data_dir, output_dir="plots"):
    """Visualize the preprocessed data used for training"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the preprocessed data
        data = torch.load(os.path.join(data_dir, 'processed_data.pt'))
        
        print("\nPreprocessed data statistics:")
        print(f"Number of nodes: {data['num_nodes']}")
        
        # Training data
        print(f"Training edges: {data['train_edge_index'].size(1)}")
        if 'train_labels' in data:
            pos_ratio = data['train_labels'].float().mean().item()
            print(f"Training positive ratio: {pos_ratio:.4f} ({int(pos_ratio * data['train_labels'].size(0))} / {data['train_labels'].size(0)})")
        
        # Validation data
        print(f"Validation edges: {data['val_edge_index'].size(1)}")
        if 'val_labels' in data:
            pos_ratio = data['val_labels'].float().mean().item()
            print(f"Validation positive ratio: {pos_ratio:.4f} ({int(pos_ratio * data['val_labels'].size(0))} / {data['val_labels'].size(0)})")
        
        # Test data
        print(f"Test edges: {data['test_edge_index'].size(1)}")
        if 'test_labels' in data:
            pos_ratio = data['test_labels'].float().mean().item()
            print(f"Test positive ratio: {pos_ratio:.4f} ({int(pos_ratio * data['test_labels'].size(0))} / {data['test_labels'].size(0)})")
            
        # Check if node features exist
        if 'node_features' in data and data['node_features'] is not None:
            print(f"Node features shape: {data['node_features'].shape}")
            
            # Visualize node features with t-SNE
            print("Creating t-SNE visualization of node features...")
            X = data['node_features'].numpy()
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7, s=10)
            plt.title('t-SNE visualization of node features')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.savefig(os.path.join(output_dir, 'node_features_tsne.png'), dpi=300)
            plt.close()
        
        # Create a network visualization from the training edges
        print("Creating training graph visualization...")
        edge_index = data['train_edge_index']
        
        # Random sample if too large
        if edge_index.size(1) > 10000:
            idx = torch.randperm(edge_index.size(1))[:10000]
            edge_index = edge_index[:, idx]
        
        G = to_networkx(torch.zeros(data['num_nodes']), edge_index.t())
        
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, node_size=5, node_color='blue', edge_color='gray', alpha=0.6, width=0.5)
        plt.title('Training Graph Structure (Sample)')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'training_graph.png'), dpi=300)
        plt.close()
        
        print(f"Preprocessed data visualizations saved to {output_dir}")
        
    except Exception as e:
        print(f"Error visualizing preprocessed data: {e}")

def extract_learning_curve(log_file):
    """Extract training and validation metrics from log file"""
    with open(log_file, 'r') as f:
        log_text = f.read()
    
    # Extract epoch information
    epoch_pattern = r"Epoch (\d+)/\d+ - Train Loss: ([\d\.]+), Val Loss: ([\d\.]+), Val Accuracy: ([\d\.]+), Val ROC-AUC: ([\d\.]+)"
    epoch_matches = re.findall(epoch_pattern, log_text)
    
    if epoch_matches:
        epochs = [int(m[0]) for m in epoch_matches]
        train_loss = [float(m[1]) for m in epoch_matches]
        val_loss = [float(m[2]) for m in epoch_matches]
        val_acc = [float(m[3]) for m in epoch_matches]
        val_roc = [float(m[4]) for m in epoch_matches]
        
        return {
            'epochs': epochs,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_roc': val_roc
        }
    
    return None

def visualize_learning_curves(results_dir, output_dir="plots"):
    """Visualize learning curves from training logs"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all result directories
    result_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    all_curves = {}
    model_types = []
    
    for result_dir in result_dirs:
        # Find the log file
        log_files = glob.glob(os.path.join(results_dir, result_dir, "*.log"))
        if not log_files:
            log_files = glob.glob(os.path.join(results_dir, result_dir, "*/*.log"))
        
        if log_files:
            log_file = log_files[0]
            model_name = result_dir
            
            # Extract curves
            curves = extract_learning_curve(log_file)
            if curves:
                all_curves[model_name] = curves
                
                # Parse model type from directory name
                parts = model_name.split('_')
                if len(parts) >= 2:
                    model_type = parts[1]  # gnn, cnn, transformer, etc.
                    if model_type not in model_types:
                        model_types.append(model_type)
    
    if not all_curves:
        print("No learning curves found in the results directory")
        return
    
    # Plot validation accuracy by model type
    plt.figure(figsize=(12, 8))
    for model_name, curves in all_curves.items():
        parts = model_name.split('_')
        if len(parts) >= 2:
            model_type = parts[1]  # gnn, cnn, transformer, etc.
            model_size = parts[0]  # small, medium, large, xlarge
            
            # Set color based on model type
            color = {'gnn': 'blue', 'transformer': 'green', 'graph': 'red', 'cnn': 'orange'}.get(model_type, 'gray')
            
            # Set line style based on model size
            linestyle = {'small': ':', 'medium': '--', 'large': '-.', 'xlarge': '-'}.get(model_size, '-')
            
            label = f"{model_size}_{model_type}"
            plt.plot(curves['epochs'], curves['val_acc'], label=label, color=color, linestyle=linestyle)
    
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'val_accuracy_comparison.png'), dpi=300)
    plt.close()
    
    # Plot validation ROC-AUC by model type
    plt.figure(figsize=(12, 8))
    for model_name, curves in all_curves.items():
        parts = model_name.split('_')
        if len(parts) >= 2:
            model_type = parts[1]  # gnn, cnn, transformer, etc.
            model_size = parts[0]  # small, medium, large, xlarge
            
            # Set color based on model type
            color = {'gnn': 'blue', 'transformer': 'green', 'graph': 'red', 'cnn': 'orange'}.get(model_type, 'gray')
            
            # Set line style based on model size
            linestyle = {'small': ':', 'medium': '--', 'large': '-.', 'xlarge': '-'}.get(model_size, '-')
            
            label = f"{model_size}_{model_type}"
            plt.plot(curves['epochs'], curves['val_roc'], label=label, color=color, linestyle=linestyle)
    
    plt.title('Validation ROC-AUC over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Validation ROC-AUC')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'val_roc_comparison.png'), dpi=300)
    plt.close()
    
    # Plot training and validation loss for each model
    for model_name, curves in all_curves.items():
        plt.figure(figsize=(12, 6))
        plt.plot(curves['epochs'], curves['train_loss'], label='Training Loss', color='blue')
        plt.plot(curves['epochs'], curves['val_loss'], label='Validation Loss', color='red')
        plt.title(f'Training and Validation Loss - {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_loss.png'), dpi=300)
        plt.close()
    
    print(f"Learning curve visualizations saved to {output_dir}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize GRN data and model results')
    parser.add_argument('--data-dir', type=str, default='processed_data',
                       help='Directory containing processed data')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing model results')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Directory to save visualization outputs')
    parser.add_argument('--visualize-network', action='store_true',
                       help='Generate network structure visualizations')
    parser.add_argument('--visualize-importance', action='store_true',
                       help='Generate importance distribution visualizations')
    parser.add_argument('--visualize-learning', action='store_true',
                       help='Generate learning curve visualizations')
    parser.add_argument('--visualize-preprocessed', action='store_true',
                       help='Generate visualizations of preprocessed data')
    parser.add_argument('--max-nodes', type=int, default=1000,
                       help='Maximum number of nodes to include in network visualization')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Execute requested visualizations
    if args.visualize_importance or args.visualize_network:
        df = load_grn_data('SCING_GRN.csv')
        
        if args.visualize_importance:
            visualize_importance_distribution(df, args.output_dir)
            
        if args.visualize_network:
            visualize_network_structure(df, args.max_nodes, args.output_dir)
    
    if args.visualize_preprocessed:
        visualize_preprocessed_data(args.data_dir, args.output_dir)
        
    if args.visualize_learning:
        visualize_learning_curves(args.results_dir, args.output_dir)

if __name__ == "__main__":
    main() 