import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
import networkx as nx
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json

def load_grn_data(file_path='SCING_GRN.csv'):
    """
    Load the GRN data from CSV file
    """
    print(f"Loading GRN data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Print dataset statistics
    print(f"Total edges: {len(df)}")
    print(f"Unique TFs: {df.TF.nunique()}")
    print(f"Unique targets: {df.target.nunique()}")
    print(f"Min importance: {df.importance.min()}")
    print(f"Max importance: {df.importance.max()}")
    print(f"Mean importance: {df.importance.mean()}")
    
    return df

def create_node_mappings(df):
    """
    Create mappings between node names and indices
    """
    # Convert all node identifiers to strings to prevent type comparison issues
    df['TF'] = df['TF'].astype(str)
    df['target'] = df['target'].astype(str)
    
    # Get all unique nodes (both TFs and targets)
    all_nodes = sorted(set(df.TF.unique()) | set(df.target.unique()))
    
    # Create mappings
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    print(f"Total unique nodes: {len(all_nodes)}")
    
    return node_to_idx, idx_to_node, all_nodes

def create_node_features(edge_index, num_nodes, importance_scores=None):
    """
    Generate node features based on network properties.
    
    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes in the graph
        importance_scores: Optional edge importance scores
        
    Returns:
        Node features tensor
    """
    # Create graph
    G = nx.DiGraph()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if importance_scores is not None:
            G.add_edge(src, dst, weight=importance_scores[i].item())
        else:
            G.add_edge(src, dst, weight=1.0)
            
    # Initialize features dictionary for all nodes
    features = {}
    for node in range(num_nodes):
        features[node] = {
            'in_degree': 0,
            'out_degree': 0,
            'total_degree': 0,
            'clustering': 0.0,
            'pagerank': 0.0,
            'hub_score': 0.0,
            'authority_score': 0.0,
            'eigenvector': 0.0
        }
    
    # Calculate basic node degrees
    for node in G.nodes():
        features[node]['in_degree'] = G.in_degree(node)
        features[node]['out_degree'] = G.out_degree(node)
        features[node]['total_degree'] = G.degree(node)
    
    # Normalize degrees by maximum
    max_in_degree = max([features[node]['in_degree'] for node in G.nodes()]) if G.nodes() else 1
    max_out_degree = max([features[node]['out_degree'] for node in G.nodes()]) if G.nodes() else 1
    max_total_degree = max([features[node]['total_degree'] for node in G.nodes()]) if G.nodes() else 1
    
    for node in features:
        features[node]['in_degree'] = features[node]['in_degree'] / max_in_degree if max_in_degree > 0 else 0
        features[node]['out_degree'] = features[node]['out_degree'] / max_out_degree if max_out_degree > 0 else 0
        features[node]['total_degree'] = features[node]['total_degree'] / max_total_degree if max_total_degree > 0 else 0
    
    # Calculate clustering coefficient (in undirected version of the graph)
    undirected_G = G.to_undirected()
    try:
        clustering = nx.clustering(undirected_G)
        for node in G.nodes():
            features[node]['clustering'] = clustering.get(node, 0.0)
    except Exception as e:
        print(f"Warning: Could not compute clustering coefficient: {e}")
    
    # Calculate PageRank with more robust parameters
    try:
        # Use more robust parameters for PageRank calculation
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-6)
        for node in G.nodes():
            features[node]['pagerank'] = pagerank.get(node, 0.0)
    except Exception as e:
        print(f"Warning: Could not compute PageRank: {e}")
        # Fallback: use normalized degree centrality instead
        if G.nodes():
            degree_centrality = nx.degree_centrality(G)
            for node in G.nodes():
                features[node]['pagerank'] = degree_centrality.get(node, 0.0)
    
    # Calculate HITS (hub and authority scores) with error handling
    try:
        # Increase max_iter and tol for better convergence
        hits = nx.hits(G, max_iter=200, tol=1e-6)
        for node in G.nodes():
            features[node]['hub_score'] = hits[0].get(node, 0.0)  # hub scores
            features[node]['authority_score'] = hits[1].get(node, 0.0)  # authority scores
    except Exception as e:
        print(f"Warning: Could not compute HITS scores: {e}")
        # Fallback: use in and out degree as approximations
        if G.nodes():
            in_degree_centrality = nx.in_degree_centrality(G)
            out_degree_centrality = nx.out_degree_centrality(G)
            for node in G.nodes():
                features[node]['authority_score'] = in_degree_centrality.get(node, 0.0)
                features[node]['hub_score'] = out_degree_centrality.get(node, 0.0)
    
    # Calculate eigenvector centrality with error handling
    try:
        # Increase max_iter and tol for better convergence
        eigenvector = nx.eigenvector_centrality(G, max_iter=200, tol=1e-6)
        for node in G.nodes():
            features[node]['eigenvector'] = eigenvector.get(node, 0.0)
    except Exception as e:
        print(f"Warning: Could not compute eigenvector centrality: {e}")
        # Fallback: use degree centrality
        if G.nodes():
            degree_centrality = nx.degree_centrality(G)
            for node in G.nodes():
                features[node]['eigenvector'] = degree_centrality.get(node, 0.0)
    
    # Convert to numpy array
    feature_matrix = np.zeros((num_nodes, 8))
    for node in range(num_nodes):
        feature_matrix[node, 0] = features[node]['in_degree']
        feature_matrix[node, 1] = features[node]['out_degree']
        feature_matrix[node, 2] = features[node]['total_degree']
        feature_matrix[node, 3] = features[node]['clustering']
        feature_matrix[node, 4] = features[node]['pagerank']
        feature_matrix[node, 5] = features[node]['hub_score']
        feature_matrix[node, 6] = features[node]['authority_score']
        feature_matrix[node, 7] = features[node]['eigenvector']
    
    # Fill NaN values with 0 (can happen for some centrality measures)
    orig_nan_count = np.isnan(feature_matrix).sum()
    if orig_nan_count > 0:
        print(f"Found {orig_nan_count} NaN values in node features, replacing with zeros")
    
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
    
    # Normalize each feature column to [0, 1] range
    for i in range(feature_matrix.shape[1]):
        column = feature_matrix[:, i]
        min_val, max_val = column.min(), column.max()
        if max_val > min_val:
            feature_matrix[:, i] = (column - min_val) / (max_val - min_val)
        else:
            # If all values are the same, set to 0 to avoid division by zero
            feature_matrix[:, i] = 0
    
    # Check for and report any remaining issues
    if np.isnan(feature_matrix).any():
        print(f"WARNING: Node features still contain {np.isnan(feature_matrix).sum()} NaN values after processing")
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
    
    if np.isinf(feature_matrix).any():
        print(f"WARNING: Node features contain {np.isinf(feature_matrix).sum()} infinite values")
        feature_matrix = np.nan_to_num(feature_matrix, posinf=1.0, neginf=0.0)
    
    # Final check for problematic values
    print(f"Node feature statistics:")
    print(f"  Min: {feature_matrix.min():.6f}")
    print(f"  Max: {feature_matrix.max():.6f}")
    print(f"  Mean: {feature_matrix.mean():.6f}")
    print(f"  NaNs: {np.isnan(feature_matrix).sum()}")
    print(f"  Infs: {np.isinf(feature_matrix).sum()}")
    
    return torch.FloatTensor(feature_matrix)

def create_edge_index_and_features(df, node_to_idx):
    """
    Create edge index and edge features for PyTorch Geometric
    """
    # Ensure TF and target are strings for consistent mapping
    df['TF'] = df['TF'].astype(str)
    df['target'] = df['target'].astype(str)
    
    # Create edge index (2 x num_edges)
    edge_index = torch.tensor([
        [node_to_idx[tf] for tf in df.TF],
        [node_to_idx[target] for target in df.target]
    ], dtype=torch.long)
    
    # Use importance as edge feature
    edge_attr = torch.tensor(df.importance.values, dtype=torch.float).reshape(-1, 1)
    
    # Check for NaN or Inf values in edge_attr
    if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
        print(f"WARNING: Found {torch.isnan(edge_attr).sum().item()} NaNs and {torch.isinf(edge_attr).sum().item()} Infs in edge attributes")
        edge_attr = torch.nan_to_num(edge_attr, nan=df.importance.median(), posinf=df.importance.max(), neginf=df.importance.min())
    
    # Normalize importance scores for better training
    edge_attr = (edge_attr - edge_attr.min()) / (edge_attr.max() - edge_attr.min())
    
    # For binary labels, use a more robust approach
    # Calculate median importance as threshold
    importance_values = df.importance.values
    
    # Print statistics to understand the distribution
    print(f"Importance value statistics before threshold calculation:")
    print(f"  Min: {np.min(importance_values)}")
    print(f"  Max: {np.max(importance_values)}")
    print(f"  Mean: {np.mean(importance_values)}")
    print(f"  Median: {np.median(importance_values)}")
    print(f"  NaNs: {np.isnan(importance_values).sum()}")
    print(f"  Infs: {np.isinf(importance_values).sum()}")
    
    # Sort values for percentile calculation
    importance_values = np.sort(importance_values)
    
    # Try using percentiles, but if that fails, use a fixed threshold
    try:
        threshold_percentiles = [50, 75, 90]  # median, 75th and 90th percentile
        thresholds = [np.percentile(importance_values, p) for p in threshold_percentiles]
        
        print(f"Successfully calculated percentiles: {threshold_percentiles}")
        print(f"Corresponding thresholds: {thresholds}")
        
        # Select the 75th percentile threshold for our binary labels
        selected_threshold = thresholds[1]  # 75th percentile
        
        # If threshold is NaN, fall back to a fixed value
        if np.isnan(selected_threshold):
            print(f"Percentile threshold is NaN, falling back to fixed threshold")
            raise ValueError("Percentile calculation returned NaN")
    except Exception as e:
        print(f"Error in percentile calculation: {e}")
        # Fallback: use a fixed threshold at 20% from the top of the range
        sorted_vals = np.sort(importance_values)
        idx = int(len(sorted_vals) * 0.8)  # Top 20% as positive examples
        selected_threshold = sorted_vals[idx]
        print(f"Using fallback threshold: {selected_threshold}")
    
    # Create binary labels
    edge_labels = torch.tensor(df.importance.values > selected_threshold, dtype=torch.float)
    
    # If we still have 0 positive examples, force the top 20% to be positive
    if edge_labels.sum().item() == 0:
        print("Warning: No positive examples using threshold. Forcing top 20% to be positive.")
        sorted_idx = np.argsort(df.importance.values)
        top_k = int(len(sorted_idx) * 0.2)  # Top 20%
        
        # Create new labels
        new_labels = torch.zeros_like(edge_labels)
        new_labels[sorted_idx[-top_k:]] = 1.0
        edge_labels = new_labels
    
    print(f"Created positive edge labels using threshold: {selected_threshold}")
    print(f"Positive edges: {edge_labels.sum().item()} ({edge_labels.sum().item()/len(edge_labels):.2%})")
    
    # Final check for NaNs in edge attributes
    if torch.isnan(edge_attr).any():
        print(f"ERROR: Edge attributes still contain NaN values after processing")
        edge_attr = torch.nan_to_num(edge_attr, nan=0.5)  # Use median value (0.5) after normalization
    
    return edge_index, edge_attr, edge_labels

def normalize_importance(importance, method='minmax'):
    """
    Normalize importance scores
    """
    if method == 'minmax':
        return (importance - importance.min()) / (importance.max() - importance.min())
    elif method == 'zscore':
        return (importance - importance.mean()) / importance.std()
    else:
        return importance

def create_adjacency_matrix(df, node_to_idx, normalize=True):
    """
    Create adjacency matrix representation of the GRN
    """
    # Ensure TF and target are strings for consistent mapping
    df['TF'] = df['TF'].astype(str)
    df['target'] = df['target'].astype(str)
    
    n_nodes = len(node_to_idx)
    adj_matrix = np.zeros((n_nodes, n_nodes))
    
    for _, row in df.iterrows():
        src_idx = node_to_idx[row['TF']]
        dst_idx = node_to_idx[row['target']]
        importance = row['importance']
        
        if normalize:
            # We'll normalize the full matrix later
            adj_matrix[src_idx, dst_idx] = importance
        else:
            adj_matrix[src_idx, dst_idx] = importance
    
    if normalize:
        # Normalize the non-zero values
        non_zero_mask = adj_matrix > 0
        if non_zero_mask.sum() > 0:  # Check if there are non-zero values
            adj_matrix[non_zero_mask] = normalize_importance(adj_matrix[non_zero_mask])
    
    # Check and fix any NaN values in the adjacency matrix
    if np.isnan(adj_matrix).any():
        print(f"Warning: Found {np.isnan(adj_matrix).sum()} NaN values in adjacency matrix, replacing with zeros")
        adj_matrix = np.nan_to_num(adj_matrix, nan=0.0)
    
    return adj_matrix

def split_edges(edge_index, edge_attr, edge_labels, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split edges into train, validation, and test sets
    with stratification by label to maintain class balance
    """
    num_edges = edge_index.shape[1]
    
    # Create indices for all edges
    edge_indices = np.arange(num_edges)
    
    # Get labels as numpy for stratification
    labels_np = edge_labels.numpy()
    
    # First split: train+val and test
    train_val_indices, test_indices = train_test_split(
        edge_indices, test_size=test_ratio, random_state=seed,
        stratify=labels_np  # Stratify by labels to maintain class distribution
    )
    
    # Second split: train and val
    train_indices, val_indices = train_test_split(
        train_val_indices, 
        test_size=val_ratio/(1-test_ratio), 
        random_state=seed,
        stratify=labels_np[train_val_indices]  # Stratify by labels
    )
    
    # Split edge index, edge attributes and labels
    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]
    test_edge_index = edge_index[:, test_indices]
    
    train_edge_attr = edge_attr[train_indices]
    val_edge_attr = edge_attr[val_indices]
    test_edge_attr = edge_attr[test_indices]
    
    train_edge_labels = edge_labels[train_indices]
    val_edge_labels = edge_labels[val_indices]
    test_edge_labels = edge_labels[test_indices]
    
    # Check for NaN values in edge attributes
    for name, attr in [("train", train_edge_attr), ("val", val_edge_attr), ("test", test_edge_attr)]:
        if torch.isnan(attr).any():
            print(f"WARNING: Found {torch.isnan(attr).sum().item()} NaN values in {name} edge attributes, replacing with 0.5")
            attr_fixed = torch.nan_to_num(attr, nan=0.5)  # Replace NaNs with 0.5 (middle value after normalization)
            if name == "train":
                train_edge_attr = attr_fixed
            elif name == "val":
                val_edge_attr = attr_fixed
            else:
                test_edge_attr = attr_fixed
    
    print(f"Train edges: {len(train_indices)} ({len(train_indices)/num_edges:.2%})")
    print(f"Validation edges: {len(val_indices)} ({len(val_indices)/num_edges:.2%})")
    print(f"Test edges: {len(test_indices)} ({len(test_indices)/num_edges:.2%})")
    
    print(f"Positive edges in train: {train_edge_labels.sum().item()} ({train_edge_labels.sum().item()/len(train_edge_labels):.2%})")
    print(f"Positive edges in val: {val_edge_labels.sum().item()} ({val_edge_labels.sum().item()/len(val_edge_labels):.2%})")
    print(f"Positive edges in test: {test_edge_labels.sum().item()} ({test_edge_labels.sum().item()/len(test_edge_labels):.2%})")
    
    # Verify negative samples
    print("\nVerifying train negative samples:")
    return (train_edge_index, train_edge_attr, train_edge_labels,
            val_edge_index, val_edge_attr, val_edge_labels,
            test_edge_index, test_edge_attr, test_edge_labels)

def create_smart_negative_samples(df, edge_index, node_features, num_nodes, num_samples, existing_edges=None, upsampling_factor=2):
    """
    Create intelligent negative samples for link prediction
    using node features to create more realistic negative edges
    
    Args:
        df: Original dataframe
        edge_index: Edge indices tensor
        node_features: Node features tensor 
        num_nodes: Number of nodes
        num_samples: Number of negative samples to generate
        existing_edges: Set of existing edges to avoid
        upsampling_factor: Factor to increase number of samples
    """
    print(f"Creating {num_samples * upsampling_factor} intelligent negative samples...")
    
    existing_edges = set() if existing_edges is None else existing_edges
    
    # Add current edges to existing edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        existing_edges.add((src, dst))
    
    # Create a networkx graph from the data
    G = nx.DiGraph()
    
    # Add all nodes 
    for i in range(num_nodes):
        G.add_node(i)
    
    # Add edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        G.add_edge(src, dst)
    
    # Get all TF nodes (nodes with out-degree > 0)
    tf_nodes = [n for n in G.nodes() if G.out_degree(n) > 0]
    
    # Get all target nodes (nodes with in-degree > 0)
    target_nodes = [n for n in G.nodes() if G.in_degree(n) > 0]
    
    # Calculate similarity between node features
    if node_features is not None:
        node_features_np = node_features.numpy()
        
        # NEW: Use 3 different negative sampling strategies for more diversity
        neg_edges = []
        
        # Strategy 1: Hard negative samples based on node feature similarity
        # Find pairs of nodes with similar features but no edge between them
        num_hard_samples = num_samples // 3
        hard_negative_candidates = []
        
        # Calculate pairwise cosine similarity between node features
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Compute similarity between TFs and target nodes
        if len(tf_nodes) > 0 and len(target_nodes) > 0:
            tf_features = node_features_np[tf_nodes]
            target_features = node_features_np[target_nodes]
            
            # Compute similarity matrix
            try:
                similarity_matrix = cosine_similarity(tf_features, target_features)
                
                # Get top K similar pairs for each TF
                K = 5  # Number of candidates per TF
                for i, tf_idx in enumerate(tf_nodes):
                    # Get indices of most similar target nodes
                    most_similar = np.argsort(-similarity_matrix[i])[:K]
                    
                    for j in most_similar:
                        target_idx = target_nodes[j]
                        if (tf_idx, target_idx) not in existing_edges and tf_idx != target_idx:
                            hard_negative_candidates.append((tf_idx, target_idx))
                            existing_edges.add((tf_idx, target_idx))
                
                # Sample from hard negatives
                if len(hard_negative_candidates) > num_hard_samples:
                    # Random sample without replacement
                    hard_indices = np.random.choice(
                        len(hard_negative_candidates), 
                        size=num_hard_samples, 
                        replace=False
                    )
                    hard_samples = [hard_negative_candidates[i] for i in hard_indices]
                else:
                    hard_samples = hard_negative_candidates
                
                if hard_samples:
                    neg_edges.extend(hard_samples)
                
                print(f"Created {len(hard_samples)} hard negative samples based on feature similarity")
            except Exception as e:
                print(f"Error computing hard negative samples: {e}")
                # Will proceed with other strategies
        
        # Strategy 2: Use preferential attachment for negative sampling
        num_preferential_samples = num_samples // 3
        preferential_candidates = []
        
        max_attempts = num_preferential_samples * 5  # Allow more attempts than needed
        attempts = 0
        
        while len(preferential_candidates) < num_preferential_samples and attempts < max_attempts:
            attempts += 1
            
            # Select a TF based on its out-degree (more likely to choose active TFs)
            if len(tf_nodes) > 0:
                tf_weights = [max(1, G.out_degree(n)) for n in tf_nodes]  # Avoid zero weights
                tf_weights = np.array(tf_weights) / np.sum(tf_weights)
                src = np.random.choice(tf_nodes, p=tf_weights)
            else:
                src = np.random.randint(0, num_nodes)
            
            # Select a target based on its in-degree (more likely to choose common targets)
            if len(target_nodes) > 0:
                target_weights = [max(1, G.in_degree(n)) for n in target_nodes]  # Avoid zero weights
                target_weights = np.array(target_weights) / np.sum(target_weights)
                dst = np.random.choice(target_nodes, p=target_weights)
            else:
                dst = np.random.randint(0, num_nodes)
            
            # Ensure edge doesn't exist and source != destination
            if (src, dst) not in existing_edges and src != dst:
                preferential_candidates.append((src, dst))
                existing_edges.add((src, dst))
        
        print(f"Created {len(preferential_candidates)} preferential attachment negative samples")
        neg_edges.extend(preferential_candidates)
        
        # Strategy 3: Community-based negative sampling
        # Connect nodes from different communities
        num_community_samples = num_samples - len(neg_edges)
        
        try:
            # Try to detect communities in the graph
            if nx.is_directed(G):
                G_undirected = G.to_undirected()
            else:
                G_undirected = G
                
            # Use a fast community detection algorithm
            communities = None
            try:
                from community import best_partition
                partition = best_partition(G_undirected)
                # Group nodes by community
                community_to_nodes = {}
                for node, comm_id in partition.items():
                    if comm_id not in community_to_nodes:
                        community_to_nodes[comm_id] = []
                    community_to_nodes[comm_id].append(node)
                communities = list(community_to_nodes.values())
            except ImportError:
                # Fallback to connected components if community detection package not available
                communities = list(nx.connected_components(G_undirected))
            
            # If we found communities
            if communities and len(communities) > 1:
                # Create edges between different communities
                community_samples = []
                
                # For each community pair
                for i in range(len(communities)):
                    for j in range(i+1, len(communities)):
                        comm_i_tfs = [n for n in communities[i] if G.out_degree(n) > 0]
                        comm_j_targets = [n for n in communities[j] if G.in_degree(n) > 0]
                        
                        # Create some random edges between communities
                        for _ in range(min(num_community_samples // (len(communities) * (len(communities)-1)//2) + 1, 10)):
                            if comm_i_tfs and comm_j_targets:
                                src = np.random.choice(comm_i_tfs)
                                dst = np.random.choice(comm_j_targets)
                                if (src, dst) not in existing_edges and src != dst:
                                    community_samples.append((src, dst))
                                    existing_edges.add((src, dst))
                
                print(f"Created {len(community_samples)} community-based negative samples")
                neg_edges.extend(community_samples)
        except Exception as e:
            print(f"Error creating community-based samples: {e}")
        
        # If we still need more negative edges, use random sampling
        remaining_samples = num_samples - len(neg_edges)
        if remaining_samples > 0:
            print(f"Creating {remaining_samples} additional random negative samples")
            random_neg_edges = create_negative_samples(
                edge_index, num_nodes, remaining_samples, existing_edges
            )
            # Convert to tuple list format
            for i in range(random_neg_edges.shape[1]):
                neg_edges.append((random_neg_edges[0, i].item(), random_neg_edges[1, i].item()))
        
        # Create the tensor with negative edges
        if neg_edges:
            neg_edges_tensor = torch.tensor(neg_edges, dtype=torch.long).t()
            return neg_edges_tensor
        else:
            print("Warning: Failed to create smart negative samples, falling back to random sampling")
            return create_negative_samples(edge_index, num_nodes, num_samples, existing_edges)
    else:
        # If no node features, fall back to standard negative sampling
        return create_negative_samples(edge_index, num_nodes, num_samples, existing_edges)

def create_negative_samples(edge_index, num_nodes, num_samples, existing_edges=None):
    """
    Create negative samples (edges that don't exist in the graph)
    """
    existing_edges = set() if existing_edges is None else existing_edges
    
    # Add current edges to existing edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        existing_edges.add((src, dst))
    
    # Create negative samples
    neg_edges = []
    while len(neg_edges) < num_samples:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        
        # Ensure the edge doesn't exist and src != dst
        if (src, dst) not in existing_edges and src != dst:
            neg_edges.append([src, dst])
            existing_edges.add((src, dst))
    
    return torch.tensor(neg_edges, dtype=torch.long).t()

def upsample_and_augment(df, upsampling_factor=2, augmentation_noise=0.05):
    """
    Upsample the dataset by replicating edges and adding small noise to importance values
    """
    print(f"Upsampling data by factor {upsampling_factor}...")
    
    # Original dataset size
    original_size = len(df)
    
    # Create upsampled dataset
    upsampled_df = df.copy()
    
    # Replicate high-importance edges with small random variations
    high_importance_df = df[df['importance'] > df['importance'].median()]
    
    for _ in range(upsampling_factor - 1):
        # Create a copy with small random variations in importance
        new_df = high_importance_df.copy()
        noise = np.random.normal(0, augmentation_noise, size=len(high_importance_df))
        
        # Add noise and ensure values remain positive
        new_df['importance'] = new_df['importance'] + new_df['importance'] * noise
        new_df['importance'] = new_df['importance'].clip(lower=df['importance'].min())
        
        # Append to the upsampled dataset
        upsampled_df = pd.concat([upsampled_df, new_df], ignore_index=True)
    
    print(f"Original dataset: {original_size} edges")
    print(f"Upsampled dataset: {len(upsampled_df)} edges")
    
    return upsampled_df

def save_processed_data(output_dir, data_dict):
    """
    Save processed data to disk
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as a single PyTorch file
    output_file = os.path.join(output_dir, "processed_data.pt")
    torch.save(data_dict, output_file)
    
    print(f"Saved processed data to {output_file}")
    
    # Optionally, save additional format as pickle files
    if False:  # Set to True if you want individual pickle files
        for name, data in data_dict.items():
            with open(os.path.join(output_dir, f"{name}.pkl"), 'wb') as f:
                pickle.dump(data, f)

def preprocess_grn_data(input_file='SCING_GRN.csv', output_dir='processed_data', 
                        val_ratio=0.2, test_ratio=0.15, neg_ratio=1.0, seed=42, 
                        use_node_features=True, upsampling_factor=1, task='link_prediction', 
                        normalize_method='minmax', feature_engineering='standard',
                        edge_weighting='continuous', class_balancing=True,
                        node_embedding_dim=64, add_self_loops=False):
    """
    Main preprocessing function
    
    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save processed data
        val_ratio: Ratio of validation set
        test_ratio: Ratio of test set
        neg_ratio: Ratio of negative samples
        seed: Random seed for reproducibility
        use_node_features: Whether to use node features
        upsampling_factor: Factor for upsampling data
        task: Task type ('link_prediction' or 'regression')
        normalize_method: Method for normalizing importance ('minmax', 'log', 'zscore', 'none')
        feature_engineering: Type of feature engineering ('standard', 'minimal', 'advanced', 'spectral')
        edge_weighting: How to weight edges ('continuous', 'binary', 'log', 'quantile')
        class_balancing: Whether to balance positive/negative classes
        node_embedding_dim: Dimension of learned node embeddings (if using 'spectral' features)
        add_self_loops: Whether to add self loops to the graph
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Log preprocessing parameters
    params = locals()
    print(f"Preprocessing with parameters: {params}")
    
    # Save parameters for later reference
    with open(os.path.join(output_dir, 'preprocessing_params.json'), 'w') as f:
        # Convert non-serializable objects to strings
        params_serializable = {k: str(v) if not isinstance(v, (bool, int, float, str, type(None))) else v 
                               for k, v in params.items()}
        json.dump(params_serializable, f, indent=2)
    
    # Load data
    df = load_grn_data(input_file)
    
    # Check for extreme outliers and handle them
    q1, q3 = df['importance'].quantile(0.25), df['importance'].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    
    # Check if there are extreme outliers
    extreme_outliers = df[df['importance'] > upper_bound]
    if len(extreme_outliers) > 0:
        print(f"Found {len(extreme_outliers)} extreme outliers (values > {upper_bound:.2f})")
        
        if normalize_method != 'none':
            print("Importance will be normalized to handle outliers")
        else:
            print("WARNING: Outliers present but normalization is disabled")
    
    # Apply normalization based on selected method
    print(f"Original importance statistics - Mean: {df['importance'].mean()}, Std: {df['importance'].std()}, Range: {df['importance'].max() - df['importance'].min()}")
    
    if normalize_method == 'minmax':
        scaler = MinMaxScaler()
        df['normalized_importance'] = scaler.fit_transform(df[['importance']])
        print("Applied min-max normalization to importance values")
    elif normalize_method == 'log':
        # Handle zero values
        min_non_zero = df['importance'][df['importance'] > 0].min() if any(df['importance'] > 0) else 1e-6
        df['normalized_importance'] = np.log1p(df['importance'] + min_non_zero * 0.1)
        
        # Re-normalize to [0,1] for consistency
        scaler = MinMaxScaler()
        df['normalized_importance'] = scaler.fit_transform(df[['normalized_importance']])
        print("Applied log transformation and scaling to importance values")
    elif normalize_method == 'zscore':
        scaler = StandardScaler()
        df['normalized_importance'] = scaler.fit_transform(df[['importance']])
        print("Applied z-score normalization to importance values")
    elif normalize_method == 'quantile':
        # Quantile transform to uniform or normal distribution
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(output_distribution='uniform', random_state=seed)
        df['normalized_importance'] = qt.fit_transform(df[['importance']])
        print("Applied quantile normalization to importance values")
    else:
        # No normalization, but create a copy for consistent code
        df['normalized_importance'] = df['importance']
        print("No normalization applied to importance values")
    
    # Log the new statistics
    print(f"Normalized importance statistics - Mean: {df['normalized_importance'].mean()}, " +
                f"Std: {df['normalized_importance'].std()}, " +
                f"Range: {df['normalized_importance'].max() - df['normalized_importance'].min()}")
    
    # Apply edge weighting strategy - IMPROVED VERSION
    if edge_weighting == 'binary':
        # Use a more informed threshold (75th percentile) for binary classification
        threshold = df['normalized_importance'].quantile(0.75)
        df['weighted_importance'] = (df['normalized_importance'] > threshold).astype(float)
        print(f"Applied binary edge weighting with threshold at 75th percentile ({threshold:.4f})")
    elif edge_weighting == 'log':
        # Log importance with improved scaling
        epsilon = 1e-5  # Small constant to avoid log(0)
        df['weighted_importance'] = np.log1p((df['normalized_importance'] + epsilon) * 10) / np.log1p(11)  # Scale to roughly [0,1]
        print("Applied improved logarithmic edge weighting")
    elif edge_weighting == 'quantile':
        # Use more quantile buckets for finer gradation
        df['weighted_importance'] = pd.qcut(df['normalized_importance'], q=10, labels=False) / 9.0
        print("Applied quantile-based edge weighting with 10 buckets")
    else:
        # Use continuous importance with slight adjustments to avoid 0s
        # If there are zeros, shift slightly above zero
        if (df['normalized_importance'] == 0).any():
            min_non_zero = df['normalized_importance'][df['normalized_importance'] > 0].min() 
            if not np.isnan(min_non_zero):
                epsilon = min_non_zero * 0.1
                df['weighted_importance'] = df['normalized_importance'] + epsilon
                # Re-normalize to [0,1]
                df['weighted_importance'] = (df['weighted_importance'] - df['weighted_importance'].min()) / (df['weighted_importance'].max() - df['weighted_importance'].min())
                print(f"Applied continuous edge weights with epsilon={epsilon:.6f} to avoid zeros")
            else:
                df['weighted_importance'] = df['normalized_importance']
                print("Using continuous edge weights")
        else:
            df['weighted_importance'] = df['normalized_importance']
            print("Using continuous edge weights")
    
    # Upsample the data if needed
    if upsampling_factor > 1:
        df = upsample_and_augment(df, upsampling_factor=upsampling_factor)
    
    # Create node mappings
    node_to_idx, idx_to_node, all_nodes = create_node_mappings(df)
    
    # Create edge index and features - use the weighted importance
    df_for_edges = df.copy()
    df_for_edges['importance'] = df['weighted_importance']
    edge_index, edge_attr, edge_labels = create_edge_index_and_features(df_for_edges, node_to_idx)
    
    # Create node features based on selected feature engineering approach
    if use_node_features:
        if feature_engineering == 'minimal':
            # Only use basic degree features
            print("Using minimal node features (degrees only)")
            # Create graph
            G = nx.DiGraph()
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                G.add_edge(src, dst)
                
            # Basic features: in-degree, out-degree, total-degree
            feature_matrix = np.zeros((len(node_to_idx), 3))
            for node in range(len(node_to_idx)):
                in_deg = G.in_degree(node) if node in G else 0
                out_deg = G.out_degree(node) if node in G else 0
                feature_matrix[node, 0] = in_deg
                feature_matrix[node, 1] = out_deg
                feature_matrix[node, 2] = in_deg + out_deg
            
            # Normalize features
            for i in range(feature_matrix.shape[1]):
                max_val = feature_matrix[:, i].max()
                if max_val > 0:
                    feature_matrix[:, i] /= max_val
            
            node_features = torch.FloatTensor(feature_matrix)
        
        elif feature_engineering == 'advanced':
            print("Using advanced node features")
            # Use more advanced features than the standard approach
            # Create graph
            G = nx.DiGraph()
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                if edge_attr is not None:
                    weight = edge_attr[i].item()
                    G.add_edge(src, dst, weight=weight)
                else:
                    G.add_edge(src, dst)
            
            # Add self-loops if specified
            if add_self_loops:
                for node in range(len(node_to_idx)):
                    G.add_edge(node, node)
            
            # Feature list:
            # 1. Normalized in-degree
            # 2. Normalized out-degree
            # 3. Normalized total degree
            # 4. Clustering coefficient
            # 5. PageRank
            # 6. HITS hub score
            # 7. HITS authority score
            # 8. Eigenvector centrality
            # 9. Closeness centrality
            # 10. Betweenness centrality
            
            feature_matrix = np.zeros((len(node_to_idx), 10))
            
            # Basic degree features
            for node in range(len(node_to_idx)):
                in_deg = G.in_degree(node) if node in G else 0
                out_deg = G.out_degree(node) if node in G else 0
                feature_matrix[node, 0] = in_deg
                feature_matrix[node, 1] = out_deg
                feature_matrix[node, 2] = in_deg + out_deg
            
            # Normalize degree features
            for i in range(3):
                max_val = feature_matrix[:, i].max()
                if max_val > 0:
                    feature_matrix[:, i] /= max_val
            
            # Try to get other centrality measures - with robust error handling
            try:
                # Clustering coefficient
                undirected_G = G.to_undirected()
                clustering = nx.clustering(undirected_G)
                for node in range(len(node_to_idx)):
                    feature_matrix[node, 3] = clustering.get(node, 0.0)
            except Exception as e:
                print(f"Could not compute clustering coefficient: {e}")
            
            try:
                # PageRank
                pagerank = nx.pagerank(G, alpha=0.85, max_iter=300)
                for node in range(len(node_to_idx)):
                    feature_matrix[node, 4] = pagerank.get(node, 0.0)
            except Exception as e:
                print(f"Could not compute PageRank: {e}")
            
            try:
                # HITS
                hits = nx.hits(G, max_iter=300)
                for node in range(len(node_to_idx)):
                    feature_matrix[node, 5] = hits[0].get(node, 0.0)  # hub
                    feature_matrix[node, 6] = hits[1].get(node, 0.0)  # authority
            except Exception as e:
                print(f"Could not compute HITS: {e}")
            
            try:
                # Eigenvector centrality
                eigenvector = nx.eigenvector_centrality(G, max_iter=300)
                for node in range(len(node_to_idx)):
                    feature_matrix[node, 7] = eigenvector.get(node, 0.0)
            except Exception as e:
                print(f"Could not compute eigenvector centrality: {e}")
            
            try:
                # Closeness centrality - may be slow on large graphs
                # Get largest weakly connected component
                largest_wcc = max(nx.weakly_connected_components(G), key=len)
                wcc_subgraph = G.subgraph(largest_wcc)
                
                closeness = nx.closeness_centrality(wcc_subgraph)
                for node in range(len(node_to_idx)):
                    feature_matrix[node, 8] = closeness.get(node, 0.0)
            except Exception as e:
                print(f"Could not compute closeness centrality: {e}")
            
            try:
                # Betweenness centrality - may be very slow on large graphs
                # Use approximate betweenness with sampling
                betweenness = nx.betweenness_centrality(G, k=min(100, len(G)), normalized=True)
                for node in range(len(node_to_idx)):
                    feature_matrix[node, 9] = betweenness.get(node, 0.0)
            except Exception as e:
                print(f"Could not compute betweenness centrality: {e}")
            
            # Handle NaN values
            feature_matrix = np.nan_to_num(feature_matrix)
            
            # Normalize all features to [0,1] range
            for i in range(feature_matrix.shape[1]):
                min_val, max_val = feature_matrix[:, i].min(), feature_matrix[:, i].max()
                if max_val > min_val:
                    feature_matrix[:, i] = (feature_matrix[:, i] - min_val) / (max_val - min_val)
            
            node_features = torch.FloatTensor(feature_matrix)
        
        elif feature_engineering == 'spectral':
            print(f"Using enhanced spectral embedding node features with dimension {node_embedding_dim}")
            # Spectral embeddings using the graph structure
            
            # Create graph
            G = nx.DiGraph()
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                # Use edge weights for weighted spectral embedding
                if edge_attr is not None:
                    weight = edge_attr[i].item()
                    G.add_edge(src, dst, weight=weight)
                else:
                    G.add_edge(src, dst)
            
            # Ensure all nodes are in the graph
            for node in range(len(node_to_idx)):
                if node not in G:
                    G.add_node(node)
            
            # Use a combined approach for better node features
            feature_matrix = np.zeros((len(node_to_idx), node_embedding_dim))
            
            # First, add some basic structural features (4 features)
            for node in range(len(node_to_idx)):
                in_deg = G.in_degree(node) if node in G else 0
                out_deg = G.out_degree(node) if node in G else 0
                
                # Basic structural features
                feature_matrix[node, 0] = in_deg  # In-degree
                feature_matrix[node, 1] = out_deg  # Out-degree
                # Two more derived features
                feature_matrix[node, 2] = in_deg + out_deg  # Total degree
                feature_matrix[node, 3] = out_deg - in_deg  # Degree differential (source vs sink)
            
            # Normalize these first 4 features
            for i in range(4):
                max_val = np.max(np.abs(feature_matrix[:, i]))
                if max_val > 0:
                    feature_matrix[:, i] /= max_val
            
            # Then try to get spectral embeddings for the remaining dimensions
            try:
                # Try DeepWalk or Node2Vec if available for better embeddings
                try:
                    # Try node2vec if available
                    from node2vec import Node2Vec
                    
                    # Convert directed graph to undirected for node2vec
                    G_undirected = G.to_undirected()
                    
                    # Create node2vec model
                    print("Using node2vec for generating node embeddings")
                    n2v_model = Node2Vec(G_undirected, 
                                        dimensions=node_embedding_dim-4,  # Account for the 4 features we already added
                                        walk_length=10, 
                                        num_walks=50, 
                                        workers=4,
                                        seed=seed)
                    
                    # Train the model
                    n2v_model = n2v_model.fit(window=5, min_count=1)
                    
                    # Get embeddings for all nodes
                    for node in range(len(node_to_idx)):
                        # Check if node is in vocabulary
                        try:
                            node_str = str(node)  # node2vec keys are strings
                            if node_str in n2v_model.wv:
                                feature_matrix[node, 4:] = n2v_model.wv[node_str]
                        except:
                            # If any error, leave as zeros
                            pass
                    
                except ImportError:
                    # Fallback to spectral embedding
                    raise ImportError("node2vec not available, falling back to spectral embedding")
                    
            except Exception as e:
                print(f"Error in advanced node embedding: {e}")
                print("Falling back to spectral embedding")
                
                try:
                    # Fallback to spectral embedding
                    # The graph is connected, use all nodes
                    undirected_G = G.to_undirected()
                    from sklearn.manifold import SpectralEmbedding
                    
                    print("Using spectral embedding for remaining node features")
                    spectral_dim = node_embedding_dim - 4  # Account for the 4 features we already added
                    
                    # Generate adjacency matrix
                    adj_matrix = nx.adjacency_matrix(undirected_G).todense()
                    
                    # Apply spectral embedding
                    embedding = SpectralEmbedding(n_components=spectral_dim, 
                                               random_state=seed).fit_transform(adj_matrix)
                    
                    # Add spectral embedding features
                    feature_matrix[:, 4:] = embedding
                    
                except Exception as e2:
                    print(f"Error in spectral embedding: {e2}")
                    print("Using random initialization for remaining features")
                    
                    # Initialize remaining features randomly if all else fails
                    np.random.seed(seed)
                    feature_matrix[:, 4:] = np.random.normal(0, 0.1, (len(node_to_idx), node_embedding_dim-4))
            
            # Make sure there are no NaN values
            feature_matrix = np.nan_to_num(feature_matrix)
            
            node_features = torch.FloatTensor(feature_matrix)
            
        else:  # standard features
            # Default to original node feature creation
            node_features = create_node_features(edge_index, len(node_to_idx), edge_attr.squeeze(1) if edge_attr is not None else None)
            
        print(f"Created node features with shape: {node_features.shape}")
    else:
        node_features = None
    
    # Split edges
    (train_edge_index, train_edge_attr, train_edge_labels,
     val_edge_index, val_edge_attr, val_edge_labels,
     test_edge_index, test_edge_attr, test_edge_labels) = split_edges(
        edge_index, edge_attr, edge_labels, val_ratio, test_ratio
    )
    
    # Create adjacency matrix
    adj_matrix = create_adjacency_matrix(df_for_edges, node_to_idx, normalize=True)
    
    # Create negative samples for link prediction tasks
    n_nodes = len(node_to_idx)
    
    # Class balancing if requested
    if class_balancing:
        # Calculate number of positive examples in each split
        n_train_pos = train_edge_labels.sum().item()
        n_val_pos = val_edge_labels.sum().item() 
        n_test_pos = test_edge_labels.sum().item()
        
        print(f"Class balancing: Creating the same number of negative examples as positive examples")
        
        # Create the same number of negative samples as positive examples
        train_neg_samples = int(n_train_pos)
        val_neg_samples = int(n_val_pos)
        test_neg_samples = int(n_test_pos)
        
        print(f"Class balancing samples: Train={train_neg_samples}, Val={val_neg_samples}, Test={test_neg_samples}")
    else:
        # Use the same number of negative samples as the total edges in each split
        train_neg_samples = train_edge_index.shape[1]
        val_neg_samples = val_edge_index.shape[1]
        test_neg_samples = test_edge_index.shape[1]
    
    # Use intelligent negative sampling
    train_neg_edge_index = create_smart_negative_samples(
        df, train_edge_index, node_features, n_nodes, 
        train_neg_samples, upsampling_factor=1
    )
    
    val_neg_edge_index = create_smart_negative_samples(
        df, torch.cat([train_edge_index, val_edge_index], dim=1),
        node_features, n_nodes, val_neg_samples,
        upsampling_factor=1
    )
    
    test_neg_edge_index = create_smart_negative_samples(
        df, torch.cat([train_edge_index, val_edge_index, test_edge_index], dim=1),
        node_features, n_nodes, test_neg_samples,
        upsampling_factor=1
    )
    
    # Create PyTorch Geometric data object with different edge options
    if add_self_loops:
        print("Adding self-loops to the graph")
        # Add self-loops to edge_index
        self_loops = torch.arange(n_nodes).repeat(2, 1)
        edge_index_with_loops = torch.cat([edge_index, self_loops], dim=1)
        
        # Add self-loop edge attributes (set to 1.0)
        self_loop_attr = torch.ones(n_nodes, 1)
        edge_attr_with_loops = torch.cat([edge_attr, self_loop_attr], dim=0) if edge_attr is not None else None
        
        data = Data(
            edge_index=edge_index_with_loops,
            edge_attr=edge_attr_with_loops,
            x=node_features,
            num_nodes=n_nodes
        )
    else:
        data = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            x=node_features,
            num_nodes=n_nodes
        )
    
    # Save processed data
    processed_data = {
        'data': data,
        'node_to_idx': node_to_idx,
        'idx_to_node': idx_to_node,
        'all_nodes': all_nodes,
        'node_features': node_features,
        'adj_matrix': adj_matrix,
        'num_nodes': n_nodes,
        'train_edge_index': train_edge_index,
        'train_edge_attr': train_edge_attr,
        'train_labels': train_edge_labels,
        'val_edge_index': val_edge_index,
        'val_edge_attr': val_edge_attr,
        'val_labels': val_edge_labels,
        'test_edge_index': test_edge_index,
        'test_edge_attr': test_edge_attr,
        'test_labels': test_edge_labels,
        'train_neg_edge_index': train_neg_edge_index,
        'val_neg_edge_index': val_neg_edge_index,
        'test_neg_edge_index': test_neg_edge_index,
        'preprocessing_params': params_serializable
    }
    
    save_processed_data(output_dir, processed_data)
    
    return processed_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess GRN data')
    parser.add_argument('--input', type=str, default='SCING_GRN.csv', help='Input CSV file')
    parser.add_argument('--output', type=str, default='processed_data', help='Output directory')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test ratio')
    parser.add_argument('--neg-ratio', type=float, default=1.0, help='Negative sampling ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use-node-features', action='store_true', help='Whether to use node features')
    parser.add_argument('--upsampling-factor', type=int, default=1, help='Upsampling factor for data augmentation')
    parser.add_argument('--task', type=str, default='link_prediction', choices=['link_prediction', 'regression'], help='Task type')
    parser.add_argument('--normalize', type=str, choices=['minmax', 'log', 'zscore', 'quantile', 'none'], default='minmax',
                        help='Method to normalize importance values')
    parser.add_argument('--feature-engineering', type=str, choices=['standard', 'minimal', 'advanced', 'spectral'], 
                        default='standard', help='Type of feature engineering to use')
    parser.add_argument('--edge-weighting', type=str, choices=['continuous', 'binary', 'log', 'quantile'], 
                        default='continuous', help='How to weight edges')
    parser.add_argument('--class-balancing', action='store_true', help='Whether to balance positive/negative classes')
    parser.add_argument('--node-embedding-dim', type=int, default=64, help='Dimension for spectral node embeddings')
    parser.add_argument('--add-self-loops', action='store_true', help='Whether to add self-loops to the graph')
    
    args = parser.parse_args()
    
    # Preprocess data
    preprocess_grn_data(
        input_file=args.input,
        output_dir=args.output,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
        use_node_features=args.use_node_features,
        upsampling_factor=args.upsampling_factor,
        task=args.task,
        normalize_method=args.normalize,
        feature_engineering=args.feature_engineering,
        edge_weighting=args.edge_weighting,
        class_balancing=args.class_balancing,
        node_embedding_dim=args.node_embedding_dim,
        add_self_loops=args.add_self_loops
    ) 