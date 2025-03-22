#!/bin/bash

echo "===== Running preprocessing with detailed logging ====="

# Set output file for logs
OUTPUT_FILE="preprocessing_log.txt"

# Clean up any previous log file
rm -f $OUTPUT_FILE

# Run preprocessing with all data verification options
python preprocess.py \
    --input SCING_GRN.csv \
    --output processed_data/verified_data \
    --task link_prediction \
    --normalize log \
    --use-node-features | tee $OUTPUT_FILE

echo "===== Preprocessing complete ====="
echo "Log saved to $OUTPUT_FILE"

# Verify the processed data structure
echo "===== Verifying processed data structure ====="

cat >> $OUTPUT_FILE << EOF

===== Data Structure Verification =====
EOF

python -c "
import torch
import numpy as np
from torch_geometric.data import Data

# Safe min/max function to handle empty tensors
def safe_min_max(tensor):
    if tensor.numel() == 0:
        return 'N/A', 'N/A'
    else:
        # Filter out NaN and Inf values for min/max calculation
        filtered = tensor
        if torch.is_floating_point(tensor):
            valid_mask = ~torch.isnan(tensor) & ~torch.isinf(tensor)
            if valid_mask.any():
                filtered = tensor[valid_mask]
            else:
                return 'N/A', 'N/A'
        
        if filtered.numel() > 0:
            return filtered.min().item(), filtered.max().item()
        else:
            return 'N/A', 'N/A'

# Load the processed data
data_path = 'processed_data/verified_data/processed_data.pt'
print(f'Loading data from {data_path}', file=open('$OUTPUT_FILE', 'a'))

try:
    data = torch.load(data_path)
    
    # Print data structure
    print('\\nData structure:', file=open('$OUTPUT_FILE', 'a'))
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            tensor_info = f'Tensor shape: {value.shape}, dtype: {value.dtype}'
            
            # Check for NaNs or infs
            if torch.is_floating_point(value):
                nan_count = torch.isnan(value).sum().item()
                inf_count = torch.isinf(value).sum().item()
                zero_count = (value == 0).sum().item()
                
                min_val, max_val = safe_min_max(value)
                
                tensor_info += f', NaNs: {nan_count}, Infs: {inf_count}, Zeros: {zero_count}, Min: {min_val}, Max: {max_val}'
            
            print(f'  {key}: {tensor_info}', file=open('$OUTPUT_FILE', 'a'))
        elif isinstance(value, np.ndarray):
            array_info = f'Array shape: {value.shape}, dtype: {value.dtype}'
            
            # Check for NaNs or infs
            nan_count = np.isnan(value).sum()
            inf_count = np.isinf(value).sum()
            zero_count = (value == 0).sum()
            
            # Safe min/max for numpy arrays
            if value.size > 0:
                valid_mask = ~np.isnan(value) & ~np.isinf(value)
                if valid_mask.any():
                    min_val = np.min(value[valid_mask])
                    max_val = np.max(value[valid_mask])
                    array_info += f', Min: {min_val}, Max: {max_val}'
            
            array_info += f', NaNs: {nan_count}, Infs: {inf_count}, Zeros: {zero_count}'
            
            print(f'  {key}: {array_info}', file=open('$OUTPUT_FILE', 'a'))
        elif isinstance(value, Data):
            print(f'  {key}: PyTorch Geometric Data object', file=open('$OUTPUT_FILE', 'a'))
            print(f'    Attributes: {value.keys}', file=open('$OUTPUT_FILE', 'a'))
            if hasattr(value, 'edge_index') and value.edge_index is not None:
                print(f'    Edge index shape: {value.edge_index.shape}', file=open('$OUTPUT_FILE', 'a'))
            if hasattr(value, 'edge_attr') and value.edge_attr is not None:
                print(f'    Edge attr shape: {value.edge_attr.shape}', file=open('$OUTPUT_FILE', 'a'))
            if hasattr(value, 'x') and value.x is not None:
                print(f'    Node features shape: {value.x.shape}', file=open('$OUTPUT_FILE', 'a'))
            if hasattr(value, 'num_nodes'):
                print(f'    Number of nodes: {value.num_nodes}', file=open('$OUTPUT_FILE', 'a'))
        else:
            print(f'  {key}: {type(value)}', file=open('$OUTPUT_FILE', 'a'))
    
    # Verify edge indices
    print('\\nEdge index verification:', file=open('$OUTPUT_FILE', 'a'))
    edge_keys = [k for k in data.keys() if k.endswith('_edge_index')]
    for key in edge_keys:
        edge_index = data[key]
        print(f'  {key}:', file=open('$OUTPUT_FILE', 'a'))
        print(f'    Shape: {edge_index.shape}', file=open('$OUTPUT_FILE', 'a'))
        
        if edge_index.shape[0] != 2:
            print(f'    WARNING: First dimension is not 2!', file=open('$OUTPUT_FILE', 'a'))
            
        # Check for invalid node indices
        if 'num_nodes' in data:
            num_nodes = data['num_nodes']
            invalid_indices = (edge_index >= num_nodes).sum().item()
            if invalid_indices > 0:
                print(f'    WARNING: Contains {invalid_indices} indices >= num_nodes ({num_nodes})', file=open('$OUTPUT_FILE', 'a'))
                
        # Check for self-loops
        if edge_index.shape[0] == 2:
            self_loops = (edge_index[0] == edge_index[1]).sum().item()
            print(f'    Self-loops: {self_loops}', file=open('$OUTPUT_FILE', 'a'))
    
    # Verify edge attributes
    print('\\nEdge attribute verification:', file=open('$OUTPUT_FILE', 'a'))
    attr_keys = [k for k in data.keys() if k.endswith('_edge_attr')]
    for key in attr_keys:
        matching_edge_index = key.replace('_edge_attr', '_edge_index')
        edge_attr = data[key]
        print(f'  {key}:', file=open('$OUTPUT_FILE', 'a'))
        print(f'    Shape: {edge_attr.shape}', file=open('$OUTPUT_FILE', 'a'))
        
        if matching_edge_index in data:
            edge_index = data[matching_edge_index]
            if edge_attr.shape[0] != edge_index.shape[1]:
                print(f'    WARNING: Number of edge attributes ({edge_attr.shape[0]}) does not match', 
                      f'number of edges ({edge_index.shape[1]})', file=open('$OUTPUT_FILE', 'a'))
        
        # Check for NaNs, Infs in edge attributes
        if torch.is_floating_point(edge_attr):
            nan_count = torch.isnan(edge_attr).sum().item()
            inf_count = torch.isinf(edge_attr).sum().item()
            if nan_count > 0 or inf_count > 0:
                print(f'    WARNING: Contains {nan_count} NaNs and {inf_count} Infs', file=open('$OUTPUT_FILE', 'a'))
            
            # Distribution of values
            valid_mask = ~torch.isnan(edge_attr) & ~torch.isinf(edge_attr)
            if valid_mask.any():
                valid_attrs = edge_attr[valid_mask]
                min_val = valid_attrs.min().item()
                max_val = valid_attrs.max().item()
                mean_val = valid_attrs.mean().item()
                std_val = valid_attrs.std().item()
                print(f'    Value distribution: Min={min_val:.6f}, Max={max_val:.6f}, Mean={mean_val:.6f}, Std={std_val:.6f}', 
                      file=open('$OUTPUT_FILE', 'a'))
    
    # Verify label balance
    print('\\nLabel distribution:', file=open('$OUTPUT_FILE', 'a'))
    
    # Access labels directly from the data dictionary
    if 'train_labels' in data:
        try:
            # Use data dictionary access consistently
            labels = data['train_labels']
            pos_count = labels.sum().item()
            total_count = len(labels)
            neg_count = total_count - pos_count
            print(f'  Train: {pos_count} positive, {neg_count} negative ({pos_count/total_count:.2%} positive)', 
                  file=open('$OUTPUT_FILE', 'a'))
        except Exception as e:
            print(f'  Error processing train labels: {e}', file=open('$OUTPUT_FILE', 'a'))
    
    if 'val_labels' in data:
        try:
            # Use data dictionary access consistently
            labels = data['val_labels']
            pos_count = labels.sum().item()
            total_count = len(labels)
            neg_count = total_count - pos_count
            print(f'  Val: {pos_count} positive, {neg_count} negative ({pos_count/total_count:.2%} positive)', 
                  file=open('$OUTPUT_FILE', 'a'))
        except Exception as e:
            print(f'  Error processing val labels: {e}', file=open('$OUTPUT_FILE', 'a'))
    
    if 'test_labels' in data:
        try:
            # Use data dictionary access consistently
            labels = data['test_labels']
            pos_count = labels.sum().item()
            total_count = len(labels)
            neg_count = total_count - pos_count
            print(f'  Test: {pos_count} positive, {neg_count} negative ({pos_count/total_count:.2%} positive)', 
                  file=open('$OUTPUT_FILE', 'a'))
        except Exception as e:
            print(f'  Error processing test labels: {e}', file=open('$OUTPUT_FILE', 'a'))
    else:
        print(f'  No test labels found in data', file=open('$OUTPUT_FILE', 'a'))
    
    # Check negative sampling
    print('\\nNegative sampling verification:', file=open('$OUTPUT_FILE', 'a'))
    neg_keys = [k for k in data.keys() if k.endswith('_neg_edge_index')]
    for key in neg_keys:
        neg_edge_index = data[key]
        print(f'  {key}:', file=open('$OUTPUT_FILE', 'a'))
        print(f'    Shape: {neg_edge_index.shape}', file=open('$OUTPUT_FILE', 'a'))
        
        # Corresponding positive edge index
        pos_key = key.replace('_neg_edge_index', '_edge_index')
        if pos_key in data:
            pos_edge_index = data[pos_key]
            print(f'    Positive edges: {pos_edge_index.shape[1]}, Negative edges: {neg_edge_index.shape[1]}', 
                  file=open('$OUTPUT_FILE', 'a'))
            
            # Check if negative edges are truly negative (not in positive edges)
            pos_edge_set = set((pos_edge_index[0, i].item(), pos_edge_index[1, i].item()) 
                              for i in range(pos_edge_index.shape[1]))
            overlap = 0
            for i in range(neg_edge_index.shape[1]):
                edge = (neg_edge_index[0, i].item(), neg_edge_index[1, i].item())
                if edge in pos_edge_set:
                    overlap += 1
            
            if overlap > 0:
                print(f'    WARNING: {overlap} negative edges appear in positive edges!', file=open('$OUTPUT_FILE', 'a'))
    
    # Check node features
    if 'node_features' in data and data['node_features'] is not None:
        print('\\nNode feature statistics:', file=open('$OUTPUT_FILE', 'a'))
        nf = data['node_features'].numpy()
        print(f'  Shape: {nf.shape}', file=open('$OUTPUT_FILE', 'a'))
        print(f'  NaNs: {np.isnan(nf).sum()}', file=open('$OUTPUT_FILE', 'a'))
        print(f'  Infs: {np.isinf(nf).sum()}', file=open('$OUTPUT_FILE', 'a'))
        print(f'  Zeros: {(nf == 0).sum()}', file=open('$OUTPUT_FILE', 'a'))
        
        if nf.size > 0:
            valid_mask = ~np.isnan(nf) & ~np.isinf(nf)
            if valid_mask.any():
                print(f'  Min: {np.min(nf[valid_mask])}', file=open('$OUTPUT_FILE', 'a'))
                print(f'  Max: {np.max(nf[valid_mask])}', file=open('$OUTPUT_FILE', 'a'))
                print(f'  Mean: {np.mean(nf[valid_mask])}', file=open('$OUTPUT_FILE', 'a'))
                print(f'  Std: {np.std(nf[valid_mask])}', file=open('$OUTPUT_FILE', 'a'))
        
        # Feature-wise statistics
        print('\\nFeature-wise statistics:', file=open('$OUTPUT_FILE', 'a'))
        for i in range(nf.shape[1]):
            print(f'  Feature {i}:', file=open('$OUTPUT_FILE', 'a'))
            print(f'    NaNs: {np.isnan(nf[:, i]).sum()}', file=open('$OUTPUT_FILE', 'a'))
            print(f'    Infs: {np.isinf(nf[:, i]).sum()}', file=open('$OUTPUT_FILE', 'a'))
            print(f'    Zeros: {(nf[:, i] == 0).sum()}', file=open('$OUTPUT_FILE', 'a'))
            
            feature_values = nf[:, i]
            valid_mask = ~np.isnan(feature_values) & ~np.isinf(feature_values)
            if valid_mask.any():
                print(f'    Min: {np.min(feature_values[valid_mask])}', file=open('$OUTPUT_FILE', 'a'))
                print(f'    Max: {np.max(feature_values[valid_mask])}', file=open('$OUTPUT_FILE', 'a'))
                print(f'    Mean: {np.mean(feature_values[valid_mask])}', file=open('$OUTPUT_FILE', 'a'))
                print(f'    Std: {np.std(feature_values[valid_mask])}', file=open('$OUTPUT_FILE', 'a'))
                
                # Value distribution (useful for checking normalization)
                print(f'    Percentiles: 10%={np.percentile(feature_values[valid_mask], 10):.4f}, ' +
                      f'25%={np.percentile(feature_values[valid_mask], 25):.4f}, ' +
                      f'50%={np.percentile(feature_values[valid_mask], 50):.4f}, ' +
                      f'75%={np.percentile(feature_values[valid_mask], 75):.4f}, ' +
                      f'90%={np.percentile(feature_values[valid_mask], 90):.4f}', 
                      file=open('$OUTPUT_FILE', 'a'))
            else:
                print(f'    No valid values for feature statistics', file=open('$OUTPUT_FILE', 'a'))
except Exception as e:
    import traceback
    print(f'Error: {e}', file=open('$OUTPUT_FILE', 'a'))
    print(traceback.format_exc(), file=open('$OUTPUT_FILE', 'a'))
" 

echo "Verification complete. Results saved to $OUTPUT_FILE" 