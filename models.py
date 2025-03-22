import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
import math

############################
# GNN Models for GRN
############################

class GNNEdgePredictor(nn.Module):
    """
    GNN-based edge predictor for GRN reconstruction.
    
    Uses a specified GNN variant (GCN, GAT, SAGE, or GIN) to generate node embeddings,
    then predicts edges by concatenating node embeddings for each edge and applying
    a multi-layer perceptron (MLP).
    
    Args:
        num_nodes (int): Number of nodes in the graph
        hidden_channels (int): Size of the hidden dimension
        num_layers (int): Number of GNN layers
        dropout (float): Dropout rate
        gnn_type (str): Which GNN variant to use: 'GCN', 'GAT', 'SAGE', 'GIN'
        use_edge_attr (bool): Whether to use edge attributes (unused here)
        task (str): 'link_prediction' or future expansions
        node_features (torch.Tensor): If provided, these features replace learned embeddings
        feature_dim (int): Dimension of node features if node_features are used
    """
    def __init__(
        self,
        num_nodes,
        hidden_channels=64,
        num_layers=2,
        dropout=0.5,
        gnn_type='GCN',
        use_edge_attr=False,
        task='link_prediction',
        node_features=None,
        feature_dim=None
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.use_edge_attr = use_edge_attr
        self.task = task
        
        # Node embedding or projection
        if node_features is not None:
            self.has_node_features = True
            self.feature_dim = node_features.shape[1]
            self.register_buffer('node_features', node_features)
            self.feature_proj = nn.Linear(self.feature_dim, hidden_channels)
        else:
            self.has_node_features = False
            self.node_emb = nn.Embedding(num_nodes, hidden_channels)
        
        # GNN layers
        self.convs = nn.ModuleList()
        if gnn_type == 'GCN':
            for _ in range(num_layers):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
        elif gnn_type == 'GAT':
            # Example: 4-attention heads, concat=False for dimension consistency
            for _ in range(num_layers):
                self.convs.append(GATConv(hidden_channels, hidden_channels, heads=4, concat=False))
        elif gnn_type == 'SAGE':
            for _ in range(num_layers):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif gnn_type == 'GIN':
            for _ in range(num_layers):
                nn_i = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                self.convs.append(GINConv(nn_i))
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Batch norm for each layer
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])
        
        # Edge predictor MLP
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, 2 * hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
        # Output activation is usually handled in the training loop

    def forward(self, edge_index, edge_attr=None):
        # Node embeddings
        if self.has_node_features:
            x = self.feature_proj(self.node_features)
        else:
            x = self.node_emb.weight
        
        # GNN passes
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        
        # Edge features
        src_idx, dst_idx = edge_index
        src_emb = x[src_idx]
        dst_emb = x[dst_idx]
        
        edge_emb = torch.cat([src_emb, dst_emb], dim=1)
        out = self.edge_predictor(edge_emb)
        return out

############################
# CNN Models for GRN
############################

class CNNEdgePredictor(nn.Module):
    """
    CNN-based edge predictor for GRN reconstruction.
    
    Learns node embeddings, then for each edge, stacks the embeddings of the two nodes as channels
    and passes them through convolutional + pooling layers, followed by an MLP for a final score.
    
    Args:
        num_nodes (int): Total number of nodes
        embedding_dim (int): Size of learned node embeddings
        num_filters (int): Number of convolutional filters
        kernel_size (int): Filter size
        num_layers (int): Number of convolutional layers
        dropout (float): Dropout rate
        task (str): 'link_prediction' or other
    """
    def __init__(
        self,
        num_nodes,
        embedding_dim=64,
        num_filters=64,
        kernel_size=3,
        num_layers=2,
        dropout=0.1,
        task='link_prediction'
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.task = task
        
        # Node embeddings
        self.node_emb = nn.Embedding(num_nodes, embedding_dim)
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        
        # First CNN layer
        self.conv_layers.append(nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(dropout)
        ))
        
        # Additional CNN layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # Global pooling
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Final MLP
        self.pred_layers = nn.Sequential(
            nn.Linear(num_filters, num_filters // 2),
            nn.BatchNorm1d(num_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters // 2, 1)
        )
        
        # Typically apply Sigmoid for link prediction
        self.output_activation = nn.Sigmoid() if self.task == 'link_prediction' else nn.Identity()

    def forward(self, edge_index, edge_attr=None):
        src_idx, dst_idx = edge_index
        
        src_emb = self.node_emb(src_idx)
        dst_emb = self.node_emb(dst_idx)
        
        # Shape: [batch_size, 2, embedding_dim]
        edge_emb = torch.stack([src_emb, dst_emb], dim=1)
        
        x = edge_emb
        for conv in self.conv_layers:
            x = conv(x)
        x = self.pool(x).squeeze(-1)
        
        out = self.pred_layers(x)
        out = self.output_activation(out).squeeze(-1)
        return out

############################
# Transformer Model for GRN
############################

class TransformerEdgePredictor(nn.Module):
    """
    Transformer-based edge predictor for GRN reconstruction.
    
    Utilizes a TransformerEncoder to process node embeddings, then concatenates
    source and target node embeddings to predict edges.
    
    Args:
        num_nodes (int): Number of nodes
        hidden_channels (int): Transformer embedding dimension
        nhead (int): Number of attention heads
        num_layers (int): Number of Transformer layers
        dropout (float): Dropout rate
        task (str): 'link_prediction' or other
    """
    def __init__(
        self,
        num_nodes,
        hidden_channels=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        task='link_prediction'
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.task = task
        
        # Node embeddings
        self.node_emb = nn.Embedding(num_nodes, hidden_channels)
        nn.init.normal_(self.node_emb.weight, mean=0.0, std=0.02)
        
        self.pos_encoder = PositionalEncoding(hidden_channels, dropout)
        
        # Transformer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=nhead,
            dim_feedforward=hidden_channels * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu',
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.pre_out_norm = nn.LayerNorm(2 * hidden_channels)
        
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        self.output_activation = nn.Sigmoid() if self.task == 'link_prediction' else nn.Identity()

    def forward(self, edge_index, edge_attr=None):
        # Identify unique nodes
        src_idx, dst_idx = edge_index
        unique_nodes = torch.unique(torch.cat([src_idx, dst_idx]))
        
        # Embed only those unique nodes
        node_emb = self.node_emb(unique_nodes)
        node_emb = self.pos_encoder(node_emb.unsqueeze(0)).squeeze(0)
        node_emb = self.transformer_encoder(node_emb.unsqueeze(0)).squeeze(0)
        
        # Map node indices back
        idx_map = {idx.item(): i for i, idx in enumerate(unique_nodes)}
        src_emb = torch.stack([node_emb[idx_map[i.item()]] for i in src_idx])
        dst_emb = torch.stack([node_emb[idx_map[i.item()]] for i in dst_idx])
        
        # Concatenate
        edge_emb = torch.cat([src_emb, dst_emb], dim=1)
        edge_emb = self.pre_out_norm(edge_emb)
        
        out = self.edge_predictor(edge_emb)
        out = self.output_activation(out).squeeze(-1)
        return out

class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for Transformers, extended for large sequences.
    
    Args:
        d_model (int): Dim of embeddings
        dropout (float): Dropout rate
        max_len (int): Max sequence length
    """
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # Extend if needed
            new_max_len = seq_len + 1000
            device = x.device
            
            new_pe = torch.zeros(1, new_max_len, self.d_model, device=device)
            position = torch.arange(0, new_max_len, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float, device=device) * 
                (-math.log(10000.0) / self.d_model)
            )
            new_pe[0, :, 0::2] = torch.sin(position * div_term)
            new_pe[0, :, 1::2] = torch.cos(position * div_term)
            self.pe = new_pe
        
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

############################
# Graph Transformer
############################

class GraphTransformerEdgePredictor(nn.Module):
    """
    Combines a GNN with a Transformer to capture both local neighborhood 
    and global relationships, then predicts edges via an MLP.
    
    Args:
        num_nodes (int): Number of nodes
        hidden_channels (int): Node embedding dim
        nhead (int): Attention heads
        gnn_layers (int): Number of GNN layers
        transformer_layers (int): Number of Transformer layers
        dropout (float): Dropout rate
        gnn_type (str): 'GCN', 'GAT', or 'SAGE'
        task (str): 'link_prediction' or other
    """
    def __init__(
        self,
        num_nodes,
        hidden_channels=64,
        nhead=4,
        gnn_layers=2,
        transformer_layers=2,
        dropout=0.1,
        gnn_type='GCN',
        task='link_prediction'
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.task = task
        
        self.node_emb = nn.Embedding(num_nodes, hidden_channels)
        nn.init.normal_(self.node_emb.weight, mean=0.0, std=0.02)
        
        # GNN
        self.gnn_layers_list = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        
        if gnn_type == 'GCN':
            for _ in range(gnn_layers):
                self.gnn_layers_list.append(GCNConv(hidden_channels, hidden_channels))
        elif gnn_type == 'GAT':
            for _ in range(gnn_layers):
                self.gnn_layers_list.append(GATConv(hidden_channels, hidden_channels))
        elif gnn_type == 'SAGE':
            for _ in range(gnn_layers):
                self.gnn_layers_list.append(SAGEConv(hidden_channels, hidden_channels))
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        for _ in range(gnn_layers):
            self.gnn_norms.append(nn.LayerNorm(hidden_channels))
        
        # Positional encoding + Transformer
        self.pos_encoder = PositionalEncoding(hidden_channels, dropout, max_len=20000)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=nhead,
            dim_feedforward=hidden_channels * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu',
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, transformer_layers)
        
        self.pre_out_norm = nn.LayerNorm(2 * hidden_channels)
        
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        self.output_activation = nn.Sigmoid() if task == 'link_prediction' else nn.Identity()

    def forward(self, edge_index, edge_attr=None):
        x = self.node_emb.weight
        
        # GNN layers with residual + layer norm
        for layer, norm in zip(self.gnn_layers_list, self.gnn_norms):
            new_x = layer(x, edge_index)
            new_x = norm(new_x)
            new_x = F.relu(new_x)
            new_x = F.dropout(new_x, p=0.5, training=self.training)
            if new_x.shape == x.shape:
                x = x + new_x
            else:
                x = new_x
        
        x = self.pos_encoder(x.unsqueeze(0))
        x = self.transformer_encoder(x).squeeze(0)
        
        # Edge features
        src_idx, dst_idx = edge_index
        src_emb = x[src_idx]
        dst_emb = x[dst_idx]
        
        edge_emb = torch.cat([src_emb, dst_emb], dim=1)
        edge_emb = self.pre_out_norm(edge_emb)
        
        out = self.edge_predictor(edge_emb)
        out = self.output_activation(out).squeeze(-1)
        return out

############################
# Model Factory
############################

def get_model(model_type, num_nodes, hidden_channels=64, task='link_prediction', **kwargs):
    """
    Instantiate a model class based on model_type. 
    model_type can be 'gnn', 'cnn', 'transformer', 'graph_transformer'.
    """
    if model_type == 'gnn':
        return GNNEdgePredictor(
            num_nodes=num_nodes,
            hidden_channels=hidden_channels,
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.5),
            gnn_type=kwargs.get('gnn_type', 'GCN'),
            task=task
        )
    elif model_type == 'cnn':
        return CNNEdgePredictor(
            num_nodes=num_nodes,
            embedding_dim=kwargs.get('embedding_dim', 64),
            num_filters=kwargs.get('num_filters', 64),
            kernel_size=kwargs.get('kernel_size', 3),
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.1),
            task=task
        )
    elif model_type == 'transformer':
        return TransformerEdgePredictor(
            num_nodes=num_nodes,
            hidden_channels=hidden_channels,
            nhead=kwargs.get('nhead', 4),
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.1),
            task=task
        )
    elif model_type == 'graph_transformer':
        return GraphTransformerEdgePredictor(
            num_nodes=num_nodes,
            hidden_channels=hidden_channels,
            nhead=kwargs.get('nhead', 4),
            gnn_layers=kwargs.get('gnn_layers', 2),
            transformer_layers=kwargs.get('transformer_layers', 2),
            dropout=kwargs.get('dropout', 0.1),
            gnn_type=kwargs.get('gnn_type', 'GCN'),
            task=task
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_model(model_type, num_nodes, task='link_prediction', **kwargs):
    """
    User-friendly wrapper around get_model.
    """
    return get_model(model_type, num_nodes, task=task, **kwargs) 