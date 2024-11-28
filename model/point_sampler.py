import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import fps 

class PointMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PointMLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CustomGNN(MessagePassing):
    def __init__(self, hidden_dim):
        super(CustomGNN, self).__init__(aggr='mean')  # Aggregate by mean
        # Define linear transformations
        self.W_c = nn.Linear(hidden_dim, hidden_dim)  # Transformation for node's own features
        self.W_n = nn.Linear(hidden_dim, hidden_dim)  # Transformation for neighbor features
        self.relu = nn.ReLU()  # ReLU activation after the GNN output

    def forward(self, x, edge_index):
        # Apply W_c to node's own features
        self_feature_transformed = self.W_c(x)
        # Message passing: aggregate neighbors' features
        out = self.propagate(edge_index, x=x)
        # Add the transformed self features to the aggregated neighbor features
        out = self_feature_transformed + out
        # Apply ReLU activation after the GNN aggregation
        out = self.relu(out)
        return out

    def message(self, x_j):
        # Apply W_n to the neighboring nodes' features
        return self.W_n(x_j)

    def update(self, aggr_out):
        # Return the final update (already aggregated by the mean)
        return aggr_out

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_neighbors=15):
        super(AttentionLayer, self).__init__()
        
        # Define MLPs
        self.mlp_concat = nn.Linear(hidden_dim + 3, hidden_dim + 3)  # Process concatenated vector (d+3)
        self.mlp_output = nn.Linear(hidden_dim + 3, 3)  # Output MLP to get a 3D vector
        self.theta_q = nn.Linear(hidden_dim, hidden_dim // 2)  # Query transformation
        self.theta_k = nn.Linear(hidden_dim, hidden_dim // 2)  # Key transformation
        self.num_neighbors = num_neighbors  # Fixed number of neighbors
    
    def forward(self, sampled_points, sampled_x, edge_index_filtered):
        # Get neighbors from filtered edge_index
        src, dst = edge_index_filtered
        
        # Initialize an empty tensor for the displacement to be applied to each point
        weighted_displacements = torch.zeros_like(sampled_points)  # Shape [n, 3]
        
        for i in range(sampled_points.size(0)):  # Loop over all points in sampled_points
            # Find neighbors of the current point i using edge_index_filtered
            neighbors_idx = dst[src == i]
            
            # If there are fewer than `num_neighbors`, pad the neighbor indices
            if len(neighbors_idx) < self.num_neighbors:
                padding = torch.zeros(self.num_neighbors - len(neighbors_idx), dtype=torch.long, device=sampled_points.device)
                neighbors_idx = torch.cat([neighbors_idx, padding], dim=0)  # Shape [num_neighbors]
            else:
                neighbors_idx = neighbors_idx[:self.num_neighbors]  # Keep only the first `num_neighbors`

            # Compute relative positions for all neighbors
            relative_positions = sampled_points[neighbors_idx] - sampled_points[i].unsqueeze(0)  # Shape [num_neighbors, 3]
            
            # Compute query and key features
            q_features = self.theta_q(sampled_x[i].unsqueeze(0))  # Shape [1, hidden_dim // 2]
            k_features = self.theta_k(sampled_x[neighbors_idx])  # Shape [num_neighbors, hidden_dim // 2]
            
            # Attention scores using scaled dot-product
            scale_factor = torch.sqrt(torch.tensor(q_features.size(-1), dtype=torch.float32, device=q_features.device)) + 1e-6
            attention_scores = (q_features * k_features).sum(dim=-1) / scale_factor  # Shape [num_neighbors]

            # Mask out the padding when calculating attention scores (assign very low scores to padding)
            mask = (neighbors_idx != 0).float()  # 1 where valid neighbors, 0 where padding
            attention_scores = F.softmax(attention_scores * mask, dim=0)  # Shape [num_neighbors]

            # Concatenate latent features and relative positions for all neighbors
            concatenated_features = torch.cat([sampled_x[neighbors_idx], relative_positions], dim=-1)  # Shape [num_neighbors, d+3]
            
            # Pass concatenated vector through MLP
            processed_features = self.mlp_concat(concatenated_features)  # Shape [num_neighbors, d+3]
            
            # Apply attention scores to processed features
            weighted_features = attention_scores.unsqueeze(-1) * processed_features  # Shape [num_neighbors, d+3]
            
            # Sum the weighted features over the neighbors
            aggregated_features = weighted_features.sum(dim=0)  # Shape [d+3]
            
            # Pass aggregated features through the final MLP to get the 3D displacement
            displacement = self.mlp_output(aggregated_features.unsqueeze(0))  # Shape [1, 3]
            
            # Update the displacement for the current point
            weighted_displacements[i] += displacement.squeeze(0)  # Shape [3]

        # Update the positions of the sampled points
        refined_positions = sampled_points + weighted_displacements
        
        return refined_positions


class PointSampler(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_neighbors=15, sample_ratio=0.5):
        super(PointSampler, self).__init__()
        
        # Initialize the point processing MLPs
        self.mlp1 = PointMLP(input_dim, hidden_dim)
        self.mlp2 = PointMLP(hidden_dim, hidden_dim)
        
        # Initialize the GNN layer
        self.gnn = CustomGNN(hidden_dim)
        
        # Initialize the attention layer
        self.attention_layer = AttentionLayer(hidden_dim, num_neighbors)
        
        # Store configuration
        self.sample_ratio = sample_ratio
        self.hidden_dim = hidden_dim
    
    def forward(self, points, edge_index, batch=None):
        """
        Args:
            points (torch.Tensor): Input points tensor of shape [N, input_dim]
            edge_index (torch.Tensor): Edge indices tensor of shape [2, E]
            batch (torch.Tensor, optional): Batch assignments for points in batch processing
        
        Returns:
            dict containing:
                - sampled_points: Tensor of sampled point coordinates
                - sampled_features: Tensor of point features
                - filtered_edges: Filtered and remapped edge indices
        """
        # Process points through MLPs
        x = self.mlp1(points)
        x = self.mlp2(x)
        
        # Apply GNN to get point embeddings
        x = self.gnn(x, edge_index)
        
        # Handle batching
        if batch is None:
            batch = torch.zeros(points.size(0), dtype=torch.long, device=points.device)
        
        # Perform FPS sampling
        sampled_idx = fps(x, batch=batch, ratio=self.sample_ratio)
        
        # Gather sampled points and their features
        sampled_points = points[sampled_idx]
        sampled_features = x[sampled_idx]
        
        # Filter and remap edges for sampled points
        filtered_edges = self._filter_edges_and_remap(edge_index, sampled_idx)
        
        # Apply attention-based refinement
        refined_positions = self.attention_layer(
            sampled_points, 
            sampled_features, 
            filtered_edges
        )
        
        return {
            'sampled_points': refined_positions,
            'sampled_features': sampled_features,
            'filtered_edges': filtered_edges
        }
    
    def _filter_edges_and_remap(self, edge_index, sampled_idx):
        """Helper method to filter edges and remap indices after sampling."""
        # Create mapping from old to new indices
        idx_map = {int(old_idx): new_idx for new_idx, old_idx in enumerate(sampled_idx.tolist())}
        
        # Filter edges
        src, dst = edge_index[0], edge_index[1]
        mask = torch.isin(src, sampled_idx) & torch.isin(dst, sampled_idx)
        
        # Apply mask and remap indices
        src_filtered = src[mask]
        dst_filtered = dst[mask]
        
        # Remap indices to new compact form
        src_remapped = torch.tensor(
            [idx_map[int(s.item())] for s in src_filtered], 
            device=edge_index.device
        )
        dst_remapped = torch.tensor(
            [idx_map[int(d.item())] for s in dst_filtered], 
            device=edge_index.device
        )
        
        return torch.stack([src_remapped, dst_remapped], dim=0)

