import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.pool import knn_graph
from torch_cluster import fps  # Import Farthest Point Sampling
from torch_geometric.utils import degree

class EdgePredictor(nn.Module):
    def __init__(self, hidden_dim=64, k_neighbors=15):
        super(EdgePredictor, self).__init__()
        self.k_neighbors = k_neighbors

        self.W_phi = nn.Linear(hidden_dim, hidden_dim)
        self.W_theta = nn.Linear(3, hidden_dim)
        self.sigma = nn.ReLU()

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, points, features, edge_index):
        # Expected shapes:
        # points: [N, 3] where N is number of points
        # features: [N, hidden_dim]
        # edge_index: [2, E] where E is number of edges
        print(f"Input shapes:")
        print(f"points shape: {points.shape}")
        print(f"features shape: {features.shape}")
        print(f"edge_index shape: {edge_index.shape}")

        # 1. Extend graph with k-NN
        knn_edges = knn_graph(points, k=self.k_neighbors)
        extended_edges = torch.cat([edge_index, knn_edges], dim=1)  # G_ext
        print(f"\nExtended edges shape: {extended_edges.shape}")
        print(f"Number of unique source nodes: {len(torch.unique(extended_edges[0]))}")
        
        # 2. DevConv processing
        dev_features = []
        for i in range(points.size(0)):
            neighbors_idx = extended_edges[1][extended_edges[0] == i]
            # print(f"\nNode {i} has {len(neighbors_idx)} neighbors")
            
            if len(neighbors_idx) == 0:
                print(f"WARNING: Node {i} has no neighbors!")
                continue
            
            coord_diff = points[neighbors_idx] - points[i].unsqueeze(0)
            # print(f"coord_diff shape for node {i}: {coord_diff.shape}")
            
            processed_diff = self.W_theta(coord_diff)
            max_feature = torch.max(processed_diff, dim=0)[0]
            f_i = self.sigma(self.W_phi(max_feature))
            
            dev_features.append(f_i)
            
        dev_features = torch.stack(dev_features)
        print(f"\nDev features shape: {dev_features.shape}")  # Should be [N, hidden_dim]

        # 3. Sparse Self-attention
        N = points.size(0)
        attention_scores = torch.zeros((N, N), device=points.device)

        for i in range(N):
            neighbors = extended_edges[1][extended_edges[0] == i]
            # print(f"\nNode {i} attention computation:")
            # print(f"Number of neighbors: {len(neighbors)}")

            if len(neighbors) == 0:
                print(f"WARNING: No neighbors for node {i} in attention computation!")
                continue

            numerators = torch.exp((self.W_q(dev_features[neighbors])).mm(self.W_k(dev_features[i]).unsqueeze(-1))).squeeze()
            denominator = torch.sum(numerators)
            # print(f"Numerators shape: {numerators.shape}")
            # print(f"Denominator value: {denominator.item()}")
            
            # Check for numerical issues
            if denominator == 0:
                print(f"WARNING: Zero denominator for node {i}!")
                continue
                
            attention_scores[i, neighbors] = numerators / denominator

        print(f"\nAttention scores stats:")
        print(f"Non-zero entries: {torch.count_nonzero(attention_scores)}")
        print(f"Max value: {torch.max(attention_scores)}")
        print(f"Min non-zero value: {torch.min(attention_scores[attention_scores > 0])}")
        
        # 4. Generate final adjacency matrix
        # Create binary adjacency matrix from extended_edges
        N = points.size(0)
        A = torch.zeros((N, N), device=points.device)
        A[extended_edges[0], extended_edges[1]] = 1

        # Compute A_s using the formula
        A_s = torch.matmul(torch.matmul(attention_scores, A), attention_scores.T)
        print(f"\nFinal adjacency matrix stats:")
        print(f"Non-zero entries in A_s: {torch.count_nonzero(A_s)}")
        
        # 5. Generate candidate triangles
        # Get all potential edges at once where A_s > threshold
        potential_edges = (A_s > 0.3).nonzero()  # Shape: [E, 2]

        if potential_edges.size(0) > 0:
            # Create an adjacency list representation using sparse tensor operations
            source_nodes = potential_edges[:, 0]  # Shape: [E]
            target_nodes = potential_edges[:, 1]  # Shape: [E]

            # Count neighbors per node
            neighbor_counts = torch.bincount(source_nodes, minlength=N)
            valid_nodes = (neighbor_counts >= 2).nonzero().squeeze(1)  # Nodes with 2+ neighbors

            if valid_nodes.size(0) > 0:
                # For each valid node, get its neighbors
                valid_source_mask = torch.isin(source_nodes, valid_nodes)
                valid_edges = potential_edges[valid_source_mask]

                # Group edges by source node
                unique_sources, counts = torch.unique(valid_edges[:, 0], return_counts=True)
                cumsum = torch.cat([torch.tensor([0], device=A_s.device), torch.cumsum(counts, 0)])

                # Generate all possible triangles using tensor operations
                triangles_list = []
                probs_list = []

                for idx in range(len(unique_sources)):
                    start_idx = cumsum[idx]
                    end_idx = cumsum[idx + 1]
                    node_neighbors = valid_edges[start_idx:end_idx, 1]

                    # Create all pairs of neighbors using tensor operations
                    n1 = node_neighbors.unsqueeze(1).expand(-1, node_neighbors.size(0))
                    n2 = node_neighbors.unsqueeze(0).expand(node_neighbors.size(0), -1)

                    # Get upper triangular part to avoid duplicates
                    mask = torch.triu(torch.ones_like(n1), diagonal=1) > 0
                    n1_filtered = n1[mask]
                    n2_filtered = n2[mask]

                    if n1_filtered.size(0) > 0:
                        # Create triangles
                        node_expanded = unique_sources[idx].expand_as(n1_filtered)
                        triangle_candidates = torch.stack([node_expanded, n1_filtered, n2_filtered], dim=1)

                        # Compute probabilities in one go
                        probs = (A_s[node_expanded, n1_filtered] + 
                                A_s[node_expanded, n2_filtered] + 
                                A_s[n1_filtered, n2_filtered]) / 3

                        # Filter positive probabilities
                        positive_mask = probs > 0
                        triangles_list.append(triangle_candidates[positive_mask])
                        probs_list.append(probs[positive_mask])

                if triangles_list:
                    triangles = torch.cat(triangles_list, dim=0)
                    triangle_probs = torch.cat(probs_list, dim=0)
                else:
                    triangles = torch.zeros((0, 3), dtype=torch.long, device=A_s.device)
                    triangle_probs = torch.zeros(0, device=A_s.device)
            else:
                triangles = torch.zeros((0, 3), dtype=torch.long, device=A_s.device)
                triangle_probs = torch.zeros(0, device=A_s.device)
        else:
            triangles = torch.zeros((0, 3), dtype=torch.long, device=A_s.device)
            triangle_probs = torch.zeros(0, device=A_s.device)

        print(f"Number of triangles found: {triangles.size(0)}")

        print(f"\nFinal output shapes:")
        print(f"A_s: {A_s.shape}")
        print(f"triangles: {triangles.shape}")
        print(f"triangle_probs: {triangle_probs.shape}")

        return A_s, triangles, triangle_probs
    

class TriConv(nn.Module):
    def __init__(self, hidden_dim):
        super(TriConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 9, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x, points, triangles, triangle_edges):
        # Print input dtypes
        # print("\nTriConv Input dtypes:")
        # print(f"x dtype: {x.dtype}")
        # print(f"points dtype: {points.dtype}")
        # print(f"triangles dtype: {triangles.dtype}")
        # print(f"triangle_edges dtype: {triangle_edges.dtype}")
        
        # Pre-compute all triangle properties in parallel
        triangle_points = points[triangles]
        
        # Compute edge vectors for all triangles at once
        e_ij = triangle_points[:, 0] - triangle_points[:, 1]
        e_ik = triangle_points[:, 0] - triangle_points[:, 2]
        e_jk = triangle_points[:, 1] - triangle_points[:, 2]
        
        edge_vectors = torch.stack([e_ij, e_ik, e_jk], dim=1)
        
        t_max = torch.max(edge_vectors, dim=1)[0]
        t_min = torch.min(edge_vectors, dim=1)[0]
        barycenters = triangle_points.mean(dim=1)
        
        # Force all geometric features to match x's dtype
        t_max = t_max.to(dtype=x.dtype)
        t_min = t_min.to(dtype=x.dtype)
        barycenters = barycenters.to(dtype=x.dtype)
        
        # Initialize output features with same dtype as input
        new_features = torch.zeros_like(x)
        # print(f"new_features dtype: {new_features.dtype}")
        
        batch_size = 512
        for batch_start in range(0, triangle_edges.size(1), batch_size):
            batch_end = min(batch_start + batch_size, triangle_edges.size(1))
            batch_edges = triangle_edges[:, batch_start:batch_end]
            
            src_idx = batch_edges[0]
            tgt_idx = batch_edges[1]
            
            r_n_k = torch.cat([
                t_min[src_idx] - t_min[tgt_idx],
                t_max[src_idx] - t_max[tgt_idx],
                barycenters[src_idx] - barycenters[tgt_idx]
            ], dim=1).to(dtype=x.dtype)
            
            feature_diff = x[src_idx] - x[tgt_idx]
            
            mlp_input = torch.cat([r_n_k, feature_diff], dim=1)
            # print(f"mlp_input dtype: {mlp_input.dtype}")
            
            mlp_output = self.mlp(mlp_input)
            # print(f"mlp_output dtype: {mlp_output.dtype}")
            
            scatter_idx = src_idx.unsqueeze(-1).expand(-1, x.size(1))
            # print(f"scatter_idx dtype: {scatter_idx.dtype}")
            
            # Convert mlp_output to match new_features dtype
            mlp_output = mlp_output.to(dtype=new_features.dtype)
            
            # Debugging before scatter_add_
            # print(f"Before scatter_add_:")
            # print(f"new_features dtype: {new_features.dtype}")
            # print(f"scatter_idx dtype: {scatter_idx.dtype}")
            # print(f"mlp_output dtype: {mlp_output.dtype}")
            
            new_features.scatter_add_(0, scatter_idx, mlp_output)
        
        return new_features

class FaceClassifier(nn.Module):
    def __init__(self, hidden_dim=128, k_neighbors=20):
        super(FaceClassifier, self).__init__()
        self.k_neighbors = k_neighbors
        self.hidden_dim = hidden_dim
        
        self.triconv1 = TriConv(hidden_dim)
        self.triconv2 = TriConv(hidden_dim)
        self.triconv3 = TriConv(hidden_dim)
        
        self.final_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, points, triangles, init_probs):
        print("\nFaceClassifier Input dtypes:")
        print(f"points dtype: {points.dtype}")
        print(f"triangles dtype: {triangles.dtype}")
        print(f"init_probs dtype: {init_probs.dtype}")
        
        if triangles.size(0) == 0:
            return torch.zeros(0, device=points.device)
        
        # Ensure consistent dtype
        dtype = torch.float32  # or torch.float16 if using half precision
        points = points.to(dtype=dtype)
        init_probs = init_probs.to(dtype=dtype)
        print('First Check Passed')
        
        # Initialize features
        features = init_probs.unsqueeze(-1).expand(-1, self.hidden_dim).clone()
        # print(f"Initial features dtype: {features.dtype}")
        print('Second Check Passed')
        
        # Compute barycenters
        barycenters = points[triangles].mean(dim=1)
        print('Third Check Passed')
        
        # Create kNN graph
        k = min(self.k_neighbors, triangles.size(0) - 1)
        
        def batched_knn_graph(x, k, batch_size=1000):
            device = x.device
            num_nodes = x.size(0)
            edge_index = []

            for i in range(0, num_nodes, batch_size):
                batch_end = min(i + batch_size, num_nodes)
                batch_x = x[i:batch_end]

                # Compute KNN for this batch
                batch_dists = torch.cdist(batch_x, x)  # [batch_size, num_nodes]
                _, batch_neighbors = batch_dists.topk(k + 1, dim=1, largest=False)  # +1 because point itself is included

                # Remove self-loops
                batch_neighbors = batch_neighbors[:, 1:]  # Remove first column (self)

                # Create edge index
                source = torch.arange(i, batch_end, device=device).view(-1, 1).expand(-1, k)
                batch_edges = torch.stack([source.reshape(-1), batch_neighbors.reshape(-1)])

                edge_index.append(batch_edges)

            return torch.cat(edge_index, dim=1)
    
        triangle_edges = batched_knn_graph(barycenters, k=k, batch_size=500)
        #triangle_edges = knn_graph(barycenters, k=k, batch=None, loop=False)
        print('Fourth Check Passed')
        
        # Apply TriConv layers
        features = self.triconv1(features, points, triangles, triangle_edges)
        # print(f"After TriConv1 dtype: {features.dtype}")
        print('Fifth Check Passed')
        
        features = self.triconv2(features, points, triangles, triangle_edges)
        # print(f"After TriConv2 dtype: {features.dtype}")
        
        features = self.triconv3(features, points, triangles, triangle_edges)
        # print(f"After TriConv3 dtype: {features.dtype}")
        
        # Final probability computation
        logits = self.final_layer(features).squeeze(-1)
        triangle_probs = torch.sigmoid(logits)
        
        # print(f"Output probabilities dtype: {triangle_probs.dtype}")
        return triangle_probs
    

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


class MeshGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, sample_ratio):
        super(MeshGNN, self).__init__()
        # Point MLP layers (as per image)
        self.mlp1 = PointMLP(3, hidden_dim)  # Point-MLP(3, D)
        self.mlp2 = PointMLP(hidden_dim, hidden_dim)  # Point-MLP(D, D)
        # Custom GNN layer with the new update rule
        self.gnn = CustomGNN(hidden_dim)
        # Farthest Point Sampling (FPS) ratio
        self.sample_ratio = sample_ratio  # Ratio of points to keep after sampling
        # Attention-based refinement layer
        self.attention_layer = AttentionLayer(hidden_dim)
        # Add EdgePredictor and FaceClassifier
        self.edge_predictor = EdgePredictor()
        self.face_classifier = FaceClassifier()
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Pass through the MLP layers
        x = self.mlp1(x)  # Point-MLP(3, D) -> BN + ReLU
        x = self.mlp2(x)  # Point-MLP(D, D) -> BN + ReLU
        # Pass through the custom GNN layer. This gives us embeddings for each point.
        x = self.gnn(x, edge_index)
        
        # Perform Farthest Point Sampling (FPS)
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)  # Handle batch
        sampled_idx = fps(x, batch=batch, ratio=self.sample_ratio).to(x.device)  # Sample based on FPS

        # Gather sampled points using the indices from FPS. This gives us the sampled points.
        sampled_points = data.x[sampled_idx]
        sampled_x = x[sampled_idx]  # Sampled embeddings
        
        # Filter `edge_index` to keep only edges between sampled vertices
        edge_index_filtered = self.filter_edges_and_remap(edge_index, sampled_idx)
        
        # Apply the attention-based refinement layer
        refined_positions = self.attention_layer(sampled_points, sampled_x, edge_index_filtered)
        print('Refined Positions',refined_positions.shape)
        
        # Edge prediction
        adjacency, candidate_triangles, triangle_probs = self.edge_predictor(
            refined_positions, sampled_x, edge_index_filtered
        )
        print('adjacency',adjacency.shape)
        print('candidate_triangles',candidate_triangles.shape)
        print('triangle_probs',triangle_probs.shape)

        # Face classification
        final_triangle_probs = self.face_classifier(
            refined_positions, candidate_triangles, triangle_probs
        )
        print('final_triangle_probs',final_triangle_probs.shape)

        # Filter triangles based on probability threshold (0.5)
        #valid_triangles = candidate_triangles[final_triangle_probs > 0.0000002]
        # Filter both triangles and probabilities using the same mask
        prob_mask = final_triangle_probs > 0.7
        valid_triangles = candidate_triangles[prob_mask]
        valid_probs = final_triangle_probs[prob_mask]
        print(f"Valid triangles shape after filtering: {valid_triangles.shape}")
        print(f"Valid probs shape after filtering: {valid_probs.shape}")

        return {
            'vertices': refined_positions,
            'faces': valid_triangles,
            'face_probabilities': valid_probs
        }
        
        # return refined_positions
    
    def filter_edges_and_remap(self, edge_index, sampled_idx):
        # Create a mapping from original vertex indices to new compacted indices
        idx_map = {int(old_idx): new_idx for new_idx, old_idx in enumerate(sampled_idx.tolist())}
        # Filter edges where both source and destination are in the sampled set
        src, dst = edge_index[0], edge_index[1]
        mask = torch.isin(src, sampled_idx) & torch.isin(dst, sampled_idx)
        # Apply the mask to filter the src and dst
        src_filtered = src[mask]
        dst_filtered = dst[mask]
        # Remap the src and dst indices to the new compacted indices in sampled_idx
        src_remapped = torch.tensor([idx_map[int(s.item())] for s in src_filtered], device=edge_index.device)
        dst_remapped = torch.tensor([idx_map[int(d.item())] for d in dst_filtered], device=edge_index.device)
        # Return the filtered and remapped edge_index
        return torch.stack([src_remapped, dst_remapped], dim=0)
