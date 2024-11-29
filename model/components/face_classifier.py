import torch
import torch.nn as nn

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
    

