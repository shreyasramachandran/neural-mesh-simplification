import torch
import torch.nn as nn
from torch_geometric.nn.pool import knn_graph

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