import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeCrossingLoss(nn.Module):
    """
    Edge Crossing Loss for mesh simplification.

    Args:
        batch_size: samples processed at a time.
    """
    def __init__(self, batch_size=1024):
        super(EdgeCrossingLoss, self).__init__()
        self.batch_size = batch_size
    
    def forward(self, vertices, faces, face_probs):
        """
        Batched computation of edge crossing loss with stability checks
        """
        # Input validation
        if torch.isnan(vertices).any():
            print("NaN detected in vertices")
            return torch.tensor(0.0, device=vertices.device)
        if torch.isnan(face_probs).any():
            print("NaN detected in face_probs")
            return torch.tensor(0.0, device=vertices.device)

        try:
            # Extract all edges from faces
            edges1 = torch.stack([faces[:, 0], faces[:, 1]], dim=1)
            edges2 = torch.stack([faces[:, 1], faces[:, 2]], dim=1)
            edges3 = torch.stack([faces[:, 2], faces[:, 0]], dim=1)
            edges = torch.cat([edges1, edges2, edges3], dim=0)

            edge_to_face = torch.arange(len(faces), device=vertices.device).repeat_interleave(3)
            edge_points = vertices[edges]  # Shape: [num_edges, 2, 3]
            edge_centroids = edge_points.mean(dim=1)  # Shape: [num_edges, 3]

            if torch.isnan(edge_centroids).any():
                print("NaN detected in edge_centroids")
                return torch.tensor(0.0, device=vertices.device)

            crossing_count = torch.zeros(len(faces), device=vertices.device)
            num_edges = len(edge_centroids)

            for i in range(0, num_edges, self.batch_size):
                batch_end = min(i + self.batch_size, num_edges)
                batch_centroids = edge_centroids[i:batch_end]

                # Add small epsilon to avoid numerical issues
                dists = torch.cdist(batch_centroids, edge_centroids)
                batch_crosses = torch.where(dists < 1.0 + 1e-6)

                batch_e1_idx = batch_crosses[0] + i
                batch_e2_idx = batch_crosses[1]

                mask = batch_e1_idx < batch_e2_idx
                batch_e1_idx = batch_e1_idx[mask]
                batch_e2_idx = batch_e2_idx[mask]

                if len(batch_e1_idx) == 0:
                    continue

                e1_start = edge_points[batch_e1_idx, 0]
                e1_end = edge_points[batch_e1_idx, 1]
                e2_start = edge_points[batch_e2_idx, 0]
                e2_end = edge_points[batch_e2_idx, 1]

                # Check for zero-length edges
                v1 = e1_end - e1_start
                v2 = e2_end - e2_start
                v3 = e2_start - e1_start

                # Add small epsilon to vectors
                v1 = v1 + 1e-8
                v2 = v2 + 1e-8
                v3 = v3 + 1e-8

                cross1 = torch.linalg.cross(v1, v3)
                cross2 = torch.linalg.cross(v1, v2)

                # Add epsilon to norms
                cross1_norm = torch.norm(cross1, dim=1) + 1e-8
                cross2_norm = torch.norm(cross2, dim=1) + 1e-8

                # More stringent check for valid pairs
                valid_pairs = cross2_norm > 1e-5

                # Initialize and compute t only for valid pairs
                t = torch.zeros_like(cross2_norm)
                if valid_pairs.any():
                    t[valid_pairs] = cross1_norm[valid_pairs] / cross2_norm[valid_pairs]

                # Clip t values to avoid numerical issues
                t = torch.clamp(t, 0.0, 1.0)
                crossings = (t >= 0) & (t <= 1) & valid_pairs

                if torch.isnan(t).any():
                    print(f"NaN detected in t values. Valid pairs: {valid_pairs.sum()}")
                    print(f"cross1_norm min/max: {cross1_norm.min()}/{cross1_norm.max()}")
                    print(f"cross2_norm min/max: {cross2_norm.min()}/{cross2_norm.max()}")
                    continue

                # Update crossing count
                crossing_count.scatter_add_(0, edge_to_face[batch_e1_idx[crossings]], 
                                          torch.ones_like(batch_e1_idx[crossings], dtype=torch.float32))
                crossing_count.scatter_add_(0, edge_to_face[batch_e2_idx[crossings]], 
                                          torch.ones_like(batch_e2_idx[crossings], dtype=torch.float32))

            # Clip crossing count to avoid explosion
            crossing_count = torch.clamp(crossing_count, 0.0, 100.0)

            if torch.isnan(crossing_count).any():
                print("NaN detected in crossing_count")
                return torch.tensor(0.0, device=vertices.device)

            # result = (face_probs * crossing_count).mean()

            # Debug prints before final calculation
            print("Before final calculation:")
            print(f"face_probs shape: {face_probs.shape}")
            print(f"face_probs min/max: {face_probs.min()}/{face_probs.max()}")
            print(f"face_probs has nan: {torch.isnan(face_probs).any()}")
            print(f"crossing_count shape: {crossing_count.shape}")
            print(f"crossing_count min/max: {crossing_count.min()}/{crossing_count.max()}")
            print(f"crossing_count has nan: {torch.isnan(crossing_count).any()}")

            # Check for infinite values
            print(f"face_probs has inf: {torch.isinf(face_probs).any()}")
            print(f"crossing_count has inf: {torch.isinf(crossing_count).any()}")

            # Try to identify problematic indices
            if torch.isnan(face_probs).any():
                nan_indices = torch.where(torch.isnan(face_probs))[0]
                print(f"NaN indices in face_probs: {nan_indices}")

            if torch.isnan(crossing_count).any():
                nan_indices = torch.where(torch.isnan(crossing_count))[0]
                print(f"NaN indices in crossing_count: {nan_indices}")

            # Add safe multiplication
            product = face_probs * crossing_count
            if torch.isnan(product).any():
                print("NaN detected in product")
                print(f"Product min/max: {product[~torch.isnan(product)].min()}/{product[~torch.isnan(product)].max()}")

            # Safe mean calculation
            if torch.all(torch.isnan(product)):
                print("All values are NaN in product")
                return torch.tensor(0.0, device=vertices.device)

            result = product[~torch.isnan(product)].mean() if torch.isnan(product).any() else product.mean()

            if torch.isnan(result):
                print("NaN detected in final result")
                return torch.tensor(0.0, device=vertices.device)

            return result


        except Exception as e:
            print(f"Error in edge crossing loss: {e}")
            return torch.tensor(0.0, device=vertices.device)