import torch
import torch.nn as nn
import torch.nn.functional as F

class TriangleOverlapLoss(nn.Module):
    """
    Triangle overlap loss for mesh simplification.

    Args:
        num_samples: samples of triangles processed at once.
        batch_size: samples processed at a time.
    """
    def __init__(self, num_samples=10, batch_size = 1024):
        super(TriangleOverlapLoss, self).__init__()
        self.num_samples = num_samples
        self.batch_size = batch_size
    
    def forward(self, vertices, faces, face_probs):
        """
        Vectorized computation of triangle overlap loss
        """
        num_faces = len(faces)
        device = vertices.device

        # Generate barycentric coordinates for all triangles at once
        alpha = torch.rand(num_faces, num_samples, device=device)
        beta = torch.rand(num_faces, num_samples, device=device) * (1 - alpha)
        gamma = 1 - alpha - beta

        # Get all triangle vertices
        v1 = vertices[faces[:, 0]]  # Shape: [num_faces, 3]
        v2 = vertices[faces[:, 1]]
        v3 = vertices[faces[:, 2]]

        # Sample points for all triangles at once
        # Shape: [num_faces, num_samples, 3]
        sampled_points = (alpha.unsqueeze(-1) * v1.unsqueeze(1) +
                         beta.unsqueeze(-1) * v2.unsqueeze(1) +
                         gamma.unsqueeze(-1) * v3.unsqueeze(1))

        # Compute triangle centroids
        centroids = vertices[faces].mean(dim=1)  # Shape: [num_faces, 3]

        # Find potentially overlapping triangles using centroid distances
        dists = torch.cdist(centroids, centroids)
        potential_overlaps = (dists < 1.0).nonzero()  # Adjust threshold as needed

        # Remove self-triangles
        mask = potential_overlaps[:, 0] != potential_overlaps[:, 1]
        potential_overlaps = potential_overlaps[mask]

        if len(potential_overlaps) == 0:
            return torch.tensor(0.0, device=device)

        overlap_count = torch.zeros(num_faces, device=device)

        # Process in batches to avoid memory issues
        for idx in range(0, len(potential_overlaps), batch_size):
            end_idx = min(idx + batch_size, len(potential_overlaps))
            batch_pairs = potential_overlaps[idx:end_idx]

            # Get relevant triangles and points
            tri_i = batch_pairs[:, 0]
            tri_j = batch_pairs[:, 1]

            # Get points from triangle i
            points = sampled_points[tri_i]  # Shape: [batch, num_samples, 3]

            # Get vertices of triangle j
            w1 = vertices[faces[tri_j, 0]]  # Shape: [batch, 3]
            w2 = vertices[faces[tri_j, 1]]
            w3 = vertices[faces[tri_j, 2]]

            # Compute vectors for barycentric coordinates
            v0 = w2 - w1  # Shape: [batch, 3]
            v1 = w3 - w1
            v2 = points - w1.unsqueeze(1)  # Shape: [batch, num_samples, 3]

            # Compute dot products
            d00 = torch.sum(v0 * v0, dim=-1, keepdim=True)  # Shape: [batch, 1]
            d01 = torch.sum(v0 * v1, dim=-1, keepdim=True)
            d11 = torch.sum(v1 * v1, dim=-1, keepdim=True)
            d20 = torch.sum(v2 * v0.unsqueeze(1), dim=-1)  # Shape: [batch, num_samples]
            d21 = torch.sum(v2 * v1.unsqueeze(1), dim=-1)

            # Compute barycentric coordinates
            denom = d00 * d11 - d01 * d01
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w

            # Check which points are inside
            inside = (u >= 0) & (v >= 0) & (w >= 0)  # Shape: [batch, num_samples]
            has_overlap = inside.any(dim=1)

            # Update overlap count
            overlap_count.scatter_add_(0, tri_i[has_overlap], 
                                     torch.ones_like(tri_i[has_overlap], dtype=torch.float32))

        return (face_probs * overlap_count).mean()