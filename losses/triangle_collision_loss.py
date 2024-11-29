import torch
import torch.nn as nn
import torch.nn.functional as F

class TriangleCollisionLoss(nn.Module):
    """
    Triangle collision loss for mesh simplification.
    """
    def __init__(self):
        super(TriangleCollisionLoss, self).__init__()
    
    def forward(self, vertices, faces, face_probs):
        """
        Vectorized computation of triangle collision loss
        """
        # Calculate face normals and centroids all at once
        v1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        v2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
        face_normals = F.normalize(torch.linalg.cross(v1, v2), p=2, dim=1)
        centroids = vertices[faces].mean(dim=1)

        # Compute pairwise distances between centroids
        dists = torch.cdist(centroids, centroids)
        nearby_mask = dists < 1.0  # Adjust threshold as needed

        # Remove self-comparisons
        nearby_mask.fill_diagonal_(False)

        # Find all pairs of triangles to check
        triangle_pairs = nearby_mask.nonzero()

        if len(triangle_pairs) == 0:
            return torch.tensor(0.0, device=vertices.device)

        # Get the relevant triangles
        triangles_i = faces[triangle_pairs[:, 0]]
        triangles_j = faces[triangle_pairs[:, 1]]
        normals_i = face_normals[triangle_pairs[:, 0]]

        # Compute signed distances for all vertices of triangle j against plane of triangle i
        points = vertices[triangles_j]  # Shape: [num_pairs, 3, 3]
        ref_points = vertices[triangles_i[:, 0]].unsqueeze(1)  # Shape: [num_pairs, 1, 3]

        # Compute signed distances for all vertices at once
        signed_dists = torch.einsum('bij,bi->bj', 
                                  points - ref_points, 
                                  normals_i)  # Shape: [num_pairs, 3]

        # Check for sign changes (indicating intersection)
        rolled_dists = torch.roll(signed_dists, 1, dims=1)
        intersections = (signed_dists * rolled_dists < 0).any(dim=1)

        # Count collisions per triangle
        collision_count = torch.zeros(len(faces), device=vertices.device)
        collision_count.scatter_add_(0, triangle_pairs[intersections, 0], 
                                   torch.ones_like(triangle_pairs[intersections, 0], 
                                                 dtype=torch.float32))

        return (face_probs * collision_count).mean()