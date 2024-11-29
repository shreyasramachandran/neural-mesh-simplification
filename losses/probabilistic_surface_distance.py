import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbabilisticSurfaceDistance(nn.Module):
    """
    Computes probabilistic surface distance as described in the paper.

    Args:
        num_points_per_face: Number of points to sample per triangle face.
        k: Number of nearest triangles to consider.
        triangle_penalty_weight: Weight for the triangle count penalty.
    """
    def __init__(self, num_points_per_face=10, k=5, triangle_penalty_weight=1.0):
        super(ProbabilisticSurfaceDistance, self).__init__()
        self.num_points_per_face = num_points_per_face
        self.k = k
        self.triangle_penalty_weight = triangle_penalty_weight
    
    def forward(self, source_vertices, source_faces, target_vertices, target_faces, face_probs):
        """
        Compute probabilistic surface distance as described in the paper
        source = generated mesh (Ss)
        target = ground truth mesh (S)
        """
        # First squeeze the batch dimension from vertices
        source_vertices = source_vertices.squeeze(0)
        target_vertices = target_vertices.squeeze(0)
        target_faces = target_faces.T

        # Calculate barycenters for forward term
        source_barycenters = source_vertices[source_faces].mean(dim=1)
        target_barycenters = target_vertices[target_faces].mean(dim=1)

        # Forward term: d_S,Ss^f
        dist_matrix_forward = compute_batched_distances(source_barycenters, target_barycenters)
        min_distances = torch.min(dist_matrix_forward, dim=1)[0]
        forward_loss = (face_probs * min_distances).sum()

        # For reverse term:
        # 1. Sample points y from generated (source) mesh
        source_points, source_point_faces = sample_points_from_triangle(
            source_vertices, source_faces, self.num_points_per_face)
        # Get probabilities for sampled points
        source_point_probs = face_probs[source_point_faces]

        # Calculate barycenters of source triangles for k-nearest triangle search
        source_tri_barycenters = source_vertices[source_faces].mean(dim=1)

        reverse_loss = 0
        batch_size = 500

        # Process source (generated) points in batches
        for i in range(0, len(source_points), batch_size):
            batch_end = min(i + batch_size, len(source_points))
            batch_source_points = source_points[i:batch_end]
            batch_source_faces = source_point_faces[i:batch_end]

            # For each point y in the batch
            for j in range(batch_end - i):
                point_idx = i + j
                y_point = batch_source_points[j]
                y_face = batch_source_faces[j]
                py = source_point_probs[point_idx]

                # Find k+1 nearest triangles to y in source mesh (using triangle barycenters)
                dists_to_tris = torch.sum((source_tri_barycenters - y_point.unsqueeze(0)) ** 2, dim=1)

                # Get k+1 nearest (including own triangle)
                tri_dists, tri_indices = torch.topk(dists_to_tris, k=self.k+1, largest=False)

                # Exclude own triangle
                mask = tri_indices != y_face
                valid_tri_indices = tri_indices[mask][:k]
                valid_tri_dists = tri_dists[mask][:k]

                if len(valid_tri_indices) > 0:
                    # Get probabilities of k-nearest triangles
                    valid_tri_probs = face_probs[valid_tri_indices]

                    # Compute distance to closest point x on target mesh
                    batch_target_dists = torch.sum((target_barycenters - y_point.unsqueeze(0)) ** 2, dim=1)
                    min_target_dist = torch.min(batch_target_dists)

                    reverse_loss += py * min_target_dist
                    reverse_loss += (1 - py) * (valid_tri_probs * valid_tri_dists).mean()  
            
        # Add triangle count penalty
        # num_source_triangles = len(source_faces)
        # num_target_triangles = len(target_faces)
        # triangle_count_loss = self.triangle_penalty_weight * abs(num_source_triangles - num_target_triangles)
        print('forward_loss',forward_loss)
        print('reverse_loss',reverse_loss)
        # print('triangle_count_loss',triangle_count_loss)

        return forward_loss + reverse_loss 

    def compute_batched_distances(self,x, y, batch_size=1000):
        """
        Compute pairwise distances in a memory-efficient way
        """
        nx = x.shape[0]
        ny = y.shape[0]
        distances = []
        
        for start in range(0, nx, batch_size):
            end = min(start + batch_size, nx)
            batch_dist = torch.cdist(x[start:end], y) ** 2
            distances.append(batch_dist)
        
        return torch.cat(distances, dim=0)

    def sample_points_from_triangle(self,vertices, faces, num_points_per_face):
        """
        Sample points from triangles using barycentric coordinates

        Args:
            vertices: (V, 3) tensor of vertex coordinates
            faces: (F, 3) tensor of face indices
            num_points_per_face: number of points to sample per triangle

        Returns:
            points: (F * num_points_per_face, 3) tensor of sampled points
            face_indices: (F * num_points_per_face) tensor mapping each point to its face
        """
        num_faces = faces.shape[0]

        # Generate random barycentric coordinates
        r1 = torch.sqrt(torch.rand(num_faces, num_points_per_face, device=vertices.device))
        r2 = torch.rand(num_faces, num_points_per_face, device=vertices.device)

        # Barycentric coordinates
        w1 = 1 - r1
        w2 = r1 * (1 - r2)
        w3 = r1 * r2

        # Get vertices for each face
        v1 = vertices[faces[:, 0]]
        v2 = vertices[faces[:, 1]]
        v3 = vertices[faces[:, 2]]

        # Compute points
        points = (w1.unsqueeze(-1) * v1.unsqueeze(1) + 
                w2.unsqueeze(-1) * v2.unsqueeze(1) + 
                w3.unsqueeze(-1) * v3.unsqueeze(1))

        # Reshape to (F * num_points_per_face, 3)
        points = points.reshape(-1, 3)

        # Create face indices for each point
        face_indices = torch.arange(num_faces, device=vertices.device).repeat_interleave(num_points_per_face)

        return points, face_indices