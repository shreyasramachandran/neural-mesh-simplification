import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_batched_distances(x, y, batch_size=1000):
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


def sample_points_from_triangle(vertices, faces, num_points_per_face):
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

def calculate_curvature(points, k_neighbors=4):
    """
    Calculate curvature values for points using the covariance matrix and eigenvalue decomposition.

    points: Tensor of shape (B, N, D) - Point cloud data, where B is the batch size, N is the number of points, and D is the dimension (3D).
    k_neighbors: The number of neighbors to consider for curvature calculation.

    Returns:
    curvatures: Tensor of shape (B, N), where each value represents the curvature of a point.
    """
    B, N, D = points.shape  # B is batch size, N is number of points, D is 3 for 3D
    curvatures = torch.zeros(B, N, device=points.device)
    # print('curvatures',curvatures)

    # Compute pairwise distances between all points in each batch
    dist_matrix = torch.cdist(points, points, p=2)  # Shape: (B, N, N)
    # print('dist_matrix',dist_matrix)

    # Get the indices of the k-nearest neighbors for each point
    # Shape: (B, N, k_neighbors)
    knn_indices = dist_matrix.topk(
        k=k_neighbors, largest=False, dim=-1).indices
    # print('knn_indices',knn_indices)

    # Gather the k-nearest neighbors for each point
    neighbors = torch.gather(points.unsqueeze(1).expand(
        B, N, N, D), 2, knn_indices.unsqueeze(-1).expand(B, N, k_neighbors, D))
    # print('neighbors',neighbors)

    # Compute the covariance matrix for each point's neighborhood
    diffs = neighbors - points.unsqueeze(2)  # Shape: (B, N, k_neighbors, D)
    # print('diffs',diffs)
    covariance_matrix = torch.einsum(
        'bnik,bnjk->bnij', diffs, diffs) / k_neighbors  # Shape: (B, N, D, D)
    # print('covariance_matrix',covariance_matrix)

    # Perform eigenvalue decomposition to get the eigenvalues of the covariance matrix
    # Shape: (B, N, D), sorted in ascending order
    eigenvalues = torch.linalg.eigvalsh(covariance_matrix)
    # print('eigenvalues',eigenvalues)

    # Curvature is the smallest eigenvalue divided by the sum of all eigenvalues
    # Set curvature to zero if the sum of eigenvalues is zero
    sum_eigenvalues = torch.sum(eigenvalues, dim=-1)
    curvatures = torch.where(sum_eigenvalues == 0, torch.zeros_like(
        # Shape: (B, N)
        sum_eigenvalues), eigenvalues[:, :, 0] / sum_eigenvalues)

    # print('curvatures',curvatures)

    return curvatures


def calculate_smoothed_curvatures(points1, curvatures1, h):
    """
    Compute smoothed curvatures using the given formula for each point in points1.

    points1: Tensor of shape (B, N, D) - Input point cloud (B batches of N points, D dimensions).
    curvatures1: Tensor of shape (B, N) - Curvatures for each point in points1.
    h: Smoothing parameter.

    Returns:
    smoothed_curvatures: Tensor of shape (B, N) - Smoothed curvatures for each point.
    """
    # Compute pairwise squared Euclidean distances between points1
    diff = points1.unsqueeze(2) - points1.unsqueeze(1)  # Shape: (B, N, N, D)
    # Shape: (B, N, N) - Pairwise squared distances
    dist_squared = (diff ** 2).sum(-1)

    # Compute the Gaussian weights based on the distances
    weights = torch.exp(-dist_squared / h)  # Shape: (B, N, N)

    # Numerator: Sum of curvature * weights for all neighbors
    numerator = (curvatures1.unsqueeze(1) *
                 weights).sum(dim=-1)  # Shape: (B, N)

    # Denominator: Sum of weights for all neighbors
    denominator = weights.sum(dim=-1)  # Shape: (B, N)

    # Smoothed curvatures
    smoothed_curvatures = numerator / denominator  # Shape: (B, N)

    return smoothed_curvatures


class MeshSimplificationLoss(nn.Module):
    def __init__(self, lambda_c=1, lambda_e=1, lambda_o=1, h=0.1):
        super(MeshSimplificationLoss, self).__init__()
        self.lambda_c = lambda_c  # Weight for collision loss
        self.lambda_e = lambda_e  # Weight for edge crossing loss
        self.lambda_o = lambda_o  # Weight for overlap loss
    
    def chamfer_loss(self, points1, points2, h=0.1, k_neighbors=10):
        """
        Compute the adaptive Chamfer Loss between two point clouds points1 and points2 with curvature weighting.

        points1: tensor of shape (B, N, D) - Input point cloud (original mesh).
        points2: tensor of shape (B, M, D) - Sampled point cloud (simplified mesh).
        h: Bandwidth parameter for local curvature weighting.
        k_neighbors: Number of neighbors for curvature calculation.

        Returns:
        Adaptive Chamfer loss value.
        """
        B, N, D = points1.shape
        M = points2.shape[1]

        # Compute distances between points1 and points2
        dist_p1_p2 = torch.cdist(points1, points2, p=2)  # Shape: (B, N, M)
        dist_p2_p1 = torch.cdist(points2, points1, p=2)  # Shape: (B, M, N)

        # First term: Nearest distance for each point in points1 to points2
        min_dist_p1_p2, _ = torch.min(dist_p1_p2, dim=-1)  # Shape: (B, N)

        # Second term: Nearest distance for each point in points2 to points1
        min_dist_p2_p1, _ = torch.min(dist_p2_p1, dim=-1)  # Shape: (B, M)

        # Calculate curvature values for points1
        curvatures1 = calculate_curvature(
            points1, k_neighbors=k_neighbors)  # Shape: (B, N)

        # Apply curvature weights using Gaussian smoothing
        smoothed_curvatures = calculate_smoothed_curvatures(
            points1, curvatures1, h)

        # First term with curvature weighting
        loss_term1 = torch.sum((min_dist_p1_p2 ** 2), dim=-1)  # Shape: (B,)

        # Second term without curvature weighting
        loss_term2 = torch.sum(min_dist_p2_p1 ** 2, dim=-1)  # Shape: (B,)

        # Total loss: sum of both terms averaged over batch
        total_loss = loss_term1 + loss_term2

        return total_loss.squeeze()

    def probabilistic_surface_distance(self, source_vertices, source_faces, target_vertices, target_faces, face_probs, 
                                     num_points_per_face=10, k=5,triangle_penalty_weight=1.0):
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
            source_vertices, source_faces, num_points_per_face)
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
                tri_dists, tri_indices = torch.topk(dists_to_tris, k=k+1, largest=False)

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
        # triangle_count_loss = triangle_penalty_weight * abs(num_source_triangles - num_target_triangles)
        print('forward_loss',forward_loss)
        print('reverse_loss',reverse_loss)
        # print('triangle_count_loss',triangle_count_loss)

        return forward_loss + reverse_loss 

    def triangle_collision_loss(self, vertices, faces, face_probs):
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

    def edge_crossing_loss(self, vertices, faces, face_probs, batch_size=1024):
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

            for i in range(0, num_edges, batch_size):
                batch_end = min(i + batch_size, num_edges)
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

    def triangle_overlap_loss(self, vertices, faces, face_probs, num_samples=10, batch_size = 1024):
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

    def forward(self, pred, target):
        """
        pred: dictionary containing:
            - vertices: simplified mesh vertices
            - faces: simplified mesh faces
            - face_probabilities: probabilities for each face
        target: dictionary containing:
            - vertices: original mesh vertices
            - faces: original mesh faces
        """
        # Unpack predictions and targets
        pred_vertices = pred['vertices']
        pred_faces = pred['faces']
        face_probs = pred['face_probabilities']

        target_vertices = target.x
        target_faces = target.faces
        print('Target Faces',target_faces.shape)

        # Add batch dimension if needed
        if len(pred_vertices.shape) == 2:
            pred_vertices = pred_vertices.unsqueeze(0)
        if len(target_vertices.shape) == 2:
            target_vertices = target_vertices.unsqueeze(0)
        print('Target Vertices',target_vertices.shape)
        print('Pred Faces',pred_faces.shape)
        print('Pred Vertices',pred_vertices.shape)

        # Chamfer loss using your implementation
        chamfer_loss = self.chamfer_loss(pred_vertices, target_vertices)
        print('chamfer_loss calculated',chamfer_loss)

        # Surface distance loss
        surface_loss = self.probabilistic_surface_distance(
            pred_vertices.squeeze(0), pred_faces,
            target_vertices.squeeze(0), target_faces,
            face_probs)
        print('surface_loss calculated',surface_loss)

        # Geometric regularity losses
        collision_loss = self.triangle_collision_loss(
            pred_vertices.squeeze(0), pred_faces, face_probs)
        print('collision_loss calculated',collision_loss)

        edge_crossing_loss = self.edge_crossing_loss(
            pred_vertices.squeeze(0), pred_faces, face_probs)
        print('edge_crossing_loss calculated',edge_crossing_loss)

        overlap_loss = self.triangle_overlap_loss(
            pred_vertices.squeeze(0), pred_faces, face_probs)
        print('overlap_loss calculated',overlap_loss)

        # Combine all losses
        total_loss = (chamfer_loss +
                      surface_loss +
                      self.lambda_c * collision_loss +
                      self.lambda_e * edge_crossing_loss +
                      self.lambda_o * overlap_loss)

        return total_loss
