import torch
import torch.nn as nn

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
    knn_indices = dist_matrix.topk(k=k_neighbors, largest=False, dim=-1).indices  # Shape: (B, N, k_neighbors)
    # print('knn_indices',knn_indices)

    # Gather the k-nearest neighbors for each point
    neighbors = torch.gather(points.unsqueeze(1).expand(B, N, N, D), 2, knn_indices.unsqueeze(-1).expand(B, N, k_neighbors, D))
    # print('neighbors',neighbors)

    # Compute the covariance matrix for each point's neighborhood
    diffs = neighbors - points.unsqueeze(2)  # Shape: (B, N, k_neighbors, D)
    # print('diffs',diffs)
    covariance_matrix = torch.einsum('bnik,bnjk->bnij', diffs, diffs) / k_neighbors  # Shape: (B, N, D, D)
    # print('covariance_matrix',covariance_matrix)

    # Perform eigenvalue decomposition to get the eigenvalues of the covariance matrix
    eigenvalues = torch.linalg.eigvalsh(covariance_matrix)  # Shape: (B, N, D), sorted in ascending order
    # print('eigenvalues',eigenvalues)

    # Curvature is the smallest eigenvalue divided by the sum of all eigenvalues
    # Set curvature to zero if the sum of eigenvalues is zero
    sum_eigenvalues = torch.sum(eigenvalues, dim=-1)
    curvatures = torch.where(sum_eigenvalues == 0, torch.zeros_like(sum_eigenvalues), eigenvalues[:, :, 0] / sum_eigenvalues) # Shape: (B, N)

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
    dist_squared = (diff ** 2).sum(-1)  # Shape: (B, N, N) - Pairwise squared distances

    # Compute the Gaussian weights based on the distances
    weights = torch.exp(-dist_squared / h)  # Shape: (B, N, N)

    # Numerator: Sum of curvature * weights for all neighbors
    numerator = (curvatures1.unsqueeze(1) * weights).sum(dim=-1)  # Shape: (B, N)

    # Denominator: Sum of weights for all neighbors
    denominator = weights.sum(dim=-1)  # Shape: (B, N)

    # Smoothed curvatures
    smoothed_curvatures = numerator / denominator  # Shape: (B, N)
    
    return smoothed_curvatures


class ChamferLoss(nn.Module):
    def __init__(self, h=0.1):
        """
        Chamfer Loss with curvature weighting.
        h: Bandwidth parameter for local curvature.
        """
        super(ChamferLoss, self).__init__()
        self.h = h
    
    def forward(self, points1, points2, h=0.1, k_neighbors=10):
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
        curvatures1 = calculate_curvature(points1, k_neighbors=k_neighbors)  # Shape: (B, N)
        
        # Apply curvature weights using Gaussian smoothing
        smoothed_curvatures = calculate_smoothed_curvatures(points1, curvatures1, h)

        # First term with curvature weighting
        loss_term1 = torch.sum((min_dist_p1_p2 ** 2), dim=-1)  # Shape: (B,)

        # Second term without curvature weighting
        loss_term2 = torch.sum(min_dist_p2_p1 ** 2, dim=-1)  # Shape: (B,)

        # Total loss: sum of both terms averaged over batch
        total_loss = loss_term1 + loss_term2

        return total_loss


