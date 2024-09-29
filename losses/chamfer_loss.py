import torch
import torch.nn as nn

class ChamferLossAdapted(nn.Module):
    def __init__(self):
        super(ChamferLossAdapted, self).__init__()

    def forward(self, points1, points2, weights):
        """
        Compute the modified Chamfer Distance between two point clouds points1 and points2,
        incorporating weights as per the paper's definition.
        
        points1: tensor of shape (B, N, D) - Input point cloud (original mesh).
        points2: tensor of shape (B, M, D) - Sampled point cloud (simplified mesh).
        weights: tensor of shape (B, N) - Weights associated with each point in points1.
        """
        # Ensure weights sum to 1 across each batch
        weights = torch.softmax(weights, dim=1)  # Normalize weights

        # Compute pairwise squared Euclidean distances between points1 and points2
        diff = points1.unsqueeze(2) - points2.unsqueeze(1)  # Broadcasting to get (B, N, M, D)
        dist = (diff ** 2).sum(-1)  # Compute squared distances (B, N, M)

        # Compute nearest neighbors in both directions
        min_dist1 = torch.min(dist, dim=2)[0]  # (B, N) - Closest points in points2 for each in points1
        min_dist2 = torch.min(dist, dim=1)[0]  # (B, M) - Closest points in points1 for each in points2

        # Apply weights to the first term (original to simplified)
        weighted_dist1 = weights * min_dist1  # Apply weights element-wise
        loss_term1 = torch.sum(weighted_dist1, dim=1)  # Sum over each batch

        # Second term (simplified to original) does not have weights
        loss_term2 = torch.mean(min_dist2, dim=1)  # Average over each batch for uniform treatment

        # Final loss is the mean of both terms
        loss = torch.mean(loss_term1 + loss_term2)  # Average over batches

        return loss
