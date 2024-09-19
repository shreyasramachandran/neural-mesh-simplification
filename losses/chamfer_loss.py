import torch
import torch.nn as nn

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, points1, points2):
        """
        Compute Chamfer Distance between two point clouds points1 and points2.
        points1: tensor of shape (B, N, D)
        points2: tensor of shape (B, M, D)
        """
        # Compute pairwise distance
        diff = points1.unsqueeze(2) - points2.unsqueeze(1)  # (B, N, M, D)
        dist = torch.norm(diff, dim=-1)  # (B, N, M)

        # Compute nearest neighbors in both directions
        min_dist1 = torch.min(dist, dim=2)[0]  # (B, N)
        min_dist2 = torch.min(dist, dim=1)[0]  # (B, M)

        # Chamfer Distance
        loss = torch.mean(min_dist1) + torch.mean(min_dist2)
        return loss