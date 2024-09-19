import torch
import torch.nn as nn

class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def evaluate(self, predicted_points, target_points):
        """
        Computes Chamfer Distance between predicted and target points.
        
        :param predicted_points: Tensor of shape (B, N, 3) where B is the batch size,
                                 N is the number of predicted points, and 3 represents (x, y, z) coordinates.
        :param target_points: Tensor of shape (B, M, 3) where B is the batch size,
                              M is the number of target points, and 3 represents (x, y, z) coordinates.
        :return: Chamfer distance averaged over the batch.
        """
        # Compute pairwise distances
        batch_size, num_pred, _ = predicted_points.shape
        num_target = target_points.shape[1]

        # Expand dimensions for broadcasting
        pred_expand = predicted_points.unsqueeze(2).repeat(1, 1, num_target, 1)  # (B, N, M, 3)
        target_expand = target_points.unsqueeze(1).repeat(1, num_pred, 1, 1)     # (B, N, M, 3)

        # Compute squared distances between all points
        distances = torch.norm(pred_expand - target_expand, dim=-1)  # (B, N, M)

        # For each point in the predicted set, find the nearest neighbor in the target set
        min_pred_to_target, _ = torch.min(distances, dim=2)  # (B, N)

        # For each point in the target set, find the nearest neighbor in the predicted set
        min_target_to_pred, _ = torch.min(distances, dim=1)  # (B, M)

        # Compute Chamfer Distance (average of both directions)
        chamfer_distance = (min_pred_to_target.mean(dim=1) + min_target_to_pred.mean(dim=1)) / 2.0

        # Return mean Chamfer Distance over the batch
        return chamfer_distance.mean()
