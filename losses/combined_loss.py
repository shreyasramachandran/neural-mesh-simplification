import torch
import torch.nn as nn
import torch.nn.functional as F
from .chamfer_loss import ChamferLoss
from .edge_crossing_loss import EdgeCrossingLoss
from .probabilistic_surface_distance import ProbabilisticSurfaceDistance
from .triangle_collision_loss import TriangleCollisionLoss
from .triangle_overlap_loss import TriangleOverlapLoss

class CombinedLoss(nn.Module):
    """
    Combined loss for mesh simplification.
    """
    def __init__(self, lambda_c=10, lambda_e=10, lambda_o=10, h=0.1):
        super(CombinedLoss, self).__init__()
        self.lambda_c = lambda_c  # Weight for collision loss
        self.lambda_e = lambda_e  # Weight for edge crossing loss
        self.lambda_o = lambda_o  # Weight for overlap loss

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