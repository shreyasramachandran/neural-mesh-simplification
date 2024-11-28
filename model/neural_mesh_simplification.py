import torch
import torch.nn as nn
import torch.nn.functional as F
from model import PointSampler, EdgePredictor, FaceClassifier

class NeuralMeshSimplification(nn.Module):
    def __init__(self, input_dim = 3, hidden_dim, sample_ratio):
        super(MeshGNN, self).__init__()
        
        # Point Sampler
        self.point_sampler = PointSampler(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            sample_ratio=sample_ratio
        )
        # Keep the rest of your components
        self.edge_predictor = EdgePredictor()
        self.face_classifier = FaceClassifier()
    
    def forward(self, data):
        # Use PointSampler to get refined positions and features
        sampler_output = self.point_sampler(
            points=data.x,
            edge_index=data.edge_index,
            batch=data.batch if hasattr(data, 'batch') else None
        )
        
        # Extract the results
        refined_positions = sampler_output['sampled_points']
        sampled_x = sampler_output['sampled_features']
        edge_index_filtered = sampler_output['filtered_edges']
        
        # Edge prediction
        adjacency, candidate_triangles, triangle_probs = self.edge_predictor(
            refined_positions, sampled_x, edge_index_filtered
        )
        print('adjacency',adjacency.shape)
        print('candidate_triangles',candidate_triangles.shape)
        print('triangle_probs',triangle_probs.shape)

        # Face classification
        final_triangle_probs = self.face_classifier(
            refined_positions, candidate_triangles, triangle_probs
        )
        print('final_triangle_probs',final_triangle_probs.shape)

        k = 8000  # Specify the number of top elements you want to select
        topk_probs, topk_indices = torch.topk(final_triangle_probs, k)

        # Create a mask for these top k probabilities
        prob_mask = torch.zeros_like(final_triangle_probs, dtype=torch.bool)
        prob_mask[topk_indices] = True
        
        valid_triangles = candidate_triangles[prob_mask]
        valid_probs = final_triangle_probs[prob_mask]
        print(f"Valid triangles shape after filtering: {valid_triangles.shape}")
        print(f"Valid probs shape after filtering: {valid_probs.shape}")

        return {
            'vertices': refined_positions,
            'faces': valid_triangles,
            'face_probabilities': valid_probs
        }