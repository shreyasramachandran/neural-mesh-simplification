import torch
from model import NeuralMeshSimplification
from data import MeshDataset
from metrics import ChamferDistance
from torch_geometric.loader import DataLoader
from utils.sampling_operations import gumbel_softmax
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description='Evaluate NeuralMeshSimplification model on a test dataset.')
parser.add_argument('--test_data_path', type=str, required=True, help='Path to the test dataset.')
parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model checkpoint.')

args = parser.parse_args()

# Hyperparameters
batch_size = 1

# Load dataset 
test_dataset = MeshDataset(root_dir=args.test_data_path)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# Load model
model = NeuralMeshSimplification(input_dim=3, hidden_dim=64, sample_ratio=0.1)
model.load_state_dict(torch.load(args.model_path, weights_only=True))
model.eval()

# Initialize metric
metric = ChamferDistance()

# Evaluation loop
total_metric = 0
for data in test_loader:
    with torch.no_grad():
        predicted_prob = model(data)  # Predict points
        
        num_points = 10000  # Subsample to 10,000 points
        indices = torch.randperm(data.x.size(0))[:num_points]
        subsampled_points = data.x[indices]
        subsampled_points = subsampled_points.unsqueeze(0)  # Add batch dimension, now (1, N, 3)
        
        # Sample using Gumbel-Softmax (differentiable)
        gumbel_weights = gumbel_softmax(predicted_prob.squeeze(), temperature=0.5)
        num_points_available = gumbel_weights.size(0)
        # Forward pass: Hard selection (discrete) for Chamfer Distance
        k = min(1000, num_points_available)  # Number of points to select
        _, selected_indices = torch.topk(gumbel_weights, k=k, dim=0)
        selected_points_hard = data.x[selected_indices].unsqueeze(0)  # Discrete point selection
        
        score = metric.evaluate(selected_points_hard, subsampled_points)  # Evaluate Chamfer Distance
        total_metric += score.item()

# Print final Chamfer Distance over the test set
print(f"Test Chamfer Distance: {total_metric / len(test_loader)}")