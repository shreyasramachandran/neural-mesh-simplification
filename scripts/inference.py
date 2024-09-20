import torch
from model import MeshGNN
from torch_geometric.data import Data
from trimesh import load
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description='Inference on a mesh using a pre-trained MeshGNN model.')
parser.add_argument('--mesh_path', type=str, required=True, help='Path to the mesh file (OBJ format) for inference.')
parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model file.')

args = parser.parse_args()

# Load the pre-trained model
model = MeshGNN(input_dim=3, hidden_dim=64, output_dim=1)
model.load_state_dict(torch.load(args.model_path, weights_only=True))
model.eval()

# Load a single mesh for inference
mesh = load(args.mesh_path)

# Prepare the data for the model
vertices = torch.tensor(mesh.vertices, dtype=torch.float)
edges = torch.tensor(mesh.edges, dtype=torch.long).t().contiguous()

# Create a PyTorch Geometric Data object
data = Data(x=vertices, edge_index=edges)

# Perform inference using the model
with torch.no_grad():
    predicted_prob = model(data)  # Pass the data object directly to the model

# Output the predicted probabilities shape
print(predicted_prob)