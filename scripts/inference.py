import os
import torch
from model import MeshGNN
from torch_geometric.data import Data
from trimesh import load, Trimesh
from utils.sampling_operations import gumbel_softmax
import argparse
import numpy as np
import random

# Set a fixed seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Argument parser setup
parser = argparse.ArgumentParser(description='Inference on a mesh using a pre-trained MeshGNN model.')
parser.add_argument('--mesh_path', type=str, required=True, help='Path to the mesh file (OBJ format) for inference.')
parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model file.')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the simplified mesh.')

args = parser.parse_args()

# Load the pre-trained model (default device)
model = MeshGNN(input_dim=3, hidden_dim=64, sample_ratio=0.1)
model.load_state_dict(torch.load(args.model_path, weights_only=True))

model.eval()  # Set model to evaluation mode

# Load a single mesh for inference
mesh = load(args.mesh_path)

# Prepare the data for the model (no explicit device movement)
vertices = torch.tensor(mesh.vertices, dtype=torch.float)
edges = torch.tensor(mesh.edges, dtype=torch.long).t().contiguous()

# Center the mesh by subtracting the mean of the vertices
vertices -= vertices.mean(dim=0)

# Scale to unit cube
max_dim = (vertices.max(dim=0)[0] - vertices.min(dim=0)[0]).max()  # Get max dimension along any axis
vertices /= max_dim  # Scale all vertices
vertices *= 5  # Adjust the scaling factor to increase the vertex range


# Create a PyTorch Geometric Data object
data = Data(x=vertices, edge_index=edges)

# Perform inference using the model
with torch.no_grad():
    sampled_vertices = model(data)  # Pass the data object directly to the model

# # Step 1: Apply Gumbel-Softmax (soft sampling) to generate new vertices
# gumbel_weights = gumbel_softmax(predicted_probs.squeeze(), temperature=0.5)  # soft sampling

# # Step 2: Perform soft selection by applying Gumbel weights to the original vertices
# selected_points_soft = (gumbel_weights * data.x)  # shape (N, 3)

# # No need for top-k because soft sampling includes all vertices, weighted by the Gumbel weights

# # Step 3: Re-use the original edges and faces (since no vertices are removed)
# original_faces = mesh.faces
# original_edges = mesh.edges

# Step 4: Create a new mesh with the soft-selected vertices but retain original edges and faces
# new_mesh = Trimesh(vertices=selected_points_soft.cpu().numpy(), edges=original_edges, faces=original_faces)
new_mesh = Trimesh(vertices=sampled_vertices.cpu().numpy())

# Step 5: Generate the output file name based on input mesh name
mesh_name = os.path.splitext(os.path.basename(args.mesh_path))[0]
output_file = os.path.join(args.output_dir, f"{mesh_name}_sampled_vertices.obj")

# Step 6: Save the new mesh to the output path
new_mesh.export(output_file)

print(f"Soft sampled mesh saved to {output_file}")
