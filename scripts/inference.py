import os
import torch
from model import MeshGNN
from torch_geometric.data import Data
from trimesh import load, Trimesh
import argparse
import numpy as np

# Argument parser setup
parser = argparse.ArgumentParser(description='Inference on a mesh using a pre-trained MeshGNN model.')
parser.add_argument('--mesh_path', type=str, required=True, help='Path to the mesh file (OBJ format) for inference.')
parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model file.')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the simplified mesh.')

args = parser.parse_args()

# Load the pre-trained model (default device)
model = MeshGNN(input_dim=3, hidden_dim=64, output_dim=1)
model.load_state_dict(torch.load(args.model_path, weights_only=True))
model.eval()

# Load a single mesh for inference
mesh = load(args.mesh_path)

# Prepare the data for the model (no explicit device movement)
vertices = torch.tensor(mesh.vertices, dtype=torch.float)
edges = torch.tensor(mesh.edges, dtype=torch.long).t().contiguous()

# Create a PyTorch Geometric Data object
data = Data(x=vertices, edge_index=edges)

# Perform inference using the model
with torch.no_grad():
    predicted_prob = model(data)  # Pass the data object directly to the model

print('Shape of predicted prob',predicted_prob.shape[0])
# Assuming predicted_prob is a tensor of shape (N, 1)
# First, squeeze it to remove the singleton dimension
predicted_prob_flat = predicted_prob.squeeze()  # Shape: (N,)

# Determine the number of vertices to keep (top 70%)
num_vertices = predicted_prob_flat.shape[0]
k = int(0.7 * num_vertices)

# Get the indices of the top k probabilities
_, vertices_to_keep = torch.topk(predicted_prob_flat, k)

print('Vertices kept size',vertices_to_keep.shape[0])

# vertices_to_keep = torch.where(predicted_prob > 0.01)[0]

# Convert indices to numpy for further mesh manipulation
vertices_to_keep_np = vertices_to_keep.numpy()

# Filter vertices
filtered_vertices = mesh.vertices[vertices_to_keep_np]

# Create a mapping from old indices to new indices
index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(vertices_to_keep_np)}

# Filter faces (remap face indices to the new set of vertices)
filtered_faces = []
for face in mesh.faces:
    if all(v in index_mapping for v in face):
        # Remap face indices using the index mapping
        new_face = [index_mapping[v] for v in face]
        filtered_faces.append(new_face)

filtered_faces = np.array(filtered_faces)

# Remap edges (keep edges where both vertices are in the selected set and remap them)
filtered_edges = []
for edge in mesh.edges:
    if all(v in index_mapping for v in edge):
        new_edge = [index_mapping[v] for v in edge]
        filtered_edges.append(new_edge)

filtered_edges = np.array(filtered_edges)

# Create a new mesh with the filtered vertices, faces, edges, and normals
filtered_mesh = Trimesh(vertices=filtered_vertices, faces=filtered_faces, edges=filtered_edges)

# Generate the output file name based on input mesh name
mesh_name = os.path.splitext(os.path.basename(args.mesh_path))[0]
output_file = os.path.join(args.output_dir, f"{mesh_name}_simplified.obj")

# Save the filtered mesh to the output path
filtered_mesh.export(output_file)

print(f"Simplified mesh saved to {output_file}")
