import torch
from model import MeshGNN
from trimesh import load

# Load model
model = MeshGNN(input_dim=3, hidden_dim=64, output_dim=1)
model.load_state_dict(torch.load('/notebooks/models/mesh_gnn_model.pth'))
model.eval()

# Load a single mesh for inference
mesh = load('/notebooks/datasets/abc/00000002_1ffb81a71e5b402e966b9341_trimesh_001.obj')

# Prepare data for the model
vertices = torch.tensor(mesh.vertices, dtype=torch.float).unsqueeze(0)  # Add batch dimension
edges = torch.tensor(mesh.edges, dtype=torch.long).t().contiguous().unsqueeze(0)  # Add batch dimension

# Predict using the model
with torch.no_grad():
    predicted_prob = model(vertices, edges)

# Output the predicted probabilities
print(predicted_prob)
