import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from model import MeshGNN
from data import MeshDataset
from losses import ChamferLoss
from metrics import ChamferDistance
from utils.sampling_operations import gumbel_softmax

# Hyperparameters
epochs = 10
learning_rate = 0.001
batch_size = 1

# Load dataset
dataset = MeshDataset(root_dir='/notebooks/datasets/abc/')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, optimizer
model = MeshGNN(input_dim=3, hidden_dim=64, output_dim=1)
criterion = ChamferLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for data in dataloader:
        optimizer.zero_grad()
        predicted_prob = model(data)  # Get predicted importance scores
        
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
        # Chamfer Distance requires discrete points, so we use `selected_points_hard`
        loss = criterion(selected_points_hard, subsampled_points)  # Calculate Chamfer Distance
        # Backward pass: Use straight-through estimator to allow gradient flow through soft selection
        selected_points_soft = (gumbel_weights.unsqueeze(-1) * data.x).sum(dim=0, keepdim=True)
        
        # Perform backpropagation using the soft points for gradient flow
        selected_points_soft.backward(torch.ones_like(selected_points_soft))
        optimizer.step()
        total_loss += loss.item()
        
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")    


# Load test dataset
test_dataset = MeshDataset(root_dir='/notebooks/datasets/abc/')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize Chamfer Distance for evaluation
metric = ChamferDistance()
    
# After training, run evaluation on the test set before saving the model
model.eval()  # Set model to evaluation mode
total_metric = 0
with torch.no_grad():
    for data in test_loader:
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
    
# Save the model
torch.save(model.state_dict(), '/notebooks/models/mesh_gnn_model.pth')