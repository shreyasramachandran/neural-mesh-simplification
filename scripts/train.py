import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from model import MeshGNN
from data import MeshDataset
from losses import ChamferLoss
from metrics import ChamferDistance
from utils.sampling_operations import gumbel_softmax
import argparse
import os
import random
import numpy as np

# Set a fixed seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Function to check GPU memory
def print_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert from bytes to GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # Reserved but not yet allocated
    free = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()  # Free within reserved

    print(f"Allocated Memory: {allocated:.2f} GB")
    print(f"Reserved Memory: {reserved:.2f} GB")
    print(f"Free Memory: {free / (1024 ** 3):.2f} GB")
    
def log_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"Layer: {name} | Gradient Norm: {grad_norm}")

# Argument parser setup
parser = argparse.ArgumentParser(description='Train MeshGNN for mesh simplification.')
parser.add_argument('--data_path', type=str, required=True, help='Path to the training dataset.')
parser.add_argument('--model_save_path', type=str, required=True, help='Path to save the trained model.')
parser.add_argument('--loss_file', type=str, default='/notebooks/models/losses.txt', help='Path to save the loss log.')

args = parser.parse_args()

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
epochs = 2
learning_rate = 0.001
batch_size = 1
max_nodes = 10000 # 32205

# Load dataset
dataset = MeshDataset(root_dir=args.data_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, optimizer
model = MeshGNN(input_dim=3, hidden_dim=3, output_dim=3)
model = model.to(device)
criterion = ChamferLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(args.loss_file), exist_ok=True)

# Training loop
for epoch in range(epochs):
    processed = 0
    skippped = 0
    model.train()
    total_loss = 0
    for data in dataloader:
        # After each iteration or batch
        num_nodes = data.x.shape[0]
        if (num_nodes > max_nodes):
            skippped += 1
            continue
        else:
            processed += 1
            data = data.to(device)
            optimizer.zero_grad()
            predicted_vertices = model(data)  # Get predicted importance scores
#             print(predicted_prob)
            subsampled_points = data.x.unsqueeze(0)
#             # Sample using Gumbel-Softmax (differentiable)
#             gumbel_weights = gumbel_softmax(predicted_prob.squeeze(), temperature=0.5)
            
#             # Forward pass: Soft selection for Chamfer Distance
#             # Here, we use soft-selected points for loss calculation to preserve gradients.
#             selected_points_soft = (gumbel_weights.unsqueeze(-1) * data.x).unsqueeze(0)  # Soft selection
#             print(selected_points_soft)
            # Chamfer Distance calculation with soft points
            loss = criterion(subsampled_points,predicted_vertices.unsqueeze(0))  # Use soft points in Chamfer loss
            # Perform backpropagation using the soft points for gradient flow
            loss.backward()  # Backpropagate through the loss
            #log_gradients(model) # Investigate gradients
            optimizer.step()
            total_loss += loss.item()            
    
    print(f'Processed {processed}, Skipped: {skippped}') 
    
    # Calculate average loss for the epoch
    avg_loss = total_loss / processed
    print(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss}, Average Loss: {avg_loss}")

    # Save total and average loss to file
    with open(args.loss_file, 'a') as f:
        f.write(f"Epoch {epoch+1}, Total Loss: {total_loss}, Average Loss: {avg_loss}\n")


checkpoint_path = f"{args.model_save_path}_epoch_{epoch+1}.pth"   
torch.save(model.state_dict(), checkpoint_path)