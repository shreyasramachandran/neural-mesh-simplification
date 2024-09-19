import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class MeshGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MeshGNN, self).__init__()
        # Define GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc_sample = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # GCN layers with ReLU
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Sampling prediction
        scores = self.fc_sample(x)
        sampling_prob = torch.sigmoid(scores)
        
        return sampling_prob