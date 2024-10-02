import os
import trimesh
import torch
from torch_geometric.data import Data, Dataset

class MeshDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.obj') and f == '00000004_1ffb81a71e5b402e966b9341_trimesh_003.obj']

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        obj_path = os.path.join(self.root_dir, self.files[idx])
        mesh = trimesh.load(obj_path)
        
        vertices = torch.tensor(mesh.vertices, dtype=torch.float)
        edges = torch.tensor(mesh.edges, dtype=torch.long).t().contiguous()
        
        # Center the mesh by subtracting the mean of the vertices
        vertices -= vertices.mean(dim=0)

        # Scale to unit cube
        max_dim = (vertices.max(dim=0)[0] - vertices.min(dim=0)[0]).max()  # Get max dimension along any axis
        vertices /= max_dim  # Scale all vertices
        vertices *= 5  # Adjust the scaling factor to increase the vertex range

        data = Data(x=vertices, edge_index=edges)
        
        if self.transform:
            data = self.transform(data)
        
        return data
