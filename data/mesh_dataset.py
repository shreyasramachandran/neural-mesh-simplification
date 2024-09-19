import os
import trimesh
import torch
from torch_geometric.data import Data, Dataset

class MeshDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.obj')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        obj_path = os.path.join(self.root_dir, self.files[idx])
        mesh = trimesh.load(obj_path)
        
        vertices = torch.tensor(mesh.vertices, dtype=torch.float)
        edges = torch.tensor(mesh.edges, dtype=torch.long).t().contiguous()

        data = Data(x=vertices, edge_index=edges)
        
        if self.transform:
            data = self.transform(data)
        
        return data
