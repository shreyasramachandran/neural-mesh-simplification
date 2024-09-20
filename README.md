# Neural Mesh Simplification

## Overview
Neural Mesh Simplification is a deep learning-based model for simplifying 3D meshes. The model leverages graph neural networks (GNNs) to reduce the number of faces in a mesh while preserving its geometric and topological properties. This approach is particularly useful for applications in computer graphics, gaming, and 3D modeling where simplified meshes are needed for faster processing and rendering without significant loss of detail.

## Features
- **Mesh Simplification**: Reduces the number of faces in a 3D mesh while retaining the original structure.
- **Neural Network-based**: Uses Graph Convolutional Networks (GCNs) to learn mesh simplification.
- **Supports all formats**: Works with meshes in the all major formats.

## Installation
To install the necessary dependencies and set up the project, run the following commands:

```bash
git clone https://github.com/shreyasramachandran/neural-mesh-simplification.git
cd neural-mesh-simplification
pip install -r requirements.txt
```

## Download Dataset

```bash
python scripts/download_dataset.py --api_key YOUR_GOOGLE_API_KEY --folder_link "YOUR_GOOGLE_DRIVE_FOLDER_LINK" --output_directory /path/to/output/directory
```

## Inference

```bash
python scripts/inference.py --input_mesh /path/to/input/mesh.obj --model_checkpoint /path/to/pretrained/model.pth
```

## Common Issues and Solutions

### 1. Module Not Found Error
If you encounter a `ModuleNotFoundError` when importing modules like `model`,`data`,`losses`,`metrics` it’s likely because Python cannot find the project folder in the `PYTHONPATH`.

To fix this, add the repository folder to your `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/neural-mesh-simplification
```