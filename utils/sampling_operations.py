import torch
import torch.nn.functional as F

def sample_gumbel(shape, eps=1e-20, device='cuda'):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax(logits, temperature=1.0):
    gumbel_noise = sample_gumbel(logits.size(), device=logits.device)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)