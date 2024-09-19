import torch
import torch.nn.functional as F

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax(logits, temperature=1.0):
    gumbel_noise = sample_gumbel(logits.size())
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)