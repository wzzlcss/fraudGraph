import torch
import torch.nn as nn
from torch import Tensor

def mask_edge(edge_index: Tensor, p: float=0.7):
    if p < 0. or p > 1.:
        raise ValueError(f'Mask probability has to be between 0 and 1 '
                         f'(got {p}')    
    e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    return edge_index[:, ~mask], edge_index[:, mask]

class MaskEdge(nn.Module):
    def __init__(self, p: float=0.7):
        super().__init__()
        self.p = p

    def forward(self, edge_index):
        remaining_edges, masked_edges = mask_edge(edge_index, p=self.p)
        return remaining_edges, masked_edges

    def extra_repr(self):
        return f"p={self.p}"
