import torch
from torch import nn
import torch.nn.functional as F

from .normalization import LayerNorm

class AdaLN(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fc = nn.Linear(dim, 2 * dim)
        self.norm = LayerNorm(dim)

    def forward(self, x, cond):
        # x is [b,n,d]
        # cond is [b,d]
        b,n,d = x.shape

        y = F.silu(cond)
        ab = self.fc(y) # [b,2d]
        ab = ab[:,None,:].expand(b,n,-1)
        a,b = ab.chunk(2,dim=-1) # each [b,n,d]

        x = self.norm(x) * (1. + a) + b
        return x

class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fc_c = nn.Linear(dim, dim)

    def forward(self, x, cond):
        # x is [b,n,d]
        # cond is [b,d]
        b,n,d = x.shape

        y = F.silu(cond)
        c = self.fc_c(y) # [b,d]
        c = c[:,None,:].expand(b,n,-1) # [b,n,d]

        return c * x