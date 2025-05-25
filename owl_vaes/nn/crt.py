import torch
from torch import nn
import torch.nn.functional as F

from .attn import StackedTransformer
from ..configs import TransformerConfig

class CRT(nn.Module):
    def __init__(self, config: 'TransformerConfig' = None):
        super().__init__()

        if config is None:
            config = TransformerConfig(
                n_layers=6
                n_heads=3
                d_model=384,
                causal=True
            )
        
        self.core = StackedTransformer(config)
    
    def forward(self, x):
        # x is [b,n,d]
        x_in = x[:,:-1]
        x_out = x[:,1:]

        pred = self.core(x_in)
        return F.mse_loss(pred, x_out)