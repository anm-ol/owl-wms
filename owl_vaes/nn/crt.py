import torch.nn.functional as F
from torch import Tensor, nn

from ..configs import TransformerConfig
from .attn import StackedTransformer


class CRT(nn.Module):
    def __init__(self, config: TransformerConfig | None = None):
        super().__init__()

        if config is None:
            config = TransformerConfig(
                n_layers=6,
                n_heads=3,
                d_model=384,
                causal=True
            )

        self.core = StackedTransformer(config)

    def forward(self, x: Tensor):
        # x is [b,n,d]
        x_in = x[:,:-1]
        x_out = x[:,1:]

        pred = self.core(x_in)
        return F.mse_loss(pred, x_out)
