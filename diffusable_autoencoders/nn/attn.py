import torch
from torch import nn
import torch.nn.functional as F

from .normalizations import LayerNorm, RMSNorm, QKNorm
from .embeddings import ImageRoPE
from .mlp import MLP

torch.backends.cuda.enable_flash_sdp(enabled = True)

class Attn(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)

        self.qk_norm = QKNorm(config.d_model // config.n_heads)

    def forward(self, x):

        q,k,v = eo.rearrange(self.qkv(x), 'b n (3 h d) -> 3 b h n d')
        q,k = self.qk_norm(q,k)
        x = F.scaled_dot_product_attention(q,k,v)
        x = eo.rearrange(x, 'b h n d -> b n (h d)')
        x = self.out(x)
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config.model

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.attn = Attn(config)
        self.mlp = MLP(config)

        # layer scale
        self.alpha = nn.Parameter(torch.randn(1)*1.0e-6)
        self.beta = nn.Parameter(torch.randn(1)*1.0e-6)

    def forward(self, x):
        res1 = x.clone()
        x = self.norm1(x)
        x = self.attn(x)
        x = res1 + self.alpha * x
        
        res2 = x.clone()
        x = self.norm2(x)
        x = self.mlp(x)
        x = res2 + self.beta * x

        return x

class StackedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        blocks = []
        for _ in range(config.n_layers):
            blocks.append(Transformer(config))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        




        

