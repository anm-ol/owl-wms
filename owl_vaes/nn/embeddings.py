import torch
from torch import nn

import einops as eo
from rotary_embedding_torch import (
    RotaryEmbedding,
    apply_rotary_emb
)

class ImageRoPE(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        pass

    def forward(self, q, k):
        # q k both [b,h,n,d]
        pass

class LearnedPosEnc(nn.Module):
    def __init__(self, n_seq, dim):
        super().__init__()

        self.p = nn.Parameter(torch.randn(n_seq,dim)*0.02)

    def forward(self, x):
        b,n,d = x.shape
        p = eo.repeat(self.p, 'n d -> b n d', b = b)
        return x + p

class RoPEEmbedding(nn.Module):
    """
    Video RoPE embedding for when latents are 3D [n,h,w]
    """
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        dim_head = config.d_model // config.n_heads
        self.pos_emb = RotaryEmbedding(
            dim = dim_head//2
        )

    def forward(self, q, k):
        # q k both [b,h,n,d]
        # k can be longer for caching
        q,k = self.pos_emb.rotate_queries_with_cached_keys(q,k)
        return q,k
        
        return q, k
