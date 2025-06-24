import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from ..nn.attn import StackedTransformer
from ..nn.audio_blocks import SnakeBeta

class AudioTransformerDecoder(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        self.config = config

        self.conv_in = weight_norm(nn.Conv2d(config.latent_channels, config.d_model, 7, 1, 3))
        self.blocks = StackedTransformer(config)
        self.act_out = SnakeBeta(config.d_model)
        self.conv_out = weight_norm(nn.Conv2d(config.latent_channels, config.channels, 7, 1, 3, bias=False))

    def forward(self, x, kv_cache):
        # x is [b,c,n]
        x = self.conv_in(x)
        x = x.transpose(-1,-2).contiguous() # [b,n,d]
        x = self.blocks(x, kv_cache)
        x = x.transpose(-1,-2).contiguous() # [b,d,n]
        x = self.act_out(x)
        x = self.conv_out(x)
        return x