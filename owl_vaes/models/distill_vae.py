import torch
from torch import nn
import torch.nn.functional as F

from ..nn.resnet import ResBlock, Downsample, Upsample, LandscapeToSquare, SquareToLandscape
from torch.nn.utils.parametrizations import weight_norm

def is_landscape(sample_size):
    h,w = sample_size
    ratio = w/h
    return abs(ratio - 16/9) < 0.01

class DistillEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv_in = LandscapeToSquare(config.sample_size, config.channels, config.ch_0) if is_landscape(config.sample_size) else weight_norm(nn.Conv2d(config.channels, config.ch_0, 3, 1, 1, bias = False))

        def ch(idx):
            return min(config.ch_0 * 2**idx, config.ch_max)

        total_blocks = len(config.encoder_blocks_per_stage)

        def res(idx):
            return ResBlock(ch(idx), total_blocks)
    
        def down(idx, stride):
            return Downsample(ch(idx), ch(idx+1), stride)

        self.blocks = nn.Sequential(*[
            res(0),
            down(0, 2), # -> 256
            res(1),
            down(1, 2), # -> 128
            res(2),
            down(2, 2), # -> 64
            res(3),
            down(3, 4), # -> 16
            res(4),
            down(4, 2), # -> 8
        ])

        self.conv_out = weight_norm(nn.Conv2d(ch(4), config.latent_channels, 3, 1, 1, bias = False))

    def forward(self, x):
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.conv_out(x)
        return x

class DistillDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv_in = weight_norm(nn.Conv2d(config.latent_channels, config.ch_0, 3, 1, 1, bias = False))

        total_blocks = len(config.decoder_blocks_per_stage)

        def ch(idx):
            idx = max(0, idx)
            return min(config.ch_0 * 2**idx, config.ch_max)

        def res(idx):
            return ResBlock(ch(idx), total_blocks)

        def up(idx, stride):
            return Upsample(ch(idx), ch(idx+1), stride)
        
        self.blocks = nn.Sequential(*[
            up(-1, 2), # -> 16
            res(0),
            up(0, 4), # -> 64
            res(1),
            up(1, 2), # -> 128
            res(2),
            up(2, 2), # -> 256
            res(3),
            up(3, 2), # -> 512
            res(4)
        ])

        self.conv_out = weight_norm(nn.Conv2d(ch(4), config.channels, 3, 1, 1, bias = False))
        
    def forward(self, x):
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.conv_out(x)
        return x

class DistillVAE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = DistillEncoder(config)
        self.decoder = DistillDecoder(config)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)