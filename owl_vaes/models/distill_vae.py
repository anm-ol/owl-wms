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

        self.conv_in = weight_norm(nn.Conv2d(config.latent_channels, config.ch_max, 3, 1, 1, bias = False))

        total_blocks = len(config.decoder_blocks_per_stage)

        def res(ch):
            return ResBlock(ch, total_blocks)

        def up(ch, next_ch, stride):
            return Upsample(ch, next_ch, stride)

        ndf = config.ch_0
        
        self.blocks = nn.Sequential(*[
            up(ndf*16, ndf*16, 2), # -> 16
            res(ndf*16),
            up(ndf*16,ndf*8, 4), # -> 64
            res(ndf*8),
            up(ndf*8,ndf*4, 2), # -> 128
            res(ndf*4),
            up(ndf*4,ndf*2, 2), # -> 256
            res(ndf*2),
            up(ndf*2, ndf, 2), # -> 512
            res(ndf)
        ])

        self.conv_out = SquareToLandscape(config.sample_size, ndf, config.channels) if is_landscape(config.sample_size) else weight_norm(nn.Conv2d(ndf, config.channels, 3, 1, 1, bias = False))
        
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