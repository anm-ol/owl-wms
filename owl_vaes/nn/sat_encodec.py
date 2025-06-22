# adapted from https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/models/encodec.py
# License can be found in LICENSE

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn.utils.parametrizations import weight_norm

import einops as eo
import math

from torchaudio.transforms import Spectrogram

def make_pad(k, d = (1,1)):
    return (
        ((k[0] - 1) * d[0]) // 2,
        ((k[1] - 1) * d[1]) // 2
    )

def make_conv(ch_in, ch_out, k, s=1, d=1, p=None):
    return weight_norm(nn.Conv2d(ch_in,ch_out,k,s,p,dilation=d))

class STFTDiscriminator(nn.Module):
    def __init__(
        self,
        ch = 64,
        channels = 2,
        n_fft = 1024, hop_length = 256, win_length = 1024,
        k=(3,9), d = [1,2,4], s = (1,1)
    ):
        super().__init__()

        self.spec = Spectrogram(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window_fn=torch.hann_window,normalized=True,center=False,
            pad_mode=None,power=None
        ) # -> 2*channels

        self.blocks = nn.ModuleList([
            make_conv(2*channels, ch, k, p=make_pad(k)),
            make_conv(ch, ch*2, k, s, (d[0],1), make_pad(k, (d[0],1))),
            make_conv(ch*2, ch*4, k, s, (d[1],1), make_pad(k, (d[1],1))),
            make_conv(ch*4, ch*8, k, s, (d[2],1), make_pad(k, (d[2],1))),
            make_conv(ch*8, ch*16, (k[0],k[0]), p=make_pad((k[0],k[0]))),
        ])
        self.conv_out = make_conv(ch*16, 1, (k[0],k[0]), p=make_pad((k[0],k[0])))

    def forward(self, x):
        h = []
        x = self.spec(x)
        x = torch.cat([x.real, x.imag], dim=1)
        x = x.transpose(-1,-2) # b c w t -> b c t w
        
        for block in self.blocks:
            x = checkpoint(block, x)
            x = F.leaky_relu(x,0.2)
            h.append(x.clone())
        
        x = checkpoint(self.conv_out, x)
        return x, h

class MultiSTFTDiscriminator(nn.Module):
    def __init__(
        self,
        ch = 64, channels = 2,
        n_ffts = [2048,1024,512], hop_lengths = [512,256,128],
        win_lengths = [2048,1024,512]
    ):
        super().__init__()

        self.discriminators = nn.ModuleList([
            STFTDiscriminator(ch, channels, n_fft, hop_length, win_length)
            for (n_fft, hop_length, win_length) in zip(n_ffts, hop_lengths, win_lengths)
        ])
    
    def forward(self, x):
        # is [b,c,n] waveforms
        scores = []
        hs = []
        for disc in self.discriminators:
            score, h = disc(x)
            
            scores.append(score.clone())
            hs.append(h.clone())

        return scores, hs