import torch
from torch import nn
import torch.nn.functional as F

from ..nn.sat_encodec import MultiSTFTDiscriminator

class EncodecDiscriminator(nn.Module):
    """
    Encodec discriminator operates at many scales.
    Returns feature matching and adversarial loss
    """
    def __init__(
        self,
        ch = 64,
        channels = 2,
        n_ffts = [2048,1024,512,256,128],
        hop_lengths = [512,256,128,64,32],
        win_lengths = [2048,1024,512,256,128]
    ):
        super().__init__()

        self.discs = MultiSTFTDiscriminator(
            ch, channels,
            n_ffts,hop_lengths,win_lengths
        )

    def forward(self, x):
        return self.discs(x)