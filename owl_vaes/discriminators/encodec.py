import torch
from torch import nn
import torch.nn.functional as F

from ..nn.sat_encodec import MultiSTFTDiscriminator

class EncodecDiscriminator(nn.Module):
    """
    Encodec discriminator operates at many scales.
    Returns feature matching and adversarial loss
    """
    def __init__(self, config):
        super().__init__()

        self.discs = MultiSTFTDiscriminator(
            config.ch, config.channels,
            config.n_ffts,config.hop_lengths,config.win_lengths
        )

    def forward(self, x):
        return self.discs(x)