from torch import nn

from vector_quantize_pytorch import LFQ

class LFQModule(nn.Module):
    def __init__(self, codebook_size = 256):
        super().__init__()

        self.quantizer = LFQ(
            codebook_size = 65536,      # codebook size, must be a power of 2
            entropy_loss_weight = 0.1,  # how much weight to place on entropy loss
            diversity_gamma = 1.        # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
        )

    def forward(self, x):
        return self.quantizer(x)[0]
