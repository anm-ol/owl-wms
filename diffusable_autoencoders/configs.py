from dataclasses import dataclass
from typing import List

@dataclass
class VAEConfig:
    sample_size : int = 512
    channels : int = 3
    latent_size : int = 16 # for 1d this is tokens, for 2d it is h/w
    latent_channels : int = 32
    
    noise_decoder_inputs : float = 0.0

@dataclass
class ResNetConfig(VAEConfig):
    ch_0 : int = 256
    ch_max : int = 2048

    encoder_blocks_per_stage : List[int] = None
    decoder_blocks_per_stage : List[int] = None

    attn_size : int = None  # When size is less than this, attention will be used

@dataclass
class TransformerConfig(VAEConfig):
    n_layers : int = 12
    n_heads : int = 12
    d_model : int = 384
    
    patch_size : int = 1