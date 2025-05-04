from dataclasses import dataclass

@dataclass
class ResNetConfig:
    pass

@dataclass
class TransformerConfig:
    n_layers: 12
    n_heads: 12
    d_model: 384

