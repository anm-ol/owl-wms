from dataclasses import dataclass
from typing import List, Optional

@dataclass
class VAEConfig:
    model_id : str = None

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

@dataclass
class TrainerConfig:
    trainer_id : str = None
    data_id : str = None

    target_batch_size : int = 128
    batch_size : int = 2

    epochs : int = 200

    opt : str = "AdamW"
    opt_kwargs : Dict = field(default_factory = lambda : {
        "lr" : 5.0e-5,
        "eps" : 1.0e-15,
        "betas" : (0.9, 0.95)
    })

    scheduler : str = None
    scheduler_kwargs : dict = None

    checkpoint_dir : str = "checkpoints/v0" # Where checkpoints saved
    
    # Distillation related
    teacher_ckpt : str = None
    teacher_cfg : str = None

    sample_interval : int = 1000
    save_interval : int = 1000

@dataclass
class WANDBConfig:
    name : str = None
    project : str = None
    run_name : str = None 