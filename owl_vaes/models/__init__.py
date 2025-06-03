from typing import Any

from .dcae import DCAE

from .dcae import DCAE, Decoder
from .dcae_vq import DCVQVAE
from .proxy_titok import ProxyTiToKVAE
from .titok import TiToKVAE
from .titok_vq import TiToKVQVAE

def get_model_cls(model_id: str) -> Any:
    return {
        "dcae": DCAE,
        "titok": TiToKVAE,
        "titok_vq": TiToKVQVAE,
        "dcae_vq": DCVQVAE,
        "proxy_titok": ProxyTiToKVAE,
        "dcae_decoder": Decoder
    }.get(model_id)
