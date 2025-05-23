from .dcae import DCAE
from .titok import TiToKVAE
from .titok_vq import TiToKVQVAE
from .dcae_vq import DCVQVAE
from .proxy_titok import ProxyTiToKVAE

def get_model_cls(model_id):
    if model_id == "dcae":
        return DCAE
    if model_id == "titok":
        return TiToKVAE
    if model_id == "titok_vq":
        return TiToKVQVAE
    if model_id == "dcae_vq":
        return DCVQVAE
    if model_id == "proxy_titok":
        return ProxyTiToKVAE