from .dcae import DCAE
from .titok import TiToKVAE

def get_model_cls(model_id):
    if model_id == "dcae":
        return DCAE
    if model_id == "titok":
        return TiToKVAE