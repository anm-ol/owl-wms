from .configs import Config
from .models import get_model_cls
from .utils import versatile_load

def from_pretrained(cfg_path, ckpt_path):
    cfg = Config.from_yaml(cfg_path).model
    model_cls = get_model_cls(cfg.model_id)
    model = model_cls(cfg)
    
    ckpt = versatile_load(ckpt_path)
    model.load_state_dict(ckpt)
    return model