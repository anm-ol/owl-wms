from .rec import RecTrainer
from .proxy import ProxyTrainer

def get_trainer_cls(trainer_id):
    if trainer_id == "rec":
        return RecTrainer
    if trainer_id == "proxy":
        return ProxyTrainer