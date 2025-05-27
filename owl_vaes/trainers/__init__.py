from typing import Literal

from .proxy import ProxyTrainer
from .rec import RecTrainer


def get_trainer_cls(trainer_id: Literal["rec", "proxy"]):
    match trainer_id:
        case "rec":
            return RecTrainer
        case "proxy":
            return ProxyTrainer
        case _:
            raise NotImplementedError
