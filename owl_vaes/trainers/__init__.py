from typing import Literal

from .audio_rec import AudioRecTrainer
from .proxy import ProxyTrainer
from .rec import RecTrainer


def get_trainer_cls(trainer_id: Literal["rec", "proxy", "audio_rec"]):
    match trainer_id:
        case "rec":
            return RecTrainer
        case "proxy":
            return ProxyTrainer
        case "audio_rec":
            return AudioRecTrainer
        case _:
            raise NotImplementedError
