from typing import Literal

from .audio_rec import AudioRecTrainer
from .proxy import ProxyTrainer
from .rec import RecTrainer
from .decoder_tune import DecTuneTrainer
from .diffdec_trainer import DiffusionDecoderTrainer

def get_trainer_cls(trainer_id: Literal["rec", "proxy", "audio_rec"]):
    match trainer_id:
        case "rec":
            return RecTrainer
        case "proxy":
            return ProxyTrainer
        case "audio_rec":
            return AudioRecTrainer
        case "dec_tune":
            return DecTuneTrainer
        case "diff_dec":
            return DiffusionDecoderTrainer
        case _:
            raise NotImplementedError
