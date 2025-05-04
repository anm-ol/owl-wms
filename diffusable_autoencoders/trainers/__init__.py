from .rec import RecTrainer

def get_trainer_cls(trainer_id):
    if trainer_id == "rec":
        return RecTrainer