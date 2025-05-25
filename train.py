from owl_vaes.configs import Config
from owl_vaes.trainers import get_trainer_cls
from owl_vaes.utils.ddp import setup, cleanup

import argparse

import os
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to config YAML file")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config_path)
    
    global_rank, local_rank, world_size = setup()
    torch.cuda.set_device(local_rank)
    trainer = get_trainer_cls(cfg.train.trainer_id)(
        cfg.train,
        cfg.wandb,
        cfg.model,
        global_rank, local_rank, world_size
    )
    trainer.train()
    cleanup()
