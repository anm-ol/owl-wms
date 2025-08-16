import argparse
import os
import torch.multiprocessing as mp

from owl_wms.configs import Config
from owl_wms.trainers import get_trainer_cls
from owl_wms.utils.ddp import cleanup, setup
from dotenv import load_dotenv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, help="Path to config YAML file")
    parser.add_argument("--nccl_timeout", type=int, default=None, help="NCCL process-group timeout in seconds")

    args = parser.parse_args()

    cfg = Config.from_yaml(args.config_path)

    # load environment variables
    load_dotenv()
    mp.set_start_method("spawn", force=True)  # avoid fork + numpy mmaps deadlocks

    global_rank, local_rank, world_size = setup(timeout=args.nccl_timeout)

    trainer = get_trainer_cls(cfg.train.trainer_id)(
        cfg.train, cfg.wandb, cfg.model, global_rank, local_rank, world_size
    )

    trainer.train()
    cleanup()