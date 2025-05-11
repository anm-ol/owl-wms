import ray
import os
from diffusable_autoencoders.configs import Config
from diffusable_autoencoders.trainers import get_trainer_cls
import torch.distributed as dist

@ray.remote(num_gpus=1)
def train_process(local_rank, node_rank, num_gpus_per_node, config_path):
    # Set environment variables for DDP
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # This should come from cluster config in real multi-node
    os.environ["MASTER_PORT"] = "29500"
    
    # Calculate proper ranks
    world_size = num_gpus_per_node * ray.cluster_resources()["num_nodes"]
    global_rank = (node_rank * num_gpus_per_node) + local_rank
    
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(global_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)

    # Load config
    cfg = Config.from_yaml(config_path)
    
    # Initialize process group
    dist.init_process_group(backend="nccl")
    
    # Create and run trainer
    trainer = get_trainer_cls(cfg.train.trainer_id)(
        cfg.train,
        cfg.wandb,
        cfg.model,
        global_rank,
        local_rank,
        world_size
    )
    trainer.train()
    
    # Cleanup
    dist.destroy_process_group()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to config YAML file")
    parser.add_argument("--gpus_per_node", type=int, default=1, help="Number of GPUs per node")
    args = parser.parse_args()

    # Initialize Ray
    ray.init()

    num_nodes = ray.cluster_resources()["num_nodes"]
    
    # Launch training processes
    futures = []
    for node_rank in range(num_nodes):
        for local_rank in range(args.gpus_per_node):
            futures.append(
                train_process.remote(
                    local_rank,
                    node_rank,
                    args.gpus_per_node,
                    args.config_path
                )
            )
    
    ray.get(futures)

if __name__ == "__main__":
    main()