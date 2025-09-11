import torch
import time
import GPUtil
import psutil
from typing import Callable, Dict, Any

def profile_sampler(
    model: torch.nn.Module,
    sampler_class: Callable,
    sampler_args: Dict[str, Any],
    data_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Profiles a given sampler with a model and data loader.

    Args:
        model: The model to use for sampling.
        sampler_class: The sampler class to instantiate.
        sampler_args: Arguments for the sampler class.
        data_loader: The data loader for providing input data.
        device: The device to run the profiling on.

    Returns:
        A dictionary containing profiling metrics.
    """
    print(F"Profiling on device: {device}")
    model.to(device)
    model.eval()

    sampler = sampler_class(**sampler_args)
    
    # Get a batch of data
    initial_latents, action_ids, _ = next(iter(data_loader))
    initial_latents = initial_latents.to(device)
    action_ids = action_ids.to(device)

    # Warm-up run
    with torch.no_grad():
        sampler(model.core, initial_latents, action_ids)

    # Start profiling
    start_time = time.time()
    
    # Memory and GPU utilization tracking
    gpus = GPUtil.getGPUs()
    gpu_start_mem = gpus[0].memoryUsed if gpus else 0
    gpu_util_start = gpus[0].load if gpus else 0
    process = psutil.Process()
    cpu_start_mem = process.memory_info().rss / (1024 ** 2)

    with torch.no_grad():
        print("Profiling TekkenActionCachingV2 sampler...")
        sampler(model.core, initial_latents, action_ids)

    end_time = time.time()

    gpu_end_mem = gpus[0].memoryUsed if gpus else 0
    gpu_util_end = gpus[0].load if gpus else 0
    cpu_end_mem = process.memory_info().rss / (1024 ** 2)

    # Collect results
    execution_time = (end_time - start_time) * 1000  # in ms
    peak_gpu_mem = gpu_end_mem - gpu_start_mem
    avg_gpu_util = (gpu_util_start + gpu_util_end) / 2
    peak_cpu_mem = cpu_end_mem - cpu_start_mem
    
    return {
        "execution_time_ms": execution_time,
        "peak_gpu_mem_mb": peak_gpu_mem,
        "avg_gpu_util_percent": avg_gpu_util * 100,
        "peak_cpu_mem_mb": peak_cpu_mem
    }
