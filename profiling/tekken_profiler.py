import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from owl_wms.models.tekken_rft_v2 import TekkenRFTV2
from owl_wms.sampling.t3_action_v2 import TekkenActionCachingV2
from owl_wms.data.tekken_latent_multi import TekkenLatentMulti
from owl_wms.configs import Config
from profiler import profile_sampler

def main():
    """
    Main function to run the Tekken sampler profiler.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    config = Config.from_yaml("configs/tekken_dmd.yml")
    model_config = config.model
    train_config = config.train

    # Initialize the model
    model = TekkenRFTV2(model_config)

    # Create dummy data loader for profiling
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, length=10):
            # The sampler takes an initial context of 16 frames and generates 60 new ones.
            # The total action sequence length needs to be at least 16 + 60 = 76.
            # We'll use 100 for a safe buffer.
            total_sequence_length = 100
            context_length = 16

            self.length = length
            self.latents = torch.randn(length, context_length, 128, 14, 23)
            self.actions = torch.randint(0, 256, (length, total_sequence_length))
            self.states = torch.randn(length, total_sequence_length, 3)

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return self.latents[idx], self.actions[idx], self.states[idx]

    dummy_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=1)

    # --- CHANGE: Explicitly set the number of frames to generate for profiling ---
    train_config.sampler_kwargs['num_frames'] = 60

    # Run the profiler
    metrics = profile_sampler(
        model,
        TekkenActionCachingV2,
        train_config.sampler_kwargs,
        dummy_loader,
        device
    )

    # Print the results
    print("--- Tekken Sampler Profiling Results (60 frames) ---")
    print(f"Execution Time: {metrics['execution_time_ms']:.2f} ms")
    print(f"Peak GPU Memory Usage: {metrics['peak_gpu_mem_mb']:.2f} MB")
    print(f"Average GPU Utilization: {metrics['avg_gpu_util_percent']:.2f}%")
    print(f"Peak CPU Memory Usage: {metrics['peak_cpu_mem_mb']:.2f} MB")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()

