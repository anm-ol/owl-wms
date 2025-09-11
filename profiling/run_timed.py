import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from owl_wms.models.tekken_rft_v2 import TekkenRFTV2
from owl_wms.configs import Config
from timed_sampler import ProfiledTekkenActionCachingV2

def main():
    """
    Main function to run the detailed Tekken sampler profiler.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    config = Config.from_yaml("configs/tekken_dmd.yml")
    model_config = config.model
    train_config = config.train

    # Initialize the model
    model = TekkenRFTV2(model_config)
    model.to(device).eval()

    # --- Create Dummy Data ---
    # The sampler takes an initial context of 16 frames and generates 60 new ones.
    # Total action sequence length needs to be at least 16 + 60 = 76.
    total_sequence_length = 100
    context_length = 16
    batch_size = 1

    initial_latents = torch.randn(batch_size, context_length, 128, 14, 23, device=device)
    action_ids = torch.randint(0, 256, (batch_size, total_sequence_length), device=device)

    # --- Run the Profiler ---
    # Create an instance of the profiled sampler
    sampler_args = train_config.sampler_kwargs
    sampler_args['num_frames'] = 60
    
    profiled_sampler = ProfiledTekkenActionCachingV2(**sampler_args)

    print("Running detailed profiling for 60 frames...")
    profiled_sampler(model.core, initial_latents, action_ids)
    
    # Print the detailed report
    profiled_sampler.profiler.report()


if __name__ == "__main__":
    main()
