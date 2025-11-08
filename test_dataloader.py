import time
import torch
import os
import sys

# Add project root and owl-vaes submodule to Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append("./owl-vaes")

# Use the multi-view latent dataloader
from owl_wms.data.tekken_latent_multiV2 import get_loader

# Configuration
ROOT_DATA_DIR = "rgb_latents"
BATCH_SIZE = 4
WINDOW_LENGTH = 16
TEMPORAL_COMPRESSION = 1


def test_loader():
    print("Starting dataloader test.\n")

    if not os.path.isdir(ROOT_DATA_DIR):
        print(f"Error: Data directory not found: {ROOT_DATA_DIR}")
        return

    print(f"Data Directory        : {ROOT_DATA_DIR}")
    print(f"Batch Size           : {BATCH_SIZE}")
    print(f"Window Length        : {WINDOW_LENGTH}")
    print(f"Temporal Compression : {TEMPORAL_COMPRESSION}\n")

    try:
        # Initialize dataloader
        print("Initializing DataLoader...")
        start_time = time.time()

        dataloader = get_loader(
            batch_size=BATCH_SIZE,
            root_dir=ROOT_DATA_DIR,
            window_length=WINDOW_LENGTH,
            temporal_compression=TEMPORAL_COMPRESSION
        )

        init_time = time.time() - start_time
        print(f"DataLoader initialized in {init_time:.2f} seconds.")

        # Load first batch
        print("\nLoading first batch...")
        start_time = time.time()

        latents, actions, states = next(iter(dataloader))

        batch_time = time.time() - start_time
        print(f"First batch loaded in {batch_time:.2f} seconds.\n")

        # Print shapes
        print("Batch Details:")
        print(f"Latents Shape: {latents.shape}, dtype={latents.dtype}")
        print(f"Actions Shape: {actions.shape}, dtype={actions.dtype}")
        print(f"States Shape : {states.shape}, dtype={states.dtype}")

        # Shape checks
        print("\nVerifying shapes...")
        assert latents.shape == (BATCH_SIZE, WINDOW_LENGTH, 64, 16, 16), f"Unexpected latent shape: {latents.shape}"
        assert actions.shape == (BATCH_SIZE, WINDOW_LENGTH, TEMPORAL_COMPRESSION), f"Unexpected action shape: {actions.shape}"
        assert states.shape[-1] == 3, "State vectors should have dimension 3."

        print("All shapes verified successfully.\n")

    except Exception:
        print("\nError: An exception occurred.\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if torch.multiprocessing.get_start_method(allow_none=True) != 'spawn':
        torch.multiprocessing.set_start_method('spawn', force=True)

    test_loader()
