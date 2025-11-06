import time
import torch
import os
import sys

# Add the project root to the Python path so we can import owl_wms
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the get_loader function from your updated script
from owl_wms.data import get_loader

# --- Configuration ---
# !! Tweak these values to match your setup !!
ROOT_DATA_DIR = "rgb_latents"  # The directory with your round_XXX folders
BATCH_SIZE = 4
WINDOW_LENGTH = 16              # Latent window length
TEMPORAL_COMPRESSION = 8        # Frames of actions per 1 latent frame
# ---------------------

def test_loader():
    print("--- Starting Dataloader Test ---")
    
    if not os.path.isdir(ROOT_DATA_DIR):
        print(f"Error: Data directory not found: {ROOT_DATA_DIR}")
        print("Please update ROOT_DATA_DIR in this script.")
        return

    print(f"Loading data from: {ROOT_DATA_DIR}")
    print(f"Batch Size: {BATCH_SIZE}, Window Length: {WINDOW_LENGTH}, Temporal Comp: {TEMPORAL_COMPRESSION}")
    
    try:
        print("\nInitializing DataLoader...")
        start_time = time.time()
        
        # Initialize the loader
        dataloader = get_loader(
            batch_size=BATCH_SIZE,
            root_dir=ROOT_DATA_DIR,
            window_length=WINDOW_LENGTH,
            temporal_compression=TEMPORAL_COMPRESSION
        )
        
        init_time = time.time() - start_time
        print(f"  Initialization & Indexing complete in {init_time:.2f}s")
        
        # --- Test getting the first batch ---
        print("\nAttempting to load first batch...")
        start_time = time.time()
        
        # Get one batch of data
        latents, actions, states = next(iter(dataloader))
        
        batch_time = time.time() - start_time
        print(f"  Successfully loaded first batch in {batch_time:.2f}s")
        
        # --- Print Shapes and Dtypes ---
        print("\n--- Batch Data Inspection ---")
        
        print(f"  Latents Shape: {latents.shape}")
        print(f"  Latents Dtype: {latents.dtype}")
        
        print(f"\n  Actions Shape: {actions.shape}")
        print(f"  Actions Dtype: {actions.dtype}")
        
        print(f"\n  States Shape: {states.shape}")
        print(f"  States Dtype: {states.dtype}")
        
        # --- Check Expected Shapes ---
        print("\n--- Verification ---")
        assert latents.shape == (BATCH_SIZE, WINDOW_LENGTH, 64, 16, 16), "Latent shape is incorrect!"
        assert actions.shape == (BATCH_SIZE, WINDOW_LENGTH, TEMPORAL_COMPRESSION), "Actions shape is incorrect!"
        assert states.shape[-1] == 3, "States final dimension should be 3!"
        
        print("✅ SUCCESS: All shapes appear correct.")

    except Exception as e:
        print(f"\n❌ FAILED: An error occurred.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set spawn method for multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    test_loader()