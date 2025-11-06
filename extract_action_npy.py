import os
import glob
import numpy as np
import sys
from tqdm import tqdm

def extract_actions_and_states():
    """
    Finds all 'round_*.npz' files, extracts 'actions_p1' and 'states'
    based on the 'valid_frames' mask, and saves them as .npy files
    in the corresponding 'rgb_latents/round_XXX/' directory.
    """
    
    # 1. Define source and destination paths
    source_dirs = ["tekken_dataset_npz/P1_WIN", "tekken_dataset_npz/P2_WIN"]
    dest_base_dir = "rgb_latents"
    
    print(f"Source directories: {source_dirs}")
    print(f"Destination base: {dest_base_dir}")
    
    # 2. Gather all .npz files from all source directories
    all_npz_files = []
    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            print(f"Warning: Source directory not found, skipping: {source_dir}")
            continue
        
        found_files = glob.glob(os.path.join(source_dir, "round_*.npz"))
        all_npz_files.extend(found_files)
        print(f"Found {len(found_files)} files in {source_dir}")
        
    if not all_npz_files:
        print("Error: No 'round_*.npz' files found in any source directory. Exiting.")
        sys.exit(1)
        
    print(f"\nFound a total of {len(all_npz_files)} files to process.")

    # 3. Process each file
    for npz_path in tqdm(all_npz_files, desc="Processing files"):
        try:
            # Get the round name (e.g., "round_001")
            round_name = os.path.basename(npz_path).replace(".npz", "")
            
            # Define the destination directory and create it
            dest_round_dir = os.path.join(dest_base_dir, round_name)
            os.makedirs(dest_round_dir, exist_ok=True)
            
            # Define final output paths
            actions_out_path = os.path.join(dest_round_dir, "actions.npy")
            states_out_path = os.path.join(dest_round_dir, "states.npy")

            # Use 'with' for safe file handling
            with np.load(npz_path) as data:
                
                if 'valid_frames' not in data:
                    print(f"Warning: 'valid_frames' key not in {npz_path}. Skipping file.")
                    continue
                    
                mask = data['valid_frames']
                
                # Find the index of the last valid frame
                valid_indices = np.where(mask == 1)[0]
                if len(valid_indices) == 0:
                    print(f"Warning: No valid frames found in {npz_path}. Skipping file.")
                    continue
                
                # The slice should go up to and *include* the last valid index
                end_slice = valid_indices[-1] + 1
                
                # Extract the data
                actions = data['actions_p1'][:end_slice]
                states = data['states'][:end_slice]

            # Save the extracted arrays as new .npy files
            np.save(actions_out_path, actions)
            np.save(states_out_path, states)
            
        except Exception as e:
            print(f"\nError processing file {npz_path}: {e}")
            
    print(f"\n--- Extraction Complete ---")
    print(f"Actions and states have been saved to subdirectories within {dest_base_dir}")

if __name__ == "__main__":
    extract_actions_and_states()