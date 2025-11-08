import numpy as np
import argparse
import os
import sys

def inspect_npy_file(file_path, num_rows=5):
    """
    Loads and prints key information about a .npy file.
    """
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    try:
        data = np.load(file_path)
        
        print(f"\n--- 🔍 Inspecting: {file_path} ---")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        
        if np.issubdtype(data.dtype, np.number):
            print("\n  Statistics:")
            if data.size == 0:
                print("    (Array is empty)")
            else:
                try:
                    print(f"    Min:  {np.min(data)}")
                    print(f"    Max:  {np.max(data)}")
                    print(f"    Mean: {np.mean(data):.4f}")
                    
                    # Check for unique values, useful for 'actions.npy'
                    unique_vals = np.unique(data)
                    if len(unique_vals) < 30: # Only show if there aren't too many
                        print(f"    Unique Values: {unique_vals}")
                    else:
                        print(f"    Unique Values: {len(unique_vals)} total")
                        
                except Exception as e:
                    print(f"    Could not compute stats: {e}")
        else:
            print("\n  Statistics: (Skipped, non-numeric data)")

        print(f"\n  Example Data (first {num_rows} rows):")
        if data.size == 0:
            print("    (Array is empty)")
        else:
            print(data[:num_rows])
        print("-" * (len(file_path) + 24))

    except Exception as e:
        print(f"Error: Could not load or inspect file {file_path}.\n{e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Inspects a .npy file and prints its shape, dtype, stats, and example data.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "file_path",
        type=str,
        help="The path to the .npy file to inspect."
    )
    
    parser.add_argument(
        "-n", "--rows",
        type=int,
        default=5,
        help="Number of example rows to print (default: 5)."
    )
    
    args = parser.parse_args()
    
    inspect_npy_file(args.file_path, args.rows)

if __name__ == "__main__":
    main()