# In owl-vaes/owl_vaes/data/t3_live_merge_loader.py
import numpy as np
import torch
import torch.nn.functional as F
import os
import random
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, DistributedSampler


def interpolate_missing_poses(all_poses):
    """Fills in None values in a list of pose arrays using linear interpolation."""
    filled_poses = [p.copy() if p is not None else None for p in all_poses]
    
    # Find all indices of valid poses
    valid_indices = [i for i, p in enumerate(filled_poses) if p is not None]
    
    if not valid_indices: # Handle case where there are no poses at all
        num_keypoints = 17 # Default to COCO keypoint count
        keypoint_dims = 3 # x, y, score
        return [np.zeros((num_keypoints, keypoint_dims)) for _ in range(len(filled_poses))]

    # Fill leading None values with the first valid pose
    first_valid_idx = valid_indices[0]
    if first_valid_idx > 0:
        for i in range(first_valid_idx):
            filled_poses[i] = filled_poses[first_valid_idx]

    # Fill trailing None values with the last valid pose
    last_valid_idx = valid_indices[-1]
    if last_valid_idx < len(filled_poses) - 1:
        for i in range(last_valid_idx + 1, len(filled_poses)):
            filled_poses[i] = filled_poses[last_valid_idx]

    # Interpolate gaps between valid poses
    for i in range(len(valid_indices) - 1):
        start_idx = valid_indices[i]
        end_idx = valid_indices[i+1]
        gap_length = end_idx - start_idx
        
        if gap_length > 1:
            start_pose = filled_poses[start_idx]
            end_pose = filled_poses[end_idx]
            for j in range(1, gap_length):
                alpha = j / gap_length
                interpolated_pose = start_pose * (1 - alpha) + end_pose * alpha
                filled_poses[start_idx + j] = interpolated_pose
                
    return filled_poses

class T3LiveMergeDataset(IterableDataset):
    def __init__(self, root="t3_data/", pose_root="t3_pose/", pose_suffix="_two_player_poses.npz", target_size=(256, 256)):
        super().__init__()
        self.root = root
        self.pose_root = pose_root
        self.pose_suffix = pose_suffix
        self.target_size = tuple(target_size)
        
        self.original_files = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith(".npz")])
        if not self.original_files:
            raise FileNotFoundError(f"No .npz files found in the directory: {root}")

    def __iter__(self):
        worker_info = get_worker_info()
        files_to_process = self.original_files
        if worker_info is not None:
            files_to_process = self.original_files[worker_info.id::worker_info.num_workers]

        while True:
            original_path = random.choice(files_to_process)
            base_name = os.path.basename(original_path)
            pose_path = os.path.join(self.pose_root, base_name.replace('.npz', self.pose_suffix))

            if not os.path.exists(pose_path):
                continue

            try:
                original_data = np.load(original_path)
                pose_data = np.load(pose_path)

                original_images = original_data['images']
                pose_images_raw = pose_data['pose_images']

                # Create a list of poses, using None for blank frames
                all_poses = []
                for pose_frame in pose_images_raw:
                    if np.any(pose_frame): # Check if the frame is not all black
                        all_poses.append(pose_frame)
                    else:
                        all_poses.append(None)
                
                # Interpolate the missing poses
                interpolated_poses = interpolate_missing_poses(all_poses)
                
                min_frames = min(original_images.shape[0], len(interpolated_poses))
                if min_frames == 0:
                    continue

                for i in range(min_frames):
                    orig_frame = torch.from_numpy(original_images[i]).float()
                    pose_frame = torch.from_numpy(interpolated_poses[i]).float()

                    # Ensure pose is single channel
                    if pose_frame.shape[0] == 3:
                        pose_frame = torch.max(pose_frame, dim=0, keepdim=True)[0]

                    combined_frame = torch.cat([orig_frame, pose_frame], dim=0)
                    resized_frame = F.interpolate(combined_frame.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
                    
                    yield resized_frame

            except Exception as e:
                print(f"Warning: Could not process file {original_path}. Error: {e}")
                continue

def collate_fn(frames):
    batch = torch.stack(frames)
    batch = (batch / 255.0) * 2.0 - 1.0
    return batch

def get_loader(batch_size, **data_kwargs):
    dataset = T3LiveMergeDataset(**data_kwargs)
    
    # IterableDatasets don't work well with standard samplers. We rely on the worker splitting logic.
    num_workers = min(os.cpu_count(), 8) 
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader
# ----------------- Main Execution Block (for direct testing) -----------------
# This block is for testing the loader independently and is not used by train.py
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Simple setup for running this file directly
    sys.path.append(str(Path(__file__).parent.parent))

    # Assuming a folder structure for testing
    TEST_ROOT = "t3_data/"
    TEST_POSE_ROOT = "t3_pose/"
#changed batch size to 16 from 4
    try:
        loader = get_loader(batch_size=4, root=TEST_ROOT, pose_root=TEST_POSE_ROOT, target_size=(256, 256))
        print("✅ Data loader initialized. Testing one batch.")
        
        # Use a single, non-overlapping progress bar for clean output
        for batch in tqdm(loader, total=1):
            print(f"Batch shape: {batch.shape}")
            print(f"Batch values range: [{batch.min():.2f}, {batch.max():.2f}]")
            break

    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Please check the paths in the script.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")