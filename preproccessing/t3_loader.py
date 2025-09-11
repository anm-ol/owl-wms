from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

import os
import random

class T3Dataset(IterableDataset):
    def __init__(self, root = "t3_data/tekken_dataset_npz"):
        super().__init__()

        self.frames = []
        print("Loading Tekken dataset...")
        
        # Get all .npz files in the root directory
        if os.path.isdir(root):
            files = os.listdir(root)
            npz_files = [f for f in files if f.endswith(".npz")]
            
            for i, npz_file in enumerate(npz_files):
                npz_path = os.path.join(root, npz_file)
                print(f"Loading {npz_file} ({i+1}/{len(npz_files)})...")
                
                # Load .npz file
                data = np.load(npz_path)
                mask = data['attention_mask']
                idx = int(np.where(mask == 1)[0][-1])
                images = data['images'][:idx]
                
                # Convert all frames to torch tensors and add to list
                for frame in images:
                    frame_tensor = torch.from_numpy(frame).float()
                    self.frames.append(frame_tensor)
                
                data.close()  # Free memory
        
        print(f"Loaded {len(self.frames)} total frames from {len(npz_files)} files")
    
    def get_item(self):
        # Pick random frame from preloaded frames
        frame_idx = random.randint(0, len(self.frames) - 1)
        frame = self.frames[frame_idx]
        
        return frame  # [c,h,w]

    def __iter__(self):
        while True:
            yield self.get_item()

def collate_fn(x):
    # x is list of frames
    x = [frame/255.0 for frame in x]  # Normalize to [0, 1]
    x = [frame *2 - 1 for frame in x]  # Scale to [-1, 1]
    res = torch.stack(x)
    return res  # [b,c,h,w]

def get_loader(batch_size, **data_kwargs):
    """
    Creates a DataLoader for the T3Dataset with the specified batch size
    
    Args:
        batch_size: Number of samples per batch
        **data_kwargs: Additional arguments to pass to T3Dataset (e.g., root path)
        
    Returns:
        DataLoader instance
    """
    dataset = T3Dataset(**data_kwargs)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    return loader

class Tekken3dDataset(Dataset):
    def __init__(self, root="t3_data/tekken_dataset_npz"):
        self.root = root
        self.rounds = []
        
        # Load all frames from the dataset
        if os.path.isdir(root):
            files = os.listdir(root)
            npz_files = [f for f in files if f.endswith(".npz")]
            
            for index, npz_file in enumerate(npz_files):
                npz_path = os.path.join(root, npz_file)
                data = np.load(npz_path)
                mask = data['attention_mask']
                end_idx = int(np.where(mask == 1)[0][-1])
                images = data['images'][:end_idx]
                
                video_tensor = torch.from_numpy(images).float()
                self.rounds.append(video_tensor)
                
                data.close()  # Free memory
        
        print(f"Loaded {len(self.rounds)} rounds from {len(npz_files)} files")

    def __len__(self):
        return sum(frame.shape[0] for frame in self.rounds)

    def __getitem__(self, idx):
        return self.rounds[idx]  # [b,c,h,w]

if __name__ == "__main__":
    import time
    loader = get_loader(4)

    start = time.time()
    batch = next(iter(loader))
    end = time.time()

    print(f"Time to load batch: {end-start:.2f}s")
    print(f"Video shape: {batch.shape}")