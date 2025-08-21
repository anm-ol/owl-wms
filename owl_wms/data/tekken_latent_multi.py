import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class TekkenLatentMulti(Dataset):
    """
    A sliding-window view over your custom nested directory structure.
    Expects structure: data_dir/round_name/{latents,actions,states}/round_name_{type}.npy
    """
    
    def __init__(
        self,
        root_dir: str,
        window_length: int,
        temporal_compression: int = 8,
        data_types: tuple = ("latents", "actions", "states"),
        min_sequence_length: int = None,
    ):
        self.data_dir = root_dir
        self.window_length = window_length
        self.temporal_compression = temporal_compression
        self.data_types = data_types
        self.min_sequence_length = min_sequence_length or window_length
        
        # Build index of all valid windows
        self._build_index()
        
        # print(f"{len(self._index)} samples qualified from {len(self.rounds)} rounds")
    
    def _build_index(self):
        """Build index of (round_dir, start_offset) pairs for all valid windows"""
        self.rounds = []
        self._index = []
        
        # Find all round directories
        round_dirs = [d for d in glob.glob(os.path.join(self.data_dir, "*")) 
                     if os.path.isdir(d)]
        
        for round_dir in sorted(round_dirs):
            round_name = os.path.basename(round_dir)
            
            # Check if all required data types exist
            data_files = {}
            valid_round = True
            
            for data_type in self.data_types:
                expected_file = os.path.join(round_dir, data_type, f"{round_name}_{data_type}.npy")
                if os.path.exists(expected_file):
                    data_files[data_type] = expected_file
                else:
                    valid_round = False
                    break
            
            if not valid_round:
                continue
            
            # Get sequence length from latents (which are temporally compressed)
            latents_file = data_files.get("latents")
            if latents_file:
                try:
                    # Load just to get shape, then close
                    sample_data = np.load(latents_file, mmap_mode='r')
                    # IMPORTANT: Get the temporal dimension after transpose
                    # Your latents are stored as (channels, time, h, w) but you transpose to (time, channels, h, w)
                    seq_len = sample_data.shape[1]  # Time dimension before transpose
                    del sample_data  # Free memory
                except:
                    continue
            else:
                # If no latents file, get from actions and divide by 8
                actions_file = data_files.get("actions")
                if actions_file:
                    try:
                        sample_data = np.load(actions_file, mmap_mode='r')
                        frame_count = sample_data.shape[0] if len(sample_data.shape) > 0 else 0
                        seq_len = frame_count // self.temporal_compression  # Convert to latent temporal resolution
                        del sample_data
                    except:
                        continue
                else:
                    continue
            
            # Skip if sequence is too short
            if seq_len < self.min_sequence_length:
                # print(f"Skipping {round_name}: seq_len {seq_len} < min_length {self.min_sequence_length}")
                continue
            
            # Add this round to our list
            self.rounds.append({
                'round_dir': round_dir,
                'round_name': round_name,
                'data_files': data_files,
                'seq_len': seq_len
            })
            
            # Create windows for this round (based on latent temporal resolution)
            # FIXED: Only create windows that fit completely within the sequence
            for start in range(0, seq_len - self.window_length + 1, self.window_length):
                self._index.append((len(self.rounds) - 1, start))
            
            # print(f"Round {round_name}: seq_len={seq_len}, created {len(range(0, seq_len - self.window_length + 1, self.window_length))} windows")
    
    def __len__(self):
        return len(self._index)
    
    def __getitem__(self, idx):
        round_idx, start = self._index[idx]
        round_info = self.rounds[round_idx]
        
        # Load windowed data for each type
        result = {}
        for data_type in self.data_types:
            file_path = round_info['data_files'][data_type]
            # Use memory mapping for efficiency
            full_data = np.load(file_path, mmap_mode='r')
            
            if data_type == "latents":
                # Latents are stored as (channels, time, h, w), transpose to (time, channels, h, w)
                transposed_data = full_data.transpose(1, 0, 2, 3)
                # Now safely window - we know this will work because of the bounds checking in _build_index
                windowed_data = transposed_data[start:start + self.window_length]
                result[data_type] = torch.from_numpy(windowed_data.copy())
                
                # Debug info (remove after testing)
                # print(f'Round: {round_info["round_name"]}, Windowing latents from {start} to {start + self.window_length}, got shape: {windowed_data.shape}')

            elif data_type == "actions":
                # Actions: reshape from (window_length * 8,) to (window_length, 8)
                frame_start = start * self.temporal_compression
                frame_end = frame_start + self.window_length * self.temporal_compression
                
                # Safe windowing with padding if needed
                if frame_end > full_data.shape[0]:
                    windowed_data = full_data[frame_start:]
                    # Pad to required length
                    windowed_data = np.pad(windowed_data, (0, frame_end - full_data.shape[0]), mode='constant')
                else:
                    windowed_data = full_data[frame_start:frame_end]
                
                # Reshape to (window_length, 8)
                windowed_data = windowed_data.reshape(self.window_length, self.temporal_compression)
                result[data_type] = torch.from_numpy(windowed_data.copy())
            
            elif data_type == "states":
                # States: reshape from (window_length * 8, 3) to (window_length, 8, 3)
                frame_start = start * self.temporal_compression
                frame_end = frame_start + self.window_length * self.temporal_compression

                # Safe windowing with padding if needed
                if frame_end > full_data.shape[0]:
                    windowed_data = full_data[frame_start:]
                    needed_frames = frame_end - full_data.shape[0]
                    windowed_data = np.pad(windowed_data, ((0, needed_frames), (0, 0)), mode='constant')
                else:
                    windowed_data = full_data[frame_start:frame_end]
                
                # Reshape to (window_length, 8, 3)
                windowed_data = windowed_data.reshape(self.window_length, self.temporal_compression, 3)
                result[data_type] = torch.from_numpy(windowed_data.copy())

            else:
                # For any other data types, assume same temporal resolution as latents
                windowed_data = full_data[start:start + self.window_length]
                result[data_type] = torch.from_numpy(windowed_data.copy())
        
        return result


# Rest of your code remains the same...

# Updated collate function for your data types
def custom_collate_fn(batch, batch_columns: list = None):
    """
    Collate function that handles your data structure.
    If batch_columns is None, returns all available data types.
    """
    if not batch:
        return []
    
    # Get all available keys if batch_columns not specified
    available_keys = list(batch[0].keys())
    columns_to_return = batch_columns if batch_columns is not None else available_keys
    
    # Stack tensors for each data type
    stacked = {}
    for key in available_keys:
        stacked[key] = torch.stack([item[key] for item in batch])
    
    # Handle dtype conversions if needed
    # You might want to customize this based on your data types
    for key, tensor in stacked.items():
        if key == "latents" and tensor.dtype == torch.float32:
            stacked[key] = tensor.to(torch.bfloat16)
        elif key in ["actions", "states"] and tensor.dtype != torch.float32:
            stacked[key] = tensor.float()
    
    # Return in specified order
    return [stacked[col] for col in columns_to_return]


# Updated get_loader function
def get_loader(batch_size, root_dir, window_length=16, **kwargs):
    batch_columns=None
    """
    Create DataLoader for your custom dataset structure.
    """
    from torch.utils.data import DataLoader
    from functools import partial
    import torch.distributed as dist
    from .cod_latent import AutoEpochDistributedSampler  # Import the sampler
    
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    ds = TekkenLatentMulti(root_dir, window_length)

    if world_size > 1:
        sampler = AutoEpochDistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        loader_kwargs = dict(sampler=sampler, shuffle=False)
    else:
        loader_kwargs = dict(shuffle=True)
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=partial(custom_collate_fn, batch_columns=batch_columns),
        num_workers=2,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        **loader_kwargs
    )