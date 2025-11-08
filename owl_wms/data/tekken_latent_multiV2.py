import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
import torch.distributed as dist
from tqdm import tqdm

try:
    from .cod_latent import AutoEpochDistributedSampler
except ImportError:
    from torch.utils.data import DistributedSampler as AutoEpochDistributedSampler


class TekkenLatentMulti(Dataset):
    """
    Dataset for loading Tekken latent data stored across multiple .pt files,
    with corresponding actions and states stored in .npy files.

    Expected directory structure:
        root_dir/
            round_xxx/
                000000_rgblatent.pt
                000001_rgblatent.pt
                ...
                actions.npy
                states.npy
    """

    def __init__(self, root_dir, window_length, temporal_compression=1, min_sequence_length=None):
        self.root_dir = root_dir
        self.window_length = window_length
        self.temporal_compression = temporal_compression
        self.min_sequence_length = min_sequence_length or window_length

        self.rounds = []
        self.samples = []

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        print(f"Building index for {self.root_dir}...")
        self._build_index()

    def _build_index(self):
        round_dirs = sorted(
            d for d in glob.glob(os.path.join(self.root_dir, "round_*"))
            if os.path.isdir(d)
        )
        if not round_dirs:
            raise FileNotFoundError(f"No 'round_*' directories found in {self.root_dir}")

        for round_dir in tqdm(round_dirs, desc="Indexing rounds"):
            actions_path = os.path.join(round_dir, "actions.npy")
            states_path = os.path.join(round_dir, "states.npy")
            latent_files = sorted(glob.glob(os.path.join(round_dir, "*_rgblatent.pt")))

            if not (os.path.exists(actions_path) and os.path.exists(states_path) and latent_files):
                continue

            segment_lens = []
            try:
                for pt_file in latent_files:
                    tensor_data = torch.load(pt_file, map_location="cpu")
                    segment_lens.append(tensor_data.shape[0])
                    del tensor_data
            except Exception:
                continue

            total_latent_frames = sum(segment_lens)

            try:
                action_frames = np.load(actions_path, mmap_mode="r").shape[0]
                expected_latent_frames = action_frames // self.temporal_compression

                if total_latent_frames != expected_latent_frames:
                    total_latent_frames = min(total_latent_frames, expected_latent_frames)
            except Exception:
                continue

            if total_latent_frames < self.min_sequence_length:
                continue

            self.rounds.append({
                "path": round_dir,
                "actions_path": actions_path,
                "states_path": states_path,
                "latent_files": latent_files,
                "segment_lens": np.array(segment_lens),
                "cumulative_lens": np.cumsum([0] + segment_lens)
            })

            for start_frame in range(total_latent_frames - self.window_length + 1):
                self.samples.append((len(self.rounds) - 1, start_frame))

        if not self.samples:
            raise RuntimeError("No valid samples found.")

        print(f"Indexed {len(self.samples)} samples from {len(self.rounds)} rounds.")

    def __len__(self):
        return len(self.samples)

    def _load_latent_window(self, round_meta, start_frame, end_frame):
        chunks = []
        start_seg = np.searchsorted(round_meta["cumulative_lens"], start_frame, side="right") - 1
        end_seg = np.searchsorted(round_meta["cumulative_lens"], end_frame - 1, side="right") - 1

        for seg_idx in range(start_seg, end_seg + 1):
            seg_path = round_meta["latent_files"][seg_idx]
            segment = torch.load(seg_path, map_location="cpu")

            seg_start = round_meta["cumulative_lens"][seg_idx]
            local_start = max(0, start_frame - seg_start)
            local_end = min(segment.shape[0], end_frame - seg_start)

            chunks.append(segment[local_start:local_end])
            del segment

        return torch.cat(chunks, dim=0)

    def __getitem__(self, idx):
        round_idx, start_latent_frame = self.samples[idx]
        meta = self.rounds[round_idx]

        end_latent_frame = start_latent_frame + self.window_length
        latents = self._load_latent_window(meta, start_latent_frame, end_latent_frame)

        if latents.shape[0] != self.window_length:
            return self.__getitem__(np.random.randint(len(self)))

        actions = np.load(meta["actions_path"], mmap_mode="r")
        states = np.load(meta["states_path"], mmap_mode="r")

        start_orig = start_latent_frame * self.temporal_compression
        end_orig = end_latent_frame * self.temporal_compression

        actions = actions[start_orig:end_orig].reshape(self.window_length, self.temporal_compression)
        states = states[start_orig:end_orig].reshape(self.window_length, self.temporal_compression, states.shape[-1])

        return {
            "latents": latents,
            "actions": torch.from_numpy(actions.copy()).long(),
            "states": torch.from_numpy(states.copy()).float()
        }


def custom_collate_fn(batch, batch_columns=None):
    if not batch:
        return []

    keys = batch[0].keys()
    stacked = {k: torch.stack([item[k] for item in batch]) for k in keys}

    return stacked["latents"], stacked["actions"], stacked["states"]


def get_loader(batch_size, root_dir, window_length=16, temporal_compression=1, **kwargs):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    ds = TekkenLatentMulti(root_dir, window_length, temporal_compression)

    if world_size > 1:
        sampler = AutoEpochDistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        loader_args = dict(sampler=sampler, shuffle=False)
    else:
        loader_args = dict(shuffle=True)

    num_workers = min(os.cpu_count() // max(world_size, 1), 8)

    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=partial(custom_collate_fn, batch_columns=None),
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        **loader_args
    )
