import time 
import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.nn.functional as F


class TekkenLatentDataset(Dataset):
    """
    Loads Tekken latents, actions, and states into sliding windows.
    """
    def __init__(self, root_dir, window_length=16, expected_buttons=11):
        self.root_dir = root_dir
        self.window_length = window_length
        self.expected_buttons = expected_buttons

        self.rounds_data = []
        self.samples = []

        if not os.path.isdir(root_dir):
            raise ValueError(f"Root directory not found or not a directory: {root_dir}")

        round_dirs = sorted(
            d for d in glob.glob(os.path.join(root_dir, 'round_*')) if os.path.isdir(d)
        )

        if not round_dirs:
            raise FileNotFoundError(f"No round directories found in {root_dir}")

        for round_dir in round_dirs:
            round_name = os.path.basename(round_dir)

            latents_path = os.path.join(round_dir, 'latents', f'{round_name}_latents.npy')
            actions_path = os.path.join(round_dir, 'actions', f'{round_name}_actions.npy')
            states_path = os.path.join(round_dir, 'states', f'{round_name}_states.npy')

            if not all(os.path.exists(p) for p in [latents_path, actions_path, states_path]):
                continue

            latents = np.load(latents_path).transpose(1, 0, 2, 3)  # (T, C, H, W)
            keep_frames = np.load(actions_path).shape[0] // 8 * 8  # multiple of 8

            # If your .npy files are already (T, 8) and (T, 8, 3), these reshape lines are OK.
            # If not, consider removing reshape and just rely on dynamic padding below.
            actions = np.load(actions_path)[:keep_frames].reshape(-1, 8)      # (T, B)
            states  = np.load(states_path)[:keep_frames].reshape(-1, 8, 3)    # (T, B, 3)
            actions = actions.astype(np.int32)
            self.rounds_data.append({
                'latents': torch.from_numpy(latents),
                'actions': torch.from_numpy(actions),
                'states':  torch.from_numpy(states)
            })

            num_frames = latents.shape[0]
            round_idx = len(self.rounds_data) - 1
            for start_frame in range(num_frames - self.window_length + 1):
                self.samples.append((round_idx, start_frame))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        round_idx, start_frame = self.samples[idx]
        end_frame = start_frame + self.window_length
        data = self.rounds_data[round_idx]

        latents_slice = data['latents'][start_frame:end_frame]               # (T, C, H, W)
        states_slice  = data['states'][start_frame:end_frame].float()        # (T, B, 3)
        actions_slice = data['actions'][start_frame:end_frame].float()       # (T, B)

        return latents_slice, states_slice, actions_slice


def collate_fn(batch):
    """
    Collates and pads sequences:
      - pre-pad TIME (start) so last frames align,
      - pad BUTTONS (end) so button dims match.
    states:  (T, B, 3) -> pad as (0,0, 0,pad_B, pad_T,0)
    actions: (T, B)    -> pad as (0,pad_B, pad_T,0)
    """
    latents_list, states_list, actions_list = zip(*batch)

    # Debug prints BEFORE padding (optional; comment out for speed)
    # for i in range(len(latents_list)):
    #     print(f"Shape of latents: {latents_list[i].shape}, states: {states_list[i].shape}, actions: {actions_list[i].shape}")

    # Max time and button sizes across the batch
    max_T = max(s.shape[0] for s in states_list)
    max_B_states = max(s.shape[1] for s in states_list)
    max_B_actions = max(a.shape[1] for a in actions_list)
    max_B = max(max_B_states, max_B_actions)

    padded_states  = []
    padded_actions = []

    for s, a in zip(states_list, actions_list):
        # States: (T, B, 3)
        pad_T_s = max_T - s.shape[0]
        pad_B_s = max_B - s.shape[1]
        s_p = F.pad(s, (0, 0,        # last dim (features=3): no pad
                        0, pad_B_s,  # buttons: pad at end
                        pad_T_s, 0)  # time: pre-pad at start
                  )
        padded_states.append(s_p)

        # Actions: (T, B)
        pad_T_a = max_T - a.shape[0]
        pad_B_a = max_B - a.shape[1]
        a_p = F.pad(a, (0, pad_B_a,  # buttons: pad at end
                        pad_T_a, 0)  # time: pre-pad at start
                  )
        padded_actions.append(a_p)

    # Latents windows are fixed-length by construction (T = window_length),
    # so we stack directly. If you ever need to pad latents in time too,
    # mirror the states/actions logic for (T, C, H, W).
    video_batch   = torch.stack(latents_list)      # (B, T, C, H, W)
    states_batch  = torch.stack(padded_states)     # (B, T, B_max, 3)
    actions_batch = torch.stack(padded_actions)    # (B, T, B_max)

    # Debug prints AFTER padding (optional)
    # print(f"Video: {video_batch.shape}, States: {states_batch.shape}, Actions: {actions_batch.shape}")

    return video_batch, states_batch, actions_batch


def get_loader(batch_size, root_dir, window_length=16, **kwargs):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    dataset = TekkenLatentDataset(root_dir=root_dir, window_length=window_length)

    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )


if __name__ == "__main__":
    path = "/home/venky/ankitd/anmol/WM/owl-wms/preproccessing/cached_data_ltx"
    dataloader_time = time.time()
    dataloader = get_loader(batch_size=128, root_dir=path, window_length=16)
    print(f'Dataloader time: {time.time() - dataloader_time:.2f} seconds')
    load_time = time.time()
    for video, states, actions in dataloader:
        print(f"Batch loaded. Video: {video.shape}, States: {states.shape}, Actions: {actions.shape}")
        # One pass is enough to verify shapes
        print(f'Time taken: {time.time() - load_time:.2f} seconds')
        load_time = time.time()  # Reset timer for next batch
        pass
        # break
