from .npy_table import NpyTable

from functools import partial
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset

from dotenv import load_dotenv
load_dotenv()


class AutoEpochDistributedSampler(DistributedSampler):
    """Ensure we shuffle every epoch"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_epoch = 0

    def __iter__(self):
        self.set_epoch(self._auto_epoch)
        self._auto_epoch += 1
        return super().__iter__()


class WindowedViewDataset(Dataset):
    """
    A sliding-window view over an NpyTable.
    Indexes into (row_idx, start_offset) pairs.
    """
    def __init__(
        self,
        table_dir: str,
        window_length: int,
        include_missing_features: bool = False,
        include_truncated: bool = True,
        meta_cols: tuple = ("tarball", "pt_idx", "missing", "truncated", "seq_len"),
        array_columns: set | None = None,
    ):
        self.window_length = window_length
        self.table = NpyTable(table_dir)

        if array_columns is None:
            self.array_columns = [c for c in self.table.columns if c not in meta_cols]
        else:
            self.array_columns = array_columns

        seq_len, missing, truncated = self.table[["seq_len", "missing", "truncated"]]

        self._index = []
        for i, (L, miss, trunc) in enumerate(zip(seq_len, missing, truncated)):
            if not include_missing_features and miss:
                continue
            if not include_truncated and trunc:
                continue
            for start in range(0, L, window_length):
                if start + window_length <= L:
                    self._index.append((i, start))

        print(f"{len(self._index)} samples qualified out of {len(seq_len)} total videos")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        row, start = self._index[idx]
        column_arrays = self.table.get(self.array_columns, rows=[row])
        return {
            col: torch.from_numpy(arr_list[0][start: start + self.window_length])
            for col, arr_list in zip(self.array_columns, column_arrays)
        }


def collate_fn(batch, batch_columns: list):
    stacked = {k: torch.stack([item[k] for item in batch]) for k in batch[0]}
    # TODO: fix hack, buttons should be preprocessed as float
    stacked = {
        k: t.bfloat16() if (t.dtype == torch.float32 or k == "buttons") else t
        for k, t in stacked.items()
    }
    return [stacked[col] for col in batch_columns]


def get_loader(batch_size, dataset_path, window_length, batch_columns):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    dataset = TekkenLatentDataset(root_dir=root_dir, window_length=window_length)

    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, batch_columns=batch_columns),
        num_workers=2,
        drop_last=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,   # workers stay alive; fewer re-forks
        prefetch_factor=2          # default is 2; keep it explicit
    )
