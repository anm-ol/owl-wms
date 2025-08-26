from pathlib import Path
from functools import partial
from diffusers import FlowMatchEulerDiscreteScheduler

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist


class AutoEpochDistributedSampler(DistributedSampler):
    """Shuffle every epoch automatically."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_epoch = 0

    def __iter__(self):
        self.set_epoch(self._auto_epoch)
        self._auto_epoch += 1
        return super().__iter__()


class WanPairDataset(Dataset):
    """
    Emits all K steps per run dir per epoch for interpolation training:
      returns dict with:
        'x_samples': [F, K, C, H, W]  # K from clean (t≈0) to noise (t≈1)
        'times':     [K]              # ascending in [0,1], aligned with K
    """

    sch = FlowMatchEulerDiscreteScheduler(shift=3, num_train_timesteps=1000)
    sch.set_timesteps(40)
    sigmas = sch.sigmas

    def __init__(self, root_dir: str):
        self.root = Path(root_dir)

        # 1) find run dirs by presence of the final step file
        self.run_dirs = sorted(
            p.parent
            for d in self.root.iterdir() if d.is_dir() and "." not in d.name
            for p in d.rglob("00000040_latents.pt")
            if all("." not in part for part in p.parent.relative_to(self.root).parts)
        )
        if not self.run_dirs:
            raise FileNotFoundError(f"No runs found under {root_dir}")

        # cache step indices per run (assume files are 000000xx_latents.pt)
        self._steps = {}
        for d in self.run_dirs:
            steps = sorted(int(fp.stem.split("_")[0]) for fp in d.glob("*_latents.pt"))
            if len(steps) >= 2:
                self._steps[d] = steps

    def __len__(self):
        # one item (all steps) per run per epoch
        return len(self.run_dirs)

    def load_item(self, run_dir: Path, steps_by_time: list[int]):
        x_samples = torch.stack([
            torch.load(
                run_dir / f"{s:08d}_latents.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,          # lazy, OS-backed reads
            )
            for s in steps_by_time
        ]).permute(2, 0, 1, 3, 4)
        sig = self.sigmas[steps_by_time].to(torch.float32)               # [K]
        times = (sig - sig.min()) / (sig.max() - sig.min() + 1e-8)       # [K] in [0,1]
        return x_samples, times

    def __getitem__(self, idx):
        run_dir = self.run_dirs[idx]
        steps_by_time = list(reversed(self._steps[run_dir]))
        x_samples, times = self.load_item(run_dir, steps_by_time)
        return {"x_samples": x_samples, "times": times}


def collate_fn(batch, batch_columns: list):
    stacked = {k: torch.stack([item[k] for item in batch]) for k in batch[0]}
    # commented out: need higher precision timesteps
    # stacked = {k: (t.bfloat16() if t.dtype == torch.float32 else t) for k, t in stacked.items()}
    return {k: v for k, v in stacked.items() if k in batch_columns}


def get_pair_loader(
    batch_size: int,
    dataset_path: str,
    batch_columns: list,
):
    """returns batches with keys ['x_samples', 'times']."""
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    ds = WanPairDataset(dataset_path)

    if world_size > 1:
        sampler = AutoEpochDistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        loader_kwargs = dict(sampler=sampler, shuffle=False)
    else:
        loader_kwargs = dict(shuffle=True)

    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, batch_columns=batch_columns),
        num_workers=4,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        **loader_kwargs,
    )


class WanGTWindowDataset(Dataset):
    """
    Returns a single tensor per item:
      x_gt: [W, C, H, W] where W is window_length (or full F if window_length=None)

    Each item is sampled from step-40 (fully denoised) of one run directory.
    """
    def __init__(self, root_dir: str, *, window_length: int | None = None, step_index: int = 40):
        self.root = Path(root_dir)
        self.window_length = window_length
        self.step_index = step_index

        # Find all run dirs that contain the target step file (e.g., 00000039_latents.pt)
        pattern = f"{step_index:08d}_latents.pt"
        self.run_dirs = sorted(
            p.parent
            for d in self.root.iterdir() if d.is_dir() and "." not in d.name
            for p in d.rglob(pattern)
            if all("." not in part for part in p.parent.relative_to(self.root).parts)
        )
        if not self.run_dirs:
            raise FileNotFoundError(f"No runs with {pattern} found under {root_dir}")

    @staticmethod
    def _load_step(run_dir: Path, step_idx: int) -> torch.Tensor:
        """Load 000000{step_idx}_latents.pt shaped [C,F,H,W], return [F,C,H,W]."""
        x_cf_hw = torch.load(run_dir / f"{step_idx:08d}_latents.pt", map_location="cpu")
        return x_cf_hw.permute(1, 0, 2, 3).contiguous()

    def __len__(self):
        # One sample per run per epoch (fresh random window each epoch if window_length is set)
        return len(self.run_dirs)

    def __getitem__(self, idx):
        run_dir = self.run_dirs[idx]
        x_fchw = self._load_step(run_dir, self.step_index)   # [F,C,H,W]
        prompt = (run_dir / "prompt.txt").read_text(encoding="utf-8").strip()
        return {"x": x_fchw[: self.window_length], "prompt": prompt}


def get_sample_loader(
    batch_size: int,
    dataset_path: str,
    batch_columns: "list[str]",
    window_length: int | None = None,
    *,
    step_index: int = 39,
):
    """
    Returns a DataLoader that yields only a single tensor per batch:
      batch: [B, W, C, H, W]  (or [B, F, C, H, W] if window_length=None)

    Notes:
      - Assumes all runs share the same F if window_length=None (typical for WAN with fixed num_frames).
      - Uses DDP sampler when initialized under torch.distributed.
    """

    # uses 39 latent from non-v2
    assert not step_index == 39
    assert not dataset_path.endswith("v2")

    ds = WanGTWindowDataset(
        dataset_path,
        window_length=window_length,
        step_index=step_index,
    )

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    if world_size > 1:
        sampler = AutoEpochDistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=True  # add drop_last=True if you want equal iters
        )
        loader_kwargs = dict(sampler=sampler, shuffle=False)
    else:
        loader_kwargs = dict(shuffle=True)

    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, batch_columns=batch_columns),
        num_workers=2,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        **loader_kwargs,
    )
