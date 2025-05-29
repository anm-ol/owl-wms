"""
Audio Reconstruction Trainer for Audio AutoEncoder - Optimized with torch.compile
"""

import einops
import numpy as np
import torch
import torch.nn.functional as F
from ema_pytorch import EMA
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import wandb

from ..data import get_loader
from ..models import get_model_cls
from ..muon import init_muon
from ..schedulers import get_scheduler_cls
from ..utils import Timer
from ..utils.get_device import DeviceManager
from ..utils.logging import LogHelper
from .base import BaseTrainer

device = DeviceManager.get_device()


@torch.compile(mode="max-autotune-no-cudagraphs")
def stft_loss(
    x_rec: Tensor,
    x_target: Tensor,
    n_fft_list: list[int] = [1024, 2048, 512],
    hop_length_ratio: float = 0.25,
) -> Tensor:
    """
    Multi-scale STFT loss for audio reconstruction

    Args:
        x_rec: Reconstructed audio (B, C, T)
        x_target: Target audio (B, C, T)
        n_fft_list: List of FFT sizes for multi-scale analysis
        hop_length_ratio: Hop length of the window as ratio of n_fft

    Returns:
        Combined STFT loss
    """
    total_loss = 0.0

    x_rec = x_rec.to(torch.float16)
    x_target = x_target.to(torch.float16)

    for n_fft in n_fft_list:
        hop_length = int(n_fft * hop_length_ratio)

        stft_rec = torch.stft(
            x_rec.view(-1, x_rec.size(-1)),
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
        )
        stft_target = torch.stft(
            x_target.view(-1, x_target.size(-1)),
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
        )

        # Magnitude loss
        mag_rec = torch.abs(stft_rec)
        mag_target = torch.abs(stft_target)
        mag_loss = F.l1_loss(mag_rec, mag_target)

        # Phase loss: MSE on real and imaginary parts (GPU-friendly)
        real_loss = F.mse_loss(stft_rec.real, stft_target.real)
        imag_loss = F.mse_loss(stft_rec.imag, stft_target.imag)
        phase_loss = real_loss + imag_loss

        total_loss = total_loss + mag_loss + phase_loss

    return total_loss / len(n_fft_list)


@torch.compile(mode="max-autotune")
def lr_to_ms(audio: Tensor) -> Tensor:
    """
    Convert Left-Right (L/R) stereo to Mid-Side (M/S)

    Args:
        audio: Input audio tensor (B, 2, T) where channel 0=L, channel 1=R

    Returns:
        Audio in M/S format (B, 2, T) where channel 0=M, channel 1=S
    """
    if audio.size(1) != 2:
        return audio

    left = audio[:, 0:1, :].clone()  # (B, 1, T)
    right = audio[:, 1:2, :].clone()  # (B, 1, T)

    mid = (
        left + right
    ) * 0.5  # Mid channel - use multiplication for better compilation
    side = (left - right) * 0.5  # Side channel

    return torch.cat([mid, side], dim=1)


@torch.compile(mode="max-autotune")
def latent_reg_loss(z: Tensor, target_var: float = 0.1) -> Tensor:
    """
    Latent regularization loss - Compiled for performance.
    """
    # z is [b,c,h,w]
    # KL divergence between N(z, 0.1) and N(0,1)
    mu = z
    log_target_var = 2 * torch.log(
        torch.tensor(target_var, device=z.device, dtype=z.dtype)
    )  # log(0.1^2)

    kl = -0.5 * (1 + log_target_var - mu.pow(2) - log_target_var.exp())
    kl = einops.reduce(kl, "b ... -> b", reduction="sum").mean()

    return kl


@torch.compile(mode="max-autotune-no-cudagraphs")
def compute_ms_loss(batch_rec: Tensor, batch: Tensor) -> Tensor:
    """
    Compute M/S component loss - Compiled for performance.
    """
    batch_ms = lr_to_ms(batch)
    batch_rec_ms = lr_to_ms(batch_rec)
    return F.mse_loss(batch_rec_ms, batch_ms)


def log_audio_to_wandb(
    original: Tensor,
    reconstructed: Tensor,
    sample_rate: int = 44100,
    max_samples: int = 4,
) -> dict[str, wandb.Audio]:
    """
    Log audio samples to Weights & Biases.

    Args:
        original: Original audio tensor (B, C, T)
        reconstructed: Reconstructed audio tensor (B, C, T)
        sample_rate: Audio sample rate
        max_samples: Maximum number of samples to log

    Returns:
        Dictionary for wandb logging
    """
    batch_size = min(original.size(0), max_samples)
    audio_logs = {}

    for i in range(batch_size):
        # Convert to numpy and ensure correct shape for wandb
        orig_audio = original[i].detach().cpu().numpy()  # (C, T)
        rec_audio = reconstructed[i].detach().cpu().numpy()  # (C, T)

        # For stereo audio, mix down to mono for logging
        if orig_audio.shape[0] == 2:
            orig_mono = np.mean(orig_audio, axis=0)
            rec_mono = np.mean(rec_audio, axis=0)
        else:
            orig_mono = orig_audio[0]
            rec_mono = rec_audio[0]

        # Ensure audio is in correct range [-1, 1]
        orig_mono = np.clip(orig_mono, -1.0, 1.0)
        rec_mono = np.clip(rec_mono, -1.0, 1.0)

        audio_logs[f"audio_original_{i}"] = wandb.Audio(
            orig_mono, sample_rate=sample_rate
        )
        audio_logs[f"audio_reconstructed_{i}"] = wandb.Audio(
            rec_mono, sample_rate=sample_rate
        )

    return audio_logs


class AudioRecTrainer(BaseTrainer):
    """
    Trainer for audio reconstruction with specialized audio losses

    Includes:
    - Reconstruction loss (MSE)
    - STFT loss (multi-scale spectral loss) - COMPILED
    - TODO: HuBERT perceptual loss)
    - KL divergence regularization (down-weighted by 1e-4) - COMPILED
    - L/R vs M/S component weighting (L/R down-weighted by 0.5) - COMPILED
    - Audio logging to W&B
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_id = self.model_cfg.model_id
        self.model: nn.Module = get_model_cls(model_id)(self.model_cfg).to(device)

        if self.rank == 0:
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"Total parameters: {param_count:,}")

        self.ema = None
        self.opt = None
        self.scheduler = None
        self.scaler = None

        self.total_step_counter = 0

    def _compile_model(self, model: nn.Module) -> nn.Module:
        print("Compiling model...")
        return torch.compile(model, mode="max-autotune-no-cudagraphs", dynamic=True)  # type: ignore

    def save(self):
        # Save the original model state, not the compiled one
        save_dict = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "steps": self.total_step_counter,
        }
        super().save(save_dict)

    def load(self):
        if self.train_cfg.resume_ckpt is not None:
            save_dict = super().load(self.train_cfg.resume_ckpt)
        else:
            return

        self.model.load_state_dict(save_dict["model"])
        self.ema.load_state_dict(save_dict["ema"])
        self.opt.load_state_dict(save_dict["opt"])
        self.scheduler.load_state_dict(save_dict["scheduler"])
        self.scaler.load_state_dict(save_dict["scaler"])
        self.total_step_counter = save_dict["steps"]

    def train(self):
        # Loss weights
        recon_weight = self.train_cfg.loss_weights.get("recon", 1.0)
        stft_weight = self.train_cfg.loss_weights.get("stft", 1.0)
        hubert_weight = self.train_cfg.loss_weights.get("hubert", 0.0)  # Placeholder
        kl_weight = self.train_cfg.loss_weights.get("kl", 1e-4)
        lr_ms_ratio = self.train_cfg.loss_weights.get(
            "lr_ms_ratio", 0.5
        )  # L/R weight relative to M/S

        # Audio-specific config
        sample_rate = getattr(self.train_cfg, "sample_rate", 44100)
        n_fft_list = getattr(self.train_cfg, "n_fft_list", [1024, 2048, 512])

        # Prepare model and EMA
        self.model = self.model.to(device).train()

        if self.world_size > 1:
            self.model = DDP(self.model)

        # Compile model after DDP setup
        self.model = self._compile_model(self.model)

        self.ema = EMA(self.model, beta=0.9999, update_after_step=0, update_every=1)

        # Set up optimizer and scheduler
        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(
                self.model,
                rank=self.rank,
                world_size=self.world_size,
                **self.train_cfg.opt_kwargs,
            )
        else:
            self.opt = getattr(torch.optim, self.train_cfg.opt)(
                self.model.parameters(), **self.train_cfg.opt_kwargs
            )

        if self.train_cfg.scheduler is not None:
            self.scheduler = get_scheduler_cls(self.train_cfg.scheduler)(
                self.opt, **self.train_cfg.scheduler_kwargs
            )

        # Grad accum setup and scaler
        accum_steps = (
            self.train_cfg.target_batch_size
            // self.train_cfg.batch_size
            // self.world_size
        )
        accum_steps = max(1, accum_steps)
        self.scaler = torch.GradScaler()
        ctx = torch.autocast(device, torch.bfloat16)

        # Timer and metrics setup
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.get_module(), log="all")

        # Dataset setup
        loader = get_loader(
            self.train_cfg.data_id, self.train_cfg.batch_size, self.train_cfg.filepath
        )

        local_step = 0
        for epoch_idx in range(self.train_cfg.epochs):
            for batch in tqdm(
                loader, desc=f"Epoch {epoch_idx + 1}/{self.train_cfg.epochs}"
            ):
                total_loss = torch.tensor(0.0, device=device)
                batch = batch[0].to(device, dtype=torch.bfloat16)

                if batch.is_cuda:
                    torch.compiler.cudagraph_mark_step_begin()

                with ctx:
                    # Forward pass: (reconstructed, latent) - Use compiled model
                    batch_rec, z = self.model(batch)
                    batch_rec, z = batch_rec.to(device), z.to(device)

                # Reconstruction loss
                if recon_weight > 0:
                    recon_loss = F.mse_loss(batch_rec, batch) / accum_steps
                    total_loss = total_loss + recon_loss * recon_weight
                    metrics.log("recon_loss", recon_loss)

                # STFT loss - Now compiled!
                if stft_weight > 0:
                    with ctx:
                        stft_l = (
                            stft_loss(batch_rec, batch, n_fft_list=n_fft_list)
                            / accum_steps
                        )
                    total_loss = total_loss + stft_l * stft_weight
                    metrics.log("stft_loss", stft_l)

                # L/R vs M/S component weighting - Now compiled!
                if lr_ms_ratio < 1.0 and batch.size(1) == 2:  # Only for stereo audio
                    with ctx:
                        ms_loss = compute_ms_loss(batch_rec, batch) / accum_steps

                    # Weight the M/S loss more heavily than L/R
                    ms_weight = (
                        1.0 + lr_ms_ratio
                    ) / 2.0  # Balance between L/R and M/S weighting
                    total_loss = total_loss + ms_loss * ms_weight * recon_weight
                    metrics.log("ms_loss", ms_loss)

                # KL divergence regularization - Now compiled!
                if kl_weight > 0:
                    kl_loss = latent_reg_loss(z) / accum_steps
                    total_loss = total_loss + kl_loss * kl_weight
                    metrics.log("kl_loss", kl_loss)

                # HuBERT loss (placeholder)
                if hubert_weight > 0:
                    # TODO: Implement HuBERT perceptual loss
                    pass

                self.scaler.scale(total_loss).backward()

                # Log latent statistics
                with torch.no_grad():
                    metrics.log_dict({
                        "z_std": z.std(),
                        "z_mean": z.mean(),
                        "z_max": z.max(),
                        "z_min": z.min(),
                    })

                local_step += 1
                if local_step % accum_steps == 0:
                    # Updates
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    self.scaler.step(self.opt)
                    self.opt.zero_grad(set_to_none=True)

                    self.scaler.update()

                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.ema.update()

                    # Logging
                    with torch.no_grad():
                        wandb_dict = metrics.pop()
                        wandb_dict["time"] = timer.hit()
                        wandb_dict["lr"] = self.opt.param_groups[0]["lr"]
                        timer.reset()

                        # Audio logging
                        if (
                            self.total_step_counter % self.train_cfg.sample_interval
                            == 0
                        ):
                            audio_logs = log_audio_to_wandb(
                                batch.detach().contiguous().float(),
                                batch_rec.detach().contiguous().float(),
                                sample_rate=sample_rate,
                                max_samples=2,
                            )
                            wandb_dict.update(audio_logs)

                        if self.rank == 0:
                            wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()

                    self.barrier()
