"""
Trainer for finetuning decoder (or distilling it) with adversarial loss
"""

import einops as eo
import torch
import torch.nn.functional as F
import wandb
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data import get_loader
from ..models import get_model_cls
from ..discriminators import get_discriminator_cls
from ..muon import init_muon
from ..nn.lpips import get_lpips_cls
from ..schedulers import get_scheduler_cls
from ..utils import Timer, freeze, unfreeze, versatile_load
from ..utils.logging import LogHelper, log_audio_to_wandb
from .base import BaseTrainer
from ..configs import Config
from ..losses.audio import stft_loss, compute_ms_loss
from torch import Tensor 

def compute_losses(
    batch_rec: Tensor,
    batch: Tensor,
    recon_weight: float,
    stft_weight: float,
    lr_ms_ratio: float,
    accum_steps: int,
    n_fft_list: list[int],
):
    """
    Compiled loss computation function
    """
    total_loss = torch.tensor(0.0, device=batch.device)
    metrics = {}

    # Reconstruction loss
    if recon_weight > 0:
        recon_loss = F.mse_loss(batch_rec, batch) / accum_steps
        total_loss = total_loss + recon_loss * recon_weight
        metrics["recon_loss"] = recon_loss

    # STFT loss
    if stft_weight > 0:
        stft_l = stft_loss(batch_rec, batch, n_fft_list=n_fft_list) / accum_steps
        total_loss = total_loss + stft_l * stft_weight
        metrics["stft_loss"] = stft_l

    # M/S loss for stereo audio
    if lr_ms_ratio < 1.0 and batch.size(1) == 2:
        ms_loss = compute_ms_loss(batch_rec, batch) / accum_steps
        ms_weight = (1.0 + lr_ms_ratio) / 2.0
        total_loss = total_loss + ms_loss * ms_weight * recon_weight
        metrics["ms_loss"] = ms_loss

    return total_loss, metrics

class AudDecTuneTrainer(BaseTrainer):
    """
    Trainer for only the decoder, with frozen encoder.

    :param train_cfg: Configuration for training
    :param logging_cfg: Configuration for logging
    :param model_cfg: Configuration for model
    :param global_rank: Rank across all devices.
    :param local_rank: Rank for current device on this process.
    :param world_size: Overall number of devices
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        teacher_ckpt_path = self.train_cfg.teacher_ckpt
        teacher_cfg_path = self.train_cfg.teacher_cfg

        teacher_ckpt = versatile_load(teacher_ckpt_path)
        teacher_cfg = Config.from_yaml(teacher_cfg_path).model

        teacher = get_model_cls(teacher_cfg.model_id)(teacher_cfg)
        teacher.load_state_dict(teacher_ckpt)

        self.encoder = teacher.encoder

        if self.train_cfg.use_teacher_decoder:
            self.model = teacher.decoder
            self.model.decoder_only = True
        else:
            del teacher.decoder
            model_id = self.model_cfg.model_id
            model = get_model_cls(model_id)(self.model_cfg)
            del model.encoder
            self.model = model.decoder

        disc_cfg = self.model_cfg.discriminator
        self.discriminator = get_discriminator_cls(disc_cfg.model_id)(disc_cfg)

        if self.rank == 0:
            model_params = sum(p.numel() for p in self.model.parameters())
            disc_params = sum(p.numel() for p in self.discriminator.parameters())
            print(f"Model parameters: {model_params:,}")
            print(f"Discriminator parameters: {disc_params:,}")

        self.ema = None
        self.opt = None
        self.d_opt = None
        self.scheduler = None
        self.scaler = None

        self.total_step_counter = 0

    def save(self):
        save_dict = {
            'model' : self.model.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'ema' : self.ema.state_dict(),
            'opt' : self.opt.state_dict(),
            'scaler' : self.scaler.state_dict(),
            'steps': self.total_step_counter
        }
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        super().save(save_dict)

    def load(self):
        if not hasattr(self.train_cfg, 'resume_ckpt') or self.train_cfg.resume_ckpt is None:
            return
        
        save_dict = super().load(self.train_cfg.resume_ckpt)
        self.model.load_state_dict(save_dict['model'])
        self.discriminator.load_state_dict(save_dict['discriminator'])
        self.ema.load_state_dict(save_dict['ema'])
        self.opt.load_state_dict(save_dict['opt'])
        if self.scheduler is not None and 'scheduler' in save_dict:
            self.scheduler.load_state_dict(save_dict['scheduler'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.total_step_counter = save_dict['steps']

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Loss weights
        recon_weight = self.train_cfg.loss_weights.get("recon", 1.0)
        stft_weight = self.train_cfg.loss_weights.get("stft", 1.0)
        lr_ms_ratio = self.train_cfg.loss_weights.get(
            "lr_ms_ratio", 0.5
        )  # L/R weight relative to M/S
        gan_weight = self.train_cfg.loss_weights.get('gan', 0.1)
        feature_matching_weight = self.train_cfg.loss_weights.get('feature_matching', 5.0)

        # Audio-specific config
        sample_rate = getattr(self.train_cfg, "sample_rate", 44100)
        n_fft_list = getattr(self.train_cfg, "n_fft_list", [1024, 2048, 512])

        # Prepare model, lpips, ema
        self.model = self.model.to(self.device).train()
        if self.world_size > 1:
            self.model = DDP(self.model)
        
        self.discriminator = self.discriminator.to(self.device).train()
        if self.world_size > 1:
            self.discriminator = DDP(self.discriminator)
        freeze(self.discriminator)

        self.encoder = self.encoder.to(self.device).bfloat16()
        freeze(self.encoder)

        self.encoder = torch.compile(self.encoder, mode='max-autotune', fullgraph=True)

        self.ema = EMA(
            self.model,
            beta = 0.9999,
            update_after_step = 0,
            update_every = 1
        )

        # Set up optimizer and scheduler
        opt_cls = getattr(torch.optim, self.train_cfg.opt)
        self.opt = opt_cls(self.model.parameters(), **self.train_cfg.opt_kwargs)
        self.d_opt = opt_cls(self.discriminator.parameters(), **self.train_cfg.opt_kwargs)

        if self.train_cfg.scheduler is not None:
            self.scheduler = get_scheduler_cls(self.train_cfg.scheduler)(self.opt, **self.train_cfg.scheduler_kwargs)

        # Grad accum setup and scaler
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)

        self.scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast(self.device, torch.bfloat16)

        self.load()

        # Timer reset
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.get_module(), log = 'all')

        # Dataset setup
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size)

        def warmup_weight():
            if self.total_step_counter < self.train_cfg.delay_adv:
                return 0.0
            else:
                x = (self.total_step_counter - self.train_cfg.delay_adv) / self.train_cfg.warmup_adv
                x = max(0.0, min(1.0, x))
                # Cosine annealing from 0 to 1
                ramp = 0.5 * (1 - torch.cos(torch.tensor(x * torch.pi)).item())
                return ramp
                
        def warmup_gan_weight(): return warmup_weight() * gan_weight
        def warmup_fm_weight(): return warmup_weight() * feature_matching_weight

        # Hardcoded for encodec
        def d_loss(d, x_fake, x_real):
            fake_outs,_ = d(x_fake)
            real_outs,_ = d(x_real)

            disc_loss = 0.
            for (fake_out, real_out) in zip(fake_outs, real_outs):
                disc_loss += torch.relu(1 - real_out).mean() + torch.relu(1 + fake_out).mean() 
            
            return disc_loss / len(fake_outs)

        def fm_reduce(x, y):
            return abs(x - y).mean()

        def g_loss(d, x_fake, x_real):
            fake_outs, hs_fake = d(x_fake)
            _, hs_real = d(x_real)

            gan_loss = 0.
            fm_loss = 0.
            for (fake_out, h_fake, h_real) in zip(fake_outs, hs_fake, hs_real):
                gan_loss += -fake_out.mean()
                fm_loss += sum(map(fm_reduce, h_real, h_fake)) / len(h_real)
            
            return gan_loss / len(fake_outs), fm_loss / len(fake_outs)

        def rec_loss(batch_rec, batch):
            return compute_losses(
                batch_rec,
                batch,
                recon_weight=recon_weight,
                stft_weight=stft_weight,
                lr_ms_ratio=lr_ms_ratio,
                accum_steps=accum_steps,
                n_fft_list=n_fft_list,  
            )

        local_step = 0
        for _ in range(self.train_cfg.epochs):
            for batch in loader:
                total_loss = 0.
                batch = batch.to(self.device).bfloat16()

                with torch.no_grad():
                    teacher_z = self.encoder.sample(batch) / self.train_cfg.latent_scale
                with ctx:
                    batch_rec = self.model(teacher_z)

                # Discriminator training 
                unfreeze(self.discriminator)
                with ctx:
                    disc_loss = d_loss(self.discriminator, batch_rec.detach(), batch.detach()) / accum_steps
                    metrics.log('disc_loss', disc_loss)

                self.scaler.scale(disc_loss).backward()
                freeze(self.discriminator)

                with ctx:
                    total_loss, loss_metrics = rec_loss(batch_rec, batch)
                    metrics.log_dict(loss_metrics)

                    if self.total_step_counter > self.train_cfg.delay_adv:
                        gan_loss, fm_loss = g_loss(self.discriminator, batch_rec, batch.detach())
                        gan_loss = gan_loss / accum_steps
                        fm_loss = fm_loss / accum_steps
                        metrics.log('g_loss', gan_loss)
                        metrics.log('fm_loss', fm_loss)

                        total_loss += gan_weight * warmup_gan_weight()
                        total_loss += feature_matching_weight * warmup_fm_weight()
                
                self.scaler.scale(total_loss).backward()

                local_step += 1
                if local_step % accum_steps == 0:
                    # Updates
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.scaler.unscale_(self.d_opt)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)

                    self.scaler.step(self.opt)
                    self.opt.zero_grad(set_to_none=True)

                    self.scaler.step(self.d_opt)
                    self.d_opt.zero_grad(set_to_none=True)
                    
                    self.scaler.update()

                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.ema.update()

                    # Do logging stuff with sampling stuff in the middle
                    with torch.no_grad():
                        wandb_dict = metrics.pop()
                        wandb_dict['time'] = timer.hit()
                        wandb_dict['lr'] = self.opt.param_groups[0]['lr']
                        timer.reset()

                        # Audio logging
                        if self.total_step_counter % self.train_cfg.sample_interval == 0:
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
