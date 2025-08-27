from ema_pytorch import EMA
from pathlib import Path
import tqdm
import wandb
import gc
import re

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .base import BaseTrainer

from ..utils import freeze, Timer
from ..schedulers import get_scheduler_cls
from ..models import get_model_cls
from ..sampling import get_sampler_cls
from ..data import get_loader
from ..utils.logging import LogHelper, to_wandb_samples
from ..utils import batch_permute_to_length
from ..muon import init_muon
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn


@torch.no_grad()
def make_batched_wan_decode_fn(vae, batch_size: int = 2):
    cfg = vae.config
    C   = int(getattr(cfg, "z_dim", getattr(cfg, "latent_channels", 16)))
    sf  = float(getattr(cfg, "scaling_factor", 1.0))
    m   = getattr(cfg, "latents_mean", None)
    s   = getattr(cfg, "latents_std",  None)

    def to_c_first_3d(z: torch.Tensor) -> torch.Tensor:
        # [B,C,H,W] -> [B,C,1,H,W]
        if z.ndim == 4:
            z = z.unsqueeze(2)
        if z.ndim != 5:
            raise ValueError(f"Expected 4D/5D latents, got {tuple(z.shape)}")
        # accept [B,C,T,H,W] or [B,T,C,H,W]
        if z.shape[1] == C and z.shape[2] != C:
            return z
        if z.shape[2] == C and z.shape[1] != C:
            return z.permute(0, 2, 1, 3, 4).contiguous()
        if z.shape[1] == C and z.shape[2] == C:
            raise ValueError(f"Ambiguous: both dim1 and dim2 equal z_dim={C}; pass [B,C,T,H,W].")
        raise ValueError(f"Neither dim1 nor dim2 equals z_dim={C}; got {tuple(z.shape)}")

    def pad_to_4k_plus_1(x: torch.Tensor) -> tuple[torch.Tensor, int]:
        # Round UP to the nearest 4k+1 in time, replicating the last latent if needed.
        T = x.shape[2]
        target = ((T - 1 + 3) // 4) * 4 + 1
        if target == T:
            return x, T
        pad = target - T
        x_pad = torch.cat([x, x[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
        return x_pad, T

    def to_vae_space(z_model: torch.Tensor) -> torch.Tensor:
        # Canonical diffusers conversion:
        #   if mean/std present:  z * (1/std) / sf + mean
        #   else:                 z / sf
        if m is not None and s is not None:
            mean = torch.as_tensor(m, device=z_model.device, dtype=z_model.dtype).view(1, C, 1, 1, 1)
            std  = torch.as_tensor(s, device=z_model.device, dtype=z_model.dtype).view(1, C, 1, 1, 1)
            return (z_model / sf) * std + mean
        return z_model / sf

    def decode(z: torch.Tensor) -> torch.Tensor:
        x = to_c_first_3d(z)
        x, orig_T = pad_to_4k_plus_1(x)
        x = x.to(torch.float32)  # keep VAE fp32
        outs = []
        for z_chunk in x.split(batch_size, dim=0):
            z_in = to_vae_space(z_chunk)
            pix = vae.decode(z_in, return_dict=True).sample  # [b,3,T,H,W]
            outs.append(pix)
        y = torch.cat(outs, dim=0)  # [B,3,Tpix,H,W] where Tpix = 1 + 4*(Tpad-1)
        # Trim back to the frames implied by the original latent length:
        want_Tpix = 1 + 4 * (orig_T - 1)
        if y.shape[2] >= want_Tpix:
            y = y[:, :, :want_Tpix]
        return y.permute(0, 2, 1, 3, 4).contiguous()  # [B,Tpix,3,H,W]

    return decode


class RFTTrainer(BaseTrainer):
    """
    Trainer for rectified flow transformer

    :param train_cfg: Configuration for training
    :param logging_cfg: Configuration for logging
    :param model_cfg: Configuration for model
    :param global_rank: Rank across all devices.
    :param local_rank: Rank for current device on this process.
    :param world_size: Overall number of devices
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_id = self.model_cfg.model_id
        self.model = get_model_cls(model_id)(self.model_cfg).train()

        # Print model size
        if self.rank == 0:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model has {n_params:,} parameters")

        self.ema = None
        self.opt = None
        self.scheduler = None

        self.total_step_counter = 0

        from diffusers import AutoencoderKLWan
        self.wan_decoder = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",  # or a local path with the same structure
            subfolder="vae",
            torch_dtype=torch.float32,  # keep VAE weights in fp32 per upstream example
        ).cuda()
        freeze(self.wan_decoder)

        self.autocast_ctx = torch.amp.autocast('cuda', torch.bfloat16)

    @staticmethod
    def get_raw_model(model):
        return getattr(model, "module", model)

    def save(self):
        if self.rank != 0:
            return

        save_dict = {
            'model': self.get_raw_model(self.model).state_dict(),
            'ema': self.get_raw_model(self.ema).state_dict(),
            'opt': self.opt.state_dict(),
            'steps': self.total_step_counter
        }
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        super().save(save_dict)

    def load(self) -> None:
        """Build runtime objects and optionally restore a checkpoint."""
        # ----- model & helpers -----
        ckpt = getattr(self.train_cfg, "resume_ckpt", None)
        state = None
        if ckpt:
            state = super().load(ckpt)

            # Allow legacy checkpoints: strip module and _orig_mod
            pat = r'^(?:(?:_orig_mod\.|module\.)+)?([^.]+\.)?(?:(?:_orig_mod\.|module\.)+)?'
            state["model"] = {re.sub(pat, r'\1', k): v for k, v in state["model"].items()}
            state["ema_model"] = {
                k.replace("module.", "").replace("_orig_mod.", "").replace("ema_model.", ""): v
                for k, v in state["ema"].items() if k.startswith("ema_model.")
            }
            state["ema_model"] = {re.sub(pat, r'\1', k): v for k, v in state["ema_model"].items()}

            self.model.load_state_dict(state["model"], strict=True)
            self.total_step_counter = state.get("steps", 0)

        self.model = self.model.cuda()
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        else:
            self.model = self.model

        self.ema = EMA(self.model, beta=0.999, update_after_step=0, update_every=1)

        self.model = torch.compile(self.model)

        self.wan_decoder = self.wan_decoder.cuda().eval()
        self.decode_fn = make_batched_wan_decode_fn(self.wan_decoder, self.train_cfg.vae_batch_size)

        # ----- optimiser, scheduler -----
        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(self.model, rank=self.rank, world_size=self.world_size, **self.train_cfg.opt_kwargs)
        else:
            self.opt = getattr(torch.optim, self.train_cfg.opt)(self.model.parameters(), **self.train_cfg.opt_kwargs)

        if self.train_cfg.scheduler:
            sched_cls = get_scheduler_cls(self.train_cfg.scheduler)
            self.scheduler = sched_cls(self.opt, **self.train_cfg.scheduler_kwargs)

        # ----- optional checkpoint restore -----
        if ckpt:
            if self.world_size > 1:
                self.ema.ema_model.module.load_state_dict(state["ema_model"])
            else:
                self.ema.ema_model.load_state_dict(state["ema_model"])
            self.opt.load_state_dict(state["opt"])
            if self.scheduler and "scheduler" in state:
                self.scheduler.load_state_dict(state["scheduler"])

        self.ema.ema_model = torch.compile(self.ema.ema_model)

        del state


    @torch.no_grad()
    def update_buffer(self, name: str, value: torch.Tensor, value_ema: torch.Tensor | None = None):
        """Set the buffer `name` (e.g. 'core.transformer.foo') across ranks and EMA."""
        online = self.model.module if isinstance(self.model, DDP) else self.model
        buf_online = online.get_buffer(name)
        buf_ema = self.ema.ema_model.get_buffer(name)

        if self.rank == 0:
            buf_online.copy_(value.to(buf_online))
        if self.world_size > 1:
            dist.broadcast(buf_online, 0)

        buf_ema.copy_(buf_online)

    def train(self):
        torch.cuda.set_device(self.local_rank)
        print(f"Device used: rank={self.rank}")

        # Grad accum setup and scaler
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)

        self.load()

        # Timer reset
        timer = Timer()
        timer.reset()
        metrics = LogHelper()

        if self.rank == 0:
            wandb.watch(self.get_module(), log='all')

        # Dataset setup
        loader = get_loader(
            self.train_cfg.data_id,
            self.train_cfg.batch_size,
            **self.train_cfg.data_kwargs
        )
        print(loader)

        n_samples = (self.train_cfg.n_samples + self.world_size - 1) // self.world_size  # round up to next world_size
        sample_loader = get_loader(
            self.train_cfg.sample_data_id,
            n_samples,
            **self.train_cfg.sample_data_kwargs
        )
        sample_loader = iter(sample_loader)

        if self.train_cfg.data_id == "cod_s3_mixed":
            loader.dataset.sleep_until_queues_filled()
            self.barrier()

        self.sampler_only_return_generated = self.train_cfg.sampler_kwargs.pop("only_return_generated")
        sampler = get_sampler_cls(self.train_cfg.sampler_id)(**self.train_cfg.sampler_kwargs)

        local_step = 0
        for epoch in range(self.train_cfg.epochs):
            for batch in tqdm.tqdm(loader, total=len(loader), disable=self.rank != 0, desc=f"Epoch: {epoch}"):
                train_steps = local_step // accum_steps

                batch_cuda = [
                    item.cuda() if isinstance(item, torch.Tensor) else item
                    for item in batch
                ]
                loss = self.fwd_step(batch_cuda, train_steps)
                loss = loss / accum_steps
                loss.backward()

                metrics.log('loss', loss)

                local_step += 1
                if local_step % accum_steps == 0:

                    # Optimizer updates
                    if self.train_cfg.opt.lower() != "muon":
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)

                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.ema.update()

                    self.log_step(metrics, timer, sample_loader, sampler)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        self.save()

                    self.barrier()

    def fwd_step(self, batch, train_step):
        vid, mouse, btn, doc_id = batch
        with self.autocast_ctx:
            loss = self.model(vid, mouse, btn, doc_id)
        return loss

    @torch.no_grad()
    def log_step(self, metrics, timer, sample_loader, sampler):
        wandb_dict = metrics.pop()
        wandb_dict['time'] = timer.hit()
        timer.reset()

        # Sampling commented out for now
        if self.total_step_counter % self.train_cfg.sample_interval == 0:
            eval_wandb_dict = self.eval_step(sample_loader, sampler)
            gc.collect()
            torch.cuda.empty_cache()
            if self.rank == 0:
                wandb_dict.update(eval_wandb_dict)

        if self.rank == 0:
            wandb.log(wandb_dict)

    def _gather_concat_cpu(self, t: torch.Tensor, dim: int = 0):
        """Gather *t* from every rank onto rank 0 and return concatenated copy."""
        if t is None:
            return None
        if self.world_size == 1:
            return t.cpu()
        if self.rank == 0:
            parts = [t.cpu()]
            scratch = torch.empty_like(t)
            for src in range(1, self.world_size):
                dist.recv(scratch, src=src)
                parts.append(scratch.cpu())
            return torch.cat(parts, dim=dim)
        dist.send(t, dst=0)

    def eval_step(self, sample_loader, sampler):
        ema_model = self.get_module(ema=True).core

        # ---- Generate Samples ----
        eval_batch = [x.cuda() for x in next(sample_loader)]
        if len(eval_batch) == 3:
            vid, mouse, btn = eval_batch
            mouse, btn = batch_permute_to_length(mouse, btn, sampler.num_frames + vid.size(1))
        else:
            vid, = eval_batch
            mouse, btn = None, None

        with self.autocast_ctx:
            latent_vid = sampler(ema_model, vid, mouse, btn)

        if self.sampler_only_return_generated:
            latent_vid, mouse, btn = (x[:, vid.size(1):] if x is not None else None for x in (latent_vid, mouse, btn))

        video_out = self.decode_fn(latent_vid.float()) if self.decode_fn is not None else None

        # ---- Optionally Save Latent Artifacts ----
        if getattr(self.train_cfg, "eval_sample_dir", None):
            latent_vid = self._gather_concat_cpu(latent_vid)
            if self.rank == 0:
                eval_dir = Path(self.train_cfg.eval_sample_dir)
                eval_dir.mkdir(parents=True, exist_ok=True)
                torch.save(latent_vid, eval_dir / f"vid.{self.total_step_counter}.pt")

        # ---- Generate Media Artifacts ----
        video_out, mouse, btn = map(self._gather_concat_cpu, (video_out, mouse, btn))
        eval_wandb_dict = to_wandb_samples(video_out, mouse, btn, fps=24) if self.rank == 0 else None
        dist.barrier()

        return eval_wandb_dict
