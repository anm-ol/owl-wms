from ema_pytorch import EMA
from pathlib import Path
from tqdm import tqdm
import wandb
import gc

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .base import BaseTrainer

from ..utils import freeze, Timer
from ..schedulers import get_scheduler_cls
from ..models import get_model_cls
from ..sampling import get_sampler_cls
from ..data import get_loader
from ..utils.logging import LogHelper, to_wandb_av
from ..utils import batch_permute_to_length
from ..muon import init_muon
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn

class TekkenRFTTrainer(BaseTrainer):
    """
    A specialized trainer for the TekkenRFT model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = get_model_cls(self.model_cfg.model_id)(self.model_cfg) # type: ignore

        if self.rank == 0:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model has {n_params:,} parameters")

        self.ema = None
        self.opt = None
        self.total_step_counter = 0

        # Load VAE decoder for sampling and visualization
        self.decoder = get_decoder_only(
            self.train_cfg.vae_id,
            self.train_cfg.vae_cfg_path,
            self.train_cfg.vae_ckpt_path
        )
        freeze(self.decoder)

    def save(self):
        save_dict = {
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
            'steps': self.total_step_counter
        }
        super().save(save_dict)

    def load(self):
        if hasattr(self.train_cfg, 'resume_ckpt') and self.train_cfg.resume_ckpt is not None:
            save_dict = super().load(self.train_cfg.resume_ckpt)
            self.model.load_state_dict(save_dict['model'])
            self.ema.load_state_dict(save_dict['ema'])
            self.opt.load_state_dict(save_dict['opt'])
            self.total_step_counter = save_dict.get('steps', 0)

    def train(self):
        torch.cuda.set_device(self.local_rank)

        self.model = self.model.cuda().train()
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        self.decoder = self.decoder.cuda().eval().bfloat16()
        decode_fn = make_batched_decode_fn(self.decoder, self.train_cfg.vae_batch_size)

        self.ema = EMA(self.model, beta=0.999, update_every=1)
        self.opt = torch.optim.AdamW(self.model.parameters(), **self.train_cfg.opt_kwargs)
        
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        ctx = torch.amp.autocast('cuda', torch.bfloat16)

        self.load()

        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.get_module(), log='all')

        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)
        print(f"Data loader created with {len(loader)} batches.")
        local_step = 0
        for epoch in range(self.train_cfg.epochs):
            for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{self.train_cfg.epochs}", disable=self.rank != 0):
                # print(f"Processing batch {local_step + 1}...")
                # The dataloader now returns (video_latents, action_ids)
                batch_vid, _, batch_actions = [t.cuda() for t in batch]
                # Scale latents according to VAE scaling factor
                action_ids = batch_actions[:, :, -1].int()
                # print(f"Batch actions dtype: {batch_actions.dtype}, shape: {batch_actions.shape}")
                batch_vid = batch_vid.bfloat16() / self.train_cfg.vae_scale
                
                with ctx:
                    loss = self.model(batch_vid, action_ids=action_ids)
                    loss = loss / accum_steps
                
                loss.backward()
                metrics.log('diffusion_loss', loss.item() * accum_steps)

                local_step += 1
                if local_step % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)
                    self.ema.update()

                    if self.rank == 0:
                        wandb_dict = metrics.pop()
                        wandb_dict['lr'] = self.opt.param_groups[0]['lr']
                        wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0 and self.rank == 0:
                        self.save()
                    
                    self.barrier()
