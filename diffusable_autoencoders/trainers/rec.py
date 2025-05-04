"""
Trainer for reconstruction only
"""

import torch
from ema_pytorch import EMA
import wandb
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .base import BaseTrainer

from ..utils import freeze, Timer
from ..schedulers import get_scheduler_cls
from ..models import get_model_cls
from ..nn.lpips import VGGLPIPS
from ..data import get_loader
from ..utils.logging import LogHelper, to_wandb

def latent_reg_loss(z):
    # z is [b,c,h,w]
    return 0.5 * torch.sum(z.pow(2), dim=[1, 2, 3]).mean()

class RecTrainer(BaseTrainer):
    """
    Trainer for reconstruction only objective.
    Only does L2 + LPIPS

    :param train_cfg: Configuration for training
    :param logging_cfg: Configuration for logging
    :param model_cfg: Configuration for model
    :param global_rank: Rank across all devices.
    :param local_rank: Rank for current device on this process.
    :param world_size: Overall number of devices
    """
    def __init__(self,*args,**kwargs):  
        super().__init__(*args,**kwargs)

        model_id = self.model_cfg.model_id
        self.model = get_model_cls(model_id)(self.model_cfg)

        self.ema = None
        self.opt = None
        self.scheduler = None
        self.scaler = None

        self.total_step_counter = 0

    def save(self):
        save_dict = {
            'model' : self.model.state_dict(),
            'ema' : self.ema.state_dict(),
            'opt' : self.opt.state_dict(),
            'scheduler' : self.scheduler.state_dict(),
            'scaler' : self.scaler.state_dict(),
            'steps': self.total_step_counter
        }
        super().save(save_dict)
    
    def load(self):
        if self.train_cfg.resume_ckpt is not None:
            save_dict = super().load(self.train_cfg.resume_ckpt)
        else:
            return
        
        self.model.load_state_dict(save_dict['model'])
        self.ema.load_state_dict(save_dict['ema'])
        self.opt.load_state_dict(save_dict['opt'])
        self.scheduler.load_state_dict(save_dict['scheduler'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.total_step_counter = save_dict['steps']

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Prepare model, lpips, ema
        self.model = self.model.cuda().train()
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])

        lpips = VGGLPIPS().cuda().eval()
        freeze(lpips)

        self.ema = EMA(
            self.model,
            beta = 0.9999,
            update_after_step = 0,
            update_every = 1
        )

        # Set up optimizer and scheduler
        self.opt = getattr(torch.optim, self.train_cfg.opt)(self.model.parameters(), **self.train_cfg.opt_kwargs)
        if self.train_cfg.scheduler is not None:
            self.scheduler = get_scheduler_cls(self.train_cfg.scheduler)(self.opt, **self.train_cfg.scheduler_kwargs)

        # Grad accum setup and scaler
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size
        accum_steps = max(1, accum_steps)
        self.scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast('cuda',torch.bfloat16)
        reg_weight =  self.train_cfg.loss_weights.get('latent_reg', 0.0)

        # Timer reset
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.get_module(), log = 'all')
        
        # Dataset setup
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size)

        local_step = 0
        for _ in range(self.train_cfg.epochs):
            for batch in loader:
                total_loss = 0.
                batch = batch.cuda().bfloat16()

                with ctx:
                    batch_rec, z = self.model(batch)

                if reg_weight > 0:
                    reg_loss = latent_reg_loss(z) / accum_steps
                    total_loss += reg_loss * reg_weight
                
                mse_loss = F.mse_loss(batch_rec, batch) / accum_steps
                total_loss += mse_loss

                lpips_loss = lpips(batch_rec, batch) / accum_steps
                total_loss += lpips_loss

                self.scaler.scale(total_loss).backward()

                metrics.log_dict({
                    'reg_loss' : reg_loss,
                    'mse_loss' : mse_loss,
                    'lpips_loss' : lpips_loss
                })

                local_step += 1
                if local_step % accum_steps == 0:
                    # Updates
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.scaler.step(self.opt)
                    self.opt.zero_grad()

                    self.scaler.update()

                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.ema.update()

                    # Do logging stuff with sampling stuff in the middle
                    wandb_dict = metrics.pop()
                    wandb_dict['time'] = timer.hit()
                    timer.reset()

                    if self.total_step_counter % self.train_cfg.sample_interval == 0:
                        wandb_dict['samples'] = to_wandb(
                            batch.detach().contiguous().bfloat16(),
                            batch_rec.detach().contiguous().bfloat16(),
                            gather = False
                        )
                    if self.rank == 0:
                        wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()
                        
                    self.barrier()
                    

                    




        




        





