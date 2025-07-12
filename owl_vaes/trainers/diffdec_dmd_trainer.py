import torch
from torch import nn
import torch.nn.functional as F
import einops as eo

import wandb
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusers import AutoencoderDC
from copy import deepcopy

from ..data import get_loader
from ..models import get_model_cls
from ..muon import init_muon
from ..schedulers import get_scheduler_cls
from ..utils import Timer, freeze, unfreeze, versatile_load
from ..utils.logging import LogHelper, to_wandb
from .base import BaseTrainer
from ..configs import Config
from ..sampling import flow_sample
from ..sampling.schedulers import get_sd3_euler

class DiffDMDTrainer(BaseTrainer):
    """
    Trainer for distilling a diffusion decoder.
    - Teacher VAE: provides latents
    - DCAE: provides proxy latents (we diffuse to make these)
    - Teacher diffusion transformer: we are training a student to match this
    - Student: same architecture as teacher but with fewer layers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- Teacher VAE ---
        teacher_vae_ckpt_path = self.train_cfg.teacher_vae_ckpt
        teacher_vae_cfg_path = self.train_cfg.teacher_vae_cfg
        teacher_vae_ckpt = versatile_load(teacher_vae_ckpt_path)
        teacher_vae_cfg = Config.from_yaml(teacher_vae_cfg_path).model
        teacher_vae = get_model_cls(teacher_vae_cfg.model_id)(teacher_vae_cfg)

        try:
            teacher_vae.load_state_dict(teacher_vae_ckpt)
        except Exception as e:
            teacher_vae.encoder.load_state_dict(teacher_vae_ckpt)

        self.encoder = teacher_vae.encoder.to(self.device).bfloat16().eval()

        # --- DCAE ---
        path = "mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers"
        self.dcae = AutoencoderDC.from_pretrained(path).bfloat16().cuda().eval()
        self.dcae = self.dcae.to(self.device).bfloat16().eval()

        # --- Teacher Diffusion Transformer ---
        teacher_diff_ckpt_path = self.train_cfg.teacher_diff_ckpt
        teacher_diff_cfg_path = self.train_cfg.teacher_diff_cfg
        teacher_diff_ckpt = versatile_load(teacher_diff_ckpt_path)
        teacher_diff_cfg = Config.from_yaml(teacher_diff_cfg_path).model
        teacher_diff = get_model_cls(teacher_diff_cfg.model_id)(teacher_diff_cfg)
        teacher_diff.load_state_dict(teacher_diff_ckpt)
        self.score_real = teacher_diff.to(self.device).bfloat16().eval()

        # --- Student Diffusion Transformer ---
        student_cfg = self.model_cfg
        self.student = get_model_cls(student_cfg.model_id)(student_cfg).core
        student_diff_ckpt_path = self.train_cfg.student_diff_ckpt
        student_diff_ckpt = versatile_load(student_diff_ckpt_path)
        self.student.load_state_dict(student_diff_ckpt)
        self.student = self.student.to(self.device).train()

        self.score_fake = get_model_cls(student_cfg.model_id)(student_cfg)
        self.score_fake.core.load_state_dict(student_diff_ckpt)
        self.score_fake = self.score_fake.to(self.device).train()

        if self.rank == 0:
            model_params = sum(p.numel() for p in self.student.parameters())
            print(f"Student model parameters: {model_params:,}")

        self.opt = None
        self.scheduler = None
        self.scaler = None
        self.total_step_counter = 0
        self.ema = None
        self.score_fake_opt = None
        self.s_fake_scaler = None

    def load_teacher_into_student(self, student_model, teacher_state_dict, teacher_cfg, student_cfg):
        """
        Copy layers from teacher to student with even spacing, so that
        first and last layers are always copied, and intermediate layers
        are mapped as evenly as possible.
        """
        import math

        # Get number of layers in teacher and student
        n_teacher = getattr(teacher_cfg, "n_layers", None)
        n_student = getattr(student_cfg, "n_layers", None)
        if n_teacher is None or n_student is None:
            raise ValueError("Both teacher_cfg and student_cfg must have n_layers attribute.")

        # Find teacher layer indices to map to student layers
        # Always include first and last
        if n_student == 1:
            teacher_indices = [0]
        else:
            teacher_indices = [
                round(i * (n_teacher - 1) / (n_student - 1))
                for i in range(n_student)
            ]

        # Now, build a new state_dict for the student, mapping teacher layers to student layers
        # Assume layers are in a module list, e.g. "blocks.blocks.{i}."
        # We'll copy all parameters for each mapped layer

        # Find the prefix for the blocks in both teacher and student
        # We'll assume "blocks.blocks.{i}." is the pattern (as in DiT)
        # If not, this may need to be adapted

        new_state_dict = {}
        for student_idx, teacher_idx in enumerate(teacher_indices):
            # For each key in teacher_state_dict, if it matches the teacher layer, map to student layer
            teacher_layer_prefix = f"blocks.blocks.{teacher_idx}."
            student_layer_prefix = f"blocks.blocks.{student_idx}."
            for k, v in teacher_state_dict.items():
                if k.startswith(teacher_layer_prefix):
                    new_k = student_layer_prefix + k[len(teacher_layer_prefix):]
                    new_state_dict[new_k] = v

        # Copy all other parameters (not in blocks.blocks) directly if they exist in student
        for k, v in teacher_state_dict.items():
            if not k.startswith("blocks.blocks."):
                new_state_dict[k] = v

        # Now, load into student
        student_model.load_state_dict(new_state_dict, strict=False)

    def get_ema_core(self):
        if self.world_size > 1:
            return self.ema.ema_model.module
        return self.ema.ema_model

    def save(self):
        save_dict = {
            'model': self.student.state_dict(),
            'score_fake': self.score_fake.state_dict(),
            'score_fake_opt': self.score_fake_opt.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
            'scaler': self.scaler.state_dict(),
            's_fake_scaler': self.s_fake_scaler.state_dict(),
            'steps': self.total_step_counter
        }
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        super().save(save_dict)

    def load(self):
        resume_ckpt = getattr(self.train_cfg, 'resume_ckpt', None)
        if resume_ckpt is None:
            return
        save_dict = super().load(resume_ckpt)
        self.student.load_state_dict(save_dict['model'], strict=False)
        self.score_fake.load_state_dict(save_dict['score_fake'])
        self.score_fake_opt.load_state_dict(save_dict['score_fake_opt'])
        self.ema.load_state_dict(save_dict['ema'])
        self.opt.load_state_dict(save_dict['opt'])
        if self.scheduler is not None and 'scheduler' in save_dict:
            self.scheduler.load_state_dict(save_dict['scheduler'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.s_fake_scaler.load_state_dict(save_dict['s_fake_scaler'])
        self.total_step_counter = save_dict['steps']

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Prepare model, ema
        self.student = self.student.to(self.device).train()
        if self.world_size > 1:
            self.student = DDP(self.student)
            self.score_fake = DDP(self.score_fake)

        freeze(self.encoder)
        freeze(self.dcae)
        freeze(self.score_real)
        freeze(self.score_fake)
        self.encoder = torch.compile(self.encoder, dynamic=False, fullgraph=True)
        self.dcae.encoder = torch.compile(self.dcae.encoder, dynamic=False, fullgraph=True)

        self.ema = EMA(
            self.student,
            beta=0.99,
            update_after_step=0,
            update_every=1
        )

        # Don't use muon for DMD
        opt_cls = getattr(torch.optim, self.train_cfg.opt)
        self.opt = opt_cls(self.student.parameters(), **self.train_cfg.opt_kwargs)
        self.score_fake_opt = opt_cls(self.score_fake.parameters(), **self.train_cfg.d_opt_kwargs)

        self.scheduler = None
        if self.train_cfg.scheduler is not None:
            self.scheduler = get_scheduler_cls(self.train_cfg.scheduler)(self.opt, **self.train_cfg.scheduler_kwargs)

        # Grad accum setup and scalers
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)

        self.scaler = torch.amp.GradScaler()
        self.s_fake_scaler = torch.amp.GradScaler()
        self.ctx = torch.amp.autocast(self.device, torch.bfloat16)

        # Dataset setup
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)
        loader = iter(loader)

        # Timer and logging
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.get_ema_core(), log='all')

        self.load()

        local_step = 0
        input_shape = (
            self.model_cfg.channels,
            self.model_cfg.sample_size[0],
            self.model_cfg.sample_size[1],
        )
        def get_dummy(z):
            return torch.randn(z.shape[0], *input_shape, device=z.device, dtype=z.dtype)
        
        def sample_from_gen(z):
            ts = torch.full((z.shape[0],), 1.0, device=z.device, dtype=z.dtype)
            x = get_dummy(z)
            preds = self.student(x, z, ts)
            return x - preds

        def get_dmd_loss(student_gen, z):
            s_real = self.score_real.core
            s_fake = self.score_fake.core if self.world_size == 1 else self.score_fake.module.core

            with torch.no_grad():
                b,c,h,w = student_gen.shape
                ts = torch.rand(b, device=z.device, dtype=z.dtype) * 0.96 + 0.02
                ts_exp = ts[:,None,None,None]

                eps = torch.randn_like(student_gen)
                lerpd = student_gen * (1. - ts_exp) + eps * ts_exp

                real_score = s_real(lerpd, z, ts)
                fake_score = s_fake(lerpd, z, ts)

                real_pred = lerpd - ts_exp * real_score
                fake_pred = lerpd - ts_exp * fake_score

                normalizer = torch.abs(student_gen - real_pred).mean(dim=[1,2,3], keepdim=True).detach()
                grad = (fake_pred - real_pred)
                grad = grad / (normalizer + 1.0e-6)
                grad = torch.nan_to_num(grad, 0.0)
            
            loss = 0.5 * F.mse_loss(student_gen.double(), (student_gen.double() - grad.double()).detach(), reduction='mean')
            
            return loss
        
        def optimizer_step(model, scaler, optimizer):
            # Assumes loss.backward() was already called
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
        
        while True:
            freeze(self.student)
            unfreeze(self.score_fake)

            for _ in range(self.train_cfg.update_ratio):
                for _ in range(accum_steps):
                    batch = next(loader)
                    batch = batch.to(self.device).bfloat16()
                    with self.ctx:
                        with torch.no_grad():
                            teacher_z = self.encoder(batch) / self.train_cfg.latent_scale
                            student_gen = sample_from_gen(teacher_z)
                        s_fake_loss = self.score_fake(student_gen, teacher_z) / accum_steps
                        metrics.log('s_fake_loss', s_fake_loss)
                        self.s_fake_scaler.scale(s_fake_loss).backward()
            
                optimizer_step(self.score_fake, self.s_fake_scaler, self.score_fake_opt)
            
            unfreeze(self.student)
            freeze(self.score_fake)

            for _ in range(accum_steps):
                batch = next(loader)
                batch = batch.to(self.device).bfloat16()
                with torch.no_grad():
                    teacher_z = self.encoder(batch) / self.train_cfg.latent_scale
                
                with self.ctx: 
                    student_gen = sample_from_gen(teacher_z)
                    dmd_loss = get_dmd_loss(student_gen, teacher_z) / accum_steps
                    metrics.log('dmd_loss', dmd_loss)
                    self.scaler.scale(dmd_loss).backward()
                
            optimizer_step(self.student, self.scaler, self.opt)
            
            if self.scheduler is not None:
                self.scheduler.step()

            self.ema.update()

            # Logging and sampling
            with torch.no_grad():
                wandb_dict = metrics.pop()
                wandb_dict['time'] = timer.hit()
                wandb_dict['lr'] = self.opt.param_groups[0]['lr']
                timer.reset()

                if self.total_step_counter % self.train_cfg.sample_interval == 0:
                    with self.ctx:
                        ema_rec = flow_sample(self.get_ema_core(), get_dummy(teacher_z), teacher_z, 1, self.dcae.decoder, scaling_factor=self.train_cfg.ldm_scale)
                    ema_rec = F.interpolate(ema_rec, size=batch.shape[2:], mode='bilinear', align_corners=False)
                    wandb_dict['samples'] = to_wandb(
                        batch[:,:3].detach().contiguous().bfloat16(),
                        ema_rec[:,:3].detach().contiguous().bfloat16(),
                        gather=False
                    )

                if self.rank == 0:
                    wandb.log(wandb_dict)

            self.total_step_counter += 1
            if self.total_step_counter % self.train_cfg.save_interval == 0:
                if self.rank == 0:
                    self.save()

            if self.world_size > 1:
                self.barrier()