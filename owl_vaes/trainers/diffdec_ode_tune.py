import torch
from torch import nn
import torch.nn.functional as F
import einops as eo

import wandb
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusers import AutoencoderDC

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

class DiffDecODETrainer(BaseTrainer):
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
        del self.dcae.encoder
        self.dcae = self.dcae.to(self.device).bfloat16().eval()

        # --- Teacher Diffusion Transformer ---
        teacher_diff_ckpt_path = self.train_cfg.teacher_diff_ckpt
        teacher_diff_cfg_path = self.train_cfg.teacher_diff_cfg
        teacher_diff_ckpt = versatile_load(teacher_diff_ckpt_path)
        teacher_diff_cfg = Config.from_yaml(teacher_diff_cfg_path).model
        teacher_diff = get_model_cls(teacher_diff_cfg.model_id)(teacher_diff_cfg)
        teacher_diff.load_state_dict(teacher_diff_ckpt)
        self.teacher_diff = teacher_diff.core.to(self.device).bfloat16().eval()

        # --- Student Diffusion Transformer ---
        student_cfg = self.model_cfg
        self.student = get_model_cls(student_cfg.model_id)(student_cfg)
        self.load_teacher_into_student(self.student, teacher_diff_ckpt, teacher_diff_cfg, student_cfg)
        self.student = self.student.core.to(self.device).train()

        if self.rank == 0:
            model_params = sum(p.numel() for p in self.student.parameters())
            print(f"Student model parameters: {model_params:,}")

        self.opt = None
        self.scheduler = None
        self.scaler = None
        self.total_step_counter = 0
        self.ema = None

    def load_teacher_into_student(self, student_model, teacher_state_dict, teacher_cfg, student_cfg):
        """
        Improved weight-transfer from teacher → student.

        1. Handles 'module.' / 'core.' prefixes in the teacher checkpoint.
        2. Copies projection / embedding / final layers verbatim.
        3. Copies a subset of transformer blocks such that the first and last
           blocks are retained and the intermediate ones are chosen with
           uniform spacing.
        4. Works for both DiffusionDecoder (wrapper) and DiffusionDecoderCore.
        """

        # ── 0. get the real core we want to load into ────────────────────────────
        student_core = student_model.core if hasattr(student_model, "core") else student_model

        # ── 1. clean the teacher state-dict prefixes ────────────────────────────
        clean_teacher = {}
        for k, v in teacher_state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("core."):
                k = k[len("core."):]
            clean_teacher[k] = v

        # ── 2. determine transformer depth of teacher & student ────────────────
        n_teacher = getattr(teacher_cfg, "n_layers", None)
        n_student = getattr(student_cfg, "n_layers", None)
        if n_teacher is None or n_student is None:
            raise ValueError("Both cfgs must expose `n_layers`")

        # Map indices: always keep first & last, interpolate the rest
        if n_student == 1:
            index_map = {0: 0}
        else:
            t_ids = [
                round(i * (n_teacher - 1) / (n_student - 1))
                for i in range(n_student)
            ]
            index_map = {s_idx: t_idx for s_idx, t_idx in enumerate(t_ids)}

        new_state_dict = {}

        # ── 3. copy projection / embedding / final layers verbatim ─────────────
        for k, v in clean_teacher.items():
            if not k.startswith("blocks.blocks."):
                new_state_dict[k] = v

        # ── 4. copy the selected transformer blocks ────────────────────────────
        for s_idx, t_idx in index_map.items():
            t_pref = f"blocks.blocks.{t_idx}."
            s_pref = f"blocks.blocks.{s_idx}."
            for k, v in clean_teacher.items():
                if k.startswith(t_pref):
                    new_state_dict[s_pref + k[len(t_pref):]] = v

        # ── 5. load into the student core ──────────────────────────────────────
        student_core.load_state_dict(new_state_dict, strict=True)

    def get_ema_core(self):
        if self.world_size > 1:
            return self.ema.ema_model.module
        return self.ema.ema_model

    def save(self):
        save_dict = {
            'model': self.student.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
            'scaler': self.scaler.state_dict(),
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
        self.ema.load_state_dict(save_dict['ema'])
        self.opt.load_state_dict(save_dict['opt'])
        if self.scheduler is not None and 'scheduler' in save_dict:
            self.scheduler.load_state_dict(save_dict['scheduler'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.total_step_counter = save_dict['steps']

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Prepare model, ema
        self.student = self.student.to(self.device).train()
        if self.world_size > 1:
            self.student = DDP(self.student)

        freeze(self.encoder)
        freeze(self.dcae)
        freeze(self.teacher_diff)
        self.encoder = torch.compile(self.encoder, mode='max-autotune', dynamic=False, fullgraph=True)
        #self.teacher_diff = torch.compile(self.teacher_diff, mode='max-autotune', dynamic=False, fullgraph=True)

        self.ema = EMA(
            self.student,
            beta=0.9999,
            update_after_step=0,
            update_every=1
        )

        # Optimizer and scheduler
        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(self.student, rank=self.rank, world_size=self.world_size, **self.train_cfg.opt_kwargs)
        else:
            opt_cls = getattr(torch.optim, self.train_cfg.opt)
            self.opt = opt_cls(self.student.parameters(), **self.train_cfg.opt_kwargs)

        self.scheduler = None
        if self.train_cfg.scheduler is not None:
            self.scheduler = get_scheduler_cls(self.train_cfg.scheduler)(self.opt, **self.train_cfg.scheduler_kwargs)

        self.scaler = torch.amp.GradScaler()
        self.ctx = torch.amp.autocast(self.device, torch.bfloat16)

        self.accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        self.accum_steps = max(1, self.accum_steps)

        # Dataset setup
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)

        # Timer and logging
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.student, log='all')

        self.load()

        local_step = 0
        input_shape = (
            self.model_cfg.channels,
            self.model_cfg.sample_size[0],
            self.model_cfg.sample_size[1],
        )

        def get_dummy(z):
            return torch.randn(z.shape[0], *input_shape, device=z.device, dtype=z.dtype)

        @torch.no_grad()
        def sample_with_teacher(z, n_steps=20, subsample=1.0):
            """
            Sample using teacher 
            """
            noisy = get_dummy(z)
            dt_list = get_sd3_euler(n_steps)
            t = torch.ones(z.shape[0], device = z.device, dtype = z.dtype)

            inputs = []
            outputs = []
            ts = []
            zs = []

            for dt in dt_list:
                pred = self.teacher_diff(noisy, z, t)

                inputs.append(noisy.clone())
                outputs.append(pred.clone())
                ts.append(t.clone())
                zs.append(z.clone())

                noisy = noisy - dt * pred
                t = t - dt
            
            # Concatenate on batch dim
            inputs = torch.cat(inputs, dim=0)
            outputs = torch.cat(outputs, dim=0)
            ts = torch.cat(ts, dim=0)
            zs = torch.cat(zs, dim=0)

            if subsample < 1.0:
                inds = torch.randperm(inputs.shape[0])[:int(inputs.shape[0] * subsample)]
                inputs = inputs[inds]
                outputs = outputs[inds]
                ts = ts[inds]
                zs = zs[inds]

            return inputs, outputs, ts, zs

        for _ in range(self.train_cfg.epochs):
            for batch in loader:
                batch = batch.to(self.device).bfloat16()
                teacher_enc_input = batch.clone()
                batch = batch[:, :3]

                with torch.no_grad():
                    teacher_z = self.encoder(teacher_enc_input) / self.train_cfg.latent_scale
                    batch = F.interpolate(batch, size=(512, 512), mode='bilinear', align_corners=False)
                    inputs, outputs, ts, zs = sample_with_teacher(teacher_z, subsample=self.train_cfg.subsample)

                with self.ctx:
                    preds = self.student(inputs, zs, ts)
                    loss = F.mse_loss(preds, outputs)
                    loss = loss / self.accum_steps

                metrics.log('loss', loss)
                self.scaler.scale(loss).backward()

                local_step += 1
                if local_step % self.accum_steps == 0:
                    # Updates
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)

                    self.scaler.step(self.opt)
                    self.opt.zero_grad(set_to_none=True)

                    self.scaler.update()

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
                                ema_rec = flow_sample(self.get_ema_core(), get_dummy(teacher_z), teacher_z, 20, self.dcae.decoder, scaling_factor=self.train_cfg.ldm_scale)

                            wandb_dict['samples'] = to_wandb(
                                batch.detach().contiguous().bfloat16(),
                                ema_rec.detach().contiguous().bfloat16(),
                                gather=False
                            )

                        if self.rank == 0:
                            wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()

                    # Barrier for distributed
                    if self.world_size > 1:
                        torch.distributed.barrier()
