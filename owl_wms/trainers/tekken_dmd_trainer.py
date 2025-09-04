import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .base import BaseTrainer
from ..configs import Config
from ..utils import versatile_load, freeze, unfreeze, Timer
from ..models import get_model_cls
from ..models.tekken_rft_v2 import action_id_to_buttons # --- MODIFICATION: Import action converter
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn
from ..utils import batch_permute_to_length
from ..nn.kv_cache import KVCache
from ..utils.logging import LogHelper, to_wandb_gif
from ..data import get_loader
from ..sampling import get_sampler_cls

from copy import deepcopy
from ema_pytorch import EMA
import random
import wandb

# MODIFICATION: This entire section is adapted from causvid_v2.py to remove audio/mouse and use action_ids

# === ROLLOUTS ===

class RolloutManager:
    def __init__(self, model_cfg, min_rollout_frames=8, rollout_steps=1):
        self.model_cfg = model_cfg
        self.min_rollout_frames = min_rollout_frames
        self.rollout_steps = rollout_steps
        self.noise_prev = 0.2
    
    def get_rollouts(
        self, 
        model,
        video_bnchw,
        action_ids_bn,
        enable_grad=False
    ):
        with torch.no_grad():
            kv_cache = KVCache(self.model_cfg)
            kv_cache.reset(video_bnchw.shape[0])

            window_length = video_bnchw.shape[1]
            rollout_frames = 16  # Fixed for simplicity in this draft
            
            # Extend control inputs for open-ended generation
            # NOTE: This permutation logic might need adjustment for your specific use case
            ext_action_ids, _ = batch_permute_to_length(action_ids_bn.unsqueeze(-1), action_ids_bn.unsqueeze(-1), window_length + rollout_frames)
            ext_action_ids = ext_action_ids.squeeze(-1)
            rollout_action_ids = ext_action_ids[:, window_length:]

            # Cache context frames
            kv_cache.enable_cache_updates()
            ts_bn = torch.zeros_like(video_bnchw[:,:,0,0,0]) # "clean" ts
            button_presses = action_id_to_buttons(action_ids_bn)
            model(video_bnchw, ts_bn, button_presses, kv_cache=kv_cache)
            kv_cache.disable_cache_updates()

        # Rollout generation
        for frame_idx in range(rollout_frames):
            ts_b1 = torch.ones_like(video_bnchw[:,0:1,0,0,0])
            frame_b1chw = torch.randn_like(video_bnchw[:,0:1])
            action_b1 = rollout_action_ids[:, frame_idx:frame_idx+1]
            button_b1 = action_id_to_buttons(action_b1)

            kv_cache.truncate(1, front=False)
            dt = 1.0 / self.rollout_steps
            end_frame = random.randint(1, self.rollout_steps) if enable_grad else self.rollout_steps

            for step_idx in range(end_frame):
                with torch.enable_grad() if (enable_grad and step_idx == end_frame - 1) else torch.no_grad():
                    vid_pred_b1chw = model(
                        frame_b1chw.detach(),
                        ts_b1.detach(),
                        button_b1.detach(),
                        kv_cache=kv_cache
                    )
            
                frame_b1chw = frame_b1chw - dt * vid_pred_b1chw
                ts_b1 = ts_b1 - dt

            with torch.no_grad():
                kv_cache.enable_cache_updates()
                model(frame_b1chw, ts_b1 * 0, button_b1, kv_cache=kv_cache)
                kv_cache.disable_cache_updates()

            video_bnchw = torch.cat([video_bnchw, frame_b1chw], dim=1)
        
        video_bnchw = video_bnchw[:, -window_length:]
        action_ids_bn = ext_action_ids[:, -window_length:]

        return video_bnchw, action_ids_bn, rollout_frames

# === LOSSES ===

def get_critic_loss(student, critic, video_bnchw, action_ids, rollout_manager):
    with torch.no_grad():
        video_bnchw, action_ids, _ = rollout_manager.get_rollouts(
            model=student, video_bnchw=video_bnchw, action_ids_bn=action_ids
        )
        button_presses = action_id_to_buttons(action_ids)

        ts_bn = torch.rand(video_bnchw.shape[0], video_bnchw.shape[1]) * 0.96 + 0.02
        ts_bn = ts_bn.to(video_bnchw.device, video_bnchw.dtype)
        ts_bn_exp = ts_bn[:,:,None,None,None]
        z_vid_bnchw = torch.randn_like(video_bnchw)
        noisy_vid_bnchw = video_bnchw * (1. - ts_bn_exp) + z_vid_bnchw * ts_bn_exp
        target_vid = (z_vid_bnchw - video_bnchw)

    pred_vid = critic(noisy_vid_bnchw, ts_bn, button_presses)
    vid_loss = F.mse_loss(pred_vid, target_vid)
    return vid_loss

def get_dmd_loss(student, critic, teacher, video_bnchw, action_ids, rollout_manager, cfg_scale=1.3):
    video_bnchw, action_ids, rollout_frames = rollout_manager.get_rollouts(
        model=student, video_bnchw=video_bnchw, action_ids_bn=action_ids, enable_grad=True
    )
    button_presses = action_id_to_buttons(action_ids)

    with torch.no_grad():
        mask_length = video_bnchw.shape[1] - rollout_frames
        ts_bn = torch.rand(video_bnchw.shape[0], video_bnchw.shape[1]) * 0.96 + 0.02
        ts_bn = ts_bn.to(video_bnchw.device, video_bnchw.dtype)
        ts_bn_exp = ts_bn[:,:,None,None,None]
        z_vid_bnchw = torch.randn_like(video_bnchw)
        noisy_vid_bnchw = video_bnchw * (1. - ts_bn_exp) + z_vid_bnchw * ts_bn_exp
        
        uncond_mask = torch.zeros(video_bnchw.shape[0], dtype=torch.bool, device=video_bnchw.device)
        cond_mask = torch.ones(video_bnchw.shape[0], dtype=torch.bool, device=video_bnchw.device)
        
        vid_pred_uncond = teacher(noisy_vid_bnchw, ts_bn, button_presses, has_controls=uncond_mask)
        vid_pred_cond = teacher(noisy_vid_bnchw, ts_bn, button_presses, has_controls=cond_mask)
        
        v_teacher_vid = vid_pred_uncond + cfg_scale * (vid_pred_cond - vid_pred_uncond)
        v_critic_vid = critic(noisy_vid_bnchw, ts_bn, button_presses)

        mu_teacher_vid = noisy_vid_bnchw - ts_bn_exp * v_teacher_vid
        mu_critic_vid = noisy_vid_bnchw - ts_bn_exp * v_critic_vid
        
        vid_normalizer = torch.abs(video_bnchw - mu_teacher_vid).mean(dim=[1,2,3,4],keepdim=True)
        grad_vid = (mu_critic_vid - mu_teacher_vid) / vid_normalizer
        grad_vid = torch.nan_to_num(grad_vid, nan=0.0)
        target_vid = (video_bnchw.double() - grad_vid.double()).detach()

        grad_mask_vid = torch.ones_like(video_bnchw, dtype=torch.bool)
        grad_mask_vid[:,:mask_length] = False

    vid_loss = 0.5 * F.mse_loss(
        video_bnchw[grad_mask_vid].double(),
        target_vid[grad_mask_vid],
        reduction='mean'
    )
    return vid_loss

class TekkenDMDTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_cfg.causal = True

        # === Init Teacher ===
        teacher_cfg_path = self.train_cfg.teacher_cfg
        teacher_ckpt_path = self.train_cfg.teacher_ckpt
        teacher_cfg = Config.from_yaml(teacher_cfg_path).model
        # Load and clean teacher state dict
        teacher_ckpt = versatile_load(teacher_ckpt_path)
        cleaned_teacher_state_dict = {}
        prefix_to_strip = '_orig_mod.core.'
        for k, v in teacher_ckpt.items():
            if k.startswith(prefix_to_strip):
                new_k = k[len(prefix_to_strip):]
                cleaned_teacher_state_dict[new_k] = v
            else:
                cleaned_teacher_state_dict[k] = v
        
        self.teacher = get_model_cls(teacher_cfg.model_id)(teacher_cfg).core
        self.teacher.load_state_dict(cleaned_teacher_state_dict)
    
        # === Init Student & Critic ===
        student_ckpt_path = self.train_cfg.student_ckpt
        # Load and clean student state dict
        student_ckpt = versatile_load(student_ckpt_path)
        cleaned_student_state_dict = {}
        prefix_to_strip = '_orig_mod.core.'
        for k, v in student_ckpt.items():
            if k.startswith(prefix_to_strip):
                new_k = k[len(prefix_to_strip):]
                cleaned_student_state_dict[new_k] = v
            else:
                cleaned_student_state_dict[k] = v
        
        self.student = get_model_cls(self.model_cfg.model_id)(self.model_cfg).core
        self.student.load_state_dict(cleaned_student_state_dict)
        self.critic = deepcopy(self.student)

        if self.rank == 0:
            print(f"Model has {sum(p.numel() for p in self.student.parameters()):,} parameters")

        self.ema, self.opt, self.critic_opt, self.scaler, self.critic_scaler = None, None, None, None, None
        self.total_step_counter = 0

    def train(self):
        # Boilerplate setup (DDP, optimizers, EMA, etc.)
        torch.cuda.set_device(self.local_rank)
        self.teacher = self.teacher.cuda().eval().bfloat16()
        self.student = self.student.cuda().train()
        self.critic = self.critic.cuda().train()

        if self.world_size > 1:
            self.student = DDP(self.student, find_unused_parameters=True)
            self.critic = DDP(self.critic, find_unused_parameters=True)

        self.ema = EMA(self.student, beta=0.99, update_every=1)
        self.opt = getattr(torch.optim, self.train_cfg.opt)(self.student.parameters(), **self.train_cfg.opt_kwargs)
        self.critic_opt = getattr(torch.optim, self.train_cfg.opt)(self.critic.parameters(), **self.train_cfg.d_opt_kwargs)
        self.scaler = torch.amp.GradScaler()
        self.critic_scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast('cuda', torch.bfloat16)

        self.load() # Implement save/load methods similar to CausVidTrainer

        metrics = LogHelper()
        if self.rank == 0: wandb.watch(self.student, log='all')
        
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)
        loader = iter(loader)
        rollout_manager = RolloutManager(self.model_cfg, self.train_cfg.min_rollout_frames, self.train_cfg.rollout_steps)
        accum_steps = max(1, self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size)

        def optimizer_step(model, scaler, optimizer):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()

        def get_batch():
            vid, actions, _ = next(loader)
            vid = vid / self.train_cfg.vae_scale
            # The Tekken dataloader provides action IDs in the last feature of the action tensor
            action_ids = actions[:,:,-1].int()
            return vid.to(self.device), action_ids.to(self.device)

        # === Training Loop ===
        while True:
            # --- Train Critic ---
            freeze(self.student); unfreeze(self.critic)
            for _ in range(self.train_cfg.update_ratio):
                for _ in range(accum_steps):
                    vid_bnchw, action_ids = get_batch()
                    with ctx:
                        critic_loss = get_critic_loss(self.student, self.critic, vid_bnchw, action_ids, rollout_manager) / accum_steps
                        metrics.log('critic_loss', critic_loss)
                    self.critic_scaler.scale(critic_loss).backward()
                optimizer_step(self.critic, self.critic_scaler, self.critic_opt)

            # --- Train Student ---
            unfreeze(self.student); freeze(self.critic)
            for _ in range(accum_steps):
                vid_bnchw, action_ids = get_batch()
                with ctx:
                    dmd_loss = get_dmd_loss(self.student, self.critic, self.teacher, vid_bnchw, action_ids, rollout_manager, self.train_cfg.cfg_scale) / accum_steps
                    metrics.log('dmd_loss', dmd_loss)
                self.scaler.scale(dmd_loss).backward()
            optimizer_step(self.student, self.scaler, self.opt)
            self.ema.update()

            # --- Logging & Saving ---
            # (This part can be copied and adapted from CausVidTrainer)
            if self.rank == 0:
                 wandb_dict = metrics.pop()
                 # ... add other metrics like time, lr ...
                 wandb.log(wandb_dict)
            self.total_step_counter += 1
            if self.total_step_counter % self.train_cfg.save_interval == 0 and self.rank == 0:
                self.save()
            self.barrier()