import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .base import BaseTrainer
from ..configs import Config
from ..utils import versatile_load, freeze, unfreeze, Timer
from ..models import get_model_cls
from ..models.tekken_rft_v2 import action_id_to_buttons
from ..utils.owl_vae_bridge import get_decoder_only, make_batched_decode_fn
from ..utils import batch_permute_to_length
from ..nn.kv_cache import KVCache
from ..utils.logging import LogHelper, to_wandb_av
from ..data import get_loader
from ..sampling import get_sampler_cls
from ..nn.rope import cast_rope_buffers_to_fp32

from copy import deepcopy
from ema_pytorch import EMA
import einops as eo
from contextlib import nullcontext
import random
import wandb
import gc
from pathlib import Path

@torch.no_grad()
def log_decoded_images(tensor, decode_fn, key, n_images=16):
    """
    Given a tensor of shape [b, c, h, w], decode and log images to wandb.
    """
    # Flatten tensor from [b, n, c, h, w] to [b * n, c, h, w] if needed
    if tensor.dim() == 5:
        b, n, c, h, w = tensor.shape
        tensor = tensor.reshape(b * n, c, h, w)
    idx = torch.randperm(tensor.shape[0])[:n_images]
    tensor = tensor[idx]
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        # Ensure tensor is on cpu and float32
        tensor = tensor.detach().cuda().bfloat16()
        # Decode to images in [-1, 1]
        imgs = decode_fn(tensor).float().cpu().squeeze(0)
        # Clamp to [-1, 1]
        imgs = torch.clamp(imgs, -1, 1)
        # Map to [0, 1]
        imgs = (imgs + 1) / 2
        # Convert to numpy for wandb
        imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()  # [b, h, w, c]
        # If single channel, repeat to 3 channels for visualization
        if imgs.shape[-1] == 1:
            imgs = imgs.repeat(1, 1, 1, 3)
        # Log as a list of wandb.Image
        wandb_imgs = [wandb.Image(img) for img in imgs]
        wandb.log({key: wandb_imgs})

# === ROLLOUTS ===

def zlerp(x, t):
    z = torch.randn_like(x)
    return x * (1. - t) + z * t

def zlerp_batched(x, t):
    z = torch.randn_like(x)
    t = t[:,:,None,None,None]
    return x * (1. - t) + z * t

def lerp_batched(x, z, t):
    t = t[:,:,None,None,None]
    return x * (1. - t) + z * t

class SoftResetIterator:
    """
    Wraps an iterable (e.g., DataLoader) so that when exhausted,
    it automatically resets and continues yielding items.
    """
    def __init__(self, iterable):
        self._iterable = iterable
        self._reset_iter()

    def _reset_iter(self):
        self._iterator = iter(self._iterable)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._reset_iter()
            return next(self._iterator)

class RolloutManager:
    def __init__(self, model_cfg, rollout_steps):
        self.model_cfg = model_cfg
        self.rollout_steps = rollout_steps
        self.noise_prev = 0.2
        self.gen_mask_p = 0.25

    def sample_discrete_ts(self, video): 
        """
        Sample discrete ts from steps relevant to sampling
        """
        valid_ts_list = torch.tensor([1.0, 0.5], device=video.device, dtype=video.dtype)
        
        # Make ts of [video.shape[0], video.shape[1]] with values in valid_ts_list
        ts = torch.randint(
            0, len(valid_ts_list),
            (video.shape[0], video.shape[1]),
            device=video.device,
            dtype=torch.long
        )
        ts = valid_ts_list[ts]
        return ts

    def get_rollouts(
        self, 
        model,
        video,
        action_ids
    ):
        with torch.no_grad():
            # Sample mask: True for frames to generate, False for context
            gen_mask = (torch.rand(video.shape[0], video.shape[1], device=video.device) < self.gen_mask_p)
            ts = self.sample_discrete_ts(video)  # [b, n]
            
            ts_full = torch.where(gen_mask, ts, torch.full_like(ts, self.noise_prev))
            noisy_video = zlerp_batched(video, ts_full)

            orig_video = video.clone()
            
            # Convert action_ids to button presses
            button_presses = action_id_to_buttons(action_ids)
        
        v_pred = model(
            noisy_video,
            ts_full,
            button_presses
        )

        video = torch.where(
            gen_mask[:,:,None,None,None],
            noisy_video - v_pred*ts_full[:,:,None,None,None],
            video
        )

        return video, action_ids, gen_mask, orig_video

# === LOSSES ===

def shift_ts(t, s):
    return t * s / (1 + (s - 1) * t) 

def get_critic_loss(
    student, critic,
    video,
    action_ids,
    rollout_manager,
    ts_shift=8
):
    # Get rollout
    with torch.no_grad():
        video, action_ids, grad_mask, _ = rollout_manager.get_rollouts(
            model=student,
            video=video,
            action_ids=action_ids
        )

        # Get ts ~ U(0.02, 0.98)
        ts = torch.randn(video.shape[0], video.shape[1], device=video.device, dtype=video.dtype).sigmoid()
        
        noise = torch.randn_like(video)
        noisy_vid = lerp_batched(video, noise, ts)
        target_vid = (noise - video)
        
        # Convert action_ids to button presses
        button_presses = action_id_to_buttons(action_ids)

    pred_vid = critic(
        noisy_vid,
        ts,
        button_presses
    )

    grad_mask_exp = grad_mask[:, :, None, None, None]
    vid_loss = F.mse_loss(pred_vid * grad_mask_exp, target_vid * grad_mask_exp)
    return vid_loss

def get_dmd_loss(
    student, critic, teacher,
    video,
    action_ids,
    rollout_manager,
    cfg_scale=1.5,
    ts_shift=8,
    decode_fn=None,
    log_predictions=False
):
    # Get rollout
    video, action_ids, grad_mask, orig_video = rollout_manager.get_rollouts(
        model=student,
        video=video,
        action_ids=action_ids
    )
    
    with torch.no_grad():
        # Get ts ~ U(0.02, 0.98)
        ts = torch.randn(video.shape[0], video.shape[1], device=video.device, dtype=video.dtype).sigmoid()

        # Get noise
        noise = torch.randn_like(video)
        noisy_vid = lerp_batched(video, noise, ts)
        
        # Convert action_ids to button presses
        button_presses = action_id_to_buttons(action_ids)
        
        # Get velocities from teacher
        if cfg_scale != 1.0:
            # Create masks for conditional/unconditional
            has_controls_uncond = torch.zeros(video.shape[0], dtype=torch.bool, device=video.device)
            has_controls_cond = torch.ones(video.shape[0], dtype=torch.bool, device=video.device)
            
            pred_vid_uncond = teacher(
                noisy_vid,
                ts,
                button_presses,
                has_controls=has_controls_uncond
            )
            
            pred_vid_cond = teacher(
                noisy_vid,
                ts,
                button_presses,
                has_controls=has_controls_cond
            )
            
            v_teacher_vid = pred_vid_uncond + cfg_scale * (pred_vid_cond - pred_vid_uncond)
        else:
            v_teacher_vid = teacher(
                noisy_vid,
                ts,
                button_presses
            )

        # Velocities from critic
        v_critic_vid = critic(
            noisy_vid,
            ts,
            button_presses
        )

        # Get predictions mu_real, mu_fake
        mu_teacher_vid = noisy_vid - ts[:,:,None,None,None] * v_teacher_vid
        mu_critic_vid = noisy_vid - ts[:,:,None,None,None] * v_critic_vid

        # Get normalizers
        vid_normalizer = torch.abs(video - mu_teacher_vid).mean(dim=[1,2,3,4], keepdim=True)

        # Get gradients
        grad_vid = (mu_critic_vid - mu_teacher_vid) / vid_normalizer

        if torch.isnan(grad_vid).any():
            print("Warning: grad_vid contains NaNs")

        grad_vid = torch.nan_to_num(grad_vid, nan=0.0)

        # Get targets
        target_vid = (video.double() - grad_vid.double()).detach()

        if log_predictions:
            log_decoded_images(video[grad_mask], decode_fn, "video")
            log_decoded_images(target_vid[grad_mask], decode_fn, "target_vid")
            log_decoded_images(mu_teacher_vid[grad_mask], decode_fn, "mu_teacher_vid")
            log_decoded_images(mu_critic_vid[grad_mask], decode_fn, "mu_critic_vid")

    grad_mask_exp = grad_mask[:, :, None, None, None]

    # Get losses, only where grad_mask is true
    dmd_loss = 0.5 * F.mse_loss(
        video.double() * grad_mask_exp,
        target_vid * grad_mask_exp,
        reduction='mean'
    )

    regression_loss = F.mse_loss(
        video * grad_mask_exp, 
        orig_video * grad_mask_exp,
        reduction='mean'
    )

    return dmd_loss, regression_loss 

class TekkenDMDTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure causal model and no cfg
        self.model_cfg.cfg_prob = 0.0
        self.model_cfg.causal = True

        # Helper function to clean checkpoint keys
        def clean_checkpoint_keys(state_dict):
            """Remove common prefixes from checkpoint keys"""
            cleaned = {}
            prefixes = ['_orig_mod.core.', '_orig_mod.', 'module.core.', 'module.', 'core.']
            
            for key, value in state_dict.items():
                new_key = key
                for prefix in prefixes:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        break
                cleaned[new_key] = value
            return cleaned

        # === init teacher ===
        teacher_cfg_path = self.train_cfg.teacher_cfg
        teacher_ckpt_path = self.train_cfg.teacher_ckpt
        teacher_cfg = Config.from_yaml(teacher_cfg_path).model
        teacher_ckpt = versatile_load(teacher_ckpt_path)
        teacher_ckpt = clean_checkpoint_keys(teacher_ckpt)

        self.teacher = get_model_cls(teacher_cfg.model_id)(teacher_cfg)
        try:
            self.teacher.load_state_dict(teacher_ckpt)
        except:
            self.teacher.core.load_state_dict(teacher_ckpt)

        # === init student ===
        student_ckpt_path = self.train_cfg.student_ckpt
        student_ckpt = versatile_load(student_ckpt_path)
        student_ckpt = clean_checkpoint_keys(student_ckpt)

        self.student = get_model_cls(self.model_cfg.model_id)(self.model_cfg)
        try:
            self.student.load_state_dict(student_ckpt)
        except:
            self.student.core.load_state_dict(student_ckpt)

        self.critic = deepcopy(self.student)

        # All models should be cores only
        self.teacher = self.teacher.core
        self.student = self.student.core
        self.critic = self.critic.core

        # Print model size for logging
        if self.rank == 0:
            n_params = sum(p.numel() for p in self.student.parameters())
            print(f"Model has {n_params:,} parameters")

        # Initialize parameters
        self.ema = None
        self.opt = None
        self.critic_opt = None
        self.scaler = None
        self.critic_scaler = None
        self.total_step_counter = 0

        # === decoders for sampling ===
        self.decoder = get_decoder_only(
            None,
            self.train_cfg.vae_cfg_path,
            self.train_cfg.vae_ckpt_path
        )

        freeze(self.teacher)
    
    @staticmethod
    def get_raw_model(model):
        """Extract the raw model from DDP/compiled wrappers"""
        # Handle _orig_mod from torch.compile
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        # Handle module from DDP
        return getattr(model, "module", model)
    
    def save(self):
        save_dict = {
            'model': self.get_raw_model(self.student).state_dict(),
            'ema': self.get_raw_model(self.ema).state_dict(),
            'opt': self.opt.state_dict(),
            'scaler': self.scaler.state_dict(),
            'critic': self.get_raw_model(self.critic).state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'critic_scaler': self.critic_scaler.state_dict(),
            'steps': self.total_step_counter
        }
        super().save(save_dict)

    def load(self):
        if not hasattr(self.train_cfg, 'resume_ckpt') or self.train_cfg.resume_ckpt is None:
            print("No checkpoint to load")
            return
        
        print(f"Loading checkpoint from: {self.train_cfg.resume_ckpt}")
        save_dict = super().load(self.train_cfg.resume_ckpt)
        
        def clean_keys(state_dict):
            """Remove common prefixes from state dict keys"""
            cleaned = {}
            prefixes = ['_orig_mod.core.', '_orig_mod.', 'module.core.', 'module.', 'core.']
            
            for key, value in state_dict.items():
                new_key = key
                for prefix in prefixes:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        break
                cleaned[new_key] = value
            return cleaned
        
        # Clean and load each component
        components = [
            ('model', self.student, 'Student model'),
            ('ema', self.ema, 'EMA'),
            ('critic', self.critic, 'Critic'),
            ('opt', self.opt, 'Optimizer'),
            ('critic_opt', self.critic_opt, 'Critic optimizer'),
            ('scaler', self.scaler, 'Scaler'),
            ('critic_scaler', self.critic_scaler, 'Critic scaler')
        ]
        
        for key, target, name in components:
            if key in save_dict and target is not None:
                try:
                    state = clean_keys(save_dict[key]) if 'model' in key or key == 'ema' or key == 'critic' else save_dict[key]
                    target.load_state_dict(state)
                    print(f"✓ {name} loaded")
                except Exception as e:
                    print(f"✗ Failed to load {name}: {e}")
        
        self.total_step_counter = save_dict.get('steps', 0)
        print(f"Loaded checkpoint at step {self.total_step_counter}")

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Inference only modules are frozen eval+cuda+bf16
        self.teacher = self.teacher.cuda().bfloat16()
        self.decoder = self.decoder.cuda().eval().bfloat16()

        cast_rope_buffers_to_fp32(self.teacher)

        # Training modules are train+cuda
        self.student = self.student.cuda().train()
        self.critic = self.critic.cuda().train()

        # DDP
        if self.world_size > 1:
            self.student = DDP(self.student, find_unused_parameters=True)
            self.critic = DDP(self.critic, find_unused_parameters=True)

        # Compile models
        self.critic.module = torch.compile(self.critic.module)
        self.teacher = torch.compile(self.teacher)

        # EMA model
        self.ema = EMA(
            self.student,
            beta=0.99,
            update_after_step=0,
            update_every=1
        )
        ema_module = lambda: self.ema.ema_model.module if self.world_size > 1 else self.ema.ema_model
        
        # Prepare decode functions for sampling
        frame_decode_fn = make_batched_decode_fn(self.decoder, batch_size=self.train_cfg.vae_batch_size)

        # Initialize optimizers and scalers
        self.opt = getattr(torch.optim, self.train_cfg.opt)(self.student.parameters(), **self.train_cfg.opt_kwargs)
        self.critic_opt = getattr(torch.optim, self.train_cfg.opt)(self.critic.parameters(), **self.train_cfg.d_opt_kwargs)
        self.scaler = torch.amp.GradScaler()
        self.critic_scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast('cuda', torch.bfloat16)

        self.load()

        # Logging helpers
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.student, log='all')
        
        # Dataset and sampling prep
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)
        loader = SoftResetIterator(loader)
        sample_loader = get_loader(self.train_cfg.sample_data_id, self.train_cfg.batch_size, **self.train_cfg.sample_data_kwargs)
        sample_loader = SoftResetIterator(sample_loader)

        # Sampler
        sampler = get_sampler_cls(self.train_cfg.sampler_id)(**self.train_cfg.sampler_kwargs)
        rollout_manager = RolloutManager(self.model_cfg, self.train_cfg.rollout_steps)

        # Gradient accumulation setup
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)

        # optimizer step
        def optimizer_step(model, scaler, optimizer):
            scaler.unscale_(optimizer)
            g_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            return g_norm

        # simplify getting batches
        def get_batch():
            vid, actions, _ = next(loader)
            vid = vid / self.train_cfg.vae_scale
            # Extract action IDs from the actions tensor (assuming last dimension)
            action_ids = actions[:,:,-1].long()
            return vid.to(self.device), action_ids.to(self.device)

        # === training loop ===
        while True:
            freeze(self.student)
            unfreeze(self.critic)

            for _ in range(self.train_cfg.update_ratio):
                for _ in range(accum_steps):
                    vid, action_ids = get_batch()
                    
                    with ctx:
                        critic_loss = get_critic_loss(
                            student=self.student,
                            critic=self.critic,
                            video=vid,
                            action_ids=action_ids,
                            rollout_manager=rollout_manager
                        ) / accum_steps
                        metrics.log('critic_loss', critic_loss)
                        self.critic_scaler.scale(critic_loss).backward()

                optimizer_step(self.critic, self.critic_scaler, self.critic_opt)

            freeze(self.critic)
            unfreeze(self.student)

            for _ in range(accum_steps):
                vid, action_ids = get_batch()
                
                with ctx:
                    dmd_loss, regression_loss = get_dmd_loss(
                        student=self.student,
                        critic=self.critic,
                        teacher=self.teacher,
                        video=vid,
                        action_ids=action_ids,
                        rollout_manager=rollout_manager,
                        decode_fn=frame_decode_fn,
                        log_predictions=(self.rank == 0 and self.total_step_counter % self.train_cfg.sample_interval == 0 and self.train_cfg.log_predictions)
                    )
                    dmd_loss = dmd_loss / accum_steps
                    regression_loss = regression_loss / accum_steps
                    metrics.log('dmd_loss', dmd_loss)
                    metrics.log('regression_loss', regression_loss)

                    gen_loss = dmd_loss + self.train_cfg.regression_weight * regression_loss
                    self.scaler.scale(gen_loss).backward()

                g_norm = optimizer_step(self.student, self.scaler, self.opt)
                metrics.log('g_norm', g_norm)

            with torch.no_grad():
                # Logging
                self.ema.update()
                wandb_dict = metrics.pop()
                wandb_dict['time'] = timer.hit()
                timer.reset()

                gc.collect()
                torch.cuda.empty_cache()

                # Sampling
                if self.total_step_counter % self.train_cfg.sample_interval == 0:
                    with ctx:
                        eval_wandb_dict = self.eval_step(sample_loader, sampler, frame_decode_fn)
                        gc.collect()
                        torch.cuda.empty_cache()
                        if self.rank == 0:
                            wandb_dict.update(eval_wandb_dict)

                if self.rank == 0:
                    wandb.log(wandb_dict)

                self.total_step_counter += 1
                if self.total_step_counter % self.train_cfg.save_interval == 0:
                    if self.rank == 0:
                        self.save()

    @torch.no_grad()
    def eval_step(self, sample_loader, sampler, decode_fn=None):
        model = self.ema.ema_model.module if self.world_size > 1 else self.ema.ema_model
        
        # Get sample data
        vid, actions, _ = next(sample_loader)
        vid = vid / self.train_cfg.vae_scale
        action_ids = actions[:,:,-1].long()

        # Collect multiple action sequences for variety
        action_ids_list = [action_ids]
        for _ in range(15):
            _, new_actions, _ = next(sample_loader)
            new_action_ids = new_actions[:,:,-1].long()
            action_ids_list.append(new_action_ids)

        action_ids_all = torch.cat(action_ids_list, dim=0)
        
        # Permute to get extended sequence
        extended_action_ids, _ = batch_permute_to_length(
            action_ids_all.unsqueeze(-1), 
            action_ids_all.unsqueeze(-1), 
            sampler.num_frames + vid.size(1)
        )
        extended_action_ids = extended_action_ids.squeeze(-1)
        action_ids = extended_action_ids[:vid.size(0)]  # Take first batch worth

        latent_vid = sampler(model, vid.cuda(), action_ids.cuda())

        # Take only generated frames
        if True:
            latent_vid = latent_vid[:, vid.size(1):]
            action_ids = action_ids[:, vid.size(1):]

        video_out = decode_fn(latent_vid * self.train_cfg.vae_scale) if decode_fn is not None else None

        gc.collect()
        torch.cuda.empty_cache()

        def gather_concat_cpu(t, dim=0):
            if self.rank == 0:
                t_gpu = t.cuda() if t.device.type == 'cpu' else t
                parts = [t_gpu.cpu()]
                scratch = torch.empty_like(t_gpu)
                for src in range(self.world_size):
                    if src == 0:
                        continue
                    dist.recv(scratch, src=src)
                    parts.append(scratch.cpu())
                return torch.cat(parts, dim=dim)
            else:
                t_gpu = t.cuda() if t.device.type == 'cpu' else t
                dist.send(t_gpu, dst=0)
                return None

        # Save latent artifacts
        if getattr(self.train_cfg, "eval_sample_dir", None):
            latent_vid = gather_concat_cpu(latent_vid)
            if self.rank == 0:
                eval_dir = Path(self.train_cfg.eval_sample_dir)
                eval_dir.mkdir(parents=True, exist_ok=True)
                torch.save(latent_vid, eval_dir / f"vid.{self.total_step_counter}.pt")
        
        # Generate media artifacts
        video_out, action_ids = [
            gather_concat_cpu(x, dim=0) for x in [video_out, action_ids]
        ]

        if self.rank == 0:
            # Convert action_ids to button presses for visualization
            button_presses = action_id_to_buttons(action_ids)
            wandb_av_out = to_wandb_av(video_out, None, None, button_presses)
            if len(wandb_av_out) == 3:
                video, depth_gif, flow_gif = wandb_av_out
                eval_wandb_dict = dict(samples=video, depth_gif=depth_gif, flow_gif=flow_gif)
            elif len(wandb_av_out) == 2:
                video, depth_gif = wandb_av_out
                eval_wandb_dict = dict(samples=video, depth_gif=depth_gif)
            else:
                eval_wandb_dict = dict(samples=wandb_av_out)
        else:
            eval_wandb_dict = None
        self.barrier()

        return eval_wandb_dict