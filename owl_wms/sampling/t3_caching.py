import torch
from tqdm import tqdm

from ..nn.kv_cache import KVCache
from .schedulers import get_sd3_euler

class TekkenCachingSampler:
    """
    An efficient sampler for the TekkenRFT model that uses KV caching to speed up
    sequential frame generation. It generates one new frame at a time by conditioning
    on a sliding window of previously generated frames.

    Args:
        n_steps (int): The number of denoising steps for each new frame.
        cfg_scale (float): The scale for classifier-free guidance. Set to 1.0 for no CFG.
        num_frames (int): The total number of new frames to generate.
        noise_prev (float): The amount of noise to add to the context frames.
        only_return_generated (bool): If True, returns only the newly generated frames.
    """
    def __init__(self, n_steps: int = 16, cfg_scale: float = 1.0, num_frames: int = 160, noise_prev: float = 0.2, only_return_generated: bool = False, **kwargs):
        self.n_steps = n_steps
        self.cfg_scale = cfg_scale
        self.num_frames = num_frames
        self.noise_prev = noise_prev
        self.only_return_generated = only_return_generated

    @staticmethod
    def zlerp(x, alpha):
        """Spherical linear interpolation with noise."""
        z = torch.randn_like(x)
        return x * (1. - alpha) + z * alpha

    @torch.no_grad()
    def denoise_frame(
        self,
        model,
        kv_cache: KVCache, # type: ignore
        prev_latents: torch.Tensor,
        prev_actions: torch.Tensor,
        curr_actions: torch.Tensor,
        dt: torch.Tensor,
    ):
        """
        Performs the full denoising loop for a single new frame using KV caching.
        """
        batch_size = prev_latents.size(0)

        # Step 1: Prepare the initial noisy state
        # Re-noise the context frames slightly and create a pure noise tensor for the new frame
        prev_latents_noised = self.zlerp(prev_latents, self.noise_prev)
        t_prev = prev_latents.new_full((batch_size, prev_latents.size(1)), self.noise_prev)
        
        new_latent = torch.randn_like(prev_latents[:, :1])
        t_new = t_prev.new_ones(batch_size, 1)

        # Step 2: Populate the KV Cache with the context
        kv_cache.enable_cache_updates()
        
        full_input_latents = torch.cat([prev_latents_noised, new_latent], dim=1)
        full_ts = torch.cat([t_prev, t_new], dim=1)
        full_actions = torch.cat([prev_actions, curr_actions], dim=1)

        # Single forward pass to populate cache (no CFG complexity)
        _ = model(full_input_latents, full_ts, full_actions, kv_cache=kv_cache)
        
        kv_cache.disable_cache_updates()
        kv_cache.truncate(1, front=True)  # Keep only the new frame being denoised

        # Take the first Euler step (similar to AVCachingSampler)
        eps_v = model(new_latent, t_new, curr_actions, kv_cache=kv_cache)
        new_latent = new_latent - eps_v * dt[0]
        t_new = t_new - dt[0]

        # Step 3: Remaining denoising steps for the new frame
        for step in range(1, self.n_steps):
            eps_v = model(new_latent, t_new, curr_actions, kv_cache=kv_cache)
            new_latent = new_latent - eps_v * dt[step]
            t_new = t_new - dt[step]

        return new_latent

    @torch.no_grad()
    def __call__(self, model, initial_latents, action_ids, decode_fn=None, vae_scale=1.0):
        model.eval()
        batch_size, init_len, c, h, w = initial_latents.shape
        dt = get_sd3_euler(self.n_steps).to(device=initial_latents.device, dtype=initial_latents.dtype)
        
        # Initialize KV cache
        kv_cache = KVCache(model.config)
        kv_cache.reset(batch_size)
        
        # This will store all generated latents, starting with the initial ones
        all_latents = [initial_latents]
        
        # Start with the initial context
        prev_latents = initial_latents
        prev_actions = action_ids[:, :init_len]

        for idx in tqdm(range(self.num_frames), desc="Sampling Tekken Frames (Caching)"):
            # Get the action for the new frame we are about to generate
            curr_actions = action_ids[:, init_len + idx: init_len + idx + 1]
            
            # Denoise the next frame
            new_latent = self.denoise_frame(
                model, kv_cache,
                prev_latents, prev_actions,
                curr_actions,
                dt=dt,
            )

            all_latents.append(new_latent)

            # Update context for next iteration - keep sliding window
            # Use the newly generated frame as the new context
            prev_latents = new_latent
            prev_actions = curr_actions

        generated_latents = torch.cat(all_latents, dim=1)

        if self.only_return_generated:
            final_latents = generated_latents[:, init_len:]
            final_actions = action_ids[:, init_len : init_len + self.num_frames]
        else:
            final_latents = generated_latents
            final_actions = action_ids[:, :generated_latents.shape[1]]

        video_out = decode_fn(final_latents * vae_scale) if decode_fn is not None else None
        
        return video_out, final_latents, final_actions
    
    
class TekkenCachingSamplerWithCFG:
    """
    Version with CFG support - use only if your model was trained with classifier-free guidance.
    """
    def __init__(self, n_steps: int = 16, cfg_scale: float = 1.3, num_frames: int = 160, noise_prev: float = 0.2, only_return_generated: bool = False, **kwargs):
        self.n_steps = n_steps
        self.cfg_scale = cfg_scale
        self.num_frames = num_frames
        self.noise_prev = noise_prev
        self.only_return_generated = only_return_generated

    @staticmethod
    def zlerp(x, alpha):
        """Spherical linear interpolation with noise."""
        z = torch.randn_like(x)
        return x * (1. - alpha) + z * alpha

    @torch.no_grad()
    def denoise_frame(
        self,
        model,
        prev_latents: torch.Tensor,
        prev_actions: torch.Tensor,
        curr_actions: torch.Tensor,
        dt: torch.Tensor,
    ):
        """
        CFG version - creates separate caches for conditional and unconditional passes.
        """
        batch_size = prev_latents.size(0)

        # Prepare noisy states
        prev_latents_noised = self.zlerp(prev_latents, self.noise_prev)
        t_prev = prev_latents.new_full((batch_size, prev_latents.size(1)), self.noise_prev)
        
        new_latent = torch.randn_like(prev_latents[:, :1])
        t_new = t_prev.new_ones(batch_size, 1)

        # Prepare full sequences
        full_input_latents = torch.cat([prev_latents_noised, new_latent], dim=1)
        full_ts = torch.cat([t_prev, t_new], dim=1)
        full_actions = torch.cat([prev_actions, curr_actions], dim=1)

        # Create and populate unconditional cache
        kv_cache_uncond = KVCache(model.config)
        kv_cache_uncond.reset(batch_size)
        kv_cache_uncond.enable_cache_updates()
        
        has_controls_uncond = torch.zeros(batch_size, device=prev_latents.device, dtype=torch.bool)
        _ = model(full_input_latents, full_ts, full_actions, has_controls=has_controls_uncond, kv_cache=kv_cache_uncond)
        
        kv_cache_uncond.disable_cache_updates()
        kv_cache_uncond.truncate(1, front=True)

        # Create and populate conditional cache
        kv_cache_cond = KVCache(model.config)
        kv_cache_cond.reset(batch_size)
        kv_cache_cond.enable_cache_updates()
        
        has_controls_cond = torch.ones(batch_size, device=prev_latents.device, dtype=torch.bool)
        _ = model(full_input_latents, full_ts, full_actions, has_controls=has_controls_cond, kv_cache=kv_cache_cond)
        
        kv_cache_cond.disable_cache_updates()
        kv_cache_cond.truncate(1, front=True)

        # Denoising loop with CFG
        for step in range(self.n_steps):
            # Unconditional prediction
            pred_uncond = model(new_latent, t_new, curr_actions, has_controls=has_controls_uncond, kv_cache=kv_cache_uncond)
            
            # Conditional prediction
            pred_cond = model(new_latent, t_new, curr_actions, has_controls=has_controls_cond, kv_cache=kv_cache_cond)
            
            # Apply CFG
            pred_velocity = pred_uncond + self.cfg_scale * (pred_cond - pred_uncond)
            
            # Euler step
            new_latent = new_latent - pred_velocity * dt[step]
            t_new = t_new - dt[step]

        return new_latent

    @torch.no_grad()
    def __call__(self, model, initial_latents, action_ids, decode_fn=None, vae_scale=1.0):
        model.eval()
        batch_size, init_len, c, h, w = initial_latents.shape
        dt = get_sd3_euler(self.n_steps).to(device=initial_latents.device, dtype=initial_latents.dtype)
        
        all_latents = [initial_latents]
        prev_latents = initial_latents
        prev_actions = action_ids[:, :init_len]

        for idx in tqdm(range(self.num_frames), desc="Sampling Tekken Frames (CFG + Caching)"):
            curr_actions = action_ids[:, init_len + idx: init_len + idx + 1]
            
            new_latent = self.denoise_frame(
                model,
                prev_latents, prev_actions,
                curr_actions,
                dt=dt,
            )

            all_latents.append(new_latent)
            
            # Update context with a true sliding window
            # Concatenate the new frame and drop the oldest one
            current_context = torch.cat([prev_latents[:, 1:], new_latent], dim=1)
            prev_latents = current_context

            # You will need to adjust the action update logic similarly
            # to maintain the correct context length for actions.
            # This is a simplified example for the latents.
            prev_actions = action_ids[:, (idx + 1) : (init_len + idx + 1)]
            
        generated_latents = torch.cat(all_latents, dim=1)

        if self.only_return_generated:
            final_latents = generated_latents[:, init_len:]
            final_actions = action_ids[:, init_len : init_len + self.num_frames]
        else:
            final_latents = generated_latents
            final_actions = action_ids[:, :generated_latents.shape[1]]

        video_out = decode_fn(final_latents * vae_scale) if decode_fn is not None else None
        
        return video_out, final_latents, final_actions

class SimpleTekkenSampler:
    """
    Simplified version following AVCachingSampler pattern exactly.
    Use this for debugging - if it works better, the issue is in the complex logic above.
    """
    def __init__(self, n_steps: int = 16, cfg_scale:float = 1.0, num_frames: int = 160, noise_prev: float = 0.2, only_return_generated: bool = False):
        self.n_steps = n_steps
        self.num_frames = num_frames
        self.noise_prev = noise_prev
        self.only_return_generated = only_return_generated

    @staticmethod
    def zlerp(x, alpha):
        z = torch.randn_like(x)
        return x * (1. - alpha) + z * alpha

    def denoise_frame(self, model, kv_cache, prev_x, prev_actions, curr_actions, dt):
        """Simplified denoising following AVCachingSampler exactly"""
        batch_size = prev_x.size(0)

        # Re-noise history
        prev_vid = self.zlerp(prev_x, self.noise_prev)
        t_prev = prev_x.new_full((batch_size, prev_vid.size(1)), self.noise_prev)

        # Create new pure-noise frame
        new_vid = torch.randn_like(prev_x[:, :1])
        t_new = t_prev.new_ones(batch_size, 1)

        # Update kv cache with context
        kv_cache.enable_cache_updates()
        eps_v = model(
            torch.cat([prev_vid, new_vid], dim=1),
            torch.cat([t_prev, t_new], dim=1),
            torch.cat([prev_actions, curr_actions], dim=1),
            kv_cache=kv_cache,
        )
        kv_cache.disable_cache_updates()
        kv_cache.truncate(1, front=True)

        # First Euler step
        new_vid = new_vid - eps_v[:, -1:] * dt[0]
        t_new = t_new - dt[0]

        # Remaining steps
        for step in range(1, self.n_steps):
            eps_vid = model(new_vid, t_new, curr_actions, kv_cache=kv_cache)
            new_vid = new_vid - eps_vid * dt[step]
            t_new = t_new - dt[step]

        return new_vid

    @torch.no_grad()
    def __call__(self, model, initial_latents, action_ids, decode_fn=None, vae_scale=1.0):
        model.eval()
        batch_size, init_len = initial_latents.shape[:2]
        dt = get_sd3_euler(self.n_steps).to(device=initial_latents.device, dtype=initial_latents.dtype)
        
        kv_cache = KVCache(model.config)
        kv_cache.reset(batch_size)
        
        latents = [initial_latents]
        prev_x = initial_latents
        prev_actions = action_ids[:, :init_len]
        
        for idx in tqdm(range(self.num_frames), desc="Simple Tekken Sampling"):
            curr_actions = action_ids[:, init_len + idx: init_len + idx + 1]
            
            new_frame = self.denoise_frame(
                model, kv_cache, prev_x, prev_actions, curr_actions, dt
            )
            latents.append(new_frame)
            
            # Update context - only keep the new frame
            prev_x = new_frame
            prev_actions = curr_actions
        
        generated_latents = torch.cat(latents, dim=1)
        
        if self.only_return_generated:
            final_latents = generated_latents[:, init_len:]
            final_actions = action_ids[:, init_len : init_len + self.num_frames]
        else:
            final_latents = generated_latents
            final_actions = action_ids[:, :generated_latents.shape[1]]

        video_out = decode_fn(final_latents * vae_scale) if decode_fn is not None else None
        
        return video_out, final_latents, final_actions
