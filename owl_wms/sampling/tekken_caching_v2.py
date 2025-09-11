import torch
from tqdm import tqdm

from ..nn.kv_cache import KVCache
from .schedulers import get_sd3_euler

def n_tokens(vid):
    """Calculate number of tokens in a video tensor."""
    return vid.size(1) * vid.size(3) * vid.size(4)

def get_deltas(custom_schedule):
    """Convert custom schedule to delta steps."""
    if custom_schedule[-1] != 0.0:
        custom_schedule.append(0.0)

    deltas = []
    crnt = custom_schedule[0]
    for nxt in custom_schedule[1:]:
        deltas.append(abs(nxt - crnt))
        crnt = nxt

    return deltas

class TekkenCachingSamplerV2:
    """
    Advanced Tekken RFT sampler with proper sliding context logic based on AV caching v2,
    but adapted for action_ids instead of mouse/btn and WITHOUT CFG support.
    
    Parameters
    ----------
    :param n_steps: Number of diffusion steps for each frame
    :param num_frames: Number of new frames to sample
    :param noise_prev: Noise level for previous frames
    :param max_window: Maximum context window size (None for unlimited)
    :param custom_schedule: Custom denoising schedule (None for default)
    :param only_return_generated: If True, returns only newly generated frames
    """
    def __init__(
        self, 
        n_steps: int = 16, 
        cfg_scale: float = 1.0,  # Keep for compatibility but ignore
        num_frames: int = 160, 
        noise_prev: float = 0.2, 
        max_window = None, 
        custom_schedule = None,
        only_return_generated: bool = False,
        **kwargs
    ) -> None:
        self.n_steps = n_steps
        self.num_frames = num_frames
        self.noise_prev = noise_prev
        self.max_window = max_window
        self.custom_schedule = custom_schedule
        self.only_return_generated = only_return_generated

    @staticmethod
    def zlerp(x, alpha):
        """Linear interpolation with noise for rectified flow."""
        z = torch.randn_like(x)
        return x * (1. - alpha) + z * alpha

    @torch.no_grad()
    def __call__(self, model, x, action_ids, decode_fn=None, vae_scale=1.0, compile_on_decode=False):
        """
        Generate new frames using KV caching and proper sliding context.
        
        Args:
            model: The TekkenRFT model
            x: Initial latent frames [B, T, C, H, W]
            action_ids: Action sequence [B, T]
            decode_fn: Optional VAE decoder function
            vae_scale: VAE scaling factor
            compile_on_decode: Whether to compile model for decoding
        
        Returns:
            Generated latent sequence or decoded videos
        """
        batch_size, init_len = x.size(0), x.size(1)

        # Setup scheduling
        if self.custom_schedule is None:
            dt = get_sd3_euler(self.n_steps).to(device=x.device, dtype=x.dtype)
        else:
            dt = torch.tensor(get_deltas(self.custom_schedule), device=x.device, dtype=x.dtype)

        # Initialize KV cache
        kv_cache = KVCache(model.config)
        kv_cache.reset(batch_size)

        # Store all latents including initial context
        latents = [x.clone()]
        
        # Setup initial context
        prev_x = x
        prev_actions = action_ids[:, :init_len]

        # ==== STEP 1: Cache initial context ====
        prev_x_noisy = self.zlerp(prev_x, self.noise_prev)
        prev_t = prev_x.new_full((batch_size, prev_x.size(1)), self.noise_prev)

        # Cache initial context
        kv_cache.enable_cache_updates()
        _ = model(
            prev_x_noisy,
            prev_t,
            prev_actions,
            kv_cache=kv_cache
        )
        kv_cache.disable_cache_updates()

        def new_xt():
            """Create new noisy frame tensor."""
            return torch.randn_like(prev_x[:, :1]), prev_t.new_ones(batch_size, 1)

        # Calculate actual frames to generate
        num_frames = min(self.num_frames, action_ids.size(1) - init_len)

        if compile_on_decode:
            model = torch.compile(model)

        # Enable decoding mode if supported
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'enable_decoding'):
            model.transformer.enable_decoding()
        
        # ==== STEP 2: Generate new frames ====
        for idx in tqdm(range(num_frames), desc="Sampling Frames..."):
            curr_x, curr_t = new_xt()

            start = init_len + idx
            curr_actions = action_ids[:, start:start+1]

            # Denoise the new frame
            for t_idx in range(self.n_steps):
                pred_v = model(
                    curr_x,
                    curr_t,
                    curr_actions,
                    kv_cache=kv_cache
                ).clone()

                # Euler step
                curr_x = curr_x - dt[t_idx] * pred_v   
                curr_t = curr_t - dt[t_idx]
            
            # ==== STEP 3: Cache the new clean frame ====
            latents.append(curr_x.clone())
            
            # Add new frame to cache
            curr_x_noisy = self.zlerp(curr_x, self.noise_prev)
            curr_t_noisy = torch.ones_like(curr_t) * self.noise_prev

            kv_cache.enable_cache_updates()
            _ = model(
                curr_x_noisy,
                curr_t_noisy,
                curr_actions,
                kv_cache=kv_cache
            )
            kv_cache.disable_cache_updates()
            
            # Manage sliding window if max_window is set
            if self.max_window is not None and len(latents) > self.max_window:
                kv_cache.truncate(1, front=False)  # Eject oldest frame

        # Disable decoding mode if it was enabled
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'disable_decoding'):
            model.transformer.disable_decoding()

        # Concatenate all latents
        result = torch.cat(latents, dim=1)
        
        # Return only generated frames if requested
        if self.only_return_generated:
            result = result[:, init_len:]
        
        # Decode if decoder function provided
        if decode_fn is not None:
            # Scale back if needed
            if vae_scale != 1.0:
                result = result * vae_scale
            
            # Decode to video
            videos = decode_fn(result)
            return videos, None, None  # Return format compatible with existing code
        
        return result, None, None  # Return format compatible with existing code


class SimpleTekkenSamplerV2:
    """
    Simplified version of TekkenCachingSamplerV2 for debugging.
    Follows the exact pattern of SimpleTekkenSampler but with v2 improvements.
    """
    def __init__(
        self, 
        n_steps: int = 16, 
        cfg_scale: float = 1.0, 
        num_frames: int = 160, 
        noise_prev: float = 0.2, 
        only_return_generated: bool = False,
        **kwargs
    ):
        self.n_steps = n_steps
        self.num_frames = num_frames
        self.noise_prev = noise_prev
        self.only_return_generated = only_return_generated

    @staticmethod
    def zlerp(x, alpha):
        """Linear interpolation with noise."""
        z = torch.randn_like(x)
        return x * (1. - alpha) + z * alpha

    def denoise_frame(self, model, kv_cache, prev_x, prev_actions, curr_actions, dt):
        """Simplified denoising following SimpleTekkenSampler exactly."""
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
        kv_cache.truncate(1, front=True)  # Remove new frame from cache

        # Euler update for step-0 (affects only the last frame)
        new_vid -= eps_v[:, -1:] * dt[0]
        t_new -= dt[0]

        # Remaining diffusion steps with cached history
        for step in range(1, self.n_steps):
            eps_vid = model(new_vid, t_new, curr_actions, kv_cache=kv_cache)
            new_vid -= eps_vid * dt[step]
            t_new -= dt[step]

        return new_vid

    @torch.no_grad()
    def __call__(self, model, initial_latents, action_ids, decode_fn=None, vae_scale=1.0):
        """Generate frames using simplified caching logic."""
        batch_size, init_len = initial_latents.size(0), initial_latents.size(1)
        
        dt = get_sd3_euler(self.n_steps).to(device=initial_latents.device, dtype=initial_latents.dtype)

        kv_cache = KVCache(model.config)
        kv_cache.reset(batch_size)

        latents = [initial_latents]
        prev_latents = initial_latents
        prev_actions = action_ids[:, :init_len]

        num_frames = min(self.num_frames, action_ids.size(1) - init_len)

        for idx in tqdm(range(num_frames), desc="Sampling frames"):
            start = init_len + idx
            curr_actions = action_ids[:, start:start+1]

            new_latent = self.denoise_frame(
                model, kv_cache, prev_latents, prev_actions, curr_actions, dt
            )

            latents.append(new_latent)
            
            # Update context with sliding window
            prev_latents = torch.cat([prev_latents[:, 1:], new_latent], dim=1)
            prev_actions = torch.cat([prev_actions[:, 1:], curr_actions], dim=1)

        result = torch.cat(latents, dim=1)
        
        if self.only_return_generated:
            result = result[:, init_len:]
        
        if decode_fn is not None:
            if vae_scale != 1.0:
                result = result * vae_scale
            videos = decode_fn(result)
            return videos, None, None
        
        return result, None, None
