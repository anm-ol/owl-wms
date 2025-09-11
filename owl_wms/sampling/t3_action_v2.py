import torch
from tqdm import tqdm
import gc

from ..nn.kv_cache import KVCache
from ..models.tekken_rft_v2 import action_id_to_buttons
from .schedulers import get_sd3_euler
from .av_caching_v2 import get_deltas

class TekkenActionCachingV2:
    """
    An efficient sampler for the TekkenRFT model that uses KV caching to speed up
    sequential frame generation, based on AVCachingSamplerV2 but using action IDs.

    Args:
        n_steps (int): Number of diffusion steps for each frame
        num_frames (int): Number of new frames to sample
        noise_prev (float): Noise previous frame
        max_window (int): Maximum context window size
        custom_schedule (list): Custom noise schedule
        only_return_generated (bool): If True, returns only the newly generated frames
    """
    def __init__(self, n_steps: int = 16, num_frames: int = 60, 
                 noise_prev: float = 0.2, max_window=None, custom_schedule=None, 
                 only_return_generated: bool = False, **kwargs):
        self.n_steps = n_steps
        self.num_frames = num_frames
        self.noise_prev = noise_prev
        self.max_window = max_window
        self.custom_schedule = custom_schedule
        self.only_return_generated = only_return_generated

    @staticmethod
    def zlerp(x, alpha):
        z = torch.randn_like(x)
        return x * (1. - alpha) + z * alpha

    @torch.no_grad()
    def __call__(self, model, x, action_ids, compile_on_decode=False, decode_fn=None, 
                 means=None, stds=None, vae_scale=1.0):
        batch_size, init_len = x.size(0), x.size(1)

        if self.custom_schedule is None:
            dt = get_sd3_euler(self.n_steps).to(device=x.device, dtype=x.dtype)
        else:
            dt = get_deltas(self.custom_schedule)

        kv_cache = KVCache(model.config)
        kv_cache.reset(batch_size)

        # At the start this is the whole video
        latents = [x.clone()]
        prev_x = x
        prev_actions = action_ids[:, :init_len]

        # ==== STEP 1: Cache context ====

        prev_x_noisy = self.zlerp(prev_x, self.noise_prev)
        prev_t = prev_x.new_full((batch_size, prev_x.size(1)), self.noise_prev)

        # Convert action IDs to button presses for caching
        prev_actions_buttons = action_id_to_buttons(prev_actions)

        kv_cache.enable_cache_updates()
        _ = model(
            prev_x_noisy,
            prev_t,
            prev_actions_buttons,
            kv_cache=kv_cache
        )
        kv_cache.disable_cache_updates()

        def new_xt():
            return torch.randn_like(prev_x[:,:1]), prev_t.new_ones(batch_size, 1)

        # START FRAME LOOP
        num_frames = min(self.num_frames, action_ids.size(1) - init_len)

        if compile_on_decode:
            model = torch.compile(model)

        model.transformer.enable_decoding()
        
        for idx in tqdm(range(num_frames), desc="Sampling Tekken Frames..."):
            curr_x, curr_t = new_xt()

            # Only pass the current frame's action - KV cache handles the context
            current_frame_action = action_ids[:, init_len + idx:init_len + idx + 1]
            current_frame_action_buttons = action_id_to_buttons(current_frame_action)
            
            print(f"Frame {idx}: current action shape={current_frame_action.shape}, "
                  f"kv_cache_frames={kv_cache.n_frames()}")

            # ==== STEP 2: Denoise the new frame ====
            for t_idx in range(self.n_steps):
                pred_v = model(
                    curr_x,
                    curr_t,
                    current_frame_action_buttons,
                    kv_cache=kv_cache
                )

                curr_x = curr_x - dt[t_idx] * pred_v   
                curr_t = curr_t - dt[t_idx]

            # ==== STEP 3: New frame generated, append and cache ====
            latents.append(curr_x.clone())
            curr_x_noisy = self.zlerp(curr_x, self.noise_prev)
            curr_t_noisy = torch.ones_like(curr_t) * self.noise_prev

            # For caching, we only need the current frame's action
            current_frame_action = action_ids[:, init_len + idx:init_len + idx + 1]
            current_frame_action_buttons = action_id_to_buttons(current_frame_action)

            kv_cache.enable_cache_updates()
            _ = model(
                curr_x_noisy,
                curr_t_noisy,
                current_frame_action_buttons,
                kv_cache=kv_cache
            )
            kv_cache.disable_cache_updates()
            
            if self.max_window is not None and len(latents) > self.max_window:
                kv_cache.truncate(1, front=False)  # Eject oldest

            gc.collect()
            torch.cuda.empty_cache()

        model.transformer.disable_decoding()

        generated_latents = torch.cat(latents, dim=1)

        if self.only_return_generated:
            final_latents = generated_latents[:, init_len:]
            final_actions = action_ids[:, init_len : init_len + self.num_frames]
        else:
            final_latents = generated_latents
            final_actions = action_ids[:, :generated_latents.shape[1]]

        # Handle normalization/scaling
        if means is not None and stds is not None:
            means = means.to(device=final_latents.device, dtype=final_latents.dtype)
            stds = stds.to(device=final_latents.device, dtype=final_latents.dtype)
            final_latents = final_latents * stds + means
        else:
            final_latents = final_latents * vae_scale
            
        video_out = decode_fn(final_latents) if decode_fn is not None else None
        
        return video_out, final_latents, final_actions