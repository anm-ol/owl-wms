import torch
from tqdm import tqdm
from .schedulers import get_sd3_euler

class TekkenSampler:
    """
    A sampler for the TekkenRFT model that generates video frames sequentially
    using classifier-free guidance.

    Args:
        n_steps (int): The number of denoising steps for each frame.
        cfg_scale (float): The scale for classifier-free guidance.
        num_frames (int): The total number of new frames to generate.
        window_length (int): The number of context frames the model sees.
        only_return_generated (bool): If True, returns only the new frames.
    """
    def __init__(self, n_steps=20, cfg_scale=1.3, num_frames=60, window_length=16, only_return_generated=False, **kwargs):
        self.n_steps = n_steps
        self.cfg_scale = cfg_scale
        self.num_frames = num_frames
        self.window_length = window_length
        self.only_return_generated = only_return_generated

    @torch.no_grad()
    def __call__(self, model, initial_latents, action_ids, decode_fn=None, vae_scale=1.0):
        """
        Generates video frames using the provided model and action sequence.
        """
        model.eval()
        dt = get_sd3_euler(self.n_steps).to(device=initial_latents.device, dtype=initial_latents.dtype)
        generated_latents = initial_latents.clone()

        # Loop to generate each new frame
        for frame_idx in tqdm(range(self.num_frames), desc="Sampling Tekken Frames"):
            context_latents = generated_latents[:, -self.window_length + 1:]
            new_latent = torch.randn_like(initial_latents[:, :1])

            start_action_idx = generated_latents.shape[1] - self.window_length + 1
            end_action_idx = generated_latents.shape[1] + 1
            current_actions = action_ids[:, start_action_idx:end_action_idx]

            # Denoising loop for the new frame
            for step_idx in range(self.n_steps):
                t = 1.0 - (step_idx / self.n_steps)
                ts = torch.full((new_latent.shape[0], self.window_length), t, device=new_latent.device, dtype=new_latent.dtype)
                
                model_input = torch.cat([context_latents, new_latent], dim=1)
                
                # --- Classifier-Free Guidance ---
                has_controls_uncond = torch.zeros(model_input.shape[0], device=model_input.device, dtype=torch.bool)
                pred_uncond = model(model_input, ts, current_actions, has_controls=has_controls_uncond)

                has_controls_cond = torch.ones(model_input.shape[0], device=model_input.device, dtype=torch.bool)
                pred_cond = model(model_input, ts, current_actions, has_controls=has_controls_cond)
                
                pred_velocity = pred_uncond[:, -1:] + self.cfg_scale * (pred_cond[:, -1:] - pred_uncond[:, -1:])
                new_latent = new_latent - pred_velocity * dt[step_idx]

            generated_latents = torch.cat([generated_latents, new_latent], dim=1)

        if self.only_return_generated:
            final_latents = generated_latents[:, initial_latents.shape[1]:]
            final_actions = action_ids[:, initial_latents.shape[1]:initial_latents.shape[1] + self.num_frames]
        else:
            final_latents = generated_latents
            final_actions = action_ids[:, :generated_latents.shape[1]]
        # print(f"Final latents shape: {final_latents.shape}, Final actions shape: {final_actions.shape}")
        video_out = decode_fn(final_latents * vae_scale) if decode_fn is not None else None
        
        return video_out, final_latents, final_actions
