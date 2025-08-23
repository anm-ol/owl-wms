import torch
from torch import nn
import torch.nn.functional as F
import einops as eo

from ..nn.embeddings import TimestepEmbedding
from ..nn.attn import DiT, FinalLayer

class TekkenRFTCore(nn.Module):
    """
    A Rectified Flow Transformer core adapted for Tekken, which uses discrete action IDs
    and latents from a temporally-compressing VAE like LTX-Video.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # The transformer backbone remains the same.
        assert config.backbone == "dit"
        self.transformer = DiT(config)

        # Timestep embedding for the diffusion process.
        self.t_embed = TimestepEmbedding(config.d_model)

        # --- KEY CHANGE ---
        # Replace separate mouse/button embeddings with a single embedding for discrete action IDs.
        # The number of actions is 256, based on the 8-button binary combinations.
        self.action_embed = nn.Embedding(config.n_actions, config.d_model)

        # Projection layers for mapping latents to/from the transformer's dimension.
        self.proj_in = nn.Linear(config.channels, config.d_model, bias=False)
        self.proj_out = FinalLayer(config.sample_size, config.d_model, config.channels)

        # The number of tokens per latent frame is its spatial area (H * W).
        self.config.tokens_per_frame = self.config.sample_size[0] * self.config.sample_size[1]
        self.uncond = config.uncond

    def forward(self, x, t, action_ids, has_controls=None, kv_cache=None):
        """
        Args:
            x (torch.Tensor): Input latent tensor of shape [B, T, C, H, W].
            t (torch.Tensor): Timestep tensor of shape [B, T].
            action_ids (torch.Tensor): Action IDs of shape [B, T].
            has_controls (torch.Tensor, optional): Mask for classifier-free guidance. Defaults to None.
            kv_cache (optional): Key-value cache for faster inference. Defaults to None.
        """
        b, n, c, h, w = x.shape
        
        # Create conditioning signal from timestep and action embeddings.
        t_cond = self.t_embed(t)

        if not self.uncond:
            action_cond = self.action_embed(action_ids)  # [B, T] -> [B, T, D]
            if has_controls is not None:
                # Zero out embeddings where has_controls is False for CFG.
                action_cond = torch.where(has_controls[:, None, None], action_cond, torch.zeros_like(action_cond))
            cond = t_cond + action_cond
        else:
            cond = t_cond

        # Reshape latent from [B, T, C, H, W] to a sequence of tokens [B, T*H*W, C]
        x = eo.rearrange(x, 'b n c h w -> b (n h w) c')

        # Project tokens into the transformer's dimension and run through the model.
        x = self.proj_in(x)
        x = self.transformer(x, cond, kv_cache)
        x = self.proj_out(x, cond)

        # Reshape back to the original latent format.
        x = eo.rearrange(x, 'b (n h w) c -> b n c h w', h=h, w=w)
        return x

class TekkenRFT(nn.Module):
    """Wrapper for the TekkenRFTCore that handles the rectified flow noise and loss calculation."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.core = TekkenRFTCore(config)

    def handle_cfg(self, has_controls=None, cfg_prob=None):
        # This function remains the same, used for classifier-free guidance.
        if cfg_prob is not None:
            print(f'cfg prob: {cfg_prob.shape}')
        
        if cfg_prob is None:
            cfg_prob = self.config.cfg_prob
        elif cfg_prob <= 0.0 or has_controls is None:
            return has_controls
        
        pct_without = 1.0 - has_controls.float().mean()
        if pct_without < cfg_prob:
            needed = cfg_prob - pct_without
            needed_frac = needed / has_controls.float().mean()
            b = has_controls.shape[0]
            mask = (torch.rand(b, device=has_controls.device) <= needed_frac) & has_controls
            has_controls = has_controls & (~mask)
        return has_controls

    def noise(self, tensor, ts):
        z = torch.randn_like(tensor)
        # Linear interpolation between clean data and noise (Rectified Flow)
        lerp = tensor * (1 - ts) + z * ts
        # The target for the model is the velocity (noise - clean_data)
        target = z - tensor
        return lerp, target

    def forward(self, x, action_ids=None, cfg_prob=None, has_controls=None):
        B, S = x.size(0), x.size(1)
        
        if has_controls is None:
            has_controls = torch.ones(B, device=x.device, dtype=torch.bool)
        if action_ids is None:
            # If no actions are provided, treat all samples as unconditional.
            has_controls = torch.zeros_like(has_controls)

        # Apply classifier-free guidance dropout.
        # has_controls = self.handle_cfg(has_controls, cfg_prob)
        
        with torch.no_grad():
            # Sample random timesteps and create noisy inputs and targets.
            ts = torch.randn(B, S, device=x.device, dtype=x.dtype).sigmoid()
            lerpd_video, target_video = self.noise(x, ts[:, :, None, None, None])

        # Get the model's prediction.
        pred_video = self.core(lerpd_video, ts, action_ids, has_controls)
        
        # Calculate the mean squared error loss.
        loss = F.mse_loss(pred_video, target_video)
        return loss
