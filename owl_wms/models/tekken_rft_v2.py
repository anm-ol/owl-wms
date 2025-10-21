import torch
from torch import nn
import torch.nn.functional as F
import einops as eo

from ..nn.embeddings import TimestepEmbedding, ActionEmbedding
from ..nn.attn import DiT, FinalLayer

class TekkenRFTCoreV2(nn.Module):
    """
    A Rectified Flow Transformer core for Tekken that uses cross-attention between
    action embeddings and video patch tokens.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_buttons = 8

        assert config.backbone == "dit", "This model requires a DiT backbone."
        self.transformer = DiT(config)

        # Timestep embedding for the diffusion process.
        self.t_embed = TimestepEmbedding(config.d_model)

        # Use the new disentangled ActionEmbedding module.
        # Note: The config must now include `n_buttons`.
        self.action_embed = ActionEmbedding(config.n_buttons, config.d_model)

        # Projection layers for mapping latents to/from the transformer's dimension.
        self.proj_in = nn.Linear(config.channels, config.d_model, bias=False)
        self.proj_out = FinalLayer(config.sample_size, config.d_model, config.channels)

        # The number of tokens per latent frame is its spatial area (H * W).
        self.config.tokens_per_frame = self.config.sample_size[0] * self.config.sample_size[1]
        self.uncond = config.uncond

    def forward(self, x, t, button_presses, has_controls=None, kv_cache=None):
        """
        Args:
            x (torch.Tensor): Input latent tensor of shape [B, T, C, H, W].
            t (torch.Tensor): Timestep tensor of shape [B, T].
            button_presses (torch.Tensor): A multi-hot tensor of button presses of shape [B, T, N_buttons].
            has_controls (torch.Tensor, optional): Mask for classifier-free guidance.
            kv_cache (optional): Key-value cache for faster inference.
        """
        b, n, c, h, w = x.shape
        
        # Time embedding is now the sole conditioning signal for AdaLN layers.
        t_cond = self.t_embed(t)  # [B, T, D_model]

        # Generate action embeddings from button presses.
        action_tokens = self.action_embed(button_presses)  # [B, T, 8, D_model]
        action_emb = action_tokens.mean(dim=2)  # [B, T, D_model]

        # if not self.uncond and has_controls is not None:
            # Zero out embeddings where has_controls is False for CFG.
            # action_emb = torch.where(has_controls[:, None, None], action_emb, torch.zeros_like(action_emb))

        cond_emb = t_cond + action_emb  # [B, T, D_model]

        # Reshape latents into a sequence of patch tokens.
        x_tokens = eo.rearrange(x, 'b t c h w -> b t (h w) c')
        x_tokens = self.proj_in(x_tokens)  # [B, T, H*W, D_model]

        # Prepend the action embedding as a special "action token" to each frame's sequence.
        # combined_tokens = torch.cat([x_tokens, action_tokens], dim=2) # [B, T, 8 + H*W, D_model]

        # Flatten the sequence for the transformer.
        b, t, s, d = x_tokens.shape
        transformer_input = x_tokens.view(b, t * s, d)
        # The AdaLN conditioning signal is just the time embedding, repeated for each token.
        # Expand t_cond to match the flattened sequence length
        cond = cond_emb.unsqueeze(2).expand(b, t, s, d).contiguous().view(b, t * s, d)

        # Pass the combined sequence through the transformer.
        processed_tokens = self.transformer(transformer_input, cond, kv_cache)

        # Separate the processed action tokens from the video tokens.
        processed_tokens = processed_tokens.view(b, t, s, d)
        # _processed_action_tokens = processed_tokens[:, :, 0, :] # We can discard this.
        # processed_video_tokens = processed_tokens[:, :, self.n_buttons:, :]

        # Reshape video tokens and project them back to the latent space.
        # processed_video_tokens = processed_tokens.reshape(b, t * (s - self.n_buttons), d)
        processed_video_tokens = processed_tokens.reshape(b, t * s, d)

        # Adjust conditioning shape for the final projection layer.
        video_cond = cond_emb.unsqueeze(2).expand(b, t, s, d).contiguous().view(b, t * s, d)
        # video_cond = t_cond.unsqueeze(2).expand(b, t, (s - self.n_buttons), d).contiguous().view(b, t * (s - self.n_buttons), d)
        output_latents = self.proj_out(processed_video_tokens, video_cond)

        # Reshape back to the original latent format [B, T, C, H, W].
        output = eo.rearrange(output_latents, 'b (t h w) c -> b t c h w', t=t, h=h, w=w)
        return output

class TekkenRFTV2(nn.Module):
    """Wrapper for the TekkenRFTV2 that handles rectified flow."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.core = TekkenRFTCoreV2(config)

    def handle_cfg(self, has_controls=None, cfg_prob=None):
        if cfg_prob is None:
            cfg_prob = self.config.cfg_prob
        if cfg_prob <= 0.0 or has_controls is None:
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
        lerp = tensor * (1 - ts) + z * ts
        target = z - tensor
        return lerp, target

    def forward(self, x, action_ids=None, cfg_prob=None, has_controls=None):
        B, S = x.size(0), x.size(1)
        
        if has_controls is None:
            has_controls = torch.ones(B, device=x.device, dtype=torch.bool)
        if action_ids is None:
            has_controls = torch.zeros_like(has_controls)
            # Create a dummy tensor if none is provided.
            button_presses = torch.zeros(B, S, self.config.n_buttons, device=x.device, dtype=torch.float)
        else:
            button_presses = action_id_to_buttons(action_ids) # (b, t) -> (b, t, 8)

        has_controls = self.handle_cfg(has_controls, cfg_prob)
        
        with torch.no_grad():
            ts = torch.randn(B, S, device=x.device, dtype=x.dtype).sigmoid()
            lerpd_video, target_video = self.noise(x, ts[:, :, None, None, None])

        
        pred_video = self.core(lerpd_video, ts, button_presses, has_controls)
        
        loss = F.mse_loss(pred_video, target_video)
        return loss
    

def action_id_to_buttons(action_id: torch.Tensor):
    """Convert action ID tensor to 8-bit button representation.
    
    Args:
        action_id (torch.Tensor): Tensor of shape [B, N, 1] containing action IDs
        
    Returns:
        torch.Tensor: Button presses tensor of shape [B, N, 8]
    """
    
    # Create a tensor for bit positions [0, 1, 2, 3, 4, 5, 6, 7]
    bit_positions = torch.arange(8, device=action_id.device, dtype=action_id.dtype)
    
    # Expand dimensions for broadcasting: [B, N] -> [B, N, 1] and [8] -> [1, 1, 8]
    action_expanded = action_id.unsqueeze(-1)  # [B, N, 1]
    bit_positions = bit_positions.unsqueeze(0).unsqueeze(0)  # [1, 1, 8]
    
    # Right shift action_id by each bit position and check the least significant bit
    buttons = (action_expanded >> bit_positions) & 1
    
    return buttons.int()

