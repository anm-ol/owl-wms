import torch
from torch import nn
import torch.nn.functional as F
import einops as eo

from ..nn.dit_v3 import DiT_V3
from ..nn.embeddings import TimestepEmbedding, ActionEmbedding
from ..nn.attn import FinalLayer

def action_id_to_buttons(action_id: torch.Tensor):
    """
    Convert action ID tensor to 8-bit button representation.
    
    Args:
        action_id (torch.Tensor): Tensor of shape [..., 1] or [...] containing action IDs.
        
    Returns:
        torch.Tensor: Button presses tensor of shape [..., 8].
    """
    # Ensure action_id is unsqueezed if it's flat
    if action_id.dim() == 2: # [B, T]
        action_expanded = action_id.unsqueeze(-1)
    else: # [B, T, M] where M is memory size
        action_expanded = action_id
        
    bit_positions = torch.arange(8, device=action_id.device, dtype=action_id.dtype)
    buttons = (action_expanded.unsqueeze(-1) >> bit_positions) & 1
    
    return buttons.int()

class TekkenRFTCoreV3(nn.Module):
    """
    The core of the Tekken World Model v3, using a DiT backbone with cross-attention
    to a memory of past actions.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_buttons = 8

        # The new transformer backbone with cross-attention blocks
        self.transformer = DiT_V3(config)

        # Embedding layers
        self.t_embed = TimestepEmbedding(config.d_model)
        self.action_embed = ActionEmbedding(config.n_buttons, config.d_model)
        
        # A learnable positional embedding for the action memory sequence
        self.action_pos_embed = nn.Parameter(torch.randn(1, config.action_memory_size, config.d_model))

        # Projection layers for video latents
        self.proj_in = nn.Linear(config.channels, config.d_model, bias=False)
        self.proj_out = FinalLayer(config.sample_size, config.d_model, config.channels)

        # Calculate tokens per frame for reshaping
        self.tokens_per_frame = self.config.sample_size[0] * self.config.sample_size[1]
        self.uncond = config.uncond

    def forward(self, x, t, button_presses_history, has_controls=None, kv_cache=None):
        """
        Args:
            x (torch.Tensor): Input latent tensor of shape [B, T, C, H, W].
            t (torch.Tensor): Timestep tensor of shape [B, T].
            button_presses_history (torch.Tensor): A history of button presses of shape [B, T, N_buttons].
            has_controls (torch.Tensor, optional): Mask for classifier-free guidance.
        """
        b, n, c, h, w = x.shape
        
        # 1. Create conditioning signal for AdaLN layers from timesteps
        t_cond = self.t_embed(t) # [B, T, D_model]

        # 2. Create the action context for cross-attention
        action_tokens = self.action_embed(button_presses_history) # [B, T, N_buttons, D]
        action_context = action_tokens.sum(dim=2)                 # Sum over simultaneous presses -> [B, T, D]
        action_context = action_context + self.action_pos_embed   # Add positional encoding

        # 3. Prepare video patch tokens
        x_tokens = eo.rearrange(x, 'b t c h w -> b t (h w) c')
        x_tokens = self.proj_in(x_tokens) # [B, T, H*W, D_model]

        # --- Reshape for Transformer ---
        # The transformer expects a single sequence, so we flatten the time dimension.
        x_tokens = eo.rearrange(x_tokens, 'b t s d -> (b t) s d')
        action_context = eo.rearrange(action_context, 'b t m d -> (b t) m d')
        cond = eo.rearrange(t_cond, 'b t d -> (b t) 1 d').expand(-1, self.tokens_per_frame, -1)
        
        # 4. Pass through the transformer
        processed_tokens = self.transformer(x_tokens, action_context, cond, kv_cache=kv_cache)

        # 5. Project back to latent space
        output_latents = self.proj_out(processed_tokens, cond)

        # Reshape back to the original video latent format
        output = eo.rearrange(output_latents, '(b t) (h w) c -> b t c h w', b=b, t=n, h=h, w=w)
        return output

class TekkenRFTV3(nn.Module):
    """
    Top-level wrapper for the TekkenRFTV3 model that handles the rectified flow process
    (adding noise and calculating loss).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.core = TekkenRFTCoreV3(config)

    def noise(self, tensor, ts):
        z = torch.randn_like(tensor)
        lerp = tensor * (1 - ts) + z * ts # Linear interpolation
        target = z - tensor              # Velocity target
        return lerp, target

    def forward(self, x, action_ids_history, has_controls=None):
        B, S = x.size(0), x.size(1)
        
        if has_controls is None:
            has_controls = torch.ones(B, device=x.device, dtype=torch.bool)
        
        # Convert action ID history to one-hot button presses
        button_presses = action_id_to_buttons(action_ids_history)

        with torch.no_grad():
            # Sample random timesteps and create noisy inputs and targets
            ts = torch.randn(B, S, device=x.device, dtype=x.dtype).sigmoid()
            lerpd_video, target_video = self.noise(x, ts[:, :, None, None, None])

        # Get the model's prediction
        pred_video = self.core(lerpd_video, ts, button_presses, has_controls)
        
        # Calculate the mean squared error loss
        loss = F.mse_loss(pred_video, target_video)
        return loss
