import torch
from torch import nn
import torch.nn.functional as F

from ..nn.embeddings import TimestepEmbedding, ControlEmbedding
from ..nn.attn import DiT, FinalLayer

import einops as eo
from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()


class GameRFTCore(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        assert config.backbone == "dit"
        self.transformer = DiT(config)

        if not config.uncond:
            self.control_embed = ControlEmbedding(config.n_buttons, config.d_model)
        self.t_embed = TimestepEmbedding(config.d_model)

        patch_size = getattr(config, "patch_size", [1, 1, 1])
        patch_stride = getattr(config, "patch_stride", patch_size)  # currently unsupported
        assert patch_size[0] == patch_stride[0] == 1, "Temporal patching not supported; use (1,*,*)."
        assert patch_size[1:] == patch_stride[1:], "For clean unpatchify, set stride==kernel on H/W."

        self.proj_in = nn.Conv3d(
            config.channels, config.d_model, kernel_size=patch_size, stride=patch_stride, bias=False)
        self.proj_out = FinalLayer(
            config.d_model, config.channels, kernel_size=patch_size, stride=patch_stride, bias=True)

        self.uncond = config.uncond

    def forward(self, x, t, mouse=None, btn=None, doc_id=None, has_controls=None, kv_cache=None):
        """
        x: [B, N, C, H, W], t: [B, N]
        """
        B, N, C, H, W = x.shape

        # per-frame conditioning
        cond = self.t_embed(t)  # [B, N, d]
        if not self.uncond:
            ctrl = self.control_embed(mouse, btn)  # [B, N, d]
            if has_controls is not None:
                ctrl = torch.where(has_controls[:, None, None], ctrl, torch.zeros_like(ctrl))
            cond = cond + ctrl

        # patchify
        x = self.proj_in(eo.rearrange(x, 'b n c h w -> b c n h w'))      # [B, D, N, H2, W2]
        B, D, N2, H2, W2 = x.shape
        assert N2 == N, "Temporal size must be preserved (patch_t=1)."
        assert self.config.tokens_per_frame == H2 * W2, \
            f"tokens_per_frame={self.config.tokens_per_frame}, got {H2 * W2}"

        tokens = eo.rearrange(x, 'b d n h w -> b (n h w) d')             # [B, N*H2*W2, D]
        tokens = self.transformer(tokens, cond, doc_id, kv_cache)
        x = eo.rearrange(tokens, 'b (n h w) d -> b d n h w', n=N2, h=H2, w=W2)

        # unpatchify
        x = self.proj_out(x, cond)
        return eo.rearrange(x, 'b c n h w -> b n c h w')



class GameRFT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.core = GameRFTCore(config)

    def handle_cfg(self, has_controls=None, cfg_prob=None):
        if cfg_prob is None:
            cfg_prob = self.config.cfg_prob
        if cfg_prob <= 0.0 or has_controls is None:
            return has_controls

        # Calculate current percentage without controls
        pct_without = 1.0 - has_controls.float().mean()

        # Only apply CFG if we need more negatives
        if pct_without < cfg_prob:
            # Calculate how many more we need
            needed = cfg_prob - pct_without
            needed_frac = needed / has_controls.float().mean()

            # Only drop controls where has_controls is True
            b = has_controls.shape[0]
            mask = (torch.rand(b, device=has_controls.device) <= needed_frac) & has_controls

            # Update has_controls based on mask
            has_controls = has_controls & (~mask)

        return has_controls

    def noise(self, tensor, ts):
        z = torch.randn_like(tensor)
        lerp = tensor * (1 - ts) + z * ts
        return lerp, z - tensor, z

    def forward(self, x, mouse=None, btn=None, doc_id=None, return_dict=False, cfg_prob=None, has_controls=None):
        B, S = x.size(0), x.size(1)
        if has_controls is None:
            has_controls = torch.ones(B, device=x.device, dtype=torch.bool)
        if mouse is None or btn is None:
            has_controls = torch.zeros_like(has_controls)

        # Apply classifier-free guidance dropout
        has_controls = self.handle_cfg(has_controls, cfg_prob)
        with torch.no_grad():
            ts = torch.randn(B, S, device=x.device, dtype=x.dtype).sigmoid()
            lerpd_video, target_video, z_video = self.noise(x, ts[:, :, None, None, None])

        pred_video = self.core(lerpd_video, ts, mouse, btn, doc_id, has_controls)
        loss = F.mse_loss(pred_video, target_video)

        if not return_dict:
            return loss
        else:
            return {
                'diffusion_loss': loss,
                'video_loss': loss,
                'lerpd_video': lerpd_video,
                'pred_video': pred_video,
                'ts': ts,
                'z_video': z_video,
                'cfg_mask': has_controls
            }
