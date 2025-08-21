from typing import Optional
from torch import Tensor

import torch
from tqdm import tqdm

from ..nn.kv_cache import KVCache
from .schedulers import get_sd3_euler, get_sd3_scheduler


class AVCachingSampler:
    """Causal video diffusion with per-step KV prefill and AR sampling."""

    def __init__(
        self,
        n_steps: int = 16,
        cfg_scale: float = 1.0,
        num_frames: int = 60,
        noise_prev=None  # TODO: remove
    ) -> None:
        if cfg_scale != 1.0:
            raise NotImplementedError("cfg_scale must be 1.0")
        self.n_steps = n_steps
        self.num_frames = num_frames
        self.fm_sched = get_sd3_scheduler(self.n_steps)

    @torch.no_grad()
    def __call__(self, model, x: torch.Tensor, mouse: Optional[Tensor] = None, btn: Optional[Tensor] = None):
        """
        Args:
            model: denoiser; signature model(x, t, mouse, btn, kv_cache=...).
            x:     [B, L0, C, H, W] clean prefix frames.
            mouse: optional control timeline [B, Lm, ...].
            btn:   optional control timeline [B, Lb, ...].

        Returns:
            Tensor [B, L0 + num_frames, C, H, W]
        """
        B, S = x.shape[:2]

        dt = get_sd3_euler(self.n_steps).to(device=x.device, dtype=x.dtype)
        pre_t = 1.0 - torch.cumsum(torch.cat([dt.new_zeros(1), dt[:-1]]), dim=0)
        pre_t = pre_t.clamp_(0, 1)

        # TODO: ####
        if not hasattr(self.fm_sched, "timesteps") or self.fm_sched.timesteps.numel() == 0:
            self.fm_sched.set_timesteps(self.n_steps, device=x.device)
        elif self.fm_sched.timesteps.device != x.device:
            self.fm_sched.set_timesteps(self.n_steps, device=x.device)
        ####

        # One KV cache per diffusion step; align device/dtype.
        kv_caches = [KVCache(model.config) for _ in range(self.n_steps)]
        for kc in kv_caches:
            kc.to(device=x.device, dtype=x.dtype).reset(B)
            kc.enable_cache_updates()

        # Normalize empty controls to None.
        if mouse is not None and mouse.size(1) == 0:
            mouse = None
        if btn is not None and btn.size(1) == 0:
            btn = None

        # 1) Prefill caches from the clean prefix.
        self.prefill_caches(model, x, mouse, btn, kv_caches, pre_t)

        # 2) Autoregressively sample new frames.
        latents = [x]
        for i in tqdm(range(self.num_frames), desc="Sampling frames"):
            # Clamp control indices to their last available frame.
            j_m = min(S + i, mouse.size(1) - 1) if mouse is not None else 0
            j_b = min(S + i, btn.size(1) - 1) if btn is not None else 0
            m = None if mouse is None else mouse[:, j_m:j_m + 1]
            b = None if btn is None else btn[:, j_b:j_b + 1]

            new_frame = self.denoise_one_frame(
                model=model,
                kv_caches=kv_caches,
                shape_like=x[:, :1],  # single-frame shape
                mouse_frame=m,
                btn_frame=b,
                dt=dt,
                pre_t=pre_t,
            )
            latents.append(new_frame)

        return torch.cat(latents, dim=1)

    def prefill_caches(
        self,
        model,
        x: torch.Tensor,
        mouse: Optional[Tensor],
        btn: Optional[Tensor],
        kv_caches: list,
        pre_t: torch.Tensor,
    ) -> None:
        """Re-noise clean prefix to each step's t and write its K/V to that step."""
        B, S = x.shape[:2]
        if S == 0:
            return

        prev_mouse = None if mouse is None else mouse[:, :S]
        prev_btn = None if btn is None else btn[:, :S]

        for s in range(self.n_steps):
            # Forward noising for flow matching:
            #   x_t = scale_noise(x0, timestep, eps)
            t_sched = self.fm_sched.timesteps[s]                  # scalar tensor
            timesteps = t_sched.expand(B).to(device=x.device)   # (B,)
            noise = torch.randn_like(x)
            x_t = self.fm_sched.scale_noise(x, timesteps, noise)  # <-- correct API
            t_arr = pre_t[s].to(device=x.device, dtype=x.dtype).expand(B, S)
            _ = model(x_t, t_arr, prev_mouse, prev_btn, kv_cache=kv_caches[s])

    def denoise_one_frame(
        self,
        model,
        kv_caches: list,
        shape_like: Tensor,
        mouse_frame: Optional[Tensor],
        btn_frame: Optional[Tensor],
        dt: torch.Tensor,
        pre_t: torch.Tensor,
    ) -> Tensor:
        """Denoise a single new frame and append its K/V at each step."""
        x_new = torch.randn_like(shape_like)
        B = x_new.size(0)
        for s in range(self.n_steps):
            t_arr = pre_t[s].to(device=x_new.device, dtype=x_new.dtype).expand(B, 1)
            eps = model(x_new, t_arr, mouse_frame, btn_frame, kv_cache=kv_caches[s])
            x_new = x_new - eps * dt[s]
        return x_new
