import sys
import os

import torch
from diffusers import AutoencoderDC, AutoencoderKLLTXVideo, AutoencoderKLWan


sys.path.append("./owl-vaes")
from owl_vaes.utils.proxy_init import load_proxy_model
from owl_vaes.models import get_model_cls
from owl_vaes.configs import Config

class WanDecoder:
    """
    Minimal encode/decode wrapper for Diffusers' AutoencoderKLWan.

    encode(rgb)  : [T,3,H,W] | [B,3,T,H,W] | [B,T,3,H,W] -> [B,T,C,H,W]  (model space)
    decode(z)    : [B,T,C,H,W] | [B,C,T,H,W] -> [B,Tpix,3,H,W]           (pixels)
                    where Tpix = 1 + 4 * (T - 1)
    """
    def __init__(self, decoder, batch_size: int = 2, dtype=torch.float32):
        decoder = torch.compile(decoder)
        self.decoder = decoder.eval()

        self.bs  = int(batch_size)
        self.dt  = dtype
        cfg = vae.config
        self.sf  = float(getattr(cfg, "scaling_factor", 1.0))
        self.m   = getattr(cfg, "latents_mean", None)
        self.s   = getattr(cfg, "latents_std",  None)
        self.C   = int(getattr(cfg, "z_dim", getattr(cfg, "latent_channels", 16)))

    # ---------- helpers ----------
    def _dev(self):
        return next(self.vae.parameters()).device

    def _rgb_to_b3thw(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [T,3,H,W], [B,3,T,H,W], [B,T,3,H,W]  ->  [B,3,T,H,W]
        if x.ndim == 4:                     # [T,3,H,W]
            if x.shape[1] != 3: raise ValueError(f"Expected [T,3,H,W], got {tuple(x.shape)}")
            return x.permute(1, 0, 2, 3).unsqueeze(0)
        if x.ndim == 5 and x.shape[1] == 3: # [B,3,T,H,W]
            return x
        if x.ndim == 5 and x.shape[2] == 3: # [B,T,3,H,W]
            return x.permute(0, 2, 1, 3, 4).contiguous()
        raise ValueError(f"RGB must be [T,3,H,W] or [B,3,T,H,W] or [B,T,3,H,W]; got {tuple(x.shape)}")

    def _to_model_space(self, z_vae: torch.Tensor) -> torch.Tensor:
        if self.m is not None and self.s is not None:
            mean = torch.as_tensor(self.m, device=z_vae.device, dtype=z_vae.dtype).view(1, -1, 1, 1, 1)
            std  = torch.as_tensor(self.s, device=z_vae.device, dtype=z_vae.dtype).view(1, -1, 1, 1, 1)
            return (z_vae - mean) * (self.sf / std)
        return z_vae * self.sf

    def _to_vae_space(self, z_model: torch.Tensor) -> torch.Tensor:
        if self.m is not None and self.s is not None:
            mean = torch.as_tensor(self.m, device=z_model.device, dtype=z_model.dtype).view(1, -1, 1, 1, 1)
            std  = torch.as_tensor(self.s, device=z_model.device, dtype=z_model.dtype).view(1, -1, 1, 1, 1)
            return (z_model / self.sf) * std + mean
        return z_model / self.sf

    def _pad_to_4k_plus_1(self, z_bcthw: torch.Tensor) -> tuple[torch.Tensor, int]:
        # WAN VAE expects latent T such that output frames = 1 + 4*(T-1)
        T = z_bcthw.shape[2]
        target = ((T - 1 + 3) // 4) * 4 + 1
        if target == T:
            return z_bcthw, T
        pad = target - T
        z_pad = torch.cat([z_bcthw, z_bcthw[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
        return z_pad, T

    @torch.no_grad()
    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        RGB -> model-space latents, returning [B,T,C,H,W]
        """
        x = self._rgb_to_b3thw(rgb)
        # Optional normalization if dataset is uint8 [0,255]
        if x.dtype == torch.uint8:
            x = x.to(torch.float32).div_(127.5).sub_(1.0)
        x = x.to(self._dev(), dtype=self.dt)
        parts = []
        for x_chunk in x.split(self.bs, dim=0):
            z_vae = self.vae.encode(x_chunk, return_dict=True).latent_dist.sample()  # [b,C,T,H,W]
            parts.append(self._to_model_space(z_vae))
        z = torch.cat(parts, dim=0)  # [B,C,T,H,W]
        return z.permute(0, 2, 1, 3, 4).contiguous()  # [B,T,C,H,W]

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Model-space latents -> RGB, returning [B,Tpix,3,H,W]
        """
        # Normalize to [B,C,T,H,W] for VAE
        if z.ndim != 5:
            raise ValueError(f"Latents must be 5D, got {tuple(z.shape)}")
        # Deterministic: accept [B,T,C,H,W] or [B,C,T,H,W] -> [B,C,T,H,W]
        if z.shape[1] == self.C:
            z_bcthw = z
        elif z.shape[2] == self.C:
            z_bcthw = z.permute(0, 2, 1, 3, 4).contiguous()
        else:
            raise ValueError(f"Neither dim1 nor dim2 equals latent C={self.C}; got {tuple(z.shape)}")
        z_bcthw = z_bcthw.to(self._dev(), dtype=self.dt)

        # Pad to valid temporal length, convert to VAE space, decode
        z_bcthw, orig_T = self._pad_to_4k_plus_1(z_bcthw)
        parts = []
        for z_chunk in z_bcthw.split(self.bs, dim=0):
            z_in = self._to_vae_space(z_chunk.to(torch.float32))  # VAE in fp32
            pix = self.vae.decode(z_in, return_dict=True).sample  # [b,3,Tpix,H,W]
            parts.append(pix)
        y = torch.cat(parts, dim=0)  # [B,3,Tpix,H,W]

        # Trim to frames implied by original latent length and return [B,Tpix,3,H,W]
        want_Tpix = 1 + 4 * (orig_T - 1)
        if y.shape[2] >= want_Tpix:
            y = y[:, :, :want_Tpix]
        return y.permute(0, 2, 1, 3, 4).contiguous()

def _get_decoder_only():
    model = load_proxy_model(
        "../checkpoints/128x_proxy_titok.yml",
        "../checkpoints/128x_proxy_titok.pt",
        "../checkpoints/16x_dcae.yml",
        "../checkpoints/16x_dcae.pt"
    )
    del model.transformer.encoder
    return model

def get_decoder_only(vae_id, cfg_path, ckpt_path):
        if vae_id == "dcae":
            model_id = "mit-han-lab/dc-ae-f64c128-mix-1.0-diffusers"
            model = AutoencoderDC.from_pretrained(model_id).bfloat16().cuda().eval()
            del model.encoder
            return model.decoder
        if vae_id == "ltx":
            model = AutoencoderKLLTXVideo.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16).cuda().eval()
            del model.encoder
            return model.decoder
        if vae_id == "wan":
            model = AutoencoderKLWan.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16).cuda().eval()
            del model.encoder
            return model.decoder
        else:
            cfg = Config.from_yaml(cfg_path).model
            model = get_model_cls(cfg.model_id)(cfg)
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location='cpu',weights_only=False))
            except:
                model.decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu',weights_only=False))
            del model.encoder
            model = model.decoder
            model = model.bfloat16().cuda().eval()
            return model

@torch.no_grad()
def make_batched_decode_fn(decoder, batch_size = 8, temporal_vae=True):
    def decode(x):
        # x is [b,n,c,h,w]
        b,n,c,h,w = x.shape
        if not temporal_vae:
            x = x.view(b*n,c,h,w).contiguous()
        else:
            x = x.permute(0, 2, 1, 3, 4)
            x = x.contiguous()
        batches = x.split(batch_size)
        batch_out = []
        for batch in batches:
            batch_out.append(decoder(batch).bfloat16())

        x = torch.cat(batch_out) # [b*n,c,h,w]
        if not temporal_vae:
            _,c,h,w = x.shape
            x = x.view(b,n,c,h,w).contiguous()
        else:
            x = x.contiguous()

        return x
    return decode

@torch.no_grad()
def make_batched_audio_decode_fn(decoder, batch_size = 8):
    def decode(x):
        # x is [b,n,c] audio samples
        x = x.transpose(1,2)
        b,c,n = x.shape

        batches = x.contiguous().split(batch_size)
        batch_out = []
        for batch in batches:
            batch_out.append(decoder(batch).bfloat16())

        x = torch.cat(batch_out) # [b,c,n]
        x = x.transpose(-1,-2).contiguous() # [b,n,2]

        return x
    return decode