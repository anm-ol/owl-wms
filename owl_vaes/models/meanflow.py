"""
MeanFlow for latent image decoding (single image, no video/audio, r and t are [b,])
"""

import torch
from torch import nn
import torch.nn.functional as F

from copy import deepcopy

from ..nn.embeddings import TimestepEmbedding, LearnedPosEnc
from ..nn.dit import DiT, FinalLayer
from ..configs import TransformerConfig

from torch.nn.attention import SDPBackend, sdpa_kernel

class MeanFlowImageCore(nn.Module):
    def __init__(self, config : TransformerConfig):
        super().__init__()

        size = config.sample_size
        n_tokens = size[0] // config.patch_size * size[1] // config.patch_size

        self.proj_in = nn.Linear(config.patch_size * config.patch_size * config.channels, config.d_model, bias = False)
        self.pos_enc_x = LearnedPosEnc(n_tokens, config.d_model)
        self.proj_out = nn.Linear(config.d_model, config.patch_size * config.patch_size * config.channels, bias = False)

        self.ts_embed = TimestepEmbedding(config.d_model)
        self.r_embed = TimestepEmbedding(config.d_model)
        
        self.proj_in_z = nn.Linear(config.latent_channels, config.d_model)
        self.pos_enc_z = LearnedPosEnc(config.latent_size**2, config.d_model)
        
        self.final = FinalLayer(config, skip_proj = True)

        self.p = config.patch_size
        self.n_p_y = config.sample_size[0] // self.p
        self.n_p_x = config.sample_size[1] // self.p

        self.blocks = DiT(config)
        self.config = config

    def forward(self, x, z, ts, r = None):
        # x is [b,c,h,w]
        # z is [b,c,h,w] but different size cause latent
        # ts is [b,] in [0,1]
        # d is [b,] in [1,2,4,...,128]

        cond = self.ts_embed(ts)
        if r is None:
            r = torch.zeros_like(ts)
        cond = cond + self.r_embed(ts - r)

        # Convert from image format [b,c,h,w] to patches [b,n_patches,patch_size*patch_size*c]
        b, c, h, w = x.shape
        x = x.view(b, c, self.n_p_y, self.p, self.n_p_x, self.p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(b, self.n_p_y * self.n_p_x, self.p * self.p * c)
        x = self.proj_in(x) # -> [b,n,d]
        x = self.pos_enc_x(x)

        # Flatten spatial dimensions: [b,c,h,w] -> [b,h*w,c]
        b, c, h, w = z.shape
        z = z.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        z = self.proj_in_z(z)
        z = self.pos_enc_z(z)
        
        n = x.shape[1]
        x = torch.cat([x,z],dim=1)

        x = self.blocks(x, cond)
        x = x[:,:n]

        x = self.final(x, cond)
        x = self.proj_out(x)
        # Convert from patches back to image format [b,n_patches,patch_size*patch_size*c] -> [b,c,h,w]
        b, n_patches, patch_dim = x.shape
        c = patch_dim // (self.p * self.p)
        x = x.view(b, self.n_p_y, self.n_p_x, self.p, self.p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(b, c, self.n_p_y * self.p, self.n_p_x * self.p)

        return x

class MeanFlowImage(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.core = MeanFlowImageCore(config)
        self.ts_mu = -0.4
        self.ts_sigma = 1.0
        self.ts_ratio = 0.25  # percent to force r = t

    @torch.no_grad()
    def sample_timesteps(self, b, device, dtype):
        # Sample r and t in [0, 1], r < t, with some probability of r == t
        mu = self.ts_mu
        sigma = self.ts_sigma
        ratio = self.ts_ratio

        eq_mask = torch.rand(b, device=device, dtype=dtype) < ratio

        t_both = torch.randn(b, 2, device=device, dtype=dtype)
        t_both = t_both * sigma + mu
        t_both = t_both.sigmoid()

        t1 = t_both[:, 0]
        t2 = t_both[:, 1]
        lesser = t1 < t2
        r = torch.where(lesser, t1, t2)
        t = torch.where(~lesser, t1, t2)
        r = torch.where(eq_mask, t, r)
        return t, r

    def forward(self, x, z, return_dict=False):
        # x: [b, c, h, w]
        # z: [b, c_z, h_z, w_z]
        b, c, h, w = x.shape

        with torch.no_grad():
            t, r = self.sample_timesteps(b, x.device, x.dtype)
            z_noise = torch.randn_like(x)
            t_exp = t[:, None, None, None]
            noisy_x = x * (1. - t_exp) + z_noise * t_exp
            v = z_noise - x

            u_pred = torch.zeros_like(v)
            u_targ = torch.zeros_like(v)

        # Branch 1: r == t (instant velocity, no JVP)
        eq_mask = (r == t)
        if eq_mask.any():
            idx = torch.where(eq_mask)[0]
            noisy_x_eq = noisy_x[idx]
            t_eq = t[idx]
            r_eq = r[idx]
            v_eq = v[idx]
            z_eq = z[idx]

            with sdpa_kernel(SDPBackend.MATH):
                with torch.autocast(device_type='cuda', enabled=False):
                    u_pred_eq = self.core(noisy_x_eq, z_eq, t_eq, r=r_eq)
            u_pred[idx] = u_pred_eq
            u_targ[idx] = v_eq

        # Branch 2: r != t (JVP, standard case)
        neq_mask = ~eq_mask
        if neq_mask.any():
            idx = torch.where(neq_mask)[0]
            noisy_x_neq = noisy_x[idx]
            t_neq = t[idx]
            r_neq = r[idx]
            v_neq = v[idx]
            z_neq = z[idx]
            ts_diff = (t_neq - r_neq)[:, None, None, None]

            def fn(noisy_x_in, r_in, t_in, z_in):
                return self.core(noisy_x_in, z_in, t_in, r=r_in)

            # Convert to float32 for JVP computation to avoid type mismatch
            primals_f32 = (
                noisy_x_neq.detach().float(), 
                r_neq.float(), 
                t_neq.float(), 
                z_neq.detach().float()
            )
            tangents_f32 = (
                v_neq.detach().float(), 
                torch.zeros_like(r_neq).float(), 
                torch.ones_like(t_neq).float(), 
                torch.zeros_like(z_neq).float()
            )
            
            with sdpa_kernel(SDPBackend.MATH):
                # Temporarily disable autocast for JVP
                with torch.autocast(device_type='cuda', enabled=False):
                    (u_out_f32, dudt_out_f32) = torch.func.jvp(fn, primals_f32, tangents_f32)
            
            # Convert back to original dtype
            u_out = u_out_f32.to(x.dtype)
            dudt_out = dudt_out_f32.to(x.dtype)
            
            
            u_pred[idx] = u_out
            u_targ[idx] = (v_neq - dudt_out * ts_diff).detach()

        return F.mse_loss(u_pred, u_targ)

        #error = u_pred - u_targ
        #error_norm = torch.norm(error.reshape(b, -1), dim=1)
        #loss = error_norm ** 2

        #return loss.mean()