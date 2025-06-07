import torch
from torch import nn
import torch.nn.functional as F
import einops as eo

from ..configs import TransformerConfig

from ..nn.dit import UViT, FinalLayer
from ..nn.embeddings import LearnedPosEnc
from ..nn.embeddings import TimestepEmbedding, StepEmbedding

class DiffusionDecoderCore(nn.Module):
    def __init__(self, config : TransformerConfig):
        super().__init__()

        patch_content = (config.patch_size ** 2) * config.channels

        self.d_embed = StepEmbedding(config.d_model)
        self.ts_embed = TimestepEmbedding(config.d_model)
        
        self.proj_in = nn.Linear(patch_content, config.d_model)
        self.proj_in_z = nn.Linear(config.latent_channels, config.d_model)
        self.proj_out = FinalLayer(config)

        self.p = config.patch_size
        self.n_p_y = config.sample_size[0] // self.p
        self.n_p_x = config.sample_size[1] // self.p

        self.blocks = UViT(config)
        self.config = config

    def sample(self, z):
        d = torch.full(len(z), 1., device = z.device, dtype = z.dtype)
        ts = torch.full(len(z), 1., device = z.device, dtype = z.dtype)
        x = torch.randn(len(z), self.config.channels, self.config.sample_size, self.config.sample_size)
        return self.forward(x, z, ts, d)

    def forward(self, x, z, ts, d):
        # x is [b,c,h,w]
        # z is [b,c,h,w] but different size cause latent
        # ts is [b,] in [0,1]
        # d is [b,] in [1,2,4,...,128]

        cond = self.ts_embed(ts) + self.d_embed(d)

        x = eo.rearrange(
            x,
            'b c (n_p_y p_y) (n_p_x p_x) -> b (n_p_y n_p_x) (p_y p_x c)',
            p_y=self.p,p_x=self.p
        )
        z = eo.rearrange(z, 'b c h w -> b (h w) c')

        x = self.proj_in(x)
        z = self.proj_in_z(z)

        n = x.shape[1]
        x = torch.cat([x,z],dim=1)
        x = self.blocks(x, cond)
        x = x[:,:n]

        x = self.proj_out(x)
        x = eo.rearrange(
            x,
            'b (n_p_y n_p_x) (p_y p_x c) -> b c (n_p_y p_y) (n_p_x p_x)',
            p_y = self.p, p_x = self.p,
            n_p_y = self.n_p_y, n_p_x = self.n_p_x
        )

        return x

def sample_discrete_timesteps(steps, eps = 1.0e-6):
    # steps is Tensor([1,4,2,64,16]) as an example
    b = len(steps)
    ts_list = []
    ts = torch.rand(b, device=steps.device, dtype=steps.dtype) * (steps - eps)
    ts = ts.clamp(eps).ceil() / steps
    """
    Example, if d was all 2, ts would be [0,2]
    so do clamp, then ceil will be 1 or 2 (0, 2]
    then do t / 2 and get 0.5 or 1.0, our desired timesteps
    """
    return ts

def sample_steps(b, device, dtype, min_val = 0):
    valid = torch.tensor([2**i for i in range(min_val, 8)]) # [1,2,...,128]
    inds = torch.randint(low=0,high=len(valid), size = (b,))
    steps = valid[inds].to(device=device,dtype=dtype)
    return steps

class DiffusionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.core = DiffusionDecoderCore(config)
        self.ema = None
        self.sc_frac = 0.25
        self.cfg_prob = 0.1
        self.cfg_strength = 1.5

    def set_ema_core(self, ema):
        if hasattr(ema.ema_model, 'module'):
            self.ema = ema.ema_model.module.core
        else:
            self.ema = ema.ema_model.core

    @torch.no_grad()
    @torch.compile()
    def get_sc_targets(self, x, z):
        steps_slow = sample_steps(len(x),x.device,x.dtype, min_val = 1)
        steps_fast = steps_slow / 2

        dt_slow = 1./steps_slow
        dt_fast = 1./steps_fast

        def expand(t):
            #b,c,h,w = x.shape
            #t = eo.repeat(t,'b -> b c h w',c=c,h=h,w=w)
            #return t
            return t[:,None,None,None]

        ts = sample_discrete_timesteps(steps_fast)
        cfg_mask = torch.isclose(steps_slow, 128)
        cfg_mask = expand(cfg_mask) # -> [b,1,1,1]

        pred_1_uncond = self.ema(x, torch.randn_like(z), ts, steps_slow)
        pred_1_cond = self.ema(x, z, ts, steps_slow)
        pred_1_cfg = pred_1_uncond + self.cfg_strength * (pred_1_cond - pred_1_uncond)
        pred_1 = torch.where(cfg_mask, pred_1_cfg, pred_1_cond)

        x_new = x - pred_1 * expand(dt_slow)
        ts_new = ts - dt_slow

        pred_2_uncond = self.ema(x_new, torch.randn_like(z), ts_new, steps_slow)
        pred_2_cond = self.ema(x_new, z, ts_new, steps_slow)
        pred_2_cfg = pred_2_uncond + self.cfg_strength * (pred_2_cond - pred_2_uncond)
        pred_2 = torch.where(cfg_mask, pred_2_cfg, pred_2_cond)

        pred = 0.5 * (pred_1 + pred_2)
        return pred, steps_fast, ts

    def get_sc_loss(self, x, z):
        target, steps, ts = get_sc_targets(x, z)
        pred = self.core(x, z, ts, steps)
        sc_loss = F.mse_loss(pred, target)
        return sc_loss
    
    def forward(self, x, z):
        b = len(x) * (1 - self.sc_frac)
        x,x_sc = x[:b], x[b:]
        z,z_sc = z[:b], z[b:]

        with torch.no_grad():
            steps = sample_steps(len(x),x.device,x.dtype)
            ts = sample_discrete_timesteps(steps)

            z = torch.randn_like(x)
            ts_exp = ts.view(-1, 1, 1, 1).expand_as(x)

            lerpd = x * (1. - ts_exp) + ts_exp * z
            target = z - x

        mask = torch.rand(len(z), device=z.device) < self.cfg_prob
        z_masked = torch.where(mask.view(-1, 1, 1, 1), torch.randn_like(z), z)

        pred = self.core(lerpd, z, ts, steps)
        diff_loss = F.mse_loss(pred, target)
        sc_loss = self.get_sc_loss(x_sc,z_sc)

        return diff_loss, sc_loss