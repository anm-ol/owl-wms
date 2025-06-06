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

def sample_steps(b, device, dtype):
    valid = torch.tensor([2**i for i in range(0, 8)]) # [1,2,...,128]
    inds = torch.randint(low=0,high=len(valid), size = (b,))
    steps = valid[inds].to(device=device,dtype=dtype)
    return steps

class DiffusionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.core = DiffusionDecoderCore(config)
    
    
    def forward(self, x):
        with torch.no_grad():
            steps = sample_steps(len(x),x.device,x.dtype)
            ts = sample_discrete_timesteps(steps)

            z = torch.randn_like(x)
            ts_exp = ts.view(-1, 1, 1, 1).expand_as(x)

            lerpd = x * (1. - ts_exp) + ts_exp * z
            target = z - x

        pred = self.core(lerpd)

            
        



