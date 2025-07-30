import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from .mlp import MLPSimple as MLP
from .attn import Attn as CosSimAttn # It brokey

from natten import NeighborhoodAttention2D
from copy import deepcopy

class HDiTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc_uv = nn.Linear(config.d_model, 8 * config.d_model)
        self.fc_out = nn.Linear(4 * config.d_model, config.d_model)

        self.act = nn.GELU()
        self.dp = nn.Dropout(config.dropout)

        nn.init.zeros_(self.fc_out.weight)
        
    def forward(self, x):
        res = x.clone()

        uv = self.fc_uv(x)
        u, v = uv.chunk(2, dim = -1)

        x = self.dp(u * v)
        x = res + self.fc_out(self.dp(u * v))

        return x

class AdaRMSParams(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.mlp = MLP(
            dim_in = config.d_model,
            dim_middle = 4 * config.d_model,
            dim_out = 2 * config.d_model
        )

    def forward(self, cond):
        # cond is [b,d]
        params = self.mlp(cond) # [b,2d]
        alpha, beta = params.chunk(2, dim = -1) # both [b,d]
        alpha = alpha[:,None,:]
        beta = beta[:,None,:] # [b,1,d] for n,h 

        return alpha, beta

class AdaRMS(nn.Module):
    def forward(self, x, gain):
        gain = (1. + gain)
        rms = (x.float().pow(2).mean(-1,keepdim=True)+1.0e-6).rsqrt().to(x.dtype)

        return x * rms * gain

def int_to_tuple(x):
    if isinstance(x, int):
        return (x,x)
    elif isinstance(x, tuple) or isinstance(x, list):
        return x
    else:
        try:
            return tuple(x)
        except:
            raise ValueError(f"Invalid input: {x}")

def get_reshape_params(config, sample_size = None):
    if sample_size is None:
        h, w = int_to_tuple(config.sample_size)
    else:
        h, w = int_to_tuple(sample_size)

    p_y, p_x = int_to_tuple(config.patch_size)
    n_p_y = h // p_y
    n_p_x = w // p_x
    n_image = n_p_y * n_p_x
    return n_p_y, n_p_x, n_image

class NeighbourAttn(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn = NeighborhoodAttention2D(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            kernel_size=7,  # 7x7 neighborhood window
            stride=1,
            dilation=1,
            is_causal=False
        )

        self.n_p_y, self.n_p_x, self.n_image = get_reshape_params(config)

    def forward(self, x):
        # x is [b,n,d]
        b,n,d = x.shape
        x = x.view(b,self.n_p_y,self.n_p_x,d).contiguous()
        x = self.attn(x)
        x = x.view(b,n,d).contiguous()
        return x

class HDiTBlock(nn.Module):
    def __init__(self, config, sample_size):
        super().__init__()

        self.norm_params = AdaRMSParams(config)
        self.norm1 = AdaRMS()
        self.norm2 = AdaRMS()

        config = deepcopy(config)
        config.sample_size = sample_size

        self.attn = CosSimAttn(config) if sample_size <= 64 else NeighbourAttn(config)
        self.mlp = HDiTMLP(config)

    def forward(self, x, cond):
        alpha, beta = self.norm_params(cond)

        res1 = x.clone()
        x = self.norm1(x, alpha)
        x = self.attn(x) + res1
        res2 = x.clone()
        x = self.norm2(x, beta)
        x = self.mlp(x) + res2
    
        return x

class HDiTUpSample(nn.Module):
    def __init__(self, config, sample_size):
        super().__init__()

        config = deepcopy(config)
        config.sample_size = sample_size

        self.proj = weight_norm(nn.Conv2d(config.d_model, config.d_model * 4, 1, 1, 0, bias = False))

        self.n_p_y, self.n_p_x, self.n_image = get_reshape_params(config)

    def forward(self, x):
        b,n,d = x.shape

        x = x.view(b,self.n_p_y,self.n_p_x,d)
        x = x.permute(0,3,1,2).contiguous() # [b,d,h,w]

        x = self.proj(x)
        x = F.pixel_shuffle(x, 2) # [b,4*d,h,w] -> [b,d,h*2,w*2]

        x = x.permute(0,2,3,1).contiguous() # [b,h,w,d]
        x = x.view(b,n*4,d)
        return x

class HDiTDownSample(nn.Module):
    def __init__(self, config, sample_size):
        super().__init__()

        config = deepcopy(config)
        config.sample_size = sample_size

        self.proj = weight_norm(nn.Conv2d(4*config.d_model, config.d_model, 1, 1, 0, bias = False))

        self.n_p_y, self.n_p_x, self.n_image = get_reshape_params(config)

    def forward(self, x):
        b,n,d = x.shape

        x = x.view(b,self.n_p_y,self.n_p_x,d)
        x = x.permute(0,3,1,2).contiguous() # [b,d,h,w]

        x = F.pixel_unshuffle(x, 2).contiguous() # [b,d,h,w] -> [b,4*d,h/2,w/2]
        x = self.proj(x)

        x = x.permute(0,2,3,1).contiguous() # [b,h*2,w*2,d])

        x = x.view(b,n//4,d)
        return x

class LayerMerge(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_scale = nn.Parameter(torch.zeros(1).squeeze())
    
    def forward(self, pre, post):
        alpha = self.layer_scale.clamp(0,1)
        return pre * alpha + (1. - alpha) * post

class StackedBlocks(nn.Module):
    def __init__(self, config, block_cls, n_blocks, sample_size):
        super().__init__()

        self.blocks = nn.ModuleList([])
        for i in range(n_blocks):
            self.blocks.append(block_cls(config, sample_size))

    def forward(self, x, cond):
        for block in self.blocks:
            x = block(x, cond)
        return x

class HDiT(nn.Module):
    def __init__(self, config):
        super().__init__()

        # We will hard code for 512x512 for now
        # We will assume we go to a 64x64 latent
        
        self.skip1 = LayerMerge()
        self.skip2 = LayerMerge()
        self.skip3 = LayerMerge()

        nattn_config = deepcopy(config)
        nattn_config.rope_impl = "image"

        mid_config = deepcopy(config)
        mid_config.rope_impl = "image+latent"

        # Down
        self.block1 = StackedBlocks(nattn_config, HDiTBlock, 2, 512)
        self.down1 = HDiTDownSample(nattn_config, 512)

        self.block2 = StackedBlocks(nattn_config, HDiTBlock, 2, 256)
        self.down2 = HDiTDownSample(nattn_config, 256)

        self.block3 = StackedBlocks(nattn_config, HDiTBlock, 2, 128)
        self.down3 = HDiTDownSample(nattn_config, 128)

        # Middle
        self.block4 = StackedBlocks(mid_config, HDiTBlock, 5, 64)

        # Up
        self.up1 = HDiTUpSample(nattn_config, 64)
        self.block5 = StackedBlocks(nattn_config, HDiTBlock, 2, 128)

        self.up2 = HDiTUpSample(nattn_config, 128)
        self.block6 = StackedBlocks(nattn_config, HDiTBlock, 2, 256)

        self.up3 = HDiTUpSample(nattn_config, 256)
        self.block7 = StackedBlocks(nattn_config, HDiTBlock, 2, 512)

        self.n_image = 256 # Assume latent is 64x64 p = 4

        self.proj_z = weight_norm(nn.Conv2d(
            config.d_model,
            config.d_model,
            config.latent_size,
            1,
            0,
            bias = False
        ))
        
        self.latent_size = config.latent_size
        self.n_latent = self.latent_size ** 2

    def forward(self, x, cond):
        # We assume the x is [x,z] per diffdec and [b,n,d]
        x, z = x[:,:-self.n_latent], x[:,-self.n_latent:]

        # z [b,n,d] to [b,d,l,l] -> conv -> flatten
        b,_,d = z.shape
        z_flat = z.view(b, self.latent_size, self.latent_size, d)
        z_flat = z_flat.permute(0,3,1,2).contiguous()
        z_flat = self.proj_z(z_flat) # [b,d,1,1]
        z_flat = z_flat.flatten(1) # [b,d]

        cond = cond + z_flat

        # Downsampling
        x = self.block1(x, cond)
        res1 = x.clone()
        x = self.down1(x)

        x = self.block2(x, cond)
        res2 = x.clone()
        x = self.down2(x)

        x = self.block3(x, cond)
        res3 = x.clone()
        x = self.down3(x) # -> 64

        # Middle
        x = torch.cat([x,z], dim = 1) # [b,n+d,d]
        x = self.block4(x, cond)
        x = x[:,:-self.n_latent]

        # Upsampling
        x = self.skip1(res3, self.up1(x))
        x = self.block5(x, cond)

        x = self.skip2(res2, self.up2(x))
        x = self.block6(x, cond)

        x = self.skip3(res1, self.up3(x))
        x = self.block7(x, cond)

        return x

if __name__ == "__main__":
    from ..configs import Config
    from .dit import DiT 

    cfg = Config.from_yaml("configs/feats_c128/cod_128x_feats_hdit.yml")
    model = HDiT(cfg.model).cuda().bfloat16()
    cfg.model.n_layers = 28
    cfg.model.rope_impl = "image+latent"
    baseline = DiT(cfg.model).cuda().bfloat16()

    # After flattening 
    n_p_y = cfg.model.sample_size[0] // cfg.model.patch_size
    n_p_x = cfg.model.sample_size[1] // cfg.model.patch_size

    xz = torch.randn(1, n_p_y * n_p_x + cfg.model.latent_size ** 2, cfg.model.d_model).cuda().bfloat16()
    cond = torch.randn(1, cfg.model.d_model).cuda().bfloat16()

    with torch.no_grad():
        out = model(xz, cond)
        out_baseline = baseline(xz, cond)[:,:-64]
    print(out.shape)
    print(out_baseline.shape)
    exit()