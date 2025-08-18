import torch
from torch import nn
import torch.nn.functional as F

from ..nn.embeddings import TimestepEmbedding, ControlEmbedding
from ..nn.attn import DiT, FinalLayer, Attn, MLP
from ..nn.normalization import rms_norm

import einops as eo
from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()


class TranslatorBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = Attn(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(rms_norm(x), block_mask=None)
        x = x + self.mlp(rms_norm(x))
        return x


"""
class _Block(nn.Module):
    def __init__(self, d, h, mlp_ratio=4):
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2  = nn.LayerNorm(d)
        self.mlp  = nn.Sequential(nn.Linear(d, d*mlp_ratio), nn.GELU(), nn.Linear(d*mlp_ratio, d))
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x

# --- unified translator (compress or expand based on G_in -> G_out) ----------

class TransformerTranslator(nn.Module):
    ""
    Input  x: [B, N_in,  C_in, H_in, W_in]
    Output y: [B, N_out, C_out, H_out, W_out]

    Pass shapes with a temporal grouping factor G:
      in_shape  = {'C': C_in,  'H': H_in,  'W': W_in,  'G': G_in}
      out_shape = {'C': C_out, 'H': H_out, 'W': W_out, 'G': G_out}

    - If G_out < G_in → compress (e.g., 4→1)
    - If G_out > G_in → expand   (e.g., 1→4)
    - If G_out = G_in → same rate (just spatial/channel remap)
    ""
    def __init__(self, in_shape, out_shape, d_model=None, n_heads=8, n_layers=2, mlp_ratio=4):
        super().__init__()

        # shapes
        self.Ci, self.Hi, self.Wi = int(in_shape['C']),  int(in_shape['H']),  int(in_shape['W'])
        self.Co, self.Ho, self.Wo = int(out_shape['C']), int(out_shape['H']), int(out_shape['W'])
        self.Gi = int(in_shape.get('G', 1))
        self.Go = int(out_shape.get('G', 1))

        # mode & grouping
        if self.Go == self.Gi:
            self.mode  = 'same'
            self.group = 1
            self.tokens_in  = self.Hi * self.Wi
            self.tokens_out = self.Ho * self.Wo
        elif self.Go < self.Gi:
            assert self.Gi % self.Go == 0, "G_in must be a multiple of G_out for compression"
            self.mode  = 'compress'
            self.group = self.Gi // self.Go                    # input frames per output frame
            self.tokens_in  = self.group * self.Hi * self.Wi
            self.tokens_out = self.Ho * self.Wo
        else:
            assert self.Go % self.Gi == 0, "G_out must be a multiple of G_in for expansion"
            self.mode  = 'expand'
            self.group = self.Go // self.Gi                    # output frames per input frame
            self.tokens_in  = self.Hi * self.Wi
            self.tokens_out = self.group * self.Ho * self.Wo

        # width & projections (channels are features)
        d = self.Co if d_model is None else int(d_model)
        assert d % n_heads == 0, "d_model must be divisible by n_heads"
        self.in_proj  = nn.Linear(self.Ci, d, bias=True)              # Ci → d (pre-attn)
        self.out_proj = nn.Identity() if d == self.Co else nn.Linear(d, self.Co, bias=True)

        # learned queries (produce target grid and, for expansion, extra frames)
        self.queries = nn.Parameter(torch.randn(self.tokens_out, d) * 0.02)

        # transformer stack
        self.blocks = nn.ModuleList([_Block(d, n_heads, mlp_ratio) for _ in range(n_layers)])
        self.ln_f   = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N_in, Ci, Hi, Wi = x.shape
        # be strict on channels, flexible on spatial (use runtime H/W)
        assert Ci == self.Ci, f"Expected C={self.Ci}, got {Ci}"

        # target frame count
        assert (N_in * self.Go) % self.Gi == 0, "Frame count not divisible given G_in/G_out"
        N_out = (N_in * self.Go) // self.Gi

        if self.mode in ('compress', 'same'):
            tokens_in = (self.group * Hi * Wi) if self.mode == 'compress' else (Hi * Wi)
            # group `group` input frames → 1 output frame
            x = x.view(B, N_out, self.group, Ci, Hi, Wi) \
                 .permute(0, 1, 2, 4, 5, 3)                               # [B,N_out,group,Hi,Wi,Ci]
            x = x.reshape(B * N_out, tokens_in, Ci)                       # [B*N_out, tokens_in, Ci]
            x = self.in_proj(x)                                           # → [B*N_out, tokens_in, d]

            q = self.queries.unsqueeze(0).expand(B * N_out, -1, -1)      # [B*N_out, tokens_out, d]
            seq = torch.cat([x, q], dim=1)                                # [B*N_out, tokens_in+out, d]
            for blk in self.blocks: seq = blk(seq)
            seq = self.ln_f(seq)

            y = seq[:, -self.tokens_out:, :]                              # take queries
            y = self.out_proj(y).reshape(B, N_out, self.Ho, self.Wo, -1)  # -1 == Co
            return y.permute(0, 1, 4, 2, 3).contiguous()                  # [B,N_out,Co,Ho,Wo]

        # expand: 1 input frame → `group` output frames
        x = x.permute(0, 1, 3, 4, 2).reshape(B * N_in, Hi * Wi, Ci)      # [B*N_in, Hi*Wi, Ci]
        x = self.in_proj(x)                                                 # → [B*N_in, tokens_in, d]

        q = self.queries.unsqueeze(0).expand(B * N_in, -1, -1)              # [B*N_in, group*Ho*Wo, d]
        seq = torch.cat([x, q], dim=1)
        for blk in self.blocks: seq = blk(seq)
        seq = self.ln_f(seq)

        y = seq[:, -self.tokens_out:, :]
        y = self.out_proj(y).reshape(B, N_in, self.group, self.Ho, self.Wo, -1)
        y = y.permute(0, 1, 2, 5, 3, 4).contiguous().view(B, N_out, -1, self.Ho, self.Wo)
        return y
"""

import einops
from ..nn.mlp import MLPCustom
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


create_block_mask = torch.compile(create_block_mask)
flex_attention = torch.compile(flex_attention)



class TransformerTranslatorBlock(nn.Module):
    def __init__(self, d_model, n_head, mlp_ratio, max_seq_len: int = 16384):
        super().__init__()
        self.mlp = MLPCustom(d_model, int(d_model * mlp_ratio), d_model)

        self.n_heads = n_head
        assert d_model % n_head == 0
        self.attn_qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)

        # rotary cache (half-truncated)
        self.head_dim = d_model // n_head
        assert self.head_dim % 4 == 0, "head_dim must be divisible by 4 for half-truncated RoPE"
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim // 4)], dim=0)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j->ij", t, angular_freq)
        self.register_buffer("pos_cos", theta.cos(), persistent=False)
        self.register_buffer("pos_sin", theta.sin(), persistent=False)

    def rope(self, x):
        # x: [B, H, T, D]
        T, D = x.size(2), x.size(-1)
        assert D % 2 == 0 and self.pos_cos.size(0) >= T, "RoPE cache too small or odd head dim"
        cos = self.pos_cos[:T].unsqueeze(0).unsqueeze(1)  # [1, 1, T, D/2]
        sin = self.pos_sin[:T].unsqueeze(0).unsqueeze(1)  # [1, 1, T, D/2]
        x1, x2 = x.to(torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), dim=-1).type_as(x)

    def attn(self, x, block_mask):
        qkv = self.attn_qkv(x)
        q, k, v = einops.rearrange(qkv, "b t (three h d) -> three b h t d", three=3, h=self.n_heads)
        q, k = rms_norm(q), rms_norm(k)
        q, k = self.rope(q), self.rope(k)
        x = flex_attention(q, k, v, block_mask=block_mask)
        x = x.permute(0, 2, 1, 3).contiguous().view(x.shape[0], x.shape[2], -1)
        return self.attn_out(x)

    def forward(self, x, block_mask):
        x = x + self.attn(rms_norm(x), block_mask)
        x = x + self.mlp(rms_norm(x))
        return x


class TransformerTranslator(nn.Module):
    """
    Attention-based frame translator.

    Input  : x [B, N, C_in, H_in, W_in]
    Output : y [B, (N//G_in)*G_out, C_out, H_out, W_out]

    Groups frames by G_in, attends over tokens, and emits G_out frames per group.
    """
    def __init__(
        self,
        in_shape: dict,      # e.g. {'C':16, 'H':60, 'W':104, 'G':1}
        out_shape: dict,     # e.g. {'C':128,'H':8, 'W':8,   'G':4}
        d_model: int = 128,
        depth: int = 4,
        nhead: int = 4,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        Cin, Hin, Win, Gin = in_shape['C'], in_shape['H'], in_shape['W'], in_shape.get('G', 1)
        Cout, Hout, Wout, Gout = out_shape['C'], out_shape['H'], out_shape['W'], out_shape.get('G', 1)

        self.Gin, self.Gout = Gin, Gout
        self.Cout, self.Hout, self.Wout = Cout, Hout, Wout

        self.tokens_in_per_group = Hin * Win * self.Gin
        self.tokens_out_per_group = Hout * Wout * self.Gout

        # Per-token projection (C_in → d_model)
        self.in_proj = nn.Linear(Cin, d_model)

        # Transformer blocks (operate on concatenated [inputs || learned outputs])
        max_seq_len = self.tokens_in_per_group + self.tokens_out_per_group
        self.blocks = nn.ModuleList([
            TransformerTranslatorBlock(d_model, n_head=nhead, mlp_ratio=mlp_ratio, max_seq_len=max_seq_len)
            for _ in range(depth)
        ])

        # Learnable output tokens per bundle (query tokens in d_model space)
        self.out_queries = nn.Parameter(torch.randn(1, self.tokens_out_per_group, d_model) * (d_model ** -0.5))

        # Project each output token to C_out
        self.out_proj = nn.Linear(d_model, Cout)

    def get_block_mask(self, in_L, out_L, device):
        S = in_L + out_L
        is_input = torch.zeros(S, dtype=torch.bool, device=device)
        is_input[:in_L] = True

        def mask_mod(b, h, q, kv):
            return ~is_input[q] | is_input[kv]

        return create_block_mask(mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C_in, H_in, W_in]
        returns: [B, (N // G_in) * G_out, C_out, H_out, W_out]
        """
        B, N, C, H, W = x.shape
        assert N % self.Gin == 0, "N must be divisible by G_in to split frames into bundles"

        # Group frames first, then flatten to a single token sequence per bundle
        tokens_in = einops.rearrange(x, 'b (n g) c h w -> (b n) (g h w) c', g=self.Gin)

        # Concatenate projected inputs | learned output embeddings
        z_in = self.in_proj(tokens_in)
        out_q = self.out_queries.expand(z_in.size(0), -1, -1)
        z = torch.cat([z_in, out_q], dim=1)

        # run encoder per bundle
        block_mask = self.get_block_mask(self.tokens_in_per_group, self.tokens_out_per_group, device=z.device)
        for blk in self.blocks:
            z = blk(z, block_mask=block_mask)
        z = rms_norm(z)

        # Keep only output tokens, project to channels, and reshape back to frames
        z_out = z[:, -self.tokens_out_per_group:, :]                 # [(B*N/Gin), Gout*Hout*Wout, d]
        z_out = self.out_proj(z_out)                                 # [(B*N/Gin), Gout*Hout*Wout, C_out]
        z_out = einops.rearrange(
            z_out, '(b n) (g ho wo) c -> b (n g) c ho wo',
            b=B, n=N // self.Gin, g=self.Gout, ho=self.Hout, wo=self.Wout
        )  # [B, (N/Gin)*Gout, C_out, H_out, W_out]
        return z_out


class GameRFTCore(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        assert config.backbone == "dit"
        self.transformer = DiT(config)

        if not config.uncond:
            self.control_embed = ControlEmbedding(config.n_buttons, config.d_model)
        self.t_embed = TimestepEmbedding(config.d_model)

        self.translate_in = TransformerTranslator(
            in_shape={'C': 16, 'H': 60, 'W': 104, 'G': 1},
            out_shape={'C': 128, 'H': 8, 'W': 8, 'G': 1},  # G: frame groups (4x as many frames)
        )
        self.translate_out = TransformerTranslator(
            in_shape={'C': 128, 'H': 8, 'W': 8, 'G': 1},
            out_shape={'C': 16, 'H': 60, 'W': 104, 'G': 1},
        )

        self.proj_in = nn.Linear(config.channels, config.d_model, bias=False)
        self.proj_out = FinalLayer(config.sample_size, config.d_model, config.channels)

        assert self.config.tokens_per_frame == self.config.sample_size**2

        self.uncond = config.uncond

    def forward(self, x, t, mouse=None, btn=None, doc_id=None, has_controls=None, kv_cache=None):
        """
        x: [b,n,c,h,w]
        t: [b,n]
        """
        x = self.translate_in(x)

        b, n, c, h, w = x.shape

        old_n = t.size(1)
        assert n % old_n == 0, "translator changed frame count by a non-integer factor"
        factor = n // old_n  # e.g., 4 when G:1->4

        t = t.repeat_interleave(factor, dim=1) if factor > 1 else t[:, :: (old_n // n)]

        t_cond = self.t_embed(t)

        if not self.uncond:
            # TODO: repeat interleave for WAN?
            ctrl_cond = self.control_embed(mouse, btn)  # [b,n,d]
            if has_controls is not None:
                ctrl_cond = torch.where(has_controls[:, None, None], ctrl_cond, torch.zeros_like(ctrl_cond))
            cond = t_cond + ctrl_cond  # [b,n,d]
        else:
            cond = t_cond

        x = eo.rearrange(x, 'b n c h w -> b (n h w) c')

        x = self.proj_in(x)
        x = self.transformer(x, cond, doc_id, kv_cache)
        x = self.proj_out(x, cond)

        x = eo.rearrange(x, 'b (n h w) c -> b n c h w', n=n, h=h, w=w)

        x = self.translate_out(x)

        return x


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
