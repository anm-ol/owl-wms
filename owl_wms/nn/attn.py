import torch
import einops
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .normalization import rms_norm
from .mlp import MLP


from .modulation import AdaLN, Gate
from .rope import get_rope_cls

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

create_block_mask = torch.compile(create_block_mask)
flex_attention = torch.compile(flex_attention)


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch_checkpoint(function, *args, **kwargs)


def create_causal_block_mask(
        n_tokens: int, tokens_per_frame: int, window_len: int | None, n_cached_tokens: int = 0, device="cpu"
):
    # Build n_tokens X n_tokens BlockMask which is causal and disallows wrapping
    assert 0 <= n_cached_tokens < n_tokens, "kv cache cannot exceed total tokens"

    frame_id = torch.arange(n_tokens, device=device, dtype=torch.int32) // tokens_per_frame
    n_frames = n_tokens // tokens_per_frame

    if window_len is None:
        window_len = n_frames

    def mask_mod(b, h, q, kv):
        abs_q = q + n_cached_tokens
        is_causal = frame_id[kv] <= frame_id[abs_q]
        is_wrap = (frame_id[abs_q] == n_frames - 1) & (frame_id[kv] == 0)
        window_mask = abs_q - kv < (window_len * tokens_per_frame)
        return is_causal & ~is_wrap & window_mask

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=n_tokens - n_cached_tokens,
        KV_LEN=n_tokens,
        device=device,
    )


class Attn(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)

        self.rope = get_rope_cls(getattr(config, "rope_impl", "ortho"))(config)

    def forward(self, x, block_mask, kv_cache=None):
        B, L, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = einops.rearrange(qkv, "b t (three h d) -> three b h t d", three=3, h=self.n_heads)
        q, k = rms_norm(q), rms_norm(k)

        # rotate new queries and keys (shared kv cache between modalities)
        offset = kv_cache.length_at(self.layer_idx) if kv_cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        # prepend cached values
        if offset > 0:
            old_k, old_v = kv_cache.get(self.layer_idx)
            k = torch.cat([old_k, k], dim=2)
            v = torch.cat([old_v, v], dim=2)

        # update cache
        if kv_cache is not None and kv_cache.should_update:
            kv_cache.update(k.clone(), v.clone(), self.layer_idx)

        attn_out = flex_attention(q, k, v, block_mask=block_mask)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(x.shape[0], L, -1)

        return self.out(attn_out)


class DiTBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        dim = config.d_model

        self.attn = Attn(config, layer_idx)
        self.mlp = MLP(config)

        self.adaln1 = AdaLN(dim)
        self.gate1 = Gate(dim)
        self.adaln2 = AdaLN(dim)
        self.gate2 = Gate(dim)

    def forward(self, x, cond, block_mask, kv_cache=None):
        residual = x
        x = self.adaln1(x, cond)
        x = self.attn(x, block_mask, kv_cache)
        x = self.gate1(x, cond)
        x = residual + x

        residual = x
        x = self.adaln2(x, cond)
        x = self.mlp(x)
        x = self.gate2(x, cond)
        x = residual + x

        return x


class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.local_layers = [(layer_idx % 4 != 0) for layer_idx in range(config.n_layers)]
        self.blocks = nn.ModuleList([DiTBlock(config, idx) for idx in range(config.n_layers)])

    def get_block_mask(self, x, kv_cache, window_len):
        if not self.config.causal:
            return None
        seq_len = x.size(1)
        offset = kv_cache.length_at(0) if kv_cache is not None else 0
        return create_causal_block_mask(
            n_tokens=seq_len + offset,
            tokens_per_frame=self.config.tokens_per_frame,
            n_cached_tokens=offset,
            window_len=window_len,
            device=x.device
        )

    def forward(self, x, cond, kv_cache=None):
        local_block_mask = self.get_block_mask(x, kv_cache, self.config.local_window)
        global_block_mask = self.get_block_mask(x, kv_cache, self.config.global_window)

        for layer_idx, block in enumerate(self.blocks):
            block_mask = local_block_mask if self.local_layers[layer_idx] else global_block_mask
            x = block(x, cond, block_mask, kv_cache)
        return x


class SkipConnection(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm = AdaLN(config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x, prev, cond):
        x = x + prev
        x = self.norm(x, cond)
        x = self.proj(x)

        return x

class UViT(nn.Module):
    get_block_mask = DiT.get_block_mask

    def __init__(self, config):
        super().__init__()

        self.tokens_per_frame = config.tokens_per_frame
        self.causal = config.causal

        blocks = []
        for i in range(config.n_layers):
            blocks.append(DiTBlock(config))
            blocks[-1].attn.layer_ind = i

        self.blocks = nn.ModuleList(blocks)

        # For odd number of layers, need linear projections for skip connections
        n_skip_connections = config.n_layers // 2
        skip_projs = []
        for _ in range(n_skip_connections):
            skip_projs.append(SkipConnection(config))
        self.skip_projs = nn.ModuleList(skip_projs)

    def forward(self, x, cond, kv_cache = None):
        block_mask = self.get_block_mask(x, kv_cache)

        # Cache early block outputs for skip connections
        early_features = []
        n_blocks = len(self.blocks)
        mid_idx = n_blocks // 2

        # Early blocks
        for i in range(mid_idx):
            x = self.blocks[i](x, cond, block_mask, kv_cache)
            early_features.append(x)

        # Middle block (if odd number of layers)
        x = self.blocks[mid_idx](x, cond, block_mask, kv_cache)

        # Late blocks with skip connections
        for i in range(mid_idx + 1, n_blocks):
            # Get corresponding early block output
            early_idx = n_blocks - 1 - i
            early_feat = early_features[early_idx]

            # Concatenate early and current features
            skip_idx = i - (mid_idx + 1)
            x = self.skip_projs[skip_idx](x, early_feat, cond)

            # Block
            x = self.blocks[i](x, cond, block_mask, kv_cache)
        return x

# === VIT Specific Layers ===

class FinalLayer(nn.Module):
    def __init__(self, sample_size, d_model, channels = 3, patch_size=1):
        super().__init__()

        self.norm = AdaLN(d_model)
        self.act = nn.SiLU()
        self.proj = nn.Linear(d_model, channels*patch_size*patch_size)

    def forward(self, x, cond):
        x = self.norm(x, cond)
        x = self.act(x)
        x = self.proj(x)

        return x

def test_attn_mask():
    total_tokens = 64
    tokens_per_frame = 8
    device = "cpu"

    block_mask = create_causal_block_mask(total_tokens, tokens_per_frame, device=device)

    # Convert to dense grid
    idx = torch.arange(total_tokens, device=device, dtype=torch.int32)
    bool_mask = block_mask.mask_mod(0, 0, idx[:, None], idx[None, :])
    dense_mask = torch.where(
        bool_mask, torch.tensor(0., device=device),
        torch.tensor(float("-inf"), device=device)
    )

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    plt.imshow(dense_mask.float().cpu().numpy(), cmap='gray')
    plt.colorbar()
    plt.title(f'Block Causal Mask ({total_tokens} tokens, {tokens_per_frame} per frame)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.savefig('test_mask.png')
    plt.close()


@torch.no_grad()
def test_kv_cache():
    from .kv_cache import KVCache
    from ..configs import TransformerConfig

    # Create test configs
    config = TransformerConfig(
        n_layers=2,
        n_heads=8,
        d_model=64,
        tokens_per_frame=8
    )

    # Create model and cache
    model = DiT(config).cuda()
    cache = KVCache(config)
    cache.to('cuda')

    # Create dummy inputs
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, config.d_model).cuda()
    cond = torch.randn(batch_size, seq_len//config.tokens_per_frame, config.d_model).cuda()

    # Test forward pass with cache
    cache.reset(batch_size)
    cache.enable_cache_updates()

    # First forward pass should populate cache
    out1 = model(x, cond, cache)

    # Second pass should use cached values
    cache.disable_cache_updates()
    out2 = model(x, cond, cache)

    # Outputs should match
    print("Max difference between outputs:", torch.max(torch.abs(out1 - out2)).item())
    print("Cache test complete")

if __name__ == "__main__":
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
    test_attn_mask()
