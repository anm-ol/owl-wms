from owl_wms.nn.attn import get_block_mask, DiT
import torch


def test_attn_mask():
    total_tokens = 64
    tokens_per_frame = 8
    device = "cpu"

    block_mask = get_block_mask(total_tokens, tokens_per_frame, device=device)

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
