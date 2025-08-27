import torch
from ..configs import TransformerConfig


def KVCache(config : TransformerConfig):
    if config.backbone in ("dit", "mmdit"):
        return SingleKVCache(config)
    else:
        raise ValueError(f"Invalid backbone: {config.backbone}")


class SingleKVCache:
    def __init__(self, config: TransformerConfig):
        self.config = config

        self.cache = None
        self.device = 'cuda'
        self.dtype = torch.bfloat16

        self.should_update = False

        self.noise_caches = 0.0
        self.offsets = [0] * self.config.n_layers

    def enable_cache_updates(self):
        self.should_update = True

    def disable_cache_updates(self):
        self.should_update = False

    def to(self, device = 'cuda', dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype
        return self

    def reset(self, batch_size = 1):
        shape = (batch_size, self.config.n_heads, 0, self.config.d_model//self.config.n_heads)
        dummy = torch.empty(*shape, device = self.device, dtype = self.dtype)
        self.cache = [[torch.empty_like(dummy), torch.empty_like(dummy)] for _ in range(self.config.n_layers)]
        self.offsets = [0] * self.config.n_layers

    def get(self, layer_ind, new_k = None, new_v = None):
        assert self.cache is not None, "Must reset cache before using"
        k,v = self.cache[layer_ind]
        if new_k is not None:
            k = torch.cat([k, new_k], dim=2)
            v = torch.cat([v, new_v], dim=2)
        return k,v

    def update(self, new_k, new_v, layer_ind):
        assert self.cache is not None, "Must reset cache before using"

        new_len = new_k.shape[2]

        self.offsets[layer_ind] += new_len

        self.cache[layer_ind] = [
            torch.cat([self.cache[layer_ind][0], new_k], dim=2),
            torch.cat([self.cache[layer_ind][1], new_v], dim=2)
        ]

    def eject(self):
        for i in range(self.config.n_layers):
            self.cache[i][0] = self.cache[i][0][:,:,self.config.tokens_per_frame:]
            self.cache[i][1] = self.cache[i][1][:,:,self.config.tokens_per_frame:]

    def length_at(self, idx):
        return self.cache[idx][0].shape[2]
    
    def get_offset(self, idx=0):
        return self.offsets[idx]

    def __len__(self):
        assert self.cache is not None, "Must reset cache before using"
        return self.cache[0][0].shape[2]

    def n_frames(self):
        assert len(self) % self.config.tokens_per_frame == 0
        return len(self) // self.config.tokens_per_frame
    
    def clone(self):
        # Clones all tensors for max-autotune to work properly
        for i in range(self.config.n_layers):
            self.cache[i] = (self.cache[i][0].clone(), self.cache[i][1].clone())
        return self
    
    def detach(self):
        for i in range(self.config.n_layers):
            self.cache[i] = (self.cache[i][0].detach(), self.cache[i][1].detach())
        return self

    @property
    def shape(self):
        return self.cache[0][0].shape

class StaticCache:
    def __init__(
        self,
        config: TransformerConfig,
        max_length = 120,
        batch_size = 1,
        device = 'cuda',
        dtype = torch.bfloat16
    ):
        self.config = config

        self.cache = None
        self.device = 'cuda'
        self.dtype = torch.bfloat16

        self.should_update = False

        self.max_length = max_length
        self.batch_size = batch_size

        self.k_cache = [torch.empty(
            batch_size,
            config.n_heads,
            max_length * config.tokens_per_frame,
            config.d_model//config.n_heads,
            device = device,
            dtype = dtype
        ) for _ in range(config.n_layers)]
        self.v_cache = [torch.empty(
            batch_size,
            config.n_heads,
            max_length * config.tokens_per_frame,
            config.d_model//config.n_heads,
            device = device,
            dtype = dtype
        ) for _ in range(config.n_layers)]

        self.tokens_per_frame = config.tokens_per_frame
        self.offsets = [0] * self.config.n_layers

    def enable_cache_updates(self):
        self.should_update = True

    def disable_cache_updates(self):
        self.should_update = False

    def to(self, device = 'cuda', dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype
        return self

    def reset(self, batch_size = 1):
        self.offsets = [0] * self.config.n_layers

    def get(self, layer_ind, new_k = None, new_v = None):
        if new_k is not None and new_v is not None:
            old_k = self.k_cache[layer_ind].clone()
            old_v = self.v_cache[layer_ind].clone()

            old_k = torch.roll(old_k, shifts = -new_k.shape[2], dims = 2)
            old_v = torch.roll(old_v, shifts = -new_v.shape[2], dims = 2)

            old_k[:,:,-new_k.shape[2]:] = new_k
            old_v[:,:,-new_v.shape[2]:] = new_v

            return old_k, old_v
        else:
            return self.k_cache[layer_ind], self.v_cache[layer_ind]

    def update(self, new_k, new_v, layer_ind):
        new_len = new_k.shape[2]
        if new_len > self.tokens_per_frame: # More than one frame, full cache
            self.k_cache[layer_ind][:,:,:,:new_len] = new_k
            self.v_cache[layer_ind][:,:,:,:new_len] = new_v
            self.offsets[layer_ind] = new_len
        else: # step forward one
            self.k_cache[layer_ind] = torch.roll(self.k_cache[layer_ind], shifts = -new_len, dims = 2)
            self.v_cache[layer_ind] = torch.roll(self.v_cache[layer_ind], shifts = -new_len, dims = 2)
            self.k_cache[layer_ind][:,:,-new_len:] = new_k
            self.v_cache[layer_ind][:,:,-new_len:] = new_v
            self.offsets[layer_ind] += new_len

    def truncate(self, truncate_amt=1, front = False):
        """
        Truncate/eject frames from the KV cache
        This isn't needed in a static cache
        """
        return
    
    def get_offset(self, idx=0):
        return self.offsets[idx]

    def clone(self):
        # Clones all tensors for max-autotune to work properly
        for i in range(self.config.n_layers):
            self.k_cache[i] = self.k_cache[i].clone()
            self.v_cache[i] = self.v_cache[i].clone()
        return self
    
    def detach(self):
        for i in range(self.config.n_layers):
            self.k_cache[i] = self.k_cache[i].detach()
            self.v_cache[i] = self.v_cache[i].detach()
        return self

    @property
    def shape(self):
        return self.k_cache[0].shape