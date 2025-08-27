from owl_wms.configs import Config
from owl_wms.data import get_loader
from owl_wms import from_pretrained
from owl_wms.nn.kv_cache import KVCache, StaticCache
from owl_wms.nn.rope import cast_rope_buffers_to_fp32

import torch.nn.functional as F
import torch

import random
import torch
from accelerate import init_empty_weights
import os
import time

def zlerp(x, alpha):
    return x * (1. - alpha) + alpha * torch.randn_like(x)

@torch.no_grad()
def to_bgr_uint8(frame, target_size=(1080,1920)):
    # frame is [rgb,h,w] in [-1,1]
    frame = frame.flip(0)
    frame = frame.permute(1,2,0)
    frame = (frame + 1) * 127.5
    frame = frame.clamp(0, 255).to(device='cpu',dtype=torch.uint32,memory_format=torch.contiguous_format,non_blocking=True)
    return frame

SAMPLING_STEPS = 2

class CausvidPipeline:
    def __init__(self, cfg_path="configs/dit_v4_dmd.yml", ckpt_path="vid_dit_v4_dmd_7k.pt"):
        cfg = Config.from_yaml(cfg_path)
        model_cfg = cfg.model
        train_cfg = cfg.train
        
        self.model, self.frame_decoder = from_pretrained(cfg_path, ckpt_path, True)
        self.model = self.model.core.cuda().bfloat16().eval()
        cast_rope_buffers_to_fp32(self.model)

        self.frame_decoder = self.frame_decoder.cuda().bfloat16().eval()

        # Store scales as instance variables
        self.frame_scale = train_cfg.vae_scale
        self.image_scale = train_cfg.vae_scale

        self.history_buffer = None
        self.mouse_buffer = None
        self.button_buffer = None

        self.alpha = 0.2

        #self.model = torch.compile(self.model)#, mode = 'max-autotune', dynamic = False, fullgraph = True)
        #self.frame_decoder = torch.compile(self.frame_decoder)#, mode = 'max-autotune', dynamic = False, fullgraph = True)
        
        self.device = 'cuda'
        
        self.cache = None
        self.init_buffers()

        self._initial_history_buffer = self.history_buffer.clone()
        self._initial_mouse_buffer = self.mouse_buffer.clone()
        self._initial_button_buffer = self.button_buffer.clone()
    
    def _build_cache(self):
        # Build cache similar to av_caching_v2.py
        self.model.transformer.disable_decoding()
        batch_size = 1
        init_len = self.history_buffer.size(1)
        
        # Initialize KV cache
        self.cache = StaticCache(self.model.config, max_length = init_len, batch_size = batch_size)
        self.cache.reset(batch_size)
        
        # Noise the history buffer for caching
        prev_x_noisy = zlerp(self.history_buffer, self.alpha)
        prev_t = self.history_buffer.new_full((batch_size, init_len), self.alpha)
        
        # Cache the context
        self.cache.enable_cache_updates()
        _ = self.model(
            prev_x_noisy,
            prev_t,
            self.mouse_buffer,
            self.button_buffer,
            kv_cache=self.cache
        )
        self.cache.disable_cache_updates()
        self.model.transformer.enable_decoding()

    def init_buffers(self, window_size=512):
        import random
        import os
        
        # Randomly select one of the 32 samples
        sample_idx = random.randint(0, 31)
        cache_path = f"data_cache/sample_{sample_idx}.pt"
        
        # Load cached tensors with memory mapping
        cache = torch.load(cache_path, map_location='cpu', mmap=True)
        
        # Extract tensors: vid [n,c,h,w], mouse [n,2], btn [n,11]
        vid = cache["vid"]
        mouse = cache["mouse"]
        button = cache["button"]
        
        # Get a random window from the sample
        seq_len = vid.size(0)
        if seq_len < window_size:
            raise ValueError(f"Sample {sample_idx} has length {seq_len} < window_size {window_size}")
        
        start_idx = random.randint(0, seq_len - window_size)
        end_idx = start_idx + window_size
        
        # Extract matching windows and add batch dimension
        self.history_buffer = vid[start_idx:end_idx].unsqueeze(0)  # [1,window_size,c,h,w]
        self.mouse_buffer = mouse[start_idx:end_idx].unsqueeze(0)  # [1,window_size,2]
        self.button_buffer = button[start_idx:end_idx].unsqueeze(0)  # [1,window_size,11]

        # Scale buffers (ensure they're on cuda and in bfloat16)
        self.history_buffer = self.history_buffer.cuda().bfloat16() / self.frame_scale
        self.mouse_buffer = self.mouse_buffer.cuda().bfloat16()
        self.button_buffer = self.button_buffer.cuda().bfloat16()

        self._build_cache()

    def restart_from_buffer(self):
        """Restore buffers to their initial state."""
        self.history_buffer = self._initial_history_buffer.clone()
        self.mouse_buffer = self._initial_mouse_buffer.clone()
        self.button_buffer = self._initial_button_buffer.clone()

        self._build_cache()

    @torch.no_grad()
    def __call__(self, new_mouse, new_btn):
        """
        new_mouse is [2,] bfloat16 tensor (assume cuda for both)
        new_btn is [11,] bool tensor indexing into [W,A,S,D,LSHIFT,SPACE,R,F,E,LMB,RMB] (i.e. true if key is currently pressed, false otherwise)

        return frame as [c,h,w] tensor in [-1,1]
        """
        new_mouse = new_mouse.bfloat16()
        new_btn = new_btn.bfloat16()

        # Prepare new frame inputs
        new_mouse_input = new_mouse[None,None,:]  # [1,1,2]
        new_btn_input = new_btn[None,None,:]      # [1,1,11]

        # Initialize new frame as noise
        curr_x = torch.randn_like(self.history_buffer[:,:1])  # [1,1,c,h,w]
        curr_t = torch.ones(1, 1, device=curr_x.device, dtype=curr_x.dtype)  # [1,1]

        dt = 1.0 / SAMPLING_STEPS

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        # Denoise the new frame using the cached context
        for _ in range(SAMPLING_STEPS):
            pred_v = self.model(
                curr_x,
                curr_t,
                new_mouse_input,
                new_btn_input,
                kv_cache=self.cache
            )
            
            curr_x = curr_x - dt * pred_v
            curr_t = curr_t - dt

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds

        # New frame generated, append and cache
        new_frame = curr_x  # [1,1,c,h,w]
        
        # Update history buffer (slide window)
        self.history_buffer = torch.cat([self.history_buffer[:,1:], new_frame], dim=1)
        
        # Update input buffers
        self.mouse_buffer = torch.cat([self.mouse_buffer[:,1:], new_mouse_input], dim=1)
        self.button_buffer = torch.cat([self.button_buffer[:,1:], new_btn_input], dim=1)

        # Add the new frame to cache with noise for next iteration
        new_frame_noisy = zlerp(new_frame, self.alpha)
        new_t_noisy = torch.ones_like(curr_t) * self.alpha

        self.cache.enable_cache_updates()
        _ = self.model(
            new_frame_noisy,
            new_t_noisy,
            new_mouse_input,
            new_btn_input,
            kv_cache=self.cache
        )
        self.cache.disable_cache_updates()

        # Decode frame for display
        x_to_dec = new_frame[0] * self.image_scale
        frame = self.frame_decoder(x_to_dec).squeeze()  # [c,h,w]
        frame = to_bgr_uint8(frame)
        
        return frame, elapsed_time
    

if __name__ == "__main__":
    pipe = CausvidPipeline()

    # INSERT_YOUR_CODE
    # Simple test: initialize buffers, print their shapes, run a forward pass and print output frame shape

    # Print buffer shapes
    print("history_buffer shape:", pipe.history_buffer.shape if pipe.history_buffer is not None else None)
    print("audio_buffer shape:", pipe.audio_buffer.shape if pipe.audio_buffer is not None else None)
    print("mouse_buffer shape:", pipe.mouse_buffer.shape if pipe.mouse_buffer is not None else None)
    print("button_buffer shape:", pipe.button_buffer.shape if pipe.button_buffer is not None else None)

    # Prepare dummy mouse and button input (matching last dimension of mouse/button buffer)
    mouse_shape = pipe.mouse_buffer.shape[-1] if pipe.mouse_buffer is not None else 2
    button_shape = pipe.button_buffer.shape[-1] if pipe.button_buffer is not None else 11
    import torch

    with torch.no_grad():
        dummy_mouse = torch.zeros(2).bfloat16().cuda()
        dummy_button = torch.zeros(11).bool().cuda()

        # Run a single forward pass
        frame = pipe(dummy_mouse, dummy_button)
        print("Generated frame shape:", frame.shape)
    