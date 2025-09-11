import time
import torch
from owl_wms.models.tekken_rft_v2 import TekkenRFTV2, action_id_to_buttons
from owl_wms.sampling.av_caching_v2 import AVCachingSamplerV2
from owl_wms.sampling.t3_action_v2 import TekkenActionCachingV2
from owl_wms.data.tekken_latent_multi import TekkenLatentMulti
import warnings

def time_function(func, *args, **kwargs):
    """Time a function call with CUDA synchronization"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = func(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    return result, (end - start) * 1000  # Return result and time in ms

# Quick test of action_id_to_buttons
action_ids_sample = action_ids[:, :1]  # Single frame
result, elapsed = time_function(action_id_to_buttons, action_ids_sample)
print(f"action_id_to_buttons: {elapsed:.2f}ms")

# Test model forward pass
x_sample = torch.randn_like(x[:, :1])
t_sample = torch.ones(1, 1)
buttons_sample = action_id_to_buttons(action_ids[:, :1])

result, elapsed = time_function(model, x_sample, t_sample, buttons_sample)
print(f"Model forward pass: {elapsed:.2f}ms")

# Compare with AV sampler
av_sampler = AVCachingSamplerV2(n_steps=16, num_frames=5)
tekken_sampler = TekkenActionCachingV2(n_steps=16, num_frames=5)

# Time both (use small num_frames for quick comparison)
_, av_time = time_function(av_sampler, model_av, x, mouse, btn)
_, tekken_time = time_function(tekken_sampler, model_tekken, x, action_ids)

print(f"AV Sampler: {av_time:.2f}ms")
print(f"Tekken Sampler: {tekken_time:.2f}ms")
print(f"Slowdown: {tekken_time/av_time:.1f}x")