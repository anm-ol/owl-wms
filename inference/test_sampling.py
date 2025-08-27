from owl_wms.data import get_loader
from owl_wms.configs import Config
from owl_wms import from_pretrained

from owl_wms.utils import batch_permute_to_length
from owl_wms.utils.owl_vae_bridge import make_batched_decode_fn
from owl_wms.utils.logging import LogHelper, to_wandb_av
from owl_wms.nn.rope import RoPE

import torch
import random

cfg_path = "configs/dit_v4_dmd.yml"
ckpt_path = "vid_dit_v4_dmd_7k.pt"

model, decoder = from_pretrained(cfg_path, ckpt_path, return_decoder=True)
model = model.core.eval().cuda().bfloat16()

# Find any RoPE modules in the model and cast their cos and sin back to fp32
def cast_rope_buffers_to_fp32(module):
    for submodule in module.modules():
        if isinstance(submodule, RoPE):
            if hasattr(submodule, "cos"):
                submodule.cos = submodule.cos.float()
            if hasattr(submodule, "sin"):
                submodule.sin = submodule.sin.float()

cast_rope_buffers_to_fp32(model)

decoder = decoder.eval().cuda().bfloat16()
#decoder = torch.compile(decoder)

cfg = Config.from_yaml(cfg_path)
train_cfg = cfg.train

print("Done Model Loading")

decode_fn = make_batched_decode_fn(decoder, train_cfg.vae_batch_size)

print("Done Decoder Loading")

import wandb

wandb.init(
    project="video_models",
    entity="shahbuland",
    name="video_dit_v4"
)

from owl_wms.sampling import get_sampler_cls

import os

sampler = get_sampler_cls(train_cfg.sampler_id)(**train_cfg.sampler_kwargs)

cache_path = "data_cache_single.pt"

# Use the same cache loading approach as CausvidPipeline
sample_idx = random.randint(0, 31)
cache_path = f"data_cache/sample_{sample_idx}.pt"

print(f"Loading cached data from {cache_path}")
cache = torch.load(cache_path, map_location='cpu', mmap=True)

# Extract tensors: vid [n,c,h,w], mouse [n,2], btn [n,11]
vid_full = cache["vid"]
mouse_full = cache["mouse"]
button_full = cache["button"]

# Get sequence length
seq_len = vid_full.size(0)

# Setup similar to CausvidPipeline ground truth mode
window_size = 240  # History window size
future_size = sampler.num_frames  # Number of future steps needed
required_len = window_size + future_size

if seq_len < required_len:
    raise ValueError(f"Sample {sample_idx} has length {seq_len} < required_len {required_len}")

start_idx = random.randint(0, seq_len - required_len)
history_end_idx = start_idx + window_size
future_end_idx = start_idx + required_len

# Extract history windows and add batch dimension
vid = vid_full[start_idx:history_end_idx].unsqueeze(0)  # [1,window_size,c,h,w]
mouse_history = mouse_full[start_idx:history_end_idx].unsqueeze(0)  # [1,window_size,2]
btn_history = button_full[start_idx:history_end_idx].unsqueeze(0)  # [1,window_size,11]

# Extract future controls
mouse_future = mouse_full[history_end_idx:future_end_idx].unsqueeze(0)  # [1,future_size,2]
btn_future = button_full[history_end_idx:future_end_idx].unsqueeze(0)  # [1,future_size,11]

# Combine history and future for full sequence
mouse = torch.cat([mouse_history, mouse_future], dim=1)  # [1, window_size + future_size, 2]
btn = torch.cat([btn_history, btn_future], dim=1)  # [1, window_size + future_size, 11]

vid = vid.cuda().bfloat16()
mouse = mouse.cuda().bfloat16()
btn = btn.cuda().bfloat16()

print(f"Loaded sample {sample_idx}: vid shape {vid.shape}, mouse shape {mouse.shape}, btn shape {btn.shape}")

with torch.no_grad():

    latent_vid = sampler(model, vid, mouse, btn, compile_on_decode = True)

    latent_vid = latent_vid[:1, vid.size(1):]
    mouse = mouse[:1, vid.size(1):]
    btn = btn[:1, vid.size(1):]

    del model

    video = decode_fn(latent_vid * train_cfg.vae_scale)

wandb_av_out = to_wandb_av(video, None, mouse, btn)

if len(wandb_av_out) == 3:
    video, depth_gif, flow_gif = wandb_av_out
    eval_wandb_dict = dict(samples=video, depth_gif=depth_gif, flow_gif=flow_gif)
elif len(wandb_av_out) == 2:
    video, depth_gif = wandb_av_out
    eval_wandb_dict = dict(samples=video, depth_gif=depth_gif)
else:
    eval_wandb_dict = dict(samples=wandb_av_out)

wandb.log(eval_wandb_dict)

print("Done")