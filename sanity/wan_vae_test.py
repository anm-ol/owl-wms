import torch
import numpy as np
import os
import glob
from preproccessing.encoder import load_vae
from preproccessing.prepare_data_v2 import save_video_ffmpeg
from preproccessing.visualize import save_video_imageio

# --- CONFIGURATION ---
# Point this to your VAE checkpoint
VAE_CKPT_DIR = "preproccessing/checkpoints/Wan2.1/vae"

# Point this to one of your raw .npz data files
RAW_NPZ_PATH = "preproccessing/data_v3/train/round_001.npz" #<-- IMPORTANT: Make sure this file exists

# Point this to the corresponding cached .npy latent file
CACHED_LATENT_PATH = "preproccessing/cached_data_v3_wan/train/round_001/latents/round_001_latents.npy" #<-- IMPORTANT: Make sure this file exists

# Your per-channel stats from the config
PER_CHANNEL_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
PER_CHANNEL_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def main():
    # --- Test 1: Direct Encode -> Decode ---
    print("--- Running Test 1: Direct Encode -> Decode from Raw NPZ ---")
    
    vae = load_vae(VAE_CKPT_DIR, device=DEVICE)
    vae.eval()

    # Load raw frames
    data = np.load(RAW_NPZ_PATH)
    mask = data['valid_frames']
    end_idx = int(np.where(mask == 1)[0][-1])
    images = torch.from_numpy(data['images'][:end_idx]).float()
    
    # Prepare frames for VAE
    n_frames_to_keep = images.shape[0] - (images.shape[0] - 1) % 4
    frames_for_vae = images[:n_frames_to_keep].to(DEVICE, dtype=torch.bfloat16)
    frames_for_vae = (frames_for_vae / 255.0) * 2.0 - 1.0 # Normalize to [-1, 1]
    frames_for_vae = frames_for_vae.permute(1, 0, 2, 3).unsqueeze(0) # [t, c, h, w] -> [1, c, t, h, w]

    # Encode and Decode
    latents = vae.encode(frames_for_vae).latent_dist.sample()
    decoded_frames = vae.decode(latents).sample.squeeze(0) # [3, t, h, w]
    
    # Post-process for saving
    decoded_frames = decoded_frames.permute(1, 0, 2, 3) # [t, 3, h, w]
    video_np = ((decoded_frames.float().cpu() + 1) / 2.0 * 255.0).to(torch.uint8).numpy()
    video_np = np.transpose(video_np, (0, 2, 3, 1)) # [t, h, w, c]
    save_video_imageio(video_np, "sanity_check_1_direct_reconstruction.mp4", fps=60)
    
    del vae, latents, decoded_frames, frames_for_vae
    torch.cuda.empty_cache()

    # --- Test 2: Decode from Cached Latent ---
    print("\n--- Running Test 2: Decode from Cached .npy Latent ---")

    vae = load_vae(VAE_CKPT_DIR, device=DEVICE)
    vae.eval()

    # Load cached latents and prepare for decoder
    cached_latents = torch.from_numpy(np.load(CACHED_LATENT_PATH)).to(DEVICE, dtype=torch.bfloat16) # [c, t, h, w]
    cached_latents = cached_latents.unsqueeze(0) # [1, c, t, h, w]

    # **CRITICAL STEP**: Denormalize the latents before decoding, as you suggested.
    C = cached_latents.shape[1]
    means_t = torch.tensor(PER_CHANNEL_MEAN, device=DEVICE, dtype=torch.bfloat16).view(1, C, 1, 1, 1)
    stds_t = torch.tensor(PER_CHANNEL_STD, device=DEVICE, dtype=torch.bfloat16).view(1, C, 1, 1, 1)
    
    # The `scaling_factor` is baked into the model's output, so we must un-scale it
    # before applying the mean/std denormalization.
    # latents_for_decode = (cached_latents ) * stds_t + means_t
    latents_for_decode = (cached_latents)
    
    # Decode
    decoded_frames_cached = vae.decode(latents_for_decode).sample.squeeze(0) # [3, t, h, w]
    
    # Post-process for saving
    decoded_frames_cached = decoded_frames_cached.permute(1, 0, 2, 3) # [t, 3, h, w]
    video_np_cached = ((decoded_frames_cached.float().cpu() + 1) / 2.0 * 255.0).to(torch.uint8).numpy()
    video_np_cached = np.transpose(video_np_cached, (0, 2, 3, 1)) # [t, h, w, c]
    save_video_imageio(video_np_cached, "sanity_check_2_cached_reconstruction.mp4", fps=60)
    
if __name__ == "__main__":
    main()