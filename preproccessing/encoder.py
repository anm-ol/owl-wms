import torch
from diffusers import AutoencoderKLLTXVideo, AutoencoderKLWan
import gc
import time
import argparse
import json
import os

# Specify the model repository ID and the device
ckpt_dir = "checkpoints/wan-vae"
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_encoder_cls(ckpt_dir):
    # load class from the config in the checkpoint directory
    config_path = os.path.join(ckpt_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    class_name = config.get("_class_name", "AutoencoderKLLTXVideo")
    
    if class_name == "AutoencoderKLWan":
        return AutoencoderKLWan
    elif class_name == "AutoencoderKLLTXVideo":
        return AutoencoderKLLTXVideo
    else:
        raise ValueError(f"Unknown autoencoder class: {class_name}")
    
def load_vae(ckpt_dir, dtype=torch.bfloat16, device=None):
    # Get the appropriate autoencoder class
    autoencoder_cls = get_encoder_cls(ckpt_dir)
    
    vae = autoencoder_cls.from_pretrained(
        ckpt_dir,
        # subfolder="wan-vae",
        torch_dtype=dtype
    )
    vae.to(device)
    print(f"✅ VAE loaded from {ckpt_dir} on {device}.")
    num_params = sum(p.numel() for p in vae.parameters())
    print(f"✅ VAE loaded with {num_params/1e6:.1f}M parameters.")
    return vae


def load_encoder(ckpt_dir, device):
    print(f"Loading VAE from {ckpt_dir}...")
    
    # Get the appropriate autoencoder class
    autoencoder_cls = get_encoder_cls(ckpt_dir)
    
    vae = autoencoder_cls.from_pretrained(
        ckpt_dir,
        # subfolder="wan-vae",
        torch_dtype=torch.bfloat16
    )

    # 2. Isolate the encoder and move it to the desired device
    encoder = vae.encoder.to(device).eval()

    # 3. IMPORTANT: Delete the full VAE object to free the decoder from memory
    del vae
    gc.collect() # Ask Python's garbage collector to run
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Ask PyTorch to release cached memory
    
    # (Optional) Freeze the encoder's parameters
    for param in encoder.parameters():
        param.requires_grad = False

    print("\n✅ VAE Encoder is now isolated in memory.")
    
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"✅ VAE Encoder loaded with {num_params/1e6:.1f}M parameters.")
    print("Waiting for 10 seconds before terminating...")
    time.sleep(10)
    
    return encoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE Encoder Isolation Script")
    parser.add_argument("--ckpt_dir", type=str, default=ckpt_dir, help="Path to the VAE checkpoint directory")
    parser.add_argument("--device", type=str, default='cuda', help="Device to load the VAE encoder on (e.g., 'cuda' or 'cpu')")
    args = parser.parse_args()
    
    encoder = load_encoder(args.ckpt_dir, args.device)